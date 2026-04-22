# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sphinx extension that adds snippet markers and templates to doctest directives.

Snippet markers (``# begin-snippet`` / ``# end-snippet``):
    Mark which lines of a ``.. testcode::`` block are rendered. The full code
    runs during doctest, but only the snippet region is displayed.

Templates (``.. testcode:: \\n   :template: name.py``):
    It is expanded into the template (replacing ``{{body}}``) to produce the
    full executable code. Templates live in ``_templates/doctest/``.

When snippet markers exist, the output uses CSS-only tabs to show a
"Snippet" tab (default) and a "Complete example" tab.

fd-level output capture:
    GPU ``printf`` (used by cuTile's in-kernel ``print``) writes directly to
    file descriptor 1, bypassing Python's ``sys.stdout``.  The standard
    doctest runner only captures ``sys.stdout``, so GPU output is invisible.
    This extension replaces the doctest runner with one that redirects fd 1
    to a temporary file before each ``run()`` call.  The runner's ``_fakeout``
    is swapped for an ``_FdCaptureOut`` whose ``write()`` goes through
    ``os.write(1, ...)`` and whose ``getvalue()`` reads back from the temp
    file after ``fflush(NULL)``.  Because ``doctest.DocTestRunner.run()``
    sets ``sys.stdout = self._fakeout``, both Python ``print()`` and GPU
    ``printf`` land on fd 1 and are captured together.
"""

import ctypes
import os
import re
import tempfile
import textwrap
from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.ext.doctest import (
    SphinxDocTestRunner,
    TestcodeDirective,
    TestoutputDirective,
    setup as doctest_setup,
)

from sphinx.util import logging as sphinx_logging
logger = sphinx_logging.getLogger(__name__)

_libc = ctypes.CDLL(None)


class _FdCaptureOut:
    """Drop-in for doctest's ``_fakeout`` that captures output via fd 1.

    ``doctest.DocTestRunner.run()`` sets ``sys.stdout = self._fakeout``.
    With this object as ``_fakeout``, every ``print()`` call goes through
    ``write()`` -> ``os.write(1, ...)`` -> the temp file that fd 1 has been
    redirected to.  GPU ``printf`` also writes to fd 1 directly.
    ``getvalue()`` flushes C stdio and reads the temp file.
    """

    encoding = 'utf-8'

    def __init__(self, tmpfile):
        self._tmpfile = tmpfile

    def write(self, s):
        data = s.encode('utf-8') if isinstance(s, str) else s
        os.write(1, data)
        return len(s)

    def flush(self):
        pass

    def getvalue(self):
        _libc.fflush(None)
        self._tmpfile.seek(0)
        return self._tmpfile.read().decode('utf-8', errors='replace')

    def truncate(self, size=0):
        self._tmpfile.seek(0)
        self._tmpfile.truncate(0)


class _TerminalLogger:
    """Logger proxy that writes directly to a terminal fd.

    Sphinx's logger writes via handlers attached to ``sys.stdout``, which is
    backed by fd 1.  When fd 1 is redirected for GPU output capture, logger
    output would go to the temp file instead of the terminal.  This proxy
    routes all logging methods to the saved terminal fd.
    """

    def __init__(self, real_logger, terminal_fd):
        self._real = real_logger
        self._terminal = os.fdopen(os.dup(terminal_fd), 'w')

    def _write(self, msg, nonl=False):
        self._terminal.write(str(msg))
        if not nonl:
            self._terminal.write('\n')
        self._terminal.flush()

    def info(self, msg, *args, nonl=False, **kwargs):
        self._write(msg, nonl=nonl)

    def warning(self, msg, *args, **kwargs):
        self._write(msg)

    def debug(self, msg, *args, **kwargs):
        self._write(msg)

    def error(self, msg, *args, **kwargs):
        self._write(msg)

    def critical(self, msg, *args, **kwargs):
        self._write(msg)


class FdCaptureDocTestRunner(SphinxDocTestRunner):
    """DocTestRunner that captures fd-level output (GPU printf).

    Redirects fd 1 to a temp file for the duration of ``run()``.
    Patches ``sphinx.ext.doctest.logger`` so diagnostic output
    (failure reports) reaches the terminal instead of the temp file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_stats = {}

    def summarize(self, out, verbose=None):
        totalf = totalt = 0
        lines = []
        only_default = set(self._test_stats.keys()) <= {'default'}
        for name, (f, t) in sorted(self._test_stats.items()):
            totalf += f
            totalt += t
            if not only_default:
                lines.append(f'{name}: {t - f} passed and {f} failed.\n')
        if totalt == 0:
            out('')
            return 0, 0
        if only_default:
            lines.append(f'{totalt - totalf} passed and {totalf} failed.\n')
        if totalf:
            lines.append(f'***Test Failed*** {totalf} failures.\n')
        out(''.join(lines))
        return totalf, totalt

    def run(self, test, compileflags=None, out=None, clear_globs=True):
        import sphinx.ext.doctest as _mod

        tmpfile = tempfile.TemporaryFile(mode='w+b')
        saved_fd = os.dup(1)
        os.dup2(tmpfile.fileno(), 1)
        save_fakeout = self._fakeout
        self._fakeout = _FdCaptureOut(tmpfile)
        orig_logger = _mod.logger
        _mod.logger = _TerminalLogger(orig_logger, saved_fd)
        try:
            result = super().run(test, compileflags, out, clear_globs)
            f, t = self._test_stats.get(test.name, (0, 0))
            self._test_stats[test.name] = (f + result.failed, t + result.attempted)
            return result
        finally:
            _mod.logger = orig_logger
            self._fakeout = save_fakeout
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
            tmpfile.close()


BEGIN_MARKER = re.compile(r'^\s*#\s*begin-snippet\s*$')
END_MARKER = re.compile(r'^\s*#\s*end-snippet\s*$')


def _extract_snippet(code) -> tuple[str, str, bool]:
    """Return (snippet_code, full_code, had_markers).

    snippet_code: only lines between markers.
    full_code: all lines with marker comments removed.
    """
    lines = code.split('\n')
    snippet_lines = []
    full_lines = []
    inside = False
    had_markers = False
    for line in lines:
        if BEGIN_MARKER.match(line):
            inside = True
            had_markers = True
            continue
        if END_MARKER.match(line):
            inside = False
            continue
        full_lines.append(line)
        if inside:
            snippet_lines.append(line)
    if not had_markers:
        return code, code, False
    return '\n'.join(snippet_lines), '\n'.join(full_lines), True


def _load_template(env, template_name):
    """Load a template file from _templates/doctest/.

    Returns (content, path) so the caller can register the dependency.
    """
    srcdir = env.srcdir
    path = os.path.join(srcdir, '_templates', 'doctest', template_name)
    with open(path) as f:
        return f.read(), path


def _expand_template(template, body):
    """Replace ``{{body}}`` placeholder in *template*.

    The indentation of the ``{{body}}`` placeholder line determines the
    indentation applied to every line of *body*.
    """
    lines = template.split('\n')
    result = []
    for line in lines:
        match = re.match(r'^(\s*)\{\{body\}\}\s*$', line)
        if match:
            indent = match.group(1)
            for bline in body.split('\n'):
                result.append(indent + bline if bline.strip() else bline)
        else:
            result.append(line)
    return '\n'.join(result)


def _make_code_block(code, groups=None, source_info=None, node_type=None):
    kwargs = dict(language='python')
    if node_type is not None:
        kwargs['testnodetype'] = node_type
        kwargs['groups'] = groups or ['default']
    node = nodes.literal_block(code, code, **kwargs)
    node['options'] = {}
    if source_info:
        node.source, node.line = source_info
    return node


def _make_tabs(env, snippet_block, full_block):
    """Build a CSS-only tabbed container with Snippet and Complete blocks.

    Uses radio inputs + labels so no JavaScript is needed.
    The Complete example tab's literal_block carries ``testnodetype`` so the
    doctest builder discovers and executes it.
    """
    if "next_tabs_id" not in env.temp_data:
        env.temp_data["next_tabs_id"] = 0
    tabs_id = env.temp_data["next_tabs_id"]
    env.temp_data["next_tabs_id"] += 1

    name = f"snippet-tab-{tabs_id}"
    id_snippet = f"snippet-tab-{tabs_id}-snippet"
    id_full = f"snippet-tab-{tabs_id}-full"

    container = nodes.container()
    container["classes"].append("snippet-tabs")

    header_html = (
        f'<input type="radio" name="{name}" id="{id_snippet}" checked>'
        f'<label for="{id_snippet}">Snippet</label>'
        f'<input type="radio" name="{name}" id="{id_full}">'
        f'<label for="{id_full}">Complete Example</label>'
    )
    container += nodes.raw('', header_html, format='html')

    panel1 = nodes.container()
    panel1["classes"].append("snippet-tab-panel")
    panel1 += snippet_block
    container += panel1

    panel2 = nodes.container()
    panel2["classes"].append("snippet-tab-panel")
    panel2["classes"].append("snippet-tab-copyable")
    copy_btn = (
        '<button class="snippet-copy-btn" title="Copy" '
        'onclick="let c=this.parentNode.querySelector(\'pre\').textContent;'
        'navigator.clipboard.writeText(c);'
        'this.textContent=\'Copied!\';'
        'setTimeout(()=>this.textContent=\'Copy\',1500)">'
        'Copy</button>'
    )
    panel2 += nodes.raw('', copy_btn, format='html')
    panel2 += full_block
    container += panel2

    return container


class SnippetTestcodeDirective(TestcodeDirective):
    """Testcode directive that supports snippet markers and templates."""

    option_spec = {
        **TestcodeDirective.option_spec,
        'template': directives.unchanged_required,
    }

    def _get_groups(self):
        if self.arguments:
            return [x.strip() for x in self.arguments[0].split(',')]
        return ['default']

    def _source_info(self):
        return self.state_machine.get_source_and_line(self.lineno)

    def run(self):
        env = self.state.document.settings.env
        template_opt = self.options.get('template')

        if template_opt:
            snippet = textwrap.dedent('\n'.join(self.content))
            template, template_path = _load_template(env, template_opt)
            env.note_dependency(template_path)
            code = _expand_template(template, snippet)
        else:
            code = '\n'.join(self.content)

        snippet, full, had_markers = _extract_snippet(code)
        full = textwrap.dedent(full)
        self.content = full.split('\n')

        if not had_markers:
            return super().run()

        snippet = textwrap.dedent(snippet)

        snippet_block = _make_code_block(snippet, groups=None, source_info=None, node_type=None)
        full_block = _make_code_block(full, self._get_groups(),
                                      source_info=self._source_info(),
                                      node_type="testcode")

        tabs = _make_tabs(env, snippet_block, full_block)
        return [tabs]


class LabeledTestoutputDirective(TestoutputDirective):
    """Testoutput directive that prepends an 'Output:' label."""

    def run(self):
        result = super().run()
        label = nodes.paragraph('', 'Output')
        label['classes'].append('testoutput-label')
        return [label] + result


_SINGLE_COLON = re.compile(r'^(\s*)\.\.\s+(testcode|testoutput):?\s*$')
_DIRECTIVE = re.compile(r'^(\s*)\.\.\s+(testcode|testoutput)::')


def _lint_lines(lines, location):
    prev_indent = 0
    for i, line in enumerate(lines, 1):
        if _SINGLE_COLON.match(line):
            logger.warning(
                "malformed directive (missing colon): %s",
                line.strip(), location=location,
            )
        m = _DIRECTIVE.match(line)
        if m:
            indent = len(m.group(1))
            if indent > prev_indent:
                logger.warning(
                    "directive indented deeper than surrounding text "
                    "(will render as blockquote): %s",
                    line.strip(), location=location,
                )
            if i < len(lines) and lines[i - 1 + 1].strip():
                # options like :template: are indented under the directive
                if not re.match(r'^\s+:', lines[i]):
                    logger.warning(
                        "missing blank line after directive: %s",
                        line.strip(), location=location,
                    )
        if line.strip():
            prev_indent = len(line) - len(line.lstrip())


def _lint_source_read(app, docname, source):
    _lint_lines(source[0].splitlines(), location=docname)


def _lint_docstring(app, what, name, obj, options, lines):
    _lint_lines(lines, location=name)


def setup(app):
    doctest_setup(app)
    app.connect('source-read', _lint_source_read)
    app.connect('autodoc-process-docstring', _lint_docstring)
    app.add_directive('testcode', SnippetTestcodeDirective, override=True)
    app.add_directive('testoutput', LabeledTestoutputDirective, override=True)

    import sphinx.ext.doctest as _doctest_mod
    _doctest_mod.SphinxDocTestRunner = FdCaptureDocTestRunner

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': False,
    }
