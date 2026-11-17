# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import inspect
import json
import os
from pathlib import Path
import subprocess
import warnings
from functools import cache
from typing import Any, Callable
import torch


from cuda.tile.tune import TuningResult


SCHEMA_VERSION = 1
TUNE_FILE = Path(__file__).with_name("benchmark_tuning.json")
GPU_NAME_ENV_VAR = "CUDA_TILE_BENCHMARK_GPU_NAME"
SAVE_TUNING_HINT = (
    "Run `pytest --tune --save-tuning ...` to tune and save configs; "
)


@cache
def get_gpu_name() -> str:
    override = os.environ.get(GPU_NAME_ENV_VAR)
    if override is not None:
        override = override.strip()
        if not override:
            raise RuntimeError(f"{GPU_NAME_ENV_VAR} must not be empty")
        return override

    command = [
        "nvidia-smi",
        "-i",
        "0",
        "--query-gpu=name",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.check_output(command, text=True)
    except Exception as exc:
        rendered = " ".join(command)
        raise RuntimeError(f"Failed to query benchmark tuning GPU with `{rendered}`") from exc
    return result.strip()


def _benchmark_name_from_tune_fn(tune_fn: Callable[..., Any]) -> str:
    name = tune_fn.__name__
    if not name.startswith("tune_"):
        raise ValueError(f"Tuning function must be named tune_<benchmark>, got {name}")
    return name[len("tune_"):]


def tune_call_kwargs(tune_fn: Callable[..., Any],
                     available_kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(tune_fn)
    kwargs: dict[str, Any] = {}
    fn_label = tune_fn.__name__
    for name, param in sig.parameters.items():
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(f"{fn_label} only supports named parameters, got {param}")
        if name in available_kwargs:
            kwargs[name] = available_kwargs[name]
        elif param.default is not inspect.Parameter.empty:
            kwargs[name] = param.default
        else:
            raise KeyError(f"Missing parameter {name} required by {fn_label}")
    return kwargs


def _canonicalize(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.Size):
        return [_canonicalize(v) for v in value]
    if isinstance(value, tuple):
        return [_canonicalize(v) for v in value]
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    if isinstance(value, dict):
        return {
            str(k): _canonicalize(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported value in benchmark tuning key/config: {value!r}")


def _tuning_params(tune_fn: Callable[..., Any],
                   available_kwargs: dict[str, Any]) -> dict[str, Any]:
    params = _canonicalize(tune_call_kwargs(tune_fn, available_kwargs))
    if not isinstance(params, dict):
        raise TypeError("Tuning params must be a JSON-serializable dict")
    return params


def _empty_database() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "gpus": {},
    }


def _load_database() -> dict[str, Any]:
    if not TUNE_FILE.exists():
        return _empty_database()

    with TUNE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported benchmark tuning schema in {TUNE_FILE}: "
            f"{data.get('schema_version')!r}"
        )
    if "gpus" not in data or not isinstance(data["gpus"], dict):
        raise RuntimeError(f"Invalid benchmark tuning database in {TUNE_FILE}: missing gpus")
    return data


def _save_database(data: dict[str, Any]) -> None:
    TUNE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = TUNE_FILE.with_suffix(TUNE_FILE.suffix + ".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_file, TUNE_FILE)


def _result_entry(result: TuningResult[Any]) -> dict[str, Any]:
    config = _canonicalize(result.best.config)
    if not isinstance(config, dict):
        raise TypeError("TuningResult.best.config must be a JSON-serializable dict")

    return {
        "config": config,
        "mean_us": result.best.mean_us,
        "error_margin_us": result.best.error_margin_us,
        "num_samples": result.best.num_samples,
    }


def record_tuning_result(tune_fn: Callable[..., Any],
                         available_kwargs: dict[str, Any],
                         result: TuningResult[Any]) -> None:
    gpu_name = get_gpu_name()
    bench_name = _benchmark_name_from_tune_fn(tune_fn)
    params = _tuning_params(tune_fn, available_kwargs)

    data = _load_database()
    gpu_entry = data.setdefault("gpus", {}).setdefault(gpu_name, {})
    bench_entries = gpu_entry.setdefault(bench_name, [])

    entry = {
        "params": params,
        **_result_entry(result),
    }
    for i, existing_entry in enumerate(bench_entries):
        if existing_entry["params"] == params:
            bench_entries[i] = entry
            break
    else:
        bench_entries.append(entry)

    _save_database(data)


def _config_from_entry(entry: Any, bench_name: str, gpu_name: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"Invalid tuned benchmark config for {bench_name} on GPU {gpu_name}: "
            "entry is not an object"
        )

    config = entry.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(
            f"Invalid tuned benchmark config for {bench_name} on GPU {gpu_name}: "
            "entry does not contain a config object"
        )
    return dict(config)


def _fallback_tuned_entry(data: dict[str, Any],
                          bench_name: str,
                          params: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    for fallback_gpu_name in sorted(data["gpus"]):
        gpu_entry = data["gpus"][fallback_gpu_name]
        entry = _find_tuned_entry(gpu_entry, bench_name, params)
        if entry is not None:
            return fallback_gpu_name, entry
    return None


def _warn_fallback(gpu_name: str,
                   fallback_gpu_name: str,
                   bench_name: str) -> None:
    warnings.warn(
        f"no tuned benchmark config for {bench_name} on GPU {gpu_name}; using "
        f"config tuned for {fallback_gpu_name}. {SAVE_TUNING_HINT}",
        stacklevel=3,
    )


def _find_tuned_entry(gpu_entry: dict[str, Any],
                      bench_name: str,
                      params: dict[str, Any]) -> dict[str, Any] | None:
    bench_entries = gpu_entry.get(bench_name)
    if bench_entries is None:
        return None
    for entry in bench_entries:
        if entry["params"] == params:
            return entry
    return None


def get_tuned_config(tune_fn: Callable[..., Any], **available_kwargs: Any) -> dict[str, Any]:
    data = _load_database()
    gpu_name = get_gpu_name()
    bench_name = _benchmark_name_from_tune_fn(tune_fn)
    params = _tuning_params(tune_fn, available_kwargs)

    gpu_entry = data["gpus"].get(gpu_name, {})
    entry = _find_tuned_entry(gpu_entry, bench_name, params)
    if entry is not None:
        return _config_from_entry(entry, bench_name, gpu_name)

    fallback = _fallback_tuned_entry(data, bench_name, params)
    if fallback is not None:
        fallback_gpu_name, fallback_entry = fallback
        _warn_fallback(gpu_name, fallback_gpu_name, bench_name)
        return _config_from_entry(fallback_entry, bench_name, fallback_gpu_name)

    raise RuntimeError(
        f"Missing tuned benchmark config for {bench_name} on GPU {gpu_name}. "
        f"{SAVE_TUNING_HINT}"
    )
