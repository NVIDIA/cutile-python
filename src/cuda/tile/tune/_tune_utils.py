# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import atexit
import itertools
import os
import secrets
import subprocess
import sys
import time
import traceback
import threading
from dataclasses import dataclass
from multiprocessing.connection import Listener, wait
from typing import Any

from cuda.tile._cext import (
    _benchmark,
    _benchmark_with_ipc_payload,
    _export_ipc_benchmark_payload,
)


class TileLaunchTimeoutError(RuntimeError):
    pass


_WORKER_BENCHMARK = "BENCHMARK"
_WORKER_STOP = "STOP"
_WORKER_START_TIMEOUT_SEC = 5.0


@dataclass
class _WorkerState:
    process: Any
    conn: Any


class _TimedBenchmarkRunner:
    def __init__(self):
        self._worker: _WorkerState | None = None
        self._next_task_id = itertools.count(0)

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.process.poll() is None

    def run(self, payload: bytes, timeout_sec: float) -> tuple[float, float]:
        worker = self._get_or_start_worker()
        task_id = next(self._next_task_id)
        wall_time_start = time.perf_counter()
        try:
            worker.conn.send((_WORKER_BENCHMARK, task_id, payload))
        except ConnectionError:
            self._exit_worker_and_report_error(worker.process)

        conn_ready = wait([worker.conn], timeout_sec)
        if not conn_ready:
            self.terminate(graceful_shutdown=False)
            raise TileLaunchTimeoutError(
                f"CUDA kernel launch exceeded timeout {timeout_sec} sec"
            )
        try:
            result_task_id, ok, value, details = worker.conn.recv()
        except (ConnectionError, EOFError):
            self._exit_worker_and_report_error(worker.process)

        if result_task_id != task_id:
            self.terminate(graceful_shutdown=False)
            raise RuntimeError(
                f"Benchmark worker returned task {result_task_id}, expected {task_id}"
            )
        if not ok:
            self.terminate(graceful_shutdown=False)
            raise RuntimeError(f"{value}: {details}")
        return value, time.perf_counter() - wall_time_start

    def _get_or_start_worker(self) -> _WorkerState:
        if self._worker is not None:
            if self._worker.process.poll() is None:
                return self._worker
            self.terminate(graceful_shutdown=False)

        authkey = secrets.token_bytes(32)
        listener = Listener(authkey=authkey)
        try:
            worker_env = os.environ.copy()
            worker_env["CUDA_TILE_IPC_BENCHMARK_WORKER"] = "1"
            process = subprocess.Popen(
                [sys.executable, "-m", "cuda.tile.tune._benchmark_worker",
                 listener.address, authkey.hex()],
                env=worker_env,
            )

            parent_conn = self._accept_worker_connection(listener, process)
        finally:
            listener.close()
        self._worker = _WorkerState(process, parent_conn)
        return self._worker

    @staticmethod
    def _accept_worker_connection(listener, process) -> Any:
        conn = None

        def _accept():
            nonlocal conn
            try:
                conn = listener.accept()
            except Exception:
                pass

        # listener.accept() doesn't support timeout, create a daemon
        # thread to wait for the connection.
        accept_thread = threading.Thread(target=_accept, daemon=True)
        accept_thread.start()
        accept_thread.join(timeout=_WORKER_START_TIMEOUT_SEC)
        if accept_thread.is_alive() or conn is None:
            listener.close()
            accept_thread.join(timeout=1)
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=1)
            except Exception:
                pass
            raise RuntimeError(f"Timed benchmark worker failed to connect "
                               f"within {_WORKER_START_TIMEOUT_SEC} sec")
        return conn

    def terminate(self, graceful_shutdown: bool):
        if self._worker is None:
            return
        worker = self._worker
        self._worker = None

        if graceful_shutdown:
            try:
                worker.conn.send((_WORKER_STOP,))
                worker.process.wait(timeout=1)
            except Exception:
                pass

        try:
            if worker.process.poll() is None:
                worker.process.terminate()
                try:
                    worker.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            if worker.process.poll() is None:
                worker.process.kill()
                worker.process.wait(timeout=1)
        finally:
            worker.conn.close()

    def _exit_worker_and_report_error(self, process, message: str = ""):
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        error_message = (f"{message} Timed benchmark worker exited unexpectedly with exit code "
                         f"{process.poll()}")
        self.terminate(graceful_shutdown=False)
        raise RuntimeError(f"{error_message}")


_timed_benchmark_runner: _TimedBenchmarkRunner | None = None
_timed_benchmark_lock = threading.Lock()


def _get_timed_benchmark_runner() -> _TimedBenchmarkRunner:
    global _timed_benchmark_runner
    if _timed_benchmark_runner is None:
        _timed_benchmark_runner = _TimedBenchmarkRunner()

    return _timed_benchmark_runner


def _terminate_timed_benchmark_runner():
    global _timed_benchmark_runner
    if _timed_benchmark_runner is not None:
        _timed_benchmark_runner.terminate(graceful_shutdown=True)
        _timed_benchmark_runner = None


atexit.register(_terminate_timed_benchmark_runner)


def _benchmark_worker_main(conn):
    while True:
        try:
            request = conn.recv()
        except (EOFError, ConnectionError):
            return  # Exit if connection is closed

        if request == (_WORKER_STOP,):
            return  # Exit if parent sent stop signal

        command, task_id, payload = request
        if command != _WORKER_BENCHMARK:
            raise ValueError(f"Unknown benchmark worker command {command!r}")

        try:
            elapsed_us = _benchmark_with_ipc_payload(payload)
        except Exception as e:
            conn.send((task_id, False, type(e).__name__, traceback.format_exc()))
        else:
            conn.send((task_id, True, elapsed_us, None))


def benchmark_with_timeout(
        stream, grid, kernel, pyargs,
        timeout_sec: float) -> tuple[float, float | None]:
    serialized_payload = _export_ipc_benchmark_payload(stream, grid, kernel, pyargs)
    if serialized_payload is None:
        return _benchmark(stream, grid, kernel, pyargs), None

    with _timed_benchmark_lock:
        runner = _get_timed_benchmark_runner()
        return runner.run(serialized_payload, timeout_sec)
