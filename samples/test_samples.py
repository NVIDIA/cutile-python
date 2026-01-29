# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys
import torch
import subprocess
import os
import pytest

# ==============================================================================
# This script uses the pytest framework to validate the functionality of
# all cuTile sample files.
#
# To run this script, simply execute `pytest` from your terminal.
#
# If a CUDA-enabled GPU is not found, all tests will be skipped automatically.
# ==============================================================================

# List of sample files to test.
SAMPLES_TO_TEST = [
    "VectorAddition.py",
    "BatchMatMul.py",
    "FFT.py",
    "MatMul.py",
    "Transpose.py",
    "AttentionFMHA.py",
    "LayerNorm.py",
    "MoE.py",
    "AllGatherMatmul.py",
]

# Get the absolute path of the current directory to ensure the script
# can find the sample files regardless of the current working directory.
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # This handles cases where __file__ is not defined (e.g., in an interactive session)
    SCRIPT_DIR = os.getcwd()

# Create a fixture to check for CUDA. pytest will automatically use this.


@pytest.fixture(scope="session", autouse=True)
def check_cuda_availability():
    """Fixture to ensure CUDA is available before running GPU tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA-enabled GPU not available on this system.")


@pytest.mark.parametrize("sample_file", SAMPLES_TO_TEST)
def test_sample_script_execution_correctness(sample_file):
    """
    A parameterized test that runs each cuTile sample script as a subprocess
    and checks for successful execution and correctness of the results.
    """
    # Construct the absolute path to the sample file.
    sample_path = os.path.join(SCRIPT_DIR, sample_file)

    # This print statement is for real-time visibility and is not captured by default.
    # To see it, run pytest with the `-s` flag.
    print(f"\nRunning test for: {sample_file} with correctness check...")

    try:
        # Use subprocess.run with check=True to automatically raise an exception
        # for non-zero exit codes.
        # The -u flag forces the Python interpreter to run in unbuffered mode.
        result = subprocess.run([sys.executable, '-u', sample_path, "--correctness-check"],
                                capture_output=True, text=True, check=True, timeout=300)

        # The output from the subprocess is captured, but we'll print it for debugging.
        # You can add specific checks here if a script's output needs validation.
        captured_output = result.stdout + result.stderr
        print("--- Captured Output ---")
        print(captured_output)
        print("--- End Captured Output ---")

        if "An unexpected error occurred" in captured_output:
            pytest.fail(f"Test for {sample_file} failed due to an unexpected error in the output.")
        # Check all the correctness checks passed and not disabled.
        if ("Correctness check failed" in captured_output or
                "Correctness check passed" not in captured_output):
            pytest.fail(
                f"Test for {sample_file} failed: "
                "correctness check disabled or failed in the output."
            )

    except subprocess.CalledProcessError as e:
        # This block is executed if the subprocess returns a non-zero exit code.
        # We use pytest.fail to explicitly mark the test as failed.
        captured_output = e.stdout + e.stderr
        pytest.fail(f"Test for {sample_file} failed with a non-zero exit code.\n"
                    f"Captured Output:\n{captured_output}")
    except subprocess.TimeoutExpired:
        # This block is executed if the subprocess runs for too long.
        pytest.fail(f"Test for {sample_file} timed out after 300 seconds.")
    except Exception as e:
        # This block catches any other unexpected Python errors.
        pytest.fail(f"Test for {sample_file} failed with an unexpected error: {e}")
