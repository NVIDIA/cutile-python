# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
import torch
import math


ConstInt = ct.Constant[int]


@ct.kernel
def fft_kernel(x_packed_in, y_packed_out,
               W0, W1, W2, T0, T1,  # W and T matrices are pre-computed as single tensors
               N: ConstInt, F0: ConstInt, F1: ConstInt, F2: ConstInt,
               BS: ConstInt, D: ConstInt):  # D is the atom_packing_dim for memory packing
    """
    cuTile kernel for a multi-dimensional FFT using tensor factorization.
    It expects packed real/imaginary parts of the input and pre-computed factors
    (W and T matrices).

    Args:
        x_packed_in: Input tensor with real/imaginary parts packed for efficient memory access.
                     Expected shape: (Batch_Size, N * 2 // D, D).
        y_packed_out: Output tensor to store FFT results, also in a packed format.
                      Expected shape: (Batch_Size, N * 2 // D, D).
        W0, W1, W2: Rotation matrices (Discrete Fourier Transform matrices) for each
                    of the three logical dimensions (F0, F1, F2). These are pre-computed.
        T0, T1: Twiddle factors for inter-dimensional permutations and phase adjustments.
                These are also pre-computed.
        N (ConstInt): Total FFT size (e.g., 256, 1024).
        F0, F1, F2 (ConstInt): Factors of N, such that N = F0 * F1 * F2. These define
                               the logical 3D shape for the FFT decomposition.
        BS (ConstInt): Per-block minibatch size — the number of batch items
                       processed by a single thread block. The full input batch
                       is split across blocks as grid_x = Batch // BS.
        D (ConstInt): Atom packing dimension. This parameter controls how the real and
                      imaginary data are interleaved and packed into memory for optimal
                      coalesced access on the GPU.
    """
    # Pre-calculate products of factors for convenience in reshaping and indexing.
    F0F1 = F0 * F1
    F1F2 = F1 * F2
    F0F2 = F0 * F2

    bid = ct.bid(0)  # Block index along the batch dimension.
    # Each block processes a contiguous minibatch of BS items from the full batch.

    # --- Load Input Data ---
    # Load this block's minibatch from `x_packed_in` (shape (Batch, N*2//D, D)).
    # `ct.load` reads a (BS, N*2//D, D) tile at minibatch index `bid`.
    # Then `ct.reshape` transforms it to (BS, N, 2) to logically separate
    # the real and imaginary components for each of the N elements.
    X_ri = ct.reshape(ct.load(x_packed_in, index=(bid, 0, 0),
                      shape=(BS, N * 2 // D, D)), (BS, N, 2))

    # Split the real (X_r) and imaginary (X_i) parts into separate tensors.
    # `ct.extract` pulls out the specific component (real at index 0, imag at index 1).
    # Reshape them into the logical 3D structure (BS, F0, F1, F2) for the FFT computation.
    X_r = ct.reshape(ct.extract(X_ri, index=(0, 0, 0), shape=(BS, N, 1)), (BS, F0, F1, F2))
    X_i = ct.reshape(ct.extract(X_ri, index=(0, 0, 1), shape=(BS, N, 1)), (BS, F0, F1, F2))

    # --- Load Rotation (W) and Twiddle (T) Matrices ---
    # These matrices are pre-computed on the host (CPU) and passed to the kernel
    # as global memory tensors. They are loaded into the kernel's local scope
    # (e.g., shared memory or registers) and their interleaved real/imaginary parts
    # are split for use in complex arithmetic.

    # W0 (F0 x F0) - Rotation matrix for the first dimension's DFT.
    # Loaded as (F0, F0*2) real/imag interleaved, then reshaped to (F0, F0, 2).
    W0_ri_loaded = ct.reshape(ct.load(W0, index=(0, 0, 0), shape=(F0, F0, 2)), (F0, F0, 2))
    # Extract and reshape real and imaginary parts. The (1, F0, F0) shape
    # allows for broadcasting during matrix multiplication with X_r/X_i.
    W0_r_tile = ct.reshape(ct.extract(W0_ri_loaded, index=(
        0, 0, 0), shape=(F0, F0, 1)), (1, F0, F0))
    W0_i_tile = ct.reshape(ct.extract(W0_ri_loaded, index=(
        0, 0, 1), shape=(F0, F0, 1)), (1, F0, F0))

    # W1 (F1 x F1) - Rotation matrix for the second dimension's DFT.
    W1_ri_loaded = ct.reshape(ct.load(W1, index=(0, 0, 0), shape=(F1, F1, 2)), (F1, F1, 2))
    W1_r_tile = ct.reshape(ct.extract(W1_ri_loaded, index=(
        0, 0, 0), shape=(F1, F1, 1)), (1, F1, F1))
    W1_i_tile = ct.reshape(ct.extract(W1_ri_loaded, index=(
        0, 0, 1), shape=(F1, F1, 1)), (1, F1, F1))

    # W2 (F2 x F2) - Rotation matrix for the third dimension's DFT.
    W2_ri_loaded = ct.reshape(ct.load(W2, index=(0, 0, 0), shape=(F2, F2, 2)), (F2, F2, 2))
    W2_r_tile = ct.reshape(ct.extract(W2_ri_loaded, index=(
        0, 0, 0), shape=(F2, F2, 1)), (1, F2, F2))
    W2_i_tile = ct.reshape(ct.extract(W2_ri_loaded, index=(
        0, 0, 1), shape=(F2, F2, 1)), (1, F2, F2))

    # T0 (F0 x F1F2) - Twiddle factors applied after the first contraction stage.
    # Loaded as (F0, F1F2*2), then reshaped to (F0, F1F2, 2).
    T0_ri_loaded = ct.reshape(ct.load(T0, index=(0, 0, 0), shape=(F0, F1F2, 2)), (F0, F1F2, 2))
    # Reshape to (N, 1) to align with the flattened data for element-wise multiplication.
    T0_r_tile = ct.reshape(ct.extract(T0_ri_loaded, index=(0, 0, 0), shape=(F0, F1F2, 1)), (N, 1))
    T0_i_tile = ct.reshape(ct.extract(T0_ri_loaded, index=(0, 0, 1), shape=(F0, F1F2, 1)), (N, 1))

    # T1 (F1 x F2) - Twiddle factors applied after the second contraction stage.
    T1_ri_loaded = ct.reshape(ct.load(T1, index=(0, 0, 0), shape=(F1, F2, 2)), (F1, F2, 2))
    # Reshape to (F1F2, 1) for element-wise multiplication.
    T1_r_tile = ct.reshape(ct.extract(T1_ri_loaded, index=(0, 0, 0), shape=(F1, F2, 1)), (F1F2, 1))
    T1_i_tile = ct.reshape(ct.extract(T1_ri_loaded, index=(0, 0, 1), shape=(F1, F2, 1)), (F1F2, 1))

    # --- CT0: Contract over the first dimension (F0) ---
    # Reshape X_r and X_i to (BS, F0, F1F2) to prepare for matrix multiplication with W0.
    X_r = ct.reshape(X_r, (BS, F0, F1F2))
    X_i = ct.reshape(X_i, (BS, F0, F1F2))
    # Perform complex matrix multiplication: (A+iB)(C+iD) = (AC-BD) + i(AD+BC).
    # The result is then reshaped back to (BS, N, 1) to align with T0 for twiddling.
    X_r_ = ct.reshape(ct.matmul(W0_r_tile, X_r) - ct.matmul(W0_i_tile, X_i), (BS, N, 1))
    X_i_ = ct.reshape(ct.matmul(W0_i_tile, X_r) + ct.matmul(W0_r_tile, X_i), (BS, N, 1))

    # --- Twiddle & Permute 0 ---
    # Apply twiddle factors T0 element-wise to the complex results.
    X_r = T0_r_tile * X_r_ - T0_i_tile * X_i_
    X_i = T0_i_tile * X_r_ + T0_r_tile * X_i_
    # Permute dimensions from (BS, F0, F1, F2) to (BS, F1, F2, F0)
    # to prepare the data for the next contraction stage.
    X_r = ct.permute(ct.reshape(X_r, (BS, F0, F1, F2)), (0, 2, 3, 1))
    X_i = ct.permute(ct.reshape(X_i, (BS, F0, F1, F2)), (0, 2, 3, 1))

    # --- CT1: Contract over the second dimension (F1) ---
    # Reshape X_r and X_i to (BS, F1, F0F2) for matrix multiplication with W1.
    X_r = ct.reshape(X_r, (BS, F1, F0F2))
    X_i = ct.reshape(X_i, (BS, F1, F0F2))
    # Perform complex matrix multiplication.
    # The result is reshaped to (BS, F1F2, F0) to align with T1 for twiddling.
    X_r_ = ct.reshape(ct.matmul(W1_r_tile, X_r) - ct.matmul(W1_i_tile, X_i), (BS, F1F2, F0))
    X_i_ = ct.reshape(ct.matmul(W1_i_tile, X_r) + ct.matmul(W1_r_tile, X_i), (BS, F1F2, F0))

    # --- Twiddle & Permute 1 ---
    # Apply twiddle factors T1 element-wise.
    X_r = T1_r_tile * X_r_ - T1_i_tile * X_i_
    X_i = T1_i_tile * X_r_ + T1_r_tile * X_i_
    # Permute dimensions from (BS, F1, F2, F0) to (BS, F2, F0, F1)
    # to prepare the data for the final contraction stage.
    X_r = ct.permute(ct.reshape(X_r, (BS, F1, F2, F0)), (0, 2, 3, 1))
    X_i = ct.permute(ct.reshape(X_i, (BS, F1, F2, F0)), (0, 2, 3, 1))

    # --- CT2: Contract over the third dimension (F2) ---
    # Reshape X_r and X_i to (BS, F2, F0F1) for matrix multiplication with W2.
    X_r = ct.reshape(X_r, (BS, F2, F0F1))
    X_i = ct.reshape(X_i, (BS, F2, F0F1))
    # Perform complex matrix multiplication.
    X_r_ = ct.matmul(W2_r_tile, X_r) - ct.matmul(W2_i_tile, X_i)
    X_i_ = ct.matmul(W2_i_tile, X_r) + ct.matmul(W2_r_tile, X_i)

    # --- Final Permutation and Reshape ---
    # Permute back to the original logical order (BS, F0, F1, F2) from (BS, F2, F0, F1).
    X_r = ct.permute(ct.reshape(X_r_, (BS, F2, F0, F1)), (0, 1, 3, 2))
    X_i = ct.permute(ct.reshape(X_i_, (BS, F2, F0, F1)), (0, 1, 3, 2))
    # Reshape to (BS, N, 1) for real and imaginary parts separately,
    # flattening the 3D logical structure back to a 1D representation per batch item.
    X_r = ct.reshape(X_r, (BS, N, 1))
    X_i = ct.reshape(X_i, (BS, N, 1))

    # --- Final Reshape and Store Output ---
    # Concatenate the real and imaginary parts along the last axis to form (BS, N, 2).
    # Then reshape to the packed output format (BS, N * 2 // D, D) for storing to global memory.
    Y_ri = ct.reshape(ct.cat((X_r, X_i), axis=-1), (BS, N * 2 // D, D))
    ct.store(y_packed_out, index=(bid, 0, 0), tile=Y_ri)


# --- Helper function for generating DFT matrices (W-matrices) ---
def twiddles(rows: int, cols: int, factor: int, device: torch.device, precision: torch.dtype):
    """
    Generates a matrix of complex exponentials ($$W_{\text{factor}}^{i dot j}$$),
    which are the core components of Discrete Fourier Transform (DFT) matrices.
    Returns it as an interleaved real/imaginary tensor.

    Args:
        rows (int): Number of rows for the generated matrix.
        cols (int): Number of columns for the generated matrix.
        factor (int): The factor used in the complex exponential denominator.
        device (torch.device): The PyTorch device to create the tensor on (e.g., 'cuda').
        precision (torch.dtype): The desired floating-point precision (e.g., torch.float32).

    Returns:
        torch.Tensor: A tensor of shape (rows, cols, 2) with interleaved real and
                      imaginary parts of the complex exponential matrix.
    """
    # Create 2D grids for indices I and J using torch.meshgrid.
    I, J = torch.meshgrid(torch.arange(rows, device=device),
                          torch.arange(cols, device=device),
                          indexing='ij')
    # Compute the complex exponential based on the FFT formula.
    W_complex = torch.exp(-2 * math.pi * 1j * (I * J) / factor)
    # Convert the complex tensor to a real tensor where the last dimension
    # stores the real and imaginary parts (e.g., [real_part, imag_part]).
    return torch.view_as_real(W_complex).to(precision).contiguous()


def make_twiddles(decomp: tuple, precision: torch.dtype, device: torch.device):
    """
    Generates mathematically correct W (rotation) and T (twiddle) matrices
    for a multi-dimensional FFT decomposition, with interleaved real/imaginary parts.

    These matrices are pre-computed on the host (CPU) and then transferred to the GPU
    for use by the cuTile kernel, avoiding re-computation on the device for each batch.

    Args:
        decomp (tuple): A tuple (F0, F1, F2) representing the factors of N,
                        where N is the total FFT size.
        precision (torch.dtype): The desired floating-point precision for the matrices.
        device (torch.device): The PyTorch device to create the tensors on.

    Returns:
        tuple: A tuple containing (W0_ri, W1_ri, W2_ri, T0_ri, T1_ri),
               where each element is a tensor with interleaved real/imaginary parts.
    """
    F0, F1, F2 = decomp
    N = F0 * F1 * F2  # Total FFT size, product of factors
    F1F2 = F1 * F2   # Product of F1 and F2, used for T0 matrix dimensions

    # Generate W matrices (rotation matrices for each dimension's DFT).
    W0_ri = twiddles(F0, F0, F0, device, precision)
    W1_ri = twiddles(F1, F1, F1, device, precision)
    W2_ri = twiddles(F2, F2, F2, device, precision)

    # Generate T matrices (twiddle factors for dimension transitions/permutations).
    # T0 applies across F0 and F1F2, with N as the overall factor.
    T0_ri = twiddles(F0, F1F2, N, device, precision)
    # T1 applies across F1 and F2, with F1F2 as the factor.
    T1_ri = twiddles(F1, F2, F1F2, device, precision)

    return (W0_ri, W1_ri, W2_ri, T0_ri, T1_ri)


# --- Wrapper function to launch the fft_kernel ---
def cutile_fft(
    x: torch.Tensor,
    factors: tuple,  # (F0, F1, F2) - factors of N
    atom_packing_dim: int = 64,  # The 'D' parameter for data packing/unpacking
    minibatch: int = 1,  # The 'BS' kernel constant: items processed per block
) -> torch.Tensor:
    """
    Performs a Batched 1D Fast Fourier Transform (FFT) using a cuTile kernel
    based on multi-dimensional factorization (similar to a Cooley-Tukey algorithm).

    This function prepares the input data, generates necessary pre-computed
    matrices, launches the cuTile kernel, and unpacks the results.

    Args:
        x (torch.Tensor): Input tensor of shape (Batch, N) containing complex64 numbers.
                          This tensor *must* be on a CUDA device.
                          N (the FFT size) must be factorable into factors[0]*factors[1]*factors[2].
        factors (tuple): A tuple (F0, F1, F2) representing the factors of N.
                         The product F0 * F1 * F2 must equal N. These factors define
                         the logical 3D shape for the FFT decomposition within the kernel.
        atom_packing_dim (int): The dimension 'D' used for data packing/unpacking
                                in the kernel. This value affects memory access patterns.
                                The total number of real/imaginary elements (N*2) must be
                                divisible by this dimension. Default is 64.
        minibatch (int): Number of batch items processed by each thread block (the
                         kernel's `BS` constant). The grid is sized as Batch // minibatch,
                         so `Batch` must be divisible by `minibatch`. Larger values let
                         a block reuse the loaded W/T matrices across more items, but
                         increase register/shared-memory pressure. Default is 1.

    Returns:
        torch.Tensor: Output tensor of shape (Batch, N) containing the FFT results.
                      The output data type will be torch.complex64.

    Raises:
        ValueError: If input tensor dimensions, device, or data type are incorrect,
                    if the provided factors do not multiply to N, if N*2 is not
                    divisible by atom_packing_dim, or if Batch is not divisible
                    by minibatch.
    """
    # --- Input Validation ---
    if x.ndim != 2:
        raise ValueError("Input tensor must be 2D (Batch, N).")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    if x.dtype != torch.complex64:
        raise ValueError("Input tensor dtype must be torch.complex64.")

    Batch = x.shape[0]  # Total batch size from the input tensor's shape.
    N = x.shape[1]      # Total FFT size from the input tensor's shape.

    F0, F1, F2 = factors
    # Validate that the provided factors correctly decompose the total FFT size N.
    if F0 * F1 * F2 != N:
        raise ValueError(f"Factors ({F0}*{F1}*{F2}={F0*F1*F2}) do not multiply to N={N}. "
                         f"Please provide factors that correctly decompose N.")

    # Determine the underlying floating-point precision (e.g., float32) for
    # the real and imaginary parts of the complex numbers.
    PRECISION_DTYPE = x.real.dtype

    # --- Prepare Input Data for Kernel (Split real/imag, pack) ---
    # Convert the complex input tensor (Batch, N) to a real tensor (Batch, N, 2)
    # where the last dimension explicitly separates real and imaginary parts.
    x_ri = torch.view_as_real(x)

    # Reshape the real/imaginary tensor to the packed format (Batch, N*2 // D, D)
    # that the kernel expects for efficient memory access.
    # This step assumes that the total number of real/imaginary elements (N*2)
    # is perfectly divisible by the `atom_packing_dim` (D).
    if (N * 2) % atom_packing_dim != 0:
        raise ValueError(f"Total real/imag elements (N*2 = {N*2}) must be divisible by "
                         f"atom_packing_dim ({atom_packing_dim}) for kernel packing.")
    if Batch % minibatch != 0:
        raise ValueError(f"Batch ({Batch}) must be divisible by minibatch ({minibatch}).")
    x_packed_in = x_ri.reshape(Batch, N * 2 // atom_packing_dim, atom_packing_dim).contiguous()

    # --- Generate W (Rotation) and T (Twiddle) Matrices ---
    # These matrices are pre-computed mathematically based on the FFT decomposition.
    # They are generated on the same device as the input tensor (CUDA) to avoid
    # costly host-to-device transfers during kernel execution.
    W0_gmem, W1_gmem, W2_gmem, T0_gmem, T1_gmem = make_twiddles(factors, PRECISION_DTYPE, x.device)

    # --- Create Output Tensor ---
    # Initialize an empty tensor with the same shape and properties as the packed input.
    # This tensor will store the results computed by the kernel.
    y_packed_out = torch.empty_like(x_packed_in)

    # --- Calculate Grid Dimensions ---
    # One thread block is launched per minibatch of size `minibatch`.
    # The grid is a 3-tuple (grid_x, grid_y, grid_z).
    grid = (Batch // minibatch, 1, 1)

    # --- Launch the cuTile Kernel ---
    # The `fft_kernel` is launched on the GPU with the calculated grid dimensions.
    # All necessary input tensors (packed data, W and T matrices) and constant parameters
    # (N, F0, F1, F2, BS=minibatch, D) are passed to the kernel.
    ct.launch(torch.cuda.current_stream(), grid, fft_kernel,
              (x_packed_in, y_packed_out,
               W0_gmem, W1_gmem, W2_gmem,
               T0_gmem, T1_gmem,
               N, F0, F1, F2, minibatch, atom_packing_dim))

    # --- Unpack Output from Kernel (Reshape, combine real/imag) ---
    # Reshape the packed output tensor back to (Batch, N, 2) to separate real/imaginary parts.
    y_ri = y_packed_out.reshape(Batch, N, 2)
    # Convert the real/imaginary pair tensor back to a complex tensor (torch.complex64).
    y_complex = torch.view_as_complex(y_ri)

    return y_complex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()
    print("--- Running cuTile FFT Example ---")

    # --- User Configuration ---
    BATCH_SIZE = 2
    # Total FFT size (N) must be factorable into factors[0] * factors[1] * factors[2].
    # For example, N = 1024 can be factored as (8, 16, 8).
    FFT_SIZE = 8  # A smaller FFT size for demonstration purposes.
    FFT_FACTORS = (2, 2, 2)  # Factors for N=8 (2 * 2 * 2 = 8).
    # The 'D' parameter for data packing/unpacking in the kernel.
    # For N=8, N*2=16. ATOM_PACKING_DIM=2 is a valid divisor (16 % 2 == 0).
    ATOM_PACKING_DIM = 2

    # Data type for the real/imaginary components (e.g., float32 for complex64).
    PRECISION_DTYPE = torch.float32

    # --- Create Sample Input Data ---
    # Generate a random input tensor of complex64 numbers, placed on the CUDA device.
    # `torch.manual_seed(0)` ensures reproducibility of the random numbers.
    torch.manual_seed(0)
    input_data_complex = torch.randn(BATCH_SIZE, FFT_SIZE, dtype=torch.complex64, device='cuda')

    print("  Configuration:")
    print(f"  FFT Size (N): {FFT_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  FFT Factors (F0,F1,F2): {FFT_FACTORS}")
    print(f"  Atom Packing Dimension (D): {ATOM_PACKING_DIM}")
    print(f"Input data shape: {input_data_complex.shape}, dtype: {input_data_complex.dtype}")

    # Perform FFT using the custom cuTile kernel.
    output_fft_cutile = cutile_fft(
        x=input_data_complex,
        factors=FFT_FACTORS,
        atom_packing_dim=ATOM_PACKING_DIM
    )
    print(
        f"""\ncuTile FFT Output shape: {output_fft_cutile.shape},
        dtype: {output_fft_cutile.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(output_fft_cutile, torch.fft.fft(input_data_complex, axis=-1))
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- cuTile FFT example execution complete ---")
