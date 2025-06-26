# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import triton
from vllm.triton_utils.trition import language as tl


################################################################################
# Triton kernel
################################################################################
@triton.jit
def row_variance_kernel(
    x_ptr,  # fp32*  input  (contiguous 2-D)
    out_ptr,  # fp32*  output (one fp32 per row)
    n_cols,  # int    number of columns (same for every row)
    stride_row,  # int    leading dimension of x   (in elements)
    BLOCK_SIZE: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # identify which logical row this program (CTA) is handling
    # -------------------------------------------------------------------------
    row = tl.program_id(0)

    # pointers that advance as we step through the row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_start = x_ptr + row * stride_row

    # 64-bit accumulator to guarantee reproducibility
    acc = tl.zeros([], dtype=tl.float64)

    # -------------------------------------------------------------------------
    # sequentially walk over the columns in deterministic order
    # -------------------------------------------------------------------------
    offs = 0
    while offs < n_cols:
        ptr = row_start + offs + col_offsets
        mask = col_offsets + offs < n_cols  # guard the last (partial) chunk
        chunk = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)  # fp32
        acc += tl.sum((chunk * chunk).to(tl.float64), axis=0)
        offs += BLOCK_SIZE

    # -------------------------------------------------------------------------
    # write out the row variance
    # -------------------------------------------------------------------------
    denom = tl.full((), n_cols, dtype=tl.float64)  # 64-bit scalar
    var = (acc / denom).to(tl.float32)
    tl.store(out_ptr + row, var)


################################################################################
# Python convenience wrapper
################################################################################
def variance_deterministic(x: torch.Tensor,
                           block_size: int = 1024) -> torch.Tensor:
    """
    Compute row-wise variance(x) = mean(x**2) with a fixed summation order.

    Arguments
    ---------
    x : (⋯, N) fp32 CUDA tensor, contiguous in the last dimension
    block_size : how many columns each loop iteration loads
    (power of two is fastest)

    Returns
    -------
    variance : (⋯, 1) fp32 CUDA tensor
    """
    assert x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
    leading_shape = x.shape[:-1]
    n_rows, n_cols = x.numel() // x.shape[-1], x.shape[-1]

    # flatten leading dims -> 2-D (rows, cols)
    x2d = x.view(n_rows, n_cols)
    out = torch.empty((n_rows, ), device=x.device, dtype=torch.float32)

    grid = (n_rows, )
    row_variance_kernel[grid](
        x2d,
        out,
        n_cols,
        x2d.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=1,  # one warp per program keeps order serial
        num_stages=1,  # one stage is enough for purely sequential loads
    )
    return out.view(*leading_shape, 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.float32)

    v1 = variance_deterministic(x)
    v2 = variance_deterministic(x)
    assert torch.equal(v1, v2), "kernel lost determinism!"
