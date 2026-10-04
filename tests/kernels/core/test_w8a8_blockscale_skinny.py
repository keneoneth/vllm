# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm._custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from vllm.platforms.rocm import on_gfx1151

pytestmark = pytest.mark.skipif(not on_gfx1151(), reason="gfx1151 only")


def _fp8_dtype() -> torch.dtype:
    return current_platform.fp8_dtype()


def _make_case(m: int, n: int, k: int = 4096):
    device = "cuda"
    fp8 = _fp8_dtype()

    # Keep values safely representable and avoid synthetic quant-path NaNs.
    a_f32 = torch.randn(m, k, device=device, dtype=torch.float32) * 0.1
    w_f32 = torch.randn(n, k, device=device, dtype=torch.float32) * 0.1

    # Explicitly build a block-scaled payload with the exact logical layouts
    # used by the single-weight design.
    a_scale = torch.rand(m, k // 128, device=device, dtype=torch.float32) * 0.05 + 0.01
    w_scale = (
        torch.rand(n // 128, k // 128, device=device, dtype=torch.float32) * 0.05 + 0.01
    )

    a_q = torch.clamp(a_f32 / a_scale.repeat_interleave(128, dim=1), -200.0, 200.0).to(
        fp8
    )
    w_q = torch.clamp(
        w_f32 / w_scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1),
        -200.0,
        200.0,
    ).to(fp8)

    return (
        a_q.contiguous(),
        w_q.contiguous(),
        a_scale.contiguous(),
        w_scale.contiguous(),
    )


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("n", [1536, 4096])
def test_wvsplitk_blockscale_matches_plain_aiter(m: int, n: int):
    a, w, a_scale, w_scale = _make_case(m, n)
    cu = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Reference is deliberately the NON-PRESHUFFLED AITER blockscale path.
    ref = rocm_aiter_ops.gemm_a8w8_blockscale(
        a,
        w,
        a_scale,
        w_scale,
        [1, 128],
        output_dtype=torch.bfloat16,
    )
    got = ops.wvSplitKQBlockScale(w, a, a_scale, w_scale, cu)

    torch.testing.assert_close(got, ref, rtol=5e-2, atol=1e-1)


@pytest.mark.parametrize("n", [1536, 4096])
def test_wvsplitk_blockscale_m6_graph_replay_changes_inputs(n: int):
    m, k = 6, 4096
    a, w, a_scale, w_scale = _make_case(m, n, k)
    cu = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Capture with persistent input buffers, then mutate them before replay.
    a_buf = a.clone()
    as_buf = a_scale.clone()

    graph = torch.cuda.CUDAGraph()
    # Initialize native dispatch before capture, on a side stream.
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        ops.wvSplitKQBlockScale(w, a_buf, as_buf, w_scale, cu)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        out = ops.wvSplitKQBlockScale(w, a_buf, as_buf, w_scale, cu)

    graph.replay()
    first = out.clone()

    a2, _, as2, _ = _make_case(m, n, k)
    a_buf.copy_(a2)
    as_buf.copy_(as2)
    graph.replay()
    second = out.clone()

    assert not torch.equal(first, second)
    ref2 = rocm_aiter_ops.gemm_a8w8_blockscale(
        a2,
        w,
        as2,
        w_scale,
        [1, 128],
        output_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(second, ref2, rtol=5e-2, atol=1e-1)
