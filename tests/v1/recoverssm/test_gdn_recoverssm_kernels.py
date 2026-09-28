# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G1 exactness: RecoverSSM record mode + commit vs the per-draft baseline."""

import pytest
import torch

from vllm.model_executor.layers.mamba import damp_runtime
from vllm.model_executor.layers.mamba.damp_gdn_update import damp_fused_update
from vllm.model_executor.layers.mamba.damp_pack import page_bytes
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")

HV, H, K, V, S = 48, 16, 128, 128, 8
DEV = "cuda"


def _inputs(n_req: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    total = n_req * S

    def r(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g, device=DEV) * scale).to(dtype)

    return dict(
        A_log=r(HV, dtype=torch.float32, scale=0.5),
        a=r(total, HV),
        b=r(total, HV),
        dt_bias=r(HV, dtype=torch.float32, scale=0.5),
        q=r(1, total, H, K),
        k=r(1, total, H, K),
        v=r(1, total, HV, V),
        cu_seqlens=torch.arange(0, total + 1, S, device=DEV, dtype=torch.int32),
    )


def _records(num_blocks: int):
    return (
        torch.zeros(num_blocks, HV, S, V, device=DEV, dtype=torch.float32),
        torch.zeros(num_blocks, HV, S, K, device=DEV, dtype=torch.float32),
        torch.zeros(num_blocks, HV, S, device=DEV, dtype=torch.float32),
    )


def _plain_baseline_and_record(n_req, seed, dtype):
    """Baseline: per-draft slots 1+r*S .. . Record: checkpoint at 1+r*S."""
    inp = _inputs(n_req, seed)
    num_blocks = 1 + n_req * S
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    state0 = (torch.randn(num_blocks, HV, V, K, generator=g, device=DEV) * 0.1).to(
        dtype
    )
    state0[0] = 0

    base_state = state0.clone()
    base_idx = torch.arange(1, num_blocks, device=DEV, dtype=torch.int32).view(n_req, S)
    ones = torch.ones(n_req, device=DEV, dtype=torch.int32)
    o_base, _ = fused_sigmoid_gating_delta_rule_update(
        initial_state=base_state,
        ssm_state_indices=base_idx,
        num_accepted_tokens=ones,
        use_qk_l2norm_in_kernel=True,
        **inp,
    )

    rec_state = state0.clone()
    rec = _records(num_blocks)
    rec_idx = base_idx[:, :1].contiguous()
    o_rec, _ = fused_sigmoid_gating_delta_rule_update(
        initial_state=rec_state,
        ssm_state_indices=rec_idx,
        num_accepted_tokens=ones,
        use_qk_l2norm_in_kernel=True,
        recoverssm_records=rec,
        **inp,
    )
    return state0, base_state, base_idx, o_base, rec_state, rec, rec_idx, o_rec


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_plain_record_mode_output_bitwise(dtype):
    state0, _, _, o_base, rec_state, rec, rec_idx, o_rec = _plain_baseline_and_record(
        n_req=2, seed=0, dtype=dtype
    )
    assert torch.equal(o_rec, o_base)
    assert torch.equal(rec_state, state0)  # record mode never stores state
    assert rec[0][rec_idx.flatten().long()].abs().sum() > 0
    assert rec[1][rec_idx.flatten().long()].abs().sum() > 0
    assert rec[2][rec_idx.flatten().long()].abs().sum() > 0


N_HI = 32


@pytest.fixture
def damp_env(tmp_path, monkeypatch):
    g = torch.Generator().manual_seed(7)
    mask = torch.zeros(HV, K, dtype=torch.bool)
    for h in range(HV):
        mask[h, torch.randperm(K, generator=g)[:N_HI]] = True
    path = tmp_path / "mask.pt"
    torch.save(mask, path)
    monkeypatch.setenv("VLLM_DAMP_STATE", "mixed")
    monkeypatch.setenv("DAMP_MASK", str(path))
    monkeypatch.setattr(damp_runtime, "_hi_idx", None)
    monkeypatch.setattr(damp_runtime, "_lo_idx", None)
    yield


def _damp_page(num_blocks: int, seed: int) -> torch.Tensor:
    page = torch.zeros(
        num_blocks, page_bytes(N_HI, K - N_HI), dtype=torch.uint8, device=DEV
    )
    damp_runtime._ensure(page.device, HV)
    hi, lo, sc = damp_runtime._views(page)
    g = torch.Generator(device=DEV).manual_seed(seed)
    hi.copy_((torch.randn(hi.shape, generator=g, device=DEV) * 0.1).half())
    lo.copy_(torch.randint(-127, 128, lo.shape, generator=g, device=DEV).to(torch.int8))
    sc.copy_(torch.rand(sc.shape, generator=g, device=DEV) * 1e-3 + 1e-4)
    page[0] = 0
    return page


def _damp_baseline_and_record(n_req, seed):
    inp = _inputs(n_req, seed)
    num_blocks = 1 + n_req * S
    page0 = _damp_page(num_blocks, seed + 1)
    base_page = page0.clone()
    base_idx = torch.arange(1, num_blocks, device=DEV, dtype=torch.int32).view(n_req, S)
    ones = torch.ones(n_req, device=DEV, dtype=torch.int32)
    o_base, _ = damp_fused_update(
        initial_state=base_page,
        ssm_state_indices=base_idx,
        num_accepted_tokens=ones,
        use_qk_l2norm_in_kernel=True,
        **inp,
    )
    rec_page = page0.clone()
    rec = _records(num_blocks)
    rec_idx = base_idx[:, :1].contiguous()
    o_rec, _ = damp_fused_update(
        initial_state=rec_page,
        ssm_state_indices=rec_idx,
        num_accepted_tokens=ones,
        use_qk_l2norm_in_kernel=True,
        recoverssm_records=rec,
        **inp,
    )
    return page0, base_page, base_idx, o_base, rec_page, rec, rec_idx, o_rec


def test_damp_record_mode_output_bitwise(damp_env):
    page0, _, _, o_base, rec_page, rec, rec_idx, o_rec = _damp_baseline_and_record(
        n_req=2, seed=3
    )
    assert torch.equal(o_rec, o_base)
    assert torch.equal(rec_page, page0)
    assert rec[1][rec_idx.flatten().long()].abs().sum() > 0


from types import SimpleNamespace  # noqa: E402

from vllm.model_executor.layers.mamba.gdn_recoverssm import (  # noqa: E402
    GDNRecoverSSMCommitContext,
)
from vllm.model_executor.layers.mamba.gdn_recoverssm_ops import (  # noqa: E402
    check_recoverssm_records,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    is_conv_state_dim_first,
)

CONV_DIM, CONV_LEN = 64, 3 + (S - 1)


def test_check_records_rejects_head_and_dim_mismatch():
    idx = torch.zeros(2, 1, dtype=torch.int32)
    ok = tuple(t.cpu() for t in _records(3))
    check_recoverssm_records(idx, ok, HV, K, V)
    for bad in (
        (ok[0][:, :-1], ok[1], ok[2]),
        (ok[0], ok[1][:, :-1], ok[2]),
        (ok[0], ok[1], ok[2][:, :-1]),
        (ok[0][..., :-1], ok[1], ok[2]),
        (ok[0], ok[1][..., :-1], ok[2]),
    ):
        with pytest.raises(ValueError):
            check_recoverssm_records(idx, bad, HV, K, V)


def _conv(num_blocks, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    c = torch.randn(num_blocks, CONV_DIM, CONV_LEN, generator=g, device=DEV).bfloat16()
    return c if is_conv_state_dim_first() else c.transpose(-1, -2).contiguous()


def _commit(state, rec, conv, rec_idx, n_acc, **align):
    layer = SimpleNamespace(kv_cache=(conv, state, *rec))
    ctx = GDNRecoverSSMCommitContext.create([layer], spec_query_len=S, max_num_reqs=8)
    n = rec_idx.shape[0]
    ctx.commit(
        torch.full((n,), n_acc, device=DEV, dtype=torch.int32),
        rec_idx[:, 0],
        torch.arange(0, n * S + 1, S, device=DEV, dtype=torch.int32),
        **align,
    )


def _max_diff(got, want):
    if got.dtype == torch.uint8:  # DAMP page: report the byte mismatch count
        return f"{(got != want).sum().item()} bytes differ"
    return f"max |diff| {(got.float() - want.float()).abs().max().item():.3e}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_acc", range(0, S + 1))
def test_plain_commit_matches_baseline_slot(dtype, n_acc):
    state0, base_state, base_idx, _, rec_state, rec, rec_idx, _ = (
        _plain_baseline_and_record(n_req=2, seed=11, dtype=dtype)
    )
    _commit(rec_state, rec, _conv(rec_state.shape[0], 5), rec_idx, n_acc)
    for r in range(2):
        got = rec_state[rec_idx[r, 0]]
        want = (
            state0[rec_idx[r, 0]] if n_acc == 0 else base_state[base_idx[r, n_acc - 1]]
        )
        assert torch.equal(got, want), f"req {r} n_acc {n_acc}: {_max_diff(got, want)}"


@pytest.mark.parametrize("n_acc", range(0, S + 1))
def test_damp_commit_matches_baseline_slot(damp_env, n_acc):
    page0, base_page, base_idx, _, rec_page, rec, rec_idx, _ = (
        _damp_baseline_and_record(n_req=2, seed=13)
    )
    _commit(rec_page, rec, _conv(rec_page.shape[0], 6), rec_idx, n_acc)
    for r in range(2):
        got = rec_page[rec_idx[r, 0]]
        want = page0[rec_idx[r, 0]] if n_acc == 0 else base_page[base_idx[r, n_acc - 1]]
        assert torch.equal(got, want), f"req {r} n_acc {n_acc}: {_max_diff(got, want)}"


@pytest.mark.parametrize("fmt", ["plain", "damp"])
def test_align_boundary_and_final(request, fmt):
    # block 16, 12 tokens computed, 6 accepted: boundary after 4 tokens.
    if fmt == "damp":
        request.getfixturevalue("damp_env")
        s0, base, base_idx, _, st, rec, rec_idx, _ = _damp_baseline_and_record(2, 17)
    else:
        s0, base, base_idx, _, st, rec, rec_idx, _ = _plain_baseline_and_record(
            2, 17, torch.float16
        )
    num_blocks = st.shape[0]
    extra = torch.arange(num_blocks, num_blocks + 2, device=DEV, dtype=torch.int32)
    st = torch.cat([st, torch.zeros_like(st[:2])])
    rec = tuple(torch.cat([t, torch.zeros_like(t[:2])]) for t in rec)
    block_table = torch.stack([rec_idx[:, 0], extra], dim=1).contiguous()
    _commit(
        st,
        rec,
        _conv(st.shape[0], 8),
        rec_idx,
        6,
        block_table=block_table,
        num_computed_tokens=torch.full((2,), 12, device=DEV, dtype=torch.int32),
        mamba_block_size=16,
    )
    for r in range(2):
        assert torch.equal(st[block_table[r, 0]], base[base_idx[r, 3]])  # boundary
        assert torch.equal(st[block_table[r, 1]], base[base_idx[r, 5]])  # final
