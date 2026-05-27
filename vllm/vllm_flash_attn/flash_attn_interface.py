# ======================================================================================
# * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
# ======================================================================================
import torch

try:
    import flash_attn_v100_cuda
    FA2_UNAVAILABLE_REASON = None
    FA2_AVAILABLE = True
except ImportError as e:
    FA2_UNAVAILABLE_REASON = str(e)
    FA2_AVAILABLE = False

FA3_AVAILABLE = False
FA3_UNAVAILABLE_REASON = "FA3 disabled (V100 patch)"

FA4_AVAILABLE = False
FA4_UNAVAILABLE_REASON = "FA4 disabled (V100 patch)"
# isort: on

DEFAULT_FA_VERSION = 2


def _is_fa2_supported() -> tuple[bool, str | None]:
    if not FA2_AVAILABLE:
        return False, f"FA2 is unavailable due to: {FA2_UNAVAILABLE_REASON}"
    return True, None


def _is_fa3_supported() -> tuple[bool, str | None]:
    return False, FA3_UNAVAILABLE_REASON


def _is_fa4_supported() -> tuple[bool, str | None]:
    return False, FA4_UNAVAILABLE_REASON


def is_fa_version_supported(fa_version: int) -> bool:
    if fa_version == 2:
        return _is_fa2_supported()[0]
    elif fa_version == 3:
        return _is_fa3_supported()[0]
    elif fa_version == 4:
        return _is_fa4_supported()[0]
    else:
        raise ValueError(f"Unsupported FA version: {fa_version}")


def fa_version_unsupported_reason(fa_version: int) -> str | None:
    if fa_version == 2:
        return _is_fa2_supported()[1]
    elif fa_version == 3:
        return _is_fa3_supported()[1]
    elif fa_version == 4:
        return _is_fa4_supported()[1]
    else:
        raise ValueError(f"Unsupported FA version: {fa_version}")


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def get_scheduler_metadata(*args, **kwargs):
    return None


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits=0,
    fa_version=DEFAULT_FA_VERSION,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if window_size is None:
        window_size = (-1, -1)

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    if cu_seqlens_k is None:
        if seqused_k is not None:
            B = cu_seqlens_q.shape[0] - 1
            cu_seqlens_k_to_pass = torch.zeros(B + 1, dtype=torch.int32, device=cu_seqlens_q.device)
            torch.cumsum(seqused_k.to(torch.int32), dim=0,
                         out=cu_seqlens_k_to_pass[1:])
        else:
            cu_seqlens_k_to_pass = cu_seqlens_q.clone()
    else:
        cu_seqlens_k_to_pass = cu_seqlens_k

    cu_seqlens_q = cu_seqlens_q.to(torch.int32)
    cu_seqlens_k_to_pass = cu_seqlens_k_to_pass.to(torch.int32)

    head_size_og = q.size(-1)
    pad_size = 0
    if head_size_og % 8 != 0:
        pad_size = 8 - head_size_og % 8
        q = torch.nn.functional.pad(q, [0, pad_size])
        k = torch.nn.functional.pad(k, [0, pad_size])
        v = torch.nn.functional.pad(v, [0, pad_size])

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if out is None:
        out_padded = torch.empty_like(q)
    elif pad_size > 0:
        out_padded = torch.empty_like(q)
    else:
        out_padded = out

    return_softmax = return_softmax_lse and dropout_p > 0

    out_padded, softmax_lse, _, rng_state = flash_attn_v100_cuda.varlen_fwd(
        q,
        k,
        v,
        out_padded,
        cu_seqlens_q,
        cu_seqlens_k_to_pass,
        seqused_k,
        None,                  # leftpad_k
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,                 # zero_tensors
        causal,
        window_size[0],
        window_size[1],
        softcap,
        return_softmax,
        None,                  # gen
        num_splits,
    )

    if pad_size > 0:
        out_result = out_padded[..., :head_size_og]
        if out is not None:
            out.copy_(out_result)
            out_result = out
    else:
        out_result = out_padded

    return (out_result, softmax_lse) if return_softmax_lse else out_result


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    seqlens_k=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_batch_idx=None,
    leftpad_k=None,
    block_table=None,
    alibi_slopes=None,
    out=None,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    is_rotary_interleaved=False,
    num_splits=0,
    return_softmax_lse=False,
    fa_version=DEFAULT_FA_VERSION, **kwargs,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if window_size is None:
        window_size = (-1, -1)

    squeeze_q = False
    if q is not None and q.dim() == 3:
        q = q.unsqueeze(1)
        squeeze_q = True

    q, k_cache, v_cache = [maybe_contiguous(x) for x in (q, k_cache, v_cache)]
    if k is not None:
        k = maybe_contiguous(k)
    if v is not None:
        v = maybe_contiguous(v)

    out_buf = torch.empty_like(q)

    out_buf, softmax_lse = flash_attn_v100_cuda.fwd_kvcache(
        q, k_cache, v_cache,
        k, v,
        seqlens_k,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        leftpad_k,
        block_table,
        alibi_slopes,
        out_buf,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        is_rotary_interleaved,
        num_splits,
    )

    if squeeze_q:
        out_buf = out_buf.squeeze(1)

    return (out_buf, softmax_lse) if return_softmax_lse else out_buf


def sparse_attn_func(*args, **kwargs):
    raise NotImplementedError("sparse_attn_func not available")


def sparse_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError("sparse_attn_varlen_func not available")