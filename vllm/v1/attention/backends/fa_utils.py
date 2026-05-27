# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# CUDA only paths
from vllm._custom_ops import reshape_and_cache_flash
from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
    flash_attn_varlen_func,
    get_scheduler_metadata,
)

def get_flash_attn_version(
    requires_alibi: bool = False, head_size: Optional[int] = None
) -> Optional[int]:
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()
        assert device_capability is not None

        # 1. default version depending on platform
        if device_capability.major == 9 and is_fa_version_supported(3):
            fa_version = 3
        elif device_capability.major == 10 and is_fa_version_supported(4):
            fa_version = 4
        else:
            fa_version = 2

        # 2. override if passed by environment or config
        from vllm.config import get_current_vllm_config_or_none
        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.attention_config.flash_attn_version is not None
        ):
            fa_version = vllm_config.attention_config.flash_attn_version

        # 3. fallback for unsupported combinations
        if device_capability.major >= 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform, "
                "defaulting to FA version 4 if supported, otherwise FA2."
            )
            fa_version = 4 if is_fa_version_supported(4) else 2

        if requires_alibi and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if requires_alibi and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if envs.VLLM_BATCH_INVARIANT and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with batch invariance, "
                "defaulting to FA version 2.",
                scope="local",
            )
            fa_version = 2

        if (
            fa_version == 4
            and device_capability.major >= 10
            and head_size is not None
            and head_size > 128
            and head_size != 192
        ):
            logger.warning_once(
                "FA4 on Blackwell does not support head_size=%d due to TMEM "
                "capacity limits, defaulting to FA version 2.",
                head_size,
            )
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error(
                "Cannot use FA version %d is not supported due to %s",
                fa_version,
                fa_version_unsupported_reason(fa_version),
            )

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None

def flash_attn_supports_fp8() -> bool:
    return (
        get_flash_attn_version() == 3
        and current_platform.is_device_capability_family(90)
    )

def flash_attn_supports_sinks() -> bool:
    return get_flash_attn_version() == 3

def flash_attn_supports_mla() -> bool:
    if current_platform.is_cuda():
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                is_fa_version_supported,
            )
            return is_fa_version_supported(
                3
            ) and current_platform.is_device_capability_family(90)
        except (ImportError, AssertionError):
            pass
    return False

def is_flash_attn_varlen_func_available() -> bool:
    """Check if flash_attn_varlen_func is available (CUDA only)."""
    if current_platform.is_cuda():
        return True
    return False
