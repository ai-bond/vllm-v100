# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_GLOBAL_RUNTIME_DATA: dict[str, str | int | bool] = {}


def set_runtime_usage_data(key: str, value: str | int | bool) -> None:
    return None


def is_usage_stats_enabled() -> bool:
    return False


class UsageContext(str, Enum):
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"
    LLM_CLASS = "LLM_CLASS"
    API_SERVER = "API_SERVER"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"
    OPENAI_BATCH_RUNNER = "OPENAI_BATCH_RUNNER"
    ENGINE_CONTEXT = "ENGINE_CONTEXT"


class UsageMessage:
    def __init__(self) -> None:
        pass

    def report_usage(
        self,
        model_architecture: str,
        usage_context: UsageContext,
        extra_kvs: dict[str, Any] | None = None,
    ) -> None:
        logger.debug(
            "Usage reporting is disabled in this build. "
            "No data collected or transmitted."
        )
        return None

    def _report_usage_worker(
        self,
        model_architecture: str,
        usage_context: UsageContext,
        extra_kvs: dict[str, Any],
    ) -> None:
        return None

    def _report_usage_once(
        self,
        model_architecture: str,
        usage_context: UsageContext,
        extra_kvs: dict[str, Any],
    ) -> None:
        return None

    def _report_continuous_usage(self) -> None:
        return None

    def _send_to_server(self, data: dict[str, Any]) -> None:
        return None

    def _write_to_file(self, data: dict[str, Any]) -> None:
        return None

usage_message = UsageMessage()
