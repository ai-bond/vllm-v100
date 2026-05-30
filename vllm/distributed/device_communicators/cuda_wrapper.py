# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from dataclasses import dataclass
from typing import Any

import torch  # noqa
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.system_utils import find_loaded_library

logger = init_logger(__name__)

cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


class CudaRTLibrary:
    exported_functions = [
        Function("cudaSetDevice", cudaError_t, [ctypes.c_int]),
        Function("cudaDeviceSynchronize", cudaError_t, []),
        Function("cudaDeviceReset", cudaError_t, []),
        Function("cudaGetErrorString", ctypes.c_char_p, [cudaError_t]),
        Function(
            "cudaMalloc",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        ),
        Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        Function(
            "cudaMemset", cudaError_t, [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        ),
        Function(
            "cudaMemcpy",
            cudaError_t,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cudaMemcpyKind],
        ),
        Function(
            "cudaIpcGetMemHandle",
            cudaError_t,
            [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p],
        ),
        Function(
            "cudaIpcOpenMemHandle",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint],
        ),
    ]

    path_to_library_cache: dict[str, Any] = {}
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str | None = None):
        if so_file is None:
            so_file = find_loaded_library("libcudart")
            if so_file is None:
                so_file = envs.VLLM_CUDART_SO_PATH
        assert so_file is not None, (
            "libcudart is not loaded in the current process, "
            "try setting VLLM_CUDART_SO_PATH"
        )

        if so_file not in CudaRTLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            CudaRTLibrary.path_to_library_cache[so_file] = lib
        self.lib = CudaRTLibrary.path_to_library_cache[so_file]

        if so_file not in CudaRTLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in CudaRTLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            CudaRTLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = CudaRTLibrary.path_to_dict_mapping[so_file]

    def CUDART_CHECK(self, result: cudaError_t) -> None:
        if result != 0:
            error_str = self.cudaGetErrorString(result)
            raise RuntimeError(f"CUDART error: {error_str}")

    def cudaGetErrorString(self, error: cudaError_t) -> str:
        return self.funcs["cudaGetErrorString"](error).decode("utf-8")

    def cudaSetDevice(self, device: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaSetDevice"](device))

    def cudaDeviceSynchronize(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceSynchronize"]())

    def cudaDeviceReset(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceReset"]())

    def cudaMalloc(self, size: int) -> ctypes.c_void_p:
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(self.funcs["cudaMalloc"](ctypes.byref(devPtr), size))
        return devPtr

    def cudaFree(self, devPtr: ctypes.c_void_p) -> None:
        self.CUDART_CHECK(self.funcs["cudaFree"](devPtr))

    def cudaMemset(self, devPtr: ctypes.c_void_p, value: int, count: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaMemset"](devPtr, value, count))

    def cudaMemcpy(
        self, dst: ctypes.c_void_p, src: ctypes.c_void_p, count: int
    ) -> None:
        cudaMemcpyDefault = 4
        kind = cudaMemcpyDefault
        self.CUDART_CHECK(self.funcs["cudaMemcpy"](dst, src, count, kind))

    def cudaIpcGetMemHandle(self, devPtr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        handle = cudaIpcMemHandle_t()
        self.CUDART_CHECK(
            self.funcs["cudaIpcGetMemHandle"](ctypes.byref(handle), devPtr)
        )
        return handle

    def cudaIpcOpenMemHandle(self, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        cudaIpcMemLazyEnablePeerAccess = 1
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(
            self.funcs["cudaIpcOpenMemHandle"](
                ctypes.byref(devPtr), handle, cudaIpcMemLazyEnablePeerAccess
            )
        )
        return devPtr