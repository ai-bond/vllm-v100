# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from itertools import product
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.logger import init_logger
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import cuda_device_count_stateless

logger = init_logger(__name__)

KiB = 1024
MiB = 1024 * 1024

# Custom AllReduce max sizes for Volta (sm_7.0).
# Conservative values since Volta lacks some of the newer IPC/NVLink features.
CUSTOM_ALL_REDUCE_MAX_SIZES = {
    "7.0": {
        2: 8 * MiB,   # 8 MB
        4: 4 * MiB,   # 4 MB
        6: 2 * MiB,   # 2 MB
        8: 1 * MiB,   # 1 MB
    },
}


def should_nccl_symm_mem_allreduce(world_size: int, input_tensor: torch.Tensor) -> bool:
    """
    NCCL Symmetric Memory requires Hopper (sm_9.0) architecture and NCCL >= 2.27.3.
    Disabled on Volta (sm_7.0). Always returns False.
    """
    return False


def producer(
    batch_src: Sequence[int],
    producer_queue,
    consumer_queue,
    result_queue,
    cuda_visible_devices: str | None = None,
):
    if cuda_visible_devices is not None:
        update_environment_variables({"CUDA_VISIBLE_DEVICES": cuda_visible_devices})
    lib = CudaRTLibrary()
    for i in batch_src:
        lib.cudaSetDevice(i)
        pointer = lib.cudaMalloc(1024)
        lib.cudaMemset(pointer, 1, 1024)
        lib.cudaDeviceSynchronize()
        handle = lib.cudaIpcGetMemHandle(pointer)
        producer_queue.put(handle)
        open_success = consumer_queue.get()
        if open_success:
            producer_queue.put(0)
            consumer_queue.get()
            host_data = (ctypes.c_char * 1024)()
            lib.cudaMemcpy(host_data, pointer, 1024)
            for i in range(1024):
                if ord(host_data[i]) != 2:
                    open_success = False
                    break
        result_queue.put(open_success)
        lib.cudaDeviceReset()


def consumer(
    batch_tgt: Sequence[int],
    producer_queue,
    consumer_queue,
    result_queue,
    cuda_visible_devices: str | None = None,
):
    if cuda_visible_devices is not None:
        update_environment_variables({"CUDA_VISIBLE_DEVICES": cuda_visible_devices})
    lib = CudaRTLibrary()
    for j in batch_tgt:
        lib.cudaSetDevice(j)
        handle = producer_queue.get()
        open_success = False
        try:
            pointer = lib.cudaIpcOpenMemHandle(handle)
            open_success = True
        except RuntimeError:
            pass
        consumer_queue.put(open_success)
        if open_success:
            lib.cudaMemset(pointer, 2, 1024)
            lib.cudaDeviceSynchronize()
            producer_queue.get()
            consumer_queue.put(0)
            host_data = (ctypes.c_char * 1024)()
            lib.cudaMemcpy(host_data, pointer, 1024)
            for i in range(1024):
                if ord(host_data[i]) != 2:
                    open_success = False
                    break
        result_queue.put(open_success)
        lib.cudaDeviceReset()


def can_actually_p2p(
    batch_src: Sequence[int],
    batch_tgt: Sequence[int],
) -> Sequence[bool]:
    """
    Perform real P2P access test via CUDA IPC to verify GPU-to-GPU
    communication actually works (driver can report false positives).
    """
    cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES

    smp = mp.get_context("spawn")
    producer_queue = smp.Queue()
    consumer_queue = smp.Queue()
    result_queue = smp.Queue()
    p_src = smp.Process(
        target=producer,
        args=(
            batch_src,
            producer_queue,
            consumer_queue,
            result_queue,
            cuda_visible_devices,
        ),
    )
    p_tgt = smp.Process(
        target=consumer,
        args=(
            batch_tgt,
            producer_queue,
            consumer_queue,
            result_queue,
            cuda_visible_devices,
        ),
    )
    p_src.start()
    p_tgt.start()
    p_src.join()
    p_tgt.join()
    assert p_src.exitcode == 0 and p_tgt.exitcode == 0
    result: list[bool] = []
    for src, tgt in zip(batch_src, batch_tgt):
        a = result_queue.get()
        b = result_queue.get()
        if a != b:
            logger.warning(
                "Two processes do not agree on the P2P access "
                "status on %d -> %d, treat as disabled.",
                src,
                tgt,
            )
            result.append(False)
        else:
            result.append(a)
    return result


_gpu_p2p_access_cache: dict[str, bool] | None = None


def gpu_p2p_access_check(src: int, tgt: int) -> bool:
    """Check if GPU src can access GPU tgt."""
    global _gpu_p2p_access_cache
    if _gpu_p2p_access_cache is not None:
        return _gpu_p2p_access_cache[f"{src}->{tgt}"]

    is_distributed = dist.is_initialized()

    num_dev = cuda_device_count_stateless()
    cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
    if cuda_visible_devices is None:
        cuda_visible_devices = ",".join(str(i) for i in range(num_dev))

    path = os.path.join(
        envs.VLLM_CACHE_ROOT, f"gpu_p2p_access_cache_for_{cuda_visible_devices}.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)

    from vllm.distributed.parallel_state import get_world_group

    if (not is_distributed or get_world_group().local_rank == 0) and (
        not os.path.exists(path)
    ):
        logger.info("generating GPU P2P access cache in %s", path)
        cache: dict[str, bool] = {}
        ids = list(range(num_dev))
        batch_src, batch_tgt = zip(*list(product(ids, ids)))

        with tempfile.NamedTemporaryFile() as output_file:
            input_bytes = pickle.dumps((batch_src, batch_tgt, output_file.name))
            returned = subprocess.run(
                [sys.executable, __file__], input=input_bytes, capture_output=True
            )
            try:
                returned.check_returncode()
            except Exception as e:
                raise RuntimeError(
                    f"Error happened when batch testing "
                    f"peer-to-peer access from {batch_src} to {batch_tgt}:\n"
                    f"{returned.stderr.decode()}"
                ) from e
            with open(output_file.name, "rb") as f:
                result = pickle.load(f)
        for _i, _j, r in zip(batch_src, batch_tgt, result):
            cache[f"{_i}->{_j}"] = r
        with open(path, "w") as f:
            json.dump(cache, f, indent=4)

    if is_distributed:
        get_world_group().barrier()

    logger.info("reading GPU P2P access cache from %s", path)
    with open(path) as f:
        cache = json.load(f)
    _gpu_p2p_access_cache = cache
    return _gpu_p2p_access_cache[f"{src}->{tgt}"]


__all__ = ["gpu_p2p_access_check", "CUSTOM_ALL_REDUCE_MAX_SIZES"]

if __name__ == "__main__":
    batch_src, batch_tgt, output_file = pickle.loads(sys.stdin.buffer.read())
    result = can_actually_p2p(batch_src, batch_tgt)
    with open(output_file, "wb") as f:
        f.write(pickle.dumps(result))