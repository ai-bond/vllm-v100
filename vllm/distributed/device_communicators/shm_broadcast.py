# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import pickle
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pickle import PickleBuffer
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import torch
import torch.distributed as dist
import zmq
from torch.distributed import ProcessGroup
from zmq import (
    IPV6,
    PUB,
    SUB,
    SUBSCRIBE,
    XPUB,
    XPUB_VERBOSE,
    Context,
)

import vllm.envs as envs
from vllm.distributed.utils import StatelessProcessGroup, sched_yield
from vllm.logger import init_logger
from vllm.utils.network_utils import (
    get_ip,
    get_open_port,
    get_open_zmq_inproc_path,
    get_open_zmq_ipc_path,
    is_valid_ipv6_address,
)

if TYPE_CHECKING:
    from _typeshed import SizedBuffer

VLLM_RINGBUFFER_WARNING_INTERVAL = envs.VLLM_RINGBUFFER_WARNING_INTERVAL

from_bytes_big = functools.partial(int.from_bytes, byteorder="big")

_memory_fence_lock = threading.Lock()


def memory_fence():
    """Full memory barrier for shared memory synchronization."""
    with _memory_fence_lock:
        pass


def to_bytes_big(value: int, size: int) -> bytes:
    return value.to_bytes(size, byteorder="big")


logger = init_logger(__name__)

LONG_WAIT_TIME_LOG_MSG = (
    "No available shared memory broadcast block found "
    "in %d seconds. This typically happens "
    "when some processes are hanging or doing some "
    "time-consuming work (e.g. compilation, "
    "weight/kv cache quantization)."
)


class SpinCondition:
    def __init__(
        self,
        is_reader: bool,
        context: zmq.Context,
        notify_address: str,
        busy_loop_s: float = 1,
    ):
        self.is_reader = is_reader

        if is_reader:
            self.last_read = time.monotonic()
            self.busy_loop_s = busy_loop_s

            self.local_notify_socket: zmq.Socket = context.socket(SUB)
            self.local_notify_socket.setsockopt(zmq.CONFLATE, 1)
            self.local_notify_socket.setsockopt_string(SUBSCRIBE, "")
            self.local_notify_socket.connect(notify_address)

            cancel_path = get_open_zmq_inproc_path()
            self.write_cancel_socket: zmq.Socket = context.socket(zmq.PAIR)
            self.write_cancel_socket.bind(cancel_path)
            self.read_cancel_socket: zmq.Socket = context.socket(zmq.PAIR)
            self.read_cancel_socket.connect(cancel_path)

            self.poller = zmq.Poller()
            self.poller.register(self.read_cancel_socket, zmq.POLLIN)
            self.poller.register(self.local_notify_socket, zmq.POLLIN)
        else:
            self.local_notify_socket: zmq.Socket = context.socket(PUB)
            self.local_notify_socket.setsockopt(zmq.SNDHWM, 1)
            self.local_notify_socket.bind(notify_address)

            self.last_read = 0
            self.busy_loop_s = 0
            self.read_cancel_socket = None
            self.write_cancel_socket = None
            self.poller = None

    def record_read(self):
        self.last_read = time.monotonic()

    def cancel(self):
        if self.is_reader:
            logger.debug("Canceling waiting reads on SHM Buffer")
            self.write_cancel_socket.send(b"\x00")

    def wait(self, timeout_ms: int | None = None) -> None:
        assert self.is_reader, "Only readers can wait"
        current_time = time.monotonic()
        if current_time <= self.last_read + self.busy_loop_s:
            sched_yield()
        else:
            events = dict(self.poller.poll(timeout=timeout_ms))
            if self.read_cancel_socket in events:
                logger.debug("Poller received cancel event")
            elif self.local_notify_socket in events:
                logger.debug("Poller received notify event")
                self.local_notify_socket.recv(flags=zmq.NOBLOCK, copy=False)
            else:
                logger.debug("Poller timed out")

    def notify(self):
        assert not self.is_reader, "Only writers can notify"
        self.local_notify_socket.send(b"\x00")


class ShmRingBuffer:
    def __init__(
        self,
        n_reader: int,
        max_chunk_bytes: int,
        max_chunks: int,
        name: str | None = None,
    ):
        self.n_reader = n_reader
        self.metadata_size = 1 + n_reader
        self.max_chunk_bytes = max_chunk_bytes
        self.max_chunks = max_chunks
        self.total_bytes_of_buffer = (
            self.max_chunk_bytes + self.metadata_size
        ) * self.max_chunks
        self.data_offset = 0
        self.metadata_offset = self.max_chunk_bytes * self.max_chunks

        if name is None:
            self.is_creator = True
            self.shared_memory = shared_memory.SharedMemory(
                create=True, size=self.total_bytes_of_buffer
            )
            assert self.shared_memory.buf is not None
            with self.shared_memory.buf[self.metadata_offset :] as metadata_buffer:
                torch.frombuffer(metadata_buffer, dtype=torch.uint8).fill_(0)
        else:
            self.is_creator = False
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                try:
                    self.shared_memory = shared_memory.SharedMemory(name=name)
                    assert self.shared_memory.size >= self.total_bytes_of_buffer
                except FileNotFoundError:
                    pass

    def handle(self):
        return (
            self.n_reader,
            self.max_chunk_bytes,
            self.max_chunks,
            self.shared_memory.name,
        )

    def __reduce__(self):
        return (self.__class__, self.handle())

    def __del__(self):
        if hasattr(self, "shared_memory"):
            self.shared_memory.close()
            if self.is_creator:
                self.shared_memory.unlink()

    @contextmanager
    def get_data(self, current_idx: int):
        start = self.data_offset + current_idx * self.max_chunk_bytes
        end = start + self.max_chunk_bytes
        assert self.shared_memory.buf is not None
        with self.shared_memory.buf[start:end] as buf:
            yield buf

    @contextmanager
    def get_metadata(self, current_idx: int):
        start = self.metadata_offset + current_idx * self.metadata_size
        end = start + self.metadata_size
        assert self.shared_memory.buf is not None
        with self.shared_memory.buf[start:end] as buf:
            yield buf


@dataclass
class Handle:
    local_reader_ranks: list[int] = field(default_factory=list)
    buffer_handle: tuple[int, int, int, str] | None = None
    local_subscribe_addr: str | None = None
    local_notify_addr: str | None = None
    remote_subscribe_addr: str | None = None
    remote_addr_ipv6: bool = False


class MessageQueue:
    class ReadTimeoutWithWarnings:
        def __init__(self, timeout: float | None, should_warn: bool) -> None:
            self.started = time.monotonic()
            self.deadline = sys.maxsize if timeout is None else self.started + timeout
            self.warning_wait_time_ms: int | None = (
                VLLM_RINGBUFFER_WARNING_INTERVAL * 1000 if should_warn else None
            )
            self._should_warn = should_warn
            self.n_warning = 1
            self.timeout = timeout

        def timeout_ms(self) -> int | None:
            warning_wait_time = self.warning_wait_time_ms
            if self.timeout is None:
                return warning_wait_time
            time_left_ms = int((self.deadline - time.monotonic()) * 1000)
            if time_left_ms <= 0:
                raise TimeoutError
            if warning_wait_time and warning_wait_time < time_left_ms:
                return warning_wait_time
            return time_left_ms

        def should_warn(self) -> bool:
            if self._should_warn:
                elapsed = time.monotonic() - self.started
                if elapsed >= VLLM_RINGBUFFER_WARNING_INTERVAL * self.n_warning:
                    self.n_warning += 1
                    return True
            return False

    def __init__(
        self,
        n_reader,
        n_local_reader,
        local_reader_ranks: list[int] | None = None,
        max_chunk_bytes: int = 1024 * 1024 * 24,
        max_chunks: int = 10,
        connect_ip: str | None = None,
    ):
        if local_reader_ranks is None:
            local_reader_ranks = list(range(n_local_reader))
        else:
            assert len(local_reader_ranks) == n_local_reader

        self.n_local_reader = n_local_reader
        n_remote_reader = n_reader - n_local_reader
        self.n_remote_reader = n_remote_reader
        self.shutting_down = False
        context = Context()

        if n_local_reader > 0:
            self.buffer = ShmRingBuffer(n_local_reader, max_chunk_bytes, max_chunks)
            self.local_socket = context.socket(XPUB)
            self.local_socket.setsockopt(XPUB_VERBOSE, True)
            local_subscribe_addr = get_open_zmq_ipc_path()
            logger.debug("Binding to %s", local_subscribe_addr)
            self.local_socket.bind(local_subscribe_addr)
            self.current_idx = 0
            local_notify_addr = get_open_zmq_ipc_path()
            self._spin_condition = SpinCondition(
                is_reader=False, context=context, notify_address=local_notify_addr
            )
        else:
            self.buffer = None
            local_subscribe_addr = None
            self.local_socket = None
            self.current_idx = -1
            local_notify_addr = None
            self._spin_condition = None

        remote_addr_ipv6 = False
        if n_remote_reader > 0:
            if not connect_ip:
                connect_ip = get_ip()
            self.remote_socket = context.socket(XPUB)
            self.remote_socket.setsockopt(XPUB_VERBOSE, True)
            remote_subscribe_port = get_open_port()
            if is_valid_ipv6_address(connect_ip):
                self.remote_socket.setsockopt(IPV6, 1)
                remote_addr_ipv6 = True
                connect_ip = f"[{connect_ip}]"
            socket_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"
            self.remote_socket.bind(socket_addr)
            remote_subscribe_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"
        else:
            remote_subscribe_addr = None
            self.remote_socket = None

        self._is_writer = True
        self._is_local_reader = False
        self.local_reader_rank = -1
        self._is_remote_reader = False

        self.handle = Handle(
            local_reader_ranks=local_reader_ranks,
            buffer_handle=self.buffer.handle() if self.buffer is not None else None,
            local_subscribe_addr=local_subscribe_addr,
            local_notify_addr=local_notify_addr,
            remote_subscribe_addr=remote_subscribe_addr,
            remote_addr_ipv6=remote_addr_ipv6,
        )
        logger.debug("vLLM message queue communication handle: %s", self.handle)

    def export_handle(self) -> Handle:
        return self.handle

    @staticmethod
    def create_from_handle(handle: Handle, rank) -> "MessageQueue":
        self = MessageQueue.__new__(MessageQueue)
        self.handle = handle
        self._is_writer = False
        context = Context()

        if rank in handle.local_reader_ranks:
            assert handle.buffer_handle is not None
            self.buffer = ShmRingBuffer(*handle.buffer_handle)
            self.current_idx = 0
            self.local_reader_rank = handle.local_reader_ranks.index(rank)
            self._is_local_reader = True
            self._is_remote_reader = False

            self.local_socket = context.socket(SUB)
            self.local_socket.setsockopt_string(SUBSCRIBE, "")
            socket_addr = handle.local_subscribe_addr
            logger.debug("Connecting to %s", socket_addr)
            self.local_socket.connect(socket_addr)
            self.remote_socket = None
            assert isinstance(handle.local_notify_addr, str)
            self._spin_condition = SpinCondition(
                is_reader=True, context=context, notify_address=handle.local_notify_addr
            )
        else:
            self.buffer = None
            self.current_idx = -1
            self.local_reader_rank = -1
            self._is_local_reader = False
            self._is_remote_reader = True
            self.local_socket = None
            self.remote_socket = context.socket(SUB)
            self.remote_socket.setsockopt_string(SUBSCRIBE, "")
            if handle.remote_addr_ipv6:
                self.remote_socket.setsockopt(IPV6, 1)
            socket_addr = handle.remote_subscribe_addr
            logger.debug("Connecting to %s", socket_addr)
            self.remote_socket.connect(socket_addr)
            self._spin_condition = None

        self.shutting_down = False
        return self

    def wait_until_ready(self):
        if self._is_writer:
            for i in range(self.n_local_reader):
                self.local_socket.recv()
            if self.n_local_reader > 0:
                self.local_socket.send(b"READY")
            for i in range(self.n_remote_reader):
                self.remote_socket.recv()
            if self.n_remote_reader > 0:
                self.remote_socket.send(b"READY")
        elif self._is_local_reader:
            recv = self.local_socket.recv()
            assert recv == b"READY"
        elif self._is_remote_reader:
            recv = self.remote_socket.recv()
            assert recv == b"READY"

    def shutdown(self):
        self.shutting_down = True
        if self._spin_condition is not None:
            self._spin_condition.cancel()

    @contextmanager
    def acquire_write(self, timeout: float | None = None):
        assert self._is_writer, "Only writers can acquire write"
        start_time = time.monotonic()
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                memory_fence()
                read_count = sum(metadata_buffer[1:])
                written_flag = metadata_buffer[0]
                if written_flag and read_count != self.buffer.n_reader:
                    sched_yield()
                    elapsed = time.monotonic() - start_time
                    if timeout is not None and elapsed > timeout:
                        raise TimeoutError
                    if elapsed > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning:
                        logger.info(
                            LONG_WAIT_TIME_LOG_MSG, VLLM_RINGBUFFER_WARNING_INTERVAL
                        )
                        n_warning += 1
                    continue

                metadata_buffer[0] = 0
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                for i in range(1, self.buffer.n_reader + 1):
                    metadata_buffer[i] = 0
                memory_fence()
                metadata_buffer[0] = 1
                memory_fence()
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                break

    @contextmanager
    def acquire_read(self, timeout: float | None = None, indefinite: bool = False):
        assert self._is_local_reader, "Only readers can acquire read"
        read_timeout = self.ReadTimeoutWithWarnings(
            timeout=timeout, should_warn=not indefinite
        )
        with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
            while True:
                memory_fence()
                read_flag = metadata_buffer[self.local_reader_rank + 1]
                written_flag = metadata_buffer[0]
                if not written_flag or read_flag:
                    self._spin_condition.wait(timeout_ms=read_timeout.timeout_ms())
                    if self.shutting_down:
                        raise RuntimeError("cancelled")
                    if read_timeout.should_warn():
                        logger.info(
                            LONG_WAIT_TIME_LOG_MSG, VLLM_RINGBUFFER_WARNING_INTERVAL
                        )
                    continue

                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                metadata_buffer[self.local_reader_rank + 1] = 1
                memory_fence()
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                self._spin_condition.record_read()
                break

    def enqueue(self, obj, timeout: float | None = None):
        assert self._is_writer, "Only writers can enqueue"
        all_buffers: list[SizedBuffer] = [b""]
        total_bytes = 6

        def oob_callback(buf: PickleBuffer) -> bool:
            raw_buf = buf.raw()
            if len(raw_buf) < 1024 * 1024:
                return True
            all_buffers.append(raw_buf)
            nonlocal total_bytes
            total_bytes += len(raw_buf) + 4
            return False

        all_buffers[0] = pickle.dumps(
            obj, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=oob_callback
        )

        if self.n_local_reader > 0:
            if total_bytes + len(all_buffers[0]) >= self.buffer.max_chunk_bytes:
                with self.acquire_write(timeout) as buf:
                    buf[0] = 1
                self.local_socket.send_multipart(all_buffers, copy=False)
            else:
                with self.acquire_write(timeout) as buf:
                    buf[0] = 0
                    offset = 3
                    buf[1:offset] = to_bytes_big(len(all_buffers), 2)
                    for buffer in all_buffers:
                        buf_len = len(buffer)
                        buf_offset = offset + 4
                        buf[offset:buf_offset] = to_bytes_big(buf_len, 4)
                        buf[buf_offset : (offset := buf_offset + buf_len)] = buffer
            self._spin_condition.notify()

        if self.n_remote_reader > 0:
            self.remote_socket.send_multipart(all_buffers, copy=False)

    def dequeue(self, timeout: float | None = None, indefinite: bool = False):
        if self._is_local_reader:
            with self.acquire_read(timeout, indefinite) as buf:
                overflow = buf[0] == 1
                if not overflow:
                    offset = 3
                    buf_count = from_bytes_big(buf[1:offset])
                    all_buffers = []
                    for i in range(buf_count):
                        buf_offset = offset + 4
                        buf_len = from_bytes_big(buf[offset:buf_offset])
                        offset = buf_offset + buf_len
                        all_buffers.append(buf[buf_offset:offset])
                    obj = pickle.loads(all_buffers[0], buffers=all_buffers[1:])
            if overflow:
                obj = MessageQueue.recv(self.local_socket, timeout)
        elif self._is_remote_reader:
            obj = MessageQueue.recv(self.remote_socket, timeout)
        else:
            raise RuntimeError("Only readers can dequeue")
        return obj

    @staticmethod
    def recv(socket: zmq.Socket, timeout: float | None) -> Any:
        timeout_ms = None if timeout is None else int(timeout * 1000)
        if not socket.poll(timeout=timeout_ms):
            raise TimeoutError
        recv, *recv_oob = socket.recv_multipart(copy=False)
        return pickle.loads(recv, buffers=recv_oob)

    def broadcast_object(self, obj=None):
        if self._is_writer:
            self.enqueue(obj)
            return obj
        return self.dequeue()

    @staticmethod
    def create_from_process_group_single_reader(
        pg: ProcessGroup,
        max_chunk_bytes,
        max_chunks,
        reader_rank: int = 0,
        blocking: bool = False,
    ) -> tuple["MessageQueue", list[Handle]]:
        # FIXED: use torch.cuda.device_count() directly instead of current_platform
        local_size = torch.cuda.device_count()
        rank = dist.get_rank()
        same_node = rank // local_size == reader_rank // local_size
        buffer_io = MessageQueue(
            n_reader=1,
            n_local_reader=1 if same_node else 0,
            max_chunk_bytes=max_chunk_bytes,
            max_chunks=max_chunks,
        )
        handle = buffer_io.export_handle()
        handles = [None] * dist.get_world_size(pg) if rank == reader_rank else None
        dist.gather_object(handle, handles, dst=reader_rank, group=pg)
        if blocking:
            buffer_io.wait_until_ready()
        return buffer_io, cast(list[Handle], handles or [])

    @staticmethod
    def create_from_process_group(
        pg: ProcessGroup | StatelessProcessGroup,
        max_chunk_bytes,
        max_chunks,
        writer_rank: int = 0,
        external_writer_handle=None,
        blocking: bool = True,
    ) -> "MessageQueue":
        if isinstance(pg, ProcessGroup):
            group_rank = dist.get_rank(pg)
            group_world_size = dist.get_world_size(pg)
            global_ranks = dist.get_process_group_ranks(pg)
        else:
            group_rank = pg.rank
            group_world_size = pg.world_size
            global_ranks = list(range(pg.world_size))

        from vllm.distributed.parallel_state import in_the_same_node_as

        status = in_the_same_node_as(pg, source_rank=writer_rank)

        if group_rank == writer_rank:
            if external_writer_handle is not None:
                buffer_io = MessageQueue.create_from_handle(
                    external_writer_handle, group_rank
                )
            else:
                same_node_ranks = [i for i, s in enumerate(status) if s]
                n_reader = group_world_size - 1
                n_local_reader = len(same_node_ranks) - 1
                local_reader_ranks = [i for i in same_node_ranks if i != writer_rank]
                buffer_io = MessageQueue(
                    n_reader=n_reader,
                    n_local_reader=n_local_reader,
                    local_reader_ranks=local_reader_ranks,
                    max_chunk_bytes=max_chunk_bytes,
                    max_chunks=max_chunks,
                )
            handle = buffer_io.export_handle()
            if isinstance(pg, ProcessGroup):
                dist.broadcast_object_list(
                    [handle], src=global_ranks[writer_rank], group=pg
                )
            else:
                pg.broadcast_obj(handle, writer_rank)
        else:
            if isinstance(pg, ProcessGroup):
                recv = [None]
                dist.broadcast_object_list(
                    recv, src=global_ranks[writer_rank], group=pg
                )
                handle = recv[0]
            else:
                handle = pg.broadcast_obj(None, writer_rank)
            buffer_io = MessageQueue.create_from_handle(handle, group_rank)

        if blocking:
            buffer_io.wait_until_ready()
        return buffer_io