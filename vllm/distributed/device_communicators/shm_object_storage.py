# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from itertools import chain
from multiprocessing import shared_memory
from multiprocessing.synchronize import Lock as LockType
from typing import Any
from unittest.mock import patch

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class SingleWriterShmRingBuffer:
    def __init__(
        self,
        data_buffer_size: int,
        name: str | None = None,
        create: bool = False,
    ):
        self.data_buffer_size = data_buffer_size
        self.is_writer = create

        self.ID_NBYTES = 4
        self.ID_MAX = 2**31
        self.SIZE_NBYTES = 4
        self.MD_SIZE = self.ID_NBYTES + self.SIZE_NBYTES
        self.monotonic_id_end = 0
        self.monotonic_id_start = 0
        self.data_buffer_start = 0
        self.data_buffer_end = 0

        if create:
            logger.debug("Creating new shared memory buffer: %s", name)
            self.metadata: dict[int, int] = {}
            self.shared_memory = shared_memory.SharedMemory(
                create=True, size=self.data_buffer_size, name=name
            )
        else:
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                self.shared_memory = shared_memory.SharedMemory(name=name)
                assert self.shared_memory.size >= self.data_buffer_size

        logger.debug(
            "Shared memory created/opened with name: %s, size: %d",
            self.shared_memory.name,
            self.data_buffer_size,
        )

    def handle(self):
        return (self.data_buffer_size, self.shared_memory.name)

    def clear(self) -> None:
        assert self.is_writer
        self.metadata.clear()
        self.monotonic_id_end = 0
        self.monotonic_id_start = 0
        self.data_buffer_start = 0
        self.data_buffer_end = 0

    def close(self) -> None:
        if hasattr(self, "shared_memory"):
            self.shared_memory.close()
            if self.is_writer:
                with suppress(FileNotFoundError):
                    self.shared_memory.unlink()

    def __del__(self):
        self.close()

    def int2byte(self, integer: int) -> bytes:
        return integer.to_bytes(self.ID_NBYTES, "little", signed=True)

    def byte2int(self, byte_data: bytes) -> int:
        return int.from_bytes(byte_data, "little", signed=True)

    def allocate_buf(self, size: int) -> tuple[int, int]:
        assert self.is_writer
        assert size > 0
        assert self.shared_memory.buf is not None
        size += self.MD_SIZE

        buffer_end_reset = self.data_buffer_end % self.data_buffer_size
        if buffer_end_reset + size > self.data_buffer_size:
            buffer_end_reset = (
                self.data_buffer_end // self.data_buffer_size + 1
            ) * self.data_buffer_size
        else:
            buffer_end_reset = self.data_buffer_end

        occupied_size_new = buffer_end_reset + size - self.data_buffer_start
        if occupied_size_new > self.data_buffer_size:
            raise MemoryError("Not enough space in the data buffer")

        self.data_buffer_end = buffer_end_reset

        buf_idx = self.data_buffer_end % self.data_buffer_size
        self.shared_memory.buf[buf_idx : buf_idx + self.ID_NBYTES] = self.int2byte(
            self.monotonic_id_end
        )
        self.shared_memory.buf[buf_idx + self.ID_NBYTES : buf_idx + self.MD_SIZE] = (
            self.int2byte(size)
        )

        self.metadata[self.monotonic_id_end % self.ID_MAX] = self.data_buffer_end

        current_buffer_end = self.data_buffer_end
        current_id_end = self.monotonic_id_end
        self.data_buffer_end += size
        self.monotonic_id_end = (self.monotonic_id_end + 1) % self.ID_MAX
        return current_buffer_end, current_id_end

    @contextmanager
    def access_buf(self, address: int):
        assert self.shared_memory.buf is not None
        buf_idx = address % self.data_buffer_size

        metadata_buff = self.shared_memory.buf[buf_idx : buf_idx + self.MD_SIZE]
        id = self.byte2int(metadata_buff[: self.ID_NBYTES])
        size = self.byte2int(metadata_buff[self.ID_NBYTES : self.MD_SIZE])

        data_buff = self.shared_memory.buf[buf_idx + self.MD_SIZE : buf_idx + size]
        with memoryview(data_buff) as data_view:
            yield data_view, (id, size)

    def free_buf(
        self,
        is_free_fn: Callable[[int, memoryview], bool],
        nbytes: int | None = None,
    ) -> Iterable[int]:
        assert self.is_writer
        logger.debug(
            "Freeing up space in the ring buffer, "
            "monotonic_id_start: %d, monotonic_id_end: %d",
            self.monotonic_id_start,
            self.monotonic_id_end,
        )
        monotonic_id_before = self.monotonic_id_start
        if nbytes is None:
            nbytes = self.data_buffer_size
        freed_bytes = 0
        while self.monotonic_id_start in self.metadata and freed_bytes < nbytes:
            address = self.metadata[self.monotonic_id_start]
            with self.access_buf(address) as (data_buff, metadata):
                if is_free_fn(self.monotonic_id_start, data_buff):
                    del self.metadata[self.monotonic_id_start]
                    self.monotonic_id_start = (
                        self.monotonic_id_start + 1
                    ) % self.ID_MAX
                    if self.monotonic_id_start in self.metadata:
                        self.data_buffer_start += (
                            self.metadata[self.monotonic_id_start]
                            - self.data_buffer_start
                        ) % self.data_buffer_size
                    else:
                        self.data_buffer_start = self.data_buffer_end = 0
                    freed_bytes += metadata[1]
                else:
                    break

        logger.debug(
            "Freed %d bytes from the ring buffer",
            freed_bytes,
        )

        if self.data_buffer_start >= self.data_buffer_size:
            self.data_buffer_start -= self.data_buffer_size
            self.data_buffer_end -= self.data_buffer_size

        monotonic_id_after = self.monotonic_id_start
        if monotonic_id_after >= monotonic_id_before:
            return range(monotonic_id_before, monotonic_id_after)
        else:
            return chain(
                range(monotonic_id_before, self.ID_MAX), range(0, monotonic_id_after)
            )


class ObjectSerde(ABC):
    @abstractmethod
    def serialize(self, value: Any) -> tuple[Any, int, bytes, int]:
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, data: memoryview) -> Any:
        raise NotImplementedError


class MsgpackSerde(ObjectSerde):
    def __init__(self):
        from vllm.multimodal.inputs import MultiModalKwargsItem
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

        self.encoder = MsgpackEncoder()
        self.tensor_decoder = MsgpackDecoder(torch.Tensor, share_mem=False)
        self.mm_decoder = MsgpackDecoder(MultiModalKwargsItem, share_mem=False)
        self._mm_kwargs_item_cls = MultiModalKwargsItem

    def serialize(self, value: Any) -> tuple[bytes | list[bytes], int, bytes, int]:
        len_arr = None
        if isinstance(value, (torch.Tensor, self._mm_kwargs_item_cls)):
            type_name = type(value).__name__
            value = self.encoder.encode(value)
            len_arr = [len(s) for s in value]
            nbytes = sum(len_arr)
        else:
            value = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            type_name = type(value).__name__
            nbytes = len(value)

        object_metadata = (type_name, nbytes, len_arr)
        serialized_metadata = pickle.dumps(
            object_metadata, protocol=pickle.HIGHEST_PROTOCOL
        )
        return value, nbytes, serialized_metadata, len(serialized_metadata)

    def deserialize(self, data_view: memoryview) -> Any:
        type_name, nbytes, len_arr = pickle.loads(data_view)
        serialized_data = data_view[-nbytes:]

        if type_name == torch.Tensor.__name__:
            obj = []
            start_idx = 0
            for length in len_arr:
                item_bytes = serialized_data[start_idx : start_idx + length]
                obj.append(item_bytes)
                start_idx += length
            obj = self.tensor_decoder.decode(obj)
        elif type_name == self._mm_kwargs_item_cls.__name__:
            obj = []
            start_idx = 0
            for length in len_arr:
                item_bytes = serialized_data[start_idx : start_idx + length]
                obj.append(item_bytes)
                start_idx += length
            obj = self.mm_decoder.decode(obj)
        elif type_name == bytes.__name__:
            obj = pickle.loads(serialized_data)
        else:
            raise ValueError(f"Unsupported object type '{type_name}' in metadata")
        return obj


@dataclass
class ShmObjectStorageHandle:
    max_object_size: int
    n_readers: int
    ring_buffer_handle: tuple[int, str]
    serde_class: type[ObjectSerde]
    reader_lock: LockType | None


class SingleWriterShmObjectStorage:
    def __init__(
        self,
        max_object_size: int,
        n_readers: int,
        ring_buffer: SingleWriterShmRingBuffer,
        serde_class: type[ObjectSerde] = MsgpackSerde,
        reader_lock: LockType | None = None,
    ):
        self.max_object_size = max_object_size
        self.n_readers = n_readers
        self.serde_class = serde_class
        self.ser_de = serde_class()
        self.ring_buffer = ring_buffer
        self.is_writer = self.ring_buffer.is_writer

        self.flag_bytes = 4

        if self.is_writer:
            self.key_index: dict[str, tuple[int, int]] = {}
            self.id_index: dict[int, str] = {}
            self.writer_flag: dict[int, int] = {}
        else:
            if reader_lock is None:
                raise ValueError("Lock must be provided for readers.")

        self._reader_lock = reader_lock

    def clear(self) -> None:
        if self.is_writer:
            self.ring_buffer.clear()
            self.key_index.clear()
            self.id_index.clear()
            self.writer_flag.clear()
            logger.debug("Object storage cleared and reinitialized.")

    def copy_to_buffer(
        self,
        data: bytes | list[bytes],
        data_bytes: int,
        metadata: bytes,
        md_bytes: int,
        data_view: memoryview,
    ) -> None:
        data_view[self.flag_bytes : self.flag_bytes + md_bytes] = metadata
        if isinstance(data, bytes):
            data_view[-data_bytes:] = data
        elif isinstance(data, list):
            start_idx = self.flag_bytes + md_bytes
            for item_bytes in data:
                item_size = len(item_bytes)
                data_view[start_idx : start_idx + item_size] = item_bytes
                start_idx += item_size
        else:
            raise ValueError(f"Unsupported data type for serialization: {type(data)}")

    def increment_writer_flag(self, id: int) -> None:
        self.writer_flag[id] = self.writer_flag.get(id, 0) + 1

    def increment_reader_flag(self, data_view: memoryview) -> None:
        reader_count = self.ring_buffer.byte2int(data_view)
        data_view[:] = self.ring_buffer.int2byte(reader_count + 1)

    def free_unused(self) -> None:
        freed_ids = self.ring_buffer.free_buf(
            self.default_is_free_check, 2 * self.max_object_size
        )
        for freed_id in freed_ids:
            key_to_free = self.id_index[freed_id]
            del self.key_index[key_to_free]
            del self.id_index[freed_id]
            del self.writer_flag[freed_id]

    def is_cached(self, key: str) -> bool:
        return key in self.key_index

    def get_cached(self, key: str) -> tuple[int, int]:
        address, monotonic_id = self.key_index[key]
        self.increment_writer_flag(monotonic_id)
        return address, monotonic_id

    def put(self, key: str, value: Any) -> tuple[int, int]:
        if key in self.key_index:
            raise ValueError(f"Key '{key}' already exists in the storage.")

        object_data, data_bytes, object_metadata, md_bytes = self.ser_de.serialize(
            value
        )
        buffer_size = self.flag_bytes + data_bytes + md_bytes

        if buffer_size > self.max_object_size:
            raise ValueError(
                f"Serialized object size ({buffer_size} bytes) exceeds "
                f"max object size ({self.max_object_size} bytes)"
            )

        try:
            address, monotonic_id = self.ring_buffer.allocate_buf(buffer_size)
        except MemoryError:
            self.free_unused()
            address, monotonic_id = self.ring_buffer.allocate_buf(buffer_size)

        with self.ring_buffer.access_buf(address) as (data_view, metadata):
            data_view[: self.flag_bytes] = self.ring_buffer.int2byte(0)
            self.copy_to_buffer(
                object_data, data_bytes, object_metadata, md_bytes, data_view
            )
        self.increment_writer_flag(monotonic_id)

        self.key_index[key] = (address, monotonic_id)
        self.id_index[monotonic_id] = key
        return address, monotonic_id

    def get(self, address: int, monotonic_id: int) -> Any:
        with self.ring_buffer.access_buf(address) as (data_view, buf_metadata):
            if buf_metadata[0] != monotonic_id:
                raise ValueError(
                    f"Data for address:id '{address}:{monotonic_id}' "
                    "has been modified or is invalid."
                )

            obj = self.ser_de.deserialize(data_view[self.flag_bytes :])

            if self._reader_lock is not None:
                with self._reader_lock:
                    self.increment_reader_flag(data_view[: self.flag_bytes])
            else:
                assert self.is_writer
        return obj

    def touch(
        self,
        key: str,
        address: int = 0,
        monotonic_id: int = 0,
    ) -> None:
        if self._reader_lock is None:
            if key not in self.key_index:
                return None
            address, monotonic_id = self.key_index[key]
            self.increment_writer_flag(monotonic_id)
        else:
            with (
                self._reader_lock,
                self.ring_buffer.access_buf(address) as (data_view, _),
            ):
                reader_count = self.ring_buffer.byte2int(data_view[: self.flag_bytes])
                if reader_count >= self.n_readers:
                    self.increment_reader_flag(data_view[: self.flag_bytes])

    def close(self) -> None:
        self.ring_buffer.close()

    def handle(self):
        return ShmObjectStorageHandle(
            max_object_size=self.max_object_size,
            n_readers=self.n_readers,
            ring_buffer_handle=self.ring_buffer.handle(),
            serde_class=self.serde_class,
            reader_lock=self._reader_lock,
        )

    @staticmethod
    def create_from_handle(
        handle: ShmObjectStorageHandle,
    ) -> "SingleWriterShmObjectStorage":
        logger.debug("Creating storage from handle: %s", handle)
        ring_buffer = SingleWriterShmRingBuffer(*handle.ring_buffer_handle)
        return SingleWriterShmObjectStorage(
            max_object_size=handle.max_object_size,
            n_readers=handle.n_readers,
            ring_buffer=ring_buffer,
            serde_class=handle.serde_class,
            reader_lock=handle.reader_lock,
        )

    def default_is_free_check(self, id: int, buf: memoryview) -> bool:
        reader_count = int.from_bytes(buf[0:4], "little", signed=True)
        writer_count = self.writer_flag[id]
        return reader_count >= writer_count * self.n_readers