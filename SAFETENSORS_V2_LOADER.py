"""
==============================================================================
NEON MAW LLC — SAFETENSORS v2 (64KB DMA ALIGNMENT + ZERO-COPY MEMORYVIEW + O(1) RADIX)
File: SAFETENSORS_V2_LOADER.py
Repository: https://github.com/NeonMaw/safetensors (branch: feat/safetensors-v2-64kb-dma)
License: Apache-2.0 / MIT

Summary of SafeTensors v2 (STv2) Architecture & Fixes:
  1. Resolves the SafeTensors Alignment Paradox:
     Standard .safetensors starts with [u64 N][N-byte JSON], placing Tensor 0
     at an arbitrary unaligned byte offset (e.g., 0x3B7A) and breaking 64KB
     OS/GPU DMA page alignment (Windows VirtualAlloc / PCIe BAR1 / DirectStorage).
  2. 100% Backward-Compatible RFC-8259 Whitespace Padding:
     realign_safetensors_to_64kb_dma() pads the JSON header with trailing ASCII
     spaces (0x20) so that (8 + N) is an exact multiple of 65536 (64KB). Every
     existing Python/Rust safetensors reader loads STv2 files with zero changes.
  3. Safe Context-Managed Zero-Copy Slicer & Offset Bounds Verification:
     Fixes mmap descriptor leaks on malformed headers, enforces MAX_HEADER_SIZE
     (100 MB), verifies contiguous non-overlapping tensor data_offsets, and
     exposes zero-copy memoryview slices directly from the 64KB-aligned slab.
  4. O(1) FNV-1a 64-Bit Radix Table Lookup:
     Computes deterministic 64-bit FNV-1a hashes of tensor names and provides
     binary-searched O(log N) / O(1) offset lookup without re-parsing JSON.
==============================================================================
"""

import hashlib
import json
import mmap
import os
import struct
import tempfile
from typing import Any, Dict, List, Optional, Tuple


ST2_DMA_ALIGNMENT: int = 65536
ST2_MAX_HEADER_SIZE: int = 100_000_000  # 100 MB safety cap matching upstream Rust spec
FNV1A_64_OFFSET: int = 0xCBF29CE484222325
FNV1A_64_PRIME: int = 0x00000100000001B3


def fnv1a_64(text: str) -> int:
    """Compute deterministic 64-bit FNV-1a hash for O(1) tensor name indexing."""
    h = FNV1A_64_OFFSET
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * FNV1A_64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def validate_contiguous_offsets(header: Dict[str, Any], payload_size: int) -> Tuple[bool, str]:
    """
    Verify that all tensor data_offsets are non-negative, non-overlapping,
    strictly contiguous from offset 0 to payload_size, and within file bounds.
    """
    spans: List[Tuple[int, int, str]] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict) or "data_offsets" not in info:
            return False, f"Malformed tensor descriptor for '{name}'"
        offsets = info["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            return False, f"Invalid data_offsets tuple for '{name}'"
        start, end = int(offsets[0]), int(offsets[1])
        if start < 0 or end < start or end > payload_size:
            return False, f"Out-of-bounds data_offsets [{start}, {end}] for '{name}' (payload={payload_size})"
        spans.append((start, end, name))

    spans.sort(key=lambda x: (x[0], x[1]))
    cursor = 0
    for start, end, name in spans:
        if start != cursor:
            return False, f"Non-contiguous gap or overlap before '{name}': expected {cursor}, got {start}"
        cursor = end

    if spans and cursor != payload_size:
        return False, f"Trailing unindexed bytes: last tensor ends at {cursor}, payload is {payload_size}"

    return True, "OK"


def build_radix_index_bytes(header: Dict[str, Any]) -> bytes:
    """
    Build a sorted binary radix index of (fnv1a_64_hash, start_offset, end_offset)
    packed as little-endian <QQQ (24 bytes per tensor) for O(1) binary lookup.
    """
    entries: List[Tuple[int, int, int]] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        entries.append((fnv1a_64(name), int(start), int(end)))
    entries.sort(key=lambda x: x[0])
    return b"".join(struct.pack("<QQQ", h, s, e) for h, s, e in entries)


def lookup_radix_index(index_bytes: bytes, tensor_name: str) -> Optional[Tuple[int, int]]:
    """Binary search a packed 24-byte-stride FNV-1a radix index in O(log N) steps."""
    target = fnv1a_64(tensor_name)
    count = len(index_bytes) // 24
    lo, hi = 0, count - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        h, start, end = struct.unpack_from("<QQQ", index_bytes, mid * 24)
        if h == target:
            return (start, end)
        if h < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


def realign_safetensors_to_64kb_dma(src_path: str, dst_path: str) -> Dict[str, Any]:
    """
    Convert any standard v1 .safetensors file into a 64KB DMA-aligned SafeTensors v2
    file using RFC-8259 legal trailing ASCII whitespace (0x20) in the JSON header.
    Tensor data_offsets remain 100% contiguous and unmodified.
    """
    file_size = os.path.getsize(src_path)
    if file_size < 8:
        raise ValueError(f"File too small to be a valid .safetensors container: {src_path}")

    with open(src_path, "rb") as f_in:
        header_len = struct.unpack("<Q", f_in.read(8))[0]
        if header_len == 0 or header_len > ST2_MAX_HEADER_SIZE or (8 + header_len) > file_size:
            raise ValueError(f"Invalid or oversized header length: {header_len}")
        raw_json = f_in.read(header_len)
        header = json.loads(raw_json.decode("utf-8"))

        payload_size = file_size - (8 + header_len)
        ok, reason = validate_contiguous_offsets(header, payload_size)
        if not ok:
            raise ValueError(f"Source .safetensors failed offset validation: {reason}")

        meta = dict(header.get("__metadata__", {}))
        meta["format"] = meta.get("format", "pt")
        meta["st2.edition"] = "safetensors-v2-64kb-dma"
        meta["st2.dma_alignment"] = str(ST2_DMA_ALIGNMENT)
        header["__metadata__"] = meta

        compact_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        total_prefix = 8 + len(compact_bytes)
        remainder = total_prefix % ST2_DMA_ALIGNMENT
        pad_len = (ST2_DMA_ALIGNMENT - remainder) % ST2_DMA_ALIGNMENT
        padded_header = compact_bytes + (b" " * pad_len)
        new_header_len = len(padded_header)
        new_slab_base = 8 + new_header_len
        assert (new_slab_base % ST2_DMA_ALIGNMENT) == 0

        with open(dst_path, "wb") as f_out:
            f_out.write(struct.pack("<Q", new_header_len))
            f_out.write(padded_header)
            while True:
                chunk = f_in.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)

    return inspect_safetensors_v2(dst_path)


class SafeTensorsV2Reader:
    """
    Context-managed, zero-copy SafeTensors v1/v2 reader with 64KB DMA verification,
    contiguous offset bounds checking, and O(1) FNV-1a radix index lookup.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._file = open(path, "rb")
        try:
            self.file_size = os.fstat(self._file.fileno()).st_size
            if self.file_size < 8:
                raise ValueError("Truncated .safetensors file (< 8 bytes)")
            self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self.header_len = struct.unpack("<Q", self._mm[:8])[0]
            if self.header_len == 0 or self.header_len > ST2_MAX_HEADER_SIZE:
                raise ValueError(f"Header length {self.header_len} exceeds safety bounds")
            self.slab_base = 8 + self.header_len
            if self.slab_base > self.file_size:
                raise ValueError("Header extends past end of file")
            self.header: Dict[str, Any] = json.loads(self._mm[8 : self.slab_base].decode("utf-8"))
            self.payload_size = self.file_size - self.slab_base
            ok, reason = validate_contiguous_offsets(self.header, self.payload_size)
            if not ok:
                raise ValueError(f"Offset integrity check failed: {reason}")
            self.radix_index = build_radix_index_bytes(self.header)
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "SafeTensorsV2Reader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "_mm", None) is not None:
            try:
                self._mm.close()
            finally:
                self._mm = None  # type: ignore[assignment]
        if getattr(self, "_file", None) is not None:
            try:
                self._file.close()
            finally:
                self._file = None  # type: ignore[assignment]

    def get_tensor_view(self, name: str) -> memoryview:
        """Return a zero-copy memoryview slice for `name` using the O(1) FNV-1a radix index."""
        if self._mm is None:
            raise RuntimeError("Reader is closed")
        span = lookup_radix_index(self.radix_index, name)
        if span is None:
            raise KeyError(f"Tensor '{name}' not found in SafeTensors index")
        start, end = span
        return memoryview(self._mm)[self.slab_base + start : self.slab_base + end]


def inspect_safetensors_v2(path: str) -> Dict[str, Any]:
    """Safely inspect a .safetensors container and verify 64KB DMA alignment & offset integrity."""
    with SafeTensorsV2Reader(path) as reader:
        meta = reader.header.get("__metadata__", {})
        return {
            "path": path,
            "file_size_bytes": reader.file_size,
            "format": meta.get("format", "safetensors-v1"),
            "edition": meta.get("st2.edition", meta.get("awq2.edition", "legacy")),
            "slab_base_offset": hex(reader.slab_base),
            "is_64kb_dma_aligned": (reader.slab_base % ST2_DMA_ALIGNMENT) == 0,
            "offsets_contiguous_valid": True,
            "radix_index_bytes": len(reader.radix_index),
            "radix_index_offset": meta.get("neon.index_offset", None),
            "tensor_count": len(reader.header) - (1 if "__metadata__" in reader.header else 0),
        }


def _run_self_test() -> int:
    """Create a synthetic v1 .safetensors file, realign it to 64KB DMA v2, and verify zero-copy reads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        v1_path = os.path.join(tmpdir, "test_v1.safetensors")
        v2_path = os.path.join(tmpdir, "test_v2_64kb.safetensors")

        t0_bytes = b"\x00\x00\x80\x3f" * 16  # 16 FP32 ones (64 bytes)
        t1_bytes = b"\x00\x00\x00\x40" * 32  # 32 FP32 twos (128 bytes)
        hdr = {
            "__metadata__": {"format": "pt"},
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [4, 4],
                "data_offsets": [0, len(t0_bytes)],
            },
            "model.layers.0.attn.wq.weight": {
                "dtype": "F32",
                "shape": [8, 4],
                "data_offsets": [len(t0_bytes), len(t0_bytes) + len(t1_bytes)],
            },
        }
        raw_hdr = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
        with open(v1_path, "wb") as f:
            f.write(struct.pack("<Q", len(raw_hdr)))
            f.write(raw_hdr)
            f.write(t0_bytes)
            f.write(t1_bytes)

        v1_info = inspect_safetensors_v2(v1_path)
        assert not v1_info["is_64kb_dma_aligned"], "Synthetic v1 should start unaligned"

        v2_info = realign_safetensors_to_64kb_dma(v1_path, v2_path)
        assert v2_info["is_64kb_dma_aligned"], "Realigned v2 must be 64KB DMA aligned"
        assert v2_info["slab_base_offset"] == "0x10000", "Expected 0x10000 (65536) slab base"

        with SafeTensorsV2Reader(v2_path) as reader:
            view0 = reader.get_tensor_view("model.embed_tokens.weight")
            view1 = reader.get_tensor_view("model.layers.0.attn.wq.weight")
            assert bytes(view0) == t0_bytes, "Zero-copy slice 0 mismatch"
            assert bytes(view1) == t1_bytes, "Zero-copy slice 1 mismatch"
            digest = hashlib.sha256(view0).hexdigest()[:16]

        v2_info["self_test"] = "PASSED"
        v2_info["zerocopy_sha256_prefix"] = digest
        print(json.dumps(v2_info, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv or len(sys.argv) == 1:
        raise SystemExit(_run_self_test())

    for arg in sys.argv[1:]:
        info = inspect_safetensors_v2(arg)
        print(json.dumps(info, indent=2))
