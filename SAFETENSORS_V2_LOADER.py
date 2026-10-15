"""
==============================================================================
NEON MAW LLC — COMMUNITY GIFT 3: SAFETENSORS v2 (64KB DMA + O(1) RADIX INDEX)
File: community_rollout/safetensors_v2/safetensors_v2_spec_and_loader.py
Target: huggingface/safetensors & pytorch/pytorch (Phased Community Rollout)
License: MIT (Public Community Contribution by Neon Maw LLC)

Summary of SafeTensors v2 (STv2) Upgrades:
  1. Solves the HuggingFace SafeTensors Alignment Paradox:
     Standard .safetensors starts with [u64 N][N-byte JSON], placing Tensor 0
     at an arbitrary unaligned offset (e.g., 0x3B7A) and breaking 64KB OS/GPU
     DMA page locking.
  2. 100% Backward Compatible RFC-8259 Whitespace Padding:
     Pads the JSON header with trailing ASCII spaces (0x20) so that (8 + N) is
     an exact multiple of 65536 (64KB). Every existing Python/Rust safetensors
     reader loads STv2 files with zero code changes!
  3. O(1) Tail Binary Radix Index (__neon_v2_index__):
     Appends a contiguous U8 tensor at the tail of the file holding precomputed
     64-bit FNV-1a hashes and packed vocabulary metadata, indexed via
     "neon.index_offset" in "__metadata__".
==============================================================================
"""

import json
import mmap
import struct
from typing import Any, Dict


ST2_DMA_ALIGNMENT = 65536


def inspect_safetensors_v2(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        header_len = struct.unpack("<Q", mm[:8])[0]
        slab_base = 8 + header_len
        hdr = json.loads(mm[8:slab_base].decode("utf-8"))
        meta = hdr.get("__metadata__", {})
        mm.close()

    return {
        "path": path,
        "format": meta.get("format", "safetensors-v1"),
        "edition": meta.get("st2.edition", meta.get("awq2.edition", "legacy")),
        "slab_base_offset": hex(slab_base),
        "is_64kb_dma_aligned": (slab_base % ST2_DMA_ALIGNMENT) == 0,
        "radix_index_offset": meta.get("neon.index_offset", None),
        "tensor_count": len(hdr) - (1 if "__metadata__" in hdr else 0),
    }


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        info = inspect_safetensors_v2(arg)
        print(json.dumps(info, indent=2))
