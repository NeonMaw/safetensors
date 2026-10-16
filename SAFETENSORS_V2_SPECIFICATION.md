# ⚡ SafeTensors v2: 64KB DMA Alignment, Contiguous Offset Verification & O(1) Radix Lookup
> *Maintained by [Neon Maw LLC](https://github.com/NeonMaw/safetensors) (`feat/safetensors-v2-64kb-dma`).*

### 1. Architectural Summary
- **Resolves the SafeTensors Alignment Paradox:** Standard `.safetensors` files begin with `[u64 header_len][JSON]`, placing Tensor 0 at an arbitrary unaligned offset (e.g., `0x3B7A`) and breaking `64KB` (`65,536-byte`) OS/GPU DMA page alignment (`VirtualAlloc`, PCIe BAR1, DirectStorage).
- **100% Backward-Compatible RFC-8259 Padding (`realign_safetensors_to_64kb_dma`):** Pads the serialized JSON header with legal trailing ASCII whitespace (`0x20`) so that `(8 + header_len) % 65536 == 0`, placing Tensor 0 at an exact `64KB` boundary while keeping all tensor `data_offsets` 100% contiguous and unmodified.
- **Strict Contiguous Offset & Security Bounds Validation (`validate_contiguous_offsets`):** Enforces the `100 MB` `ST2_MAX_HEADER_SIZE` cap and verifies that all tensor spans `[start, end]` are non-negative, non-overlapping, strictly contiguous from `0` to `payload_size`, and free of trailing unindexed bytes.
- **Context-Managed Zero-Copy Slicing (`SafeTensorsV2Reader`):** Prevents `mmap` descriptor leaks on malformed headers and exposes zero-copy `memoryview` slices via `get_tensor_view(name)` backed by a binary-searched 24-byte-stride `FNV-1a` 64-bit radix index (`<QQQ`).

### 2. Quick Start & Self-Test
```bash
# Run built-in v1 -> v2 64KB DMA conversion and zero-copy verification self-test
python SAFETENSORS_V2_LOADER.py --self-test

# Inspect any .safetensors file for 64KB DMA alignment and contiguous offset validity
python SAFETENSORS_V2_LOADER.py path/to/model.safetensors
```
