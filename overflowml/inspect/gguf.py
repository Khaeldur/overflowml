"""GGUF metadata reader — header + tensor table only, no weights loaded, no dependencies."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

GGUF_MAGIC = b"GGUF"

_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_STRING, _ARRAY = 8, 9
_BLOCK_RE = re.compile(r"^blk\.(\d+)\.")


@dataclass
class GGUFInfo:
    path: str
    file_bytes: int
    architecture: str = ""
    name: str = ""
    n_layers: int = 0
    n_experts: int = 0
    n_experts_used: int = 0
    n_kv_layers: int = 0          # layers with a KV cache (hybrid/linear-attention models have fewer)
    n_head_kv: int = 0
    key_length: int = 0
    value_length: int = 0
    context_length: int = 0
    layer_bytes: list[int] = field(default_factory=list)         # per repeating block, all tensors
    layer_expert_bytes: list[int] = field(default_factory=list)  # per block, *_exps tensors only
    output_bytes: int = 0         # output head + final norm (offloaded with ngl > n_layers)
    embed_bytes: int = 0          # token embeddings (llama.cpp keeps these on CPU)

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 1 and sum(self.layer_expert_bytes) > 0

    @property
    def weights_bytes(self) -> int:
        return sum(self.layer_bytes) + self.output_bytes + self.embed_bytes

    def kv_bytes_per_token(self, cache_type: str = "f16") -> float:
        per_elem = {"f16": 2.0, "bf16": 2.0, "f32": 4.0, "q8_0": 34 / 32, "q5_1": 24 / 32,
                    "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32}[cache_type]
        return self.n_kv_layers * self.n_head_kv * (self.key_length + self.value_length) * per_elem


def _read(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def _read_str(f) -> str:
    return f.read(_read(f, "<Q")).decode("utf-8", errors="replace")


def _read_value(f, vtype):
    if vtype in _SCALARS:
        return _read(f, _SCALARS[vtype])
    if vtype == _STRING:
        return _read_str(f)
    if vtype == _ARRAY:
        item_type, n = _read(f, "<I"), _read(f, "<Q")
        if item_type in _SCALARS and item_type != 7:
            fmt = _SCALARS[item_type]
            size = struct.calcsize(fmt)
            data = f.read(size * n)
            return list(struct.unpack(f"<{n}{fmt[1]}", data)) if n <= 4096 else n
        items = [_read_value(f, item_type) for _ in range(n)]
        return items if n <= 4096 else n
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf(path: str | Path) -> GGUFInfo:
    path = Path(path)
    file_bytes = path.stat().st_size
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        version = _read(f, "<I")
        if version < 2:
            raise ValueError(f"GGUF v{version} not supported")
        n_tensors, n_kv = _read(f, "<Q"), _read(f, "<Q")

        meta = {}
        for _ in range(n_kv):
            key = _read_str(f)
            meta[key] = _read_value(f, _read(f, "<I"))

        tensors = []
        for _ in range(n_tensors):
            name = _read_str(f)
            n_dims = _read(f, "<I")
            f.read(8 * n_dims)
            _read(f, "<I")
            tensors.append((name, _read(f, "<Q")))

        align = meta.get("general.alignment", 32)
        data_start = (f.tell() + align - 1) // align * align

    arch = meta.get("general.architecture", "")

    def m(key, default=0):
        v = meta.get(f"{arch}.{key}", default)
        return max(v) if isinstance(v, list) and v else v

    info = GGUFInfo(
        path=str(path), file_bytes=file_bytes, architecture=arch,
        name=meta.get("general.name", ""),
        n_layers=m("block_count"), n_experts=m("expert_count"), n_experts_used=m("expert_used_count"),
        n_head_kv=m("attention.head_count_kv"), context_length=m("context_length"),
    )
    head_dim = m("embedding_length") // max(1, m("attention.head_count", 1))
    info.key_length = m("attention.key_length", head_dim)
    info.value_length = m("attention.value_length", head_dim)

    # Tensor sizes from offset deltas: exact, and independent of the quant type table.
    tensors.sort(key=lambda t: t[1])
    data_bytes = file_bytes - data_start
    info.layer_bytes = [0] * info.n_layers
    info.layer_expert_bytes = [0] * info.n_layers
    kv_layers, ssm_layers = set(), set()
    for i, (name, off) in enumerate(tensors):
        size = (tensors[i + 1][1] if i + 1 < len(tensors) else data_bytes) - off
        mb = _BLOCK_RE.match(name)
        if mb and int(mb.group(1)) < info.n_layers:
            b = int(mb.group(1))
            info.layer_bytes[b] += size
            if "_exps" in name:
                info.layer_expert_bytes[b] += size
            if ".attn_k." in name or ".attn_qkv." in name or ".attn_kv_a" in name:
                kv_layers.add(b)
            if ".ssm_" in name:
                ssm_layers.add(b)
        elif name.startswith("token_embd"):
            info.embed_bytes += size
        else:
            info.output_bytes += size
    # Hybrid models (e.g. Qwen3.5/3.6 Gated DeltaNet) keep a small recurrent state, not a KV cache, in ssm layers
    info.n_kv_layers = len(kv_layers - ssm_layers) or info.n_layers
    return info
