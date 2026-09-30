"""GGUF metadata reader — header + tensor table only, no weights loaded, no dependencies.

Every count read from a file is checked against the bytes actually left in it before anything
is read or allocated, so a hostile or truncated file fails fast with GGUFError instead of
exhausting memory. Split models (`name-00001-of-00004.gguf`) are read as one model.
"""

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
_SPLIT_RE = re.compile(r"^(?P<stem>.*)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")

MAX_STRING_BYTES = 16 * 1024 * 1024   # longest single metadata string accepted
MAX_ARRAY_DEPTH = 3                   # arrays of arrays of arrays
MAX_LAYERS = 4096                     # far beyond any real model; sizes a per-layer list
MAX_DIMS = 8                          # GGML_MAX_DIMS is 4
MAX_ALIGNMENT = 1024 * 1024
MAX_SHARDS = 1024


class GGUFError(ValueError):
    """The file is not a readable GGUF model (malformed, truncated, hostile, or a shard is missing)."""


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
    shards: list[str] = field(default_factory=list)  # every file of a split model, in order

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


class _Reader:
    """Bounded reader: refuses any read or count larger than what is left in the file."""

    def __init__(self, f, size: int):
        self.f, self.size = f, size

    @property
    def remaining(self) -> int:
        return self.size - self.f.tell()

    def raw(self, n: int) -> bytes:
        if n < 0 or n > self.remaining:
            raise GGUFError(f"truncated: needs {n} bytes, {self.remaining} left")
        return self.f.read(n)

    def scalar(self, fmt: str):
        return struct.unpack(fmt, self.raw(struct.calcsize(fmt)))[0]

    def count(self, what: str, min_item_bytes: int) -> int:
        n = self.scalar("<Q")
        if n * max(1, min_item_bytes) > self.remaining:
            raise GGUFError(f"{what} count {n} exceeds the file ({self.remaining} bytes left)")
        return n

    def string(self) -> str:
        n = self.scalar("<Q")
        if n > MAX_STRING_BYTES:
            raise GGUFError(f"string of {n} bytes exceeds the {MAX_STRING_BYTES} byte limit")
        return self.raw(n).decode("utf-8", errors="replace")

    def value(self, vtype: int, depth: int = 0):
        if vtype in _SCALARS:
            return self.scalar(_SCALARS[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            if depth >= MAX_ARRAY_DEPTH:
                raise GGUFError("arrays nested too deeply")
            item_type = self.scalar("<I")
            if item_type in _SCALARS and item_type != 7:
                fmt = _SCALARS[item_type]
                n = self.count("array", struct.calcsize(fmt))
                if n > 4096:  # tokenizer tables etc.: skip, only the length is useful
                    self.f.seek(n * struct.calcsize(fmt), 1)
                    return n
                return list(struct.unpack(f"<{n}{fmt[1]}", self.raw(n * struct.calcsize(fmt))))
            n = self.count("array", 1)
            items = [self.value(item_type, depth + 1) for _ in range(n)]
            return items if n <= 4096 else n
        raise GGUFError(f"unknown GGUF value type {vtype}")


def _read_file(path: Path) -> tuple[dict, list[tuple[str, int]], int]:
    """Parse one GGUF file: (metadata, [(tensor name, size in bytes)], file size)."""
    try:
        file_bytes = path.stat().st_size
        with open(path, "rb") as f:
            r = _Reader(f, file_bytes)
            if r.raw(4) != GGUF_MAGIC:
                raise GGUFError(f"{path} is not a GGUF file")
            version = r.scalar("<I")
            if version < 2:
                raise GGUFError(f"GGUF v{version} not supported")
            n_tensors = r.count("tensor", 24)  # name length + dims + type + offset
            n_kv = r.count("metadata", 12)    # key length + value type

            meta = {}
            for _ in range(n_kv):
                key = r.string()
                meta[key] = r.value(r.scalar("<I"))

            offsets = []
            for _ in range(n_tensors):
                name = r.string()
                n_dims = r.scalar("<I")
                if n_dims > MAX_DIMS:
                    raise GGUFError(f"tensor {name!r} has {n_dims} dimensions")
                r.raw(8 * n_dims)
                r.scalar("<I")
                offsets.append((name, r.scalar("<Q")))

            align = meta.get("general.alignment", 32)
            if not isinstance(align, int) or isinstance(align, bool) or not 1 <= align <= MAX_ALIGNMENT:
                raise GGUFError(f"invalid general.alignment {align!r}")
            data_start = (f.tell() + align - 1) // align * align
    except OSError as e:
        raise GGUFError(f"cannot read {path}: {e}") from e
    except (struct.error, UnicodeError, OverflowError, MemoryError) as e:
        raise GGUFError(f"malformed GGUF {path}: {e}") from e

    # Tensor sizes from offset deltas: exact, and independent of the quant type table
    data_bytes = file_bytes - data_start
    offsets.sort(key=lambda t: t[1])
    tensors = []
    for i, (name, off) in enumerate(offsets):
        end = offsets[i + 1][1] if i + 1 < len(offsets) else data_bytes
        if off > data_bytes:
            raise GGUFError(f"tensor {name!r} lies past the end of {path.name} (truncated download?)")
        tensors.append((name, end - off))
    return meta, tensors, file_bytes


def _shard_paths(path: Path, meta: dict) -> list[Path]:
    """All files of a split model, in order; raises if any is missing."""
    m = _SPLIT_RE.match(path.name)
    split_count = meta.get("split.count", 0)
    if not m:
        if isinstance(split_count, int) and split_count > 1:
            raise GGUFError(f"{path.name} is shard of a {split_count}-file model but not named -NNNNN-of-NNNNN.gguf")
        return [path]
    count = int(m.group("count"))
    if not 1 <= count <= MAX_SHARDS:
        raise GGUFError(f"invalid shard count {count} in {path.name}")
    paths = [path.with_name(f"{m.group('stem')}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise GGUFError(f"split model is incomplete — missing {', '.join(missing)}")
    return paths


def read_gguf(path: str | Path) -> GGUFInfo:
    """Read a GGUF model's metadata and tensor sizes. Any shard of a split model may be given.

    Raises GGUFError (a ValueError) for malformed files or missing shards.
    """
    path = Path(path)
    meta, tensors, file_bytes = _read_file(path)
    shards = _shard_paths(path, meta)
    if len(shards) > 1:
        if shards[0] != path:  # metadata lives in the first shard
            meta, tensors, file_bytes = _read_file(shards[0])
        declared = meta.get("split.count")
        if declared not in (None, len(shards)):
            raise GGUFError(f"{shards[0].name} declares split.count={declared} but {len(shards)} files are named")
        for p in shards[1:]:
            _, more, size = _read_file(p)
            tensors += more
            file_bytes += size

    arch = meta.get("general.architecture", "")
    if not isinstance(arch, str):
        raise GGUFError("general.architecture is not a string")

    def m(key, default=0):
        v = meta.get(f"{arch}.{key}", default)
        if isinstance(v, list):
            v = max(v) if v else default
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 2**31:
            raise GGUFError(f"{arch}.{key} is not a valid count: {v!r}")
        return v

    name = meta.get("general.name", "")
    info = GGUFInfo(
        path=str(shards[0]), file_bytes=file_bytes, architecture=arch,
        name=name if isinstance(name, str) else "",
        n_layers=m("block_count"), n_experts=m("expert_count"), n_experts_used=m("expert_used_count"),
        n_head_kv=m("attention.head_count_kv"), context_length=m("context_length"),
        shards=[str(p) for p in shards],
    )
    if info.n_layers > MAX_LAYERS:
        raise GGUFError(f"block_count {info.n_layers} exceeds {MAX_LAYERS}")
    head_dim = m("embedding_length") // max(1, m("attention.head_count", 1))
    info.key_length = m("attention.key_length", head_dim)
    info.value_length = m("attention.value_length", head_dim)

    info.layer_bytes = [0] * info.n_layers
    info.layer_expert_bytes = [0] * info.n_layers
    kv_layers, ssm_layers = set(), set()
    for tname, size in tensors:
        mb = _BLOCK_RE.match(tname)
        if mb and int(mb.group(1)) < info.n_layers:
            b = int(mb.group(1))
            info.layer_bytes[b] += size
            if "_exps" in tname:
                info.layer_expert_bytes[b] += size
            if ".attn_k." in tname or ".attn_qkv." in tname or ".attn_kv_a" in tname:
                kv_layers.add(b)
            if ".ssm_" in tname:
                ssm_layers.add(b)
        elif tname.startswith("token_embd"):
            info.embed_bytes += size
        else:
            info.output_bytes += size
    # Hybrid models (e.g. Qwen3.5/3.6 Gated DeltaNet) keep a small recurrent state, not a KV cache, in ssm layers
    info.n_kv_layers = len(kv_layers - ssm_layers) or info.n_layers
    return info
