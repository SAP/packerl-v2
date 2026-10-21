from __future__ import annotations

import hashlib
import math
import struct
import unicodedata
from typing import Any, Iterable, Tuple
from src.utils.utils import colored

# Optional deps
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    np = None
    _HAS_NUMPY = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    nx = None
    _HAS_NX = False


# ---------- Type tags ----------
_T_NONE      = b"N"
_T_BOOL      = b"B"
_T_INT       = b"I"
_T_FLOAT     = b"F"
_T_STR       = b"S"
_T_BYTES     = b"Y"
_T_LIST      = b"L"
_T_TUPLE     = b"T"
_T_SET       = b"E"   # (think "ensemble")
_T_FSET      = b"R"   # (frozenset)
_T_DICT      = b"D"
_T_NDARRAY   = b"A"   # NumPy ndarray
_T_TORCH     = b"P"   # PyTorch tensor

# Canonical NaN payloads (big-endian)
_CANON_QNAN_64 = struct.pack(">Q", 0x7FF8000000000000)
_CANON_QNAN_32 = struct.pack(">I", 0x7FC00000)
_CANON_QNAN_16 = struct.pack(">H", 0x7E00)


# ---------- Utilities ----------
def _varint(n: int) -> bytes:
    if n < 0:
        raise ValueError("varint requires non-negative integer")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(0x80 | b)
        else:
            out.append(b)
            break
    return bytes(out)


def _int_bytes(n: int) -> Tuple[bytes, bytes]:
    if n == 0:
        return b"\x00", b"\x00"
    sign = b"\x00" if n >= 0 else b"\x01"
    m = abs(n)
    mag = bytearray()
    while m:
        mag.append(m & 0xFF)
        m >>= 8
    mag = bytes(reversed(mag))
    return sign, mag


def _float_bytes(x: float) -> bytes:
    if math.isnan(x):
        return _CANON_QNAN_64
    return struct.pack(">d", x)


def _norm_str(s: str) -> bytes:
    return unicodedata.normalize("NFC", s).encode("utf-8", errors="strict")


class _SeenStackError(RuntimeError):
    pass


# ---------- NumPy ----------
def _encode_ndarray(arr, _seen_ids: set) -> bytes:
    assert _HAS_NUMPY
    oid = id(arr)
    if oid in _seen_ids:
        raise _SeenStackError("Cycle detected during canonical encoding")
    _seen_ids.add(oid)
    try:
        arr = np.ascontiguousarray(arr)  # C order for uniqueness
        dt = arr.dtype
        ndim = arr.ndim
        shape = arr.shape

        # Object arrays: encode elements individually
        if dt == np.dtype("O"):
            parts = [ _T_NDARRAY, b"O", _varint(ndim) ]
            for d in shape:
                parts.append(_varint(int(d)))
            parts.append(_encode(("dtype", "object"), _seen_ids))
            parts.append(_varint(arr.size))
            for x in arr.flat:
                parts.append(_encode(x, _seen_ids))
            return b"".join(parts)

        # Normalize to big-endian
        # Build a big-endian dtype (no-op for non-endian types like '|u1', '|S', etc.)
        be_dt = dt if dt.byteorder == "|" else dt.newbyteorder(">")

        # Decide whether we must byteswap the data to get big-endian bytes
        if dt.byteorder == "|":
            need_swap = False
        elif dt.byteorder == "=":
            # Native-endian: swap on little-endian hosts
            need_swap = np.little_endian
        elif dt.byteorder == "<":
            need_swap = True
        else:  # dt.byteorder == ">"
            need_swap = False

        arr_be = (arr.byteswap().view(be_dt)) if need_swap else arr.view(be_dt)

        # Canonicalize NaNs for float/complex kinds
        if be_dt.kind in ("f", "c"):
            if be_dt.kind == "f":
                if be_dt.itemsize == 8:
                    view = arr_be.view(np.dtype(">u8"))
                    mask = np.isnan(arr_be)
                    view[mask] = 0x7FF8000000000000
                elif be_dt.itemsize == 4:
                    view = arr_be.view(np.dtype(">u4"))
                    mask = np.isnan(arr_be)
                    view[mask] = 0x7FC00000
                elif be_dt.itemsize == 2:
                    view = arr_be.view(np.dtype(">u2"))
                    mask = np.isnan(arr_be)
                    view[mask] = 0x7E00
            else:  # complex
                if be_dt.itemsize == 16:
                    fdt, udt, patt = np.dtype(">f8"), np.dtype(">u8"), 0x7FF8000000000000
                elif be_dt.itemsize == 8:
                    fdt, udt, patt = np.dtype(">f4"), np.dtype(">u4"), 0x7FC00000
                elif be_dt.itemsize == 4:
                    fdt, udt, patt = np.dtype(">f2"), np.dtype(">u2"), 0x7E00
                fview = arr_be.view(fdt)
                uview = fview.view(udt)
                uview[np.isnan(fview)] = patt

        raw = arr_be.tobytes()

        # Dtype metadata
        if be_dt.names is not None:
            dtype_meta = ("descr", be_dt.descr)
        else:
            dtype_meta = ("str", be_dt.str)

        parts = [ _T_NDARRAY, b"P", _varint(ndim) ]
        for d in shape:
            parts.append(_varint(int(d)))
        parts.append(_encode(dtype_meta, _seen_ids))
        parts.append(_varint(len(raw)))
        parts.append(raw)
        return b"".join(parts)
    finally:
        _seen_ids.discard(oid)


# ---------- PyTorch ----------
def _torch_dense_to_numpy(t):
    # Detach, move to CPU, enforce contiguous
    return t.detach().cpu().contiguous().numpy()

def _encode_torch_tensor(t, _seen_ids: set) -> bytes:
    assert _HAS_TORCH
    # Handle sparse COO explicitly for stability
    if t.layout == torch.sparse_coo:
        tc = t.coalesce()
        indices = tc.indices()
        values = tc.values()
        size = tuple(tc.size())
        payload = (
            ("layout", "sparse_coo"),
            ("size", size),
            ("indices", _torch_dense_to_numpy(indices)),
            ("values", _torch_dense_to_numpy(values)),
        )
        return _T_TORCH + _encode(payload, _seen_ids)
    else:
        arr = _torch_dense_to_numpy(t)
        return _T_TORCH + _encode(arr, _seen_ids)  # falls through to ndarray encoder if NumPy is present


# ---------- Core encoder ----------
def _encode(obj: Any, _seen_ids: set) -> bytes:
    # Container-cycle protection
    oid = id(obj)
    is_container = isinstance(obj, (dict, list, tuple, set, frozenset))
    if is_container:
        if oid in _seen_ids:
            raise _SeenStackError("Cycle detected during canonical encoding")
        _seen_ids.add(oid)

    try:
        # Primitives
        if obj is None:
            return _T_NONE
        if isinstance(obj, bool):
            return _T_BOOL + (b"\x01" if obj else b"\x00")
        if isinstance(obj, int):
            sign, mag = _int_bytes(obj)
            return _T_INT + sign + _varint(len(mag)) + mag
        if isinstance(obj, float):
            return _T_FLOAT + _float_bytes(obj)
        if isinstance(obj, (bytes, bytearray, memoryview)):
            bts = bytes(obj)
            return _T_BYTES + _varint(len(bts)) + bts
        if isinstance(obj, str):
            bts = _norm_str(obj)
            return _T_STR + _varint(len(bts)) + bts

        # NumPy scalars/arrays
        if _HAS_NUMPY:
            if isinstance(obj, np.ndarray):
                return _encode_ndarray(obj, _seen_ids)
            if isinstance(obj, np.generic):
                return _encode(obj.item(), _seen_ids)

        # PyTorch tensors
        if _HAS_TORCH and isinstance(obj, torch.Tensor):
            return _encode_torch_tensor(obj, _seen_ids)

        # Containers
        if isinstance(obj, list):
            parts = [_encode(v, _seen_ids) for v in obj]
            return _T_LIST + _varint(len(parts)) + b"".join(parts)
        if isinstance(obj, tuple):
            parts = [_encode(v, _seen_ids) for v in obj]
            return _T_TUPLE + _varint(len(parts)) + b"".join(parts)
        if isinstance(obj, set):
            enc = sorted((_encode(v, _seen_ids) for v in obj))
            return _T_SET + _varint(len(enc)) + b"".join(enc)
        if isinstance(obj, frozenset):
            enc = sorted((_encode(v, _seen_ids) for v in obj))
            return _T_FSET + _varint(len(enc)) + b"".join(enc)
        if isinstance(obj, dict):
            items = []
            for k, v in obj.items():
                ek = _encode(k, _seen_ids)
                ev = _encode(v, _seen_ids)
                items.append((ek, ev))
            items.sort(key=lambda kv: kv[0])  # sort by encoded key bytes
            out = bytearray()
            out += _T_DICT + _varint(len(items))
            for ek, ev in items:
                out += ek + ev
            return bytes(out)

        # Dataclasses / generic objects: fall back to a sorted attribute dict
        # (only use if you *really* want to include the whole object)
        try:
            import dataclasses
            if dataclasses.is_dataclass(obj):
                data = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
                return _encode(("__dataclass__", obj.__class__.__name__, data), _seen_ids)
        except Exception:
            pass

        if hasattr(obj, "__dict__"):
            data = {k: getattr(obj, k) for k in sorted(obj.__dict__.keys())}
            return _encode(("__obj__", obj.__class__.__name__, data), _seen_ids)

        raise TypeError(f"Unsupported type for canonical encoding: {type(obj).__name__}")
    finally:
        if is_container:
            _seen_ids.discard(oid)


# ---------- NetworkX integration ----------


def _graph_type_tag(G) -> str:
    if G.is_multigraph():
        return "MultiDiGraph" if G.is_directed() else "MultiGraph"
    return "DiGraph" if G.is_directed() else "Graph"


def _sorted_nodes(G) -> Iterable[Any]:
    return [n for n, _ in sorted(((n, canonical_bytes(n)) for n in G.nodes), key=lambda pair: pair[1])]


def _sorted_edges(G) -> Iterable[Tuple[Any, Any, dict]]:
    if G.is_multigraph():
        iterator = G.edges(keys=True, data=True)
        if G.is_directed():
            edges = [((u, v, k), d) for u, v, k, d in iterator]
            edges.sort(key=lambda item: canonical_bytes(item[0]))
            return [(u, v, {"__key__": k, **dict(d)}) for (u, v, k), d in edges]
        else:
            normed = []
            for u, v, k, d in iterator:
                eu, ev = canonical_bytes(u), canonical_bytes(v)
                a, b = (u, v) if eu <= ev else (v, u)
                normed.append(((a, b, k), d))
            normed.sort(key=lambda item: canonical_bytes(item[0]))
            return [(a, b, {"__key__": k, **dict(d)}) for (a, b, k), d in normed]
    else:
        if G.is_directed():
            edges = [((u, v), d) for u, v, d in G.edges(data=True)]
            edges.sort(key=lambda item: canonical_bytes(item[0]))
            return [(u, v, dict(d)) for (u, v), d in edges]
        else:
            normed = []
            for u, v, d in G.edges(data=True):
                eu, ev = canonical_bytes(u), canonical_bytes(v)
                a, b = (u, v) if eu <= ev else (v, u)
                normed.append(((a, b), d))
            normed.sort(key=lambda item: canonical_bytes(item[0]))
            return [(a, b, dict(d)) for (a, b), d in normed]


def nx_graph_to_canonical_bytes(G) -> bytes:
    """
    Canonically encode a NetworkX graph (nodes/edges/global attributes) to bytes.
    Stable across platforms/Python versions and safe for hashing.

    Supported attribute types (recursively):
      None, bool, int, float (NaNs canonicalized), str (NFC UTF-8), bytes,
      list/tuple/set/frozenset/dict, NumPy scalars, NumPy ndarrays (incl. structured/complex),
      PyTorch tensors.
    """
    if not _HAS_NX:
        raise RuntimeError("networkx is not available")
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("Expected a NetworkX Graph/DiGraph/MultiGraph/MultiDiGraph")

    gtype = _graph_type_tag(G)
    gattrs = dict(G.graph)
    if "name" in gattrs:  # delete the graph's name (not relevant for structure; it's okay to delete because this is a copy)
        del gattrs["name"]

    node_list = []
    for n in _sorted_nodes(G):
        node_list.append((n, dict(G.nodes[n])))

    edge_list = []
    for (u, v, d) in _sorted_edges(G):
        edge_list.append(((u, v), d))

    outer = (
        "NX-CANON",
        1,                    # version
        gtype,
        gattrs,
        tuple(node_list),
        tuple(edge_list),
    )

    try:
        return canonical_bytes(outer)
    except _SeenStackError as e:
        raise RuntimeError("Cannot encode cyclic attributes") from e


# --- Canonical encode and pretty hash ---


def canonical_bytes(obj: Any) -> bytes:
    """Canonical, platform-stable bytes for `obj`."""
    return _encode(obj, set())


def canonical_hash(parts: Iterable[Any],
                   algo: str = "sha256",
                   show_full: bool = False,
                   show_in_color: bool = True) -> str:
    """
    Hash an iterable of parts/items (bytes or any supported Python/NumPy/Torch/NetworkX objects)
    using the canonical encoder.
    """
    h = hashlib.new(algo, usedforsecurity=False)
    for x in parts:
        if isinstance(x, (bytes, bytearray, memoryview)):
            h.update(bytes(x))
        else:
            h.update(canonical_bytes(x))

    hex_str = h.hexdigest()
    out_hash = "-".join(hex_str[i:i+4] for i in range(0, len(hex_str), 4))
    out_hash = out_hash if show_full else out_hash[:9]  # short display
    out_hash = colored(out_hash, color=130) if show_in_color else out_hash  # colorize
    return out_hash
