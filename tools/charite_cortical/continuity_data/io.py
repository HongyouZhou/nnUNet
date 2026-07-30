"""Dependency-light readers and deterministic sidecar writers."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import numpy as np

from .schema import canonical_json_bytes


_SEGMENT_FIELD_RE = re.compile(r"segment(\d+)_(.+)", re.IGNORECASE)
_DIRECTION_TOKEN_RE = re.compile(r"none|\([^)]*\)", re.IGNORECASE)


@dataclass(frozen=True)
class Segment:
    header_index: int
    name: str
    layer: int
    label_value: int
    extent_xyz: tuple[int, int, int, int, int, int] | None


@dataclass(frozen=True)
class NrrdLayout:
    path: Path
    header: Mapping[str, str]
    payload_offset: int
    encoding: str
    dtype: np.dtype[Any]
    layer_count: int
    spatial_shape_xyz: tuple[int, int, int]
    segments: tuple[Segment, ...]


@dataclass(frozen=True)
class NiftiGeometry:
    path: Path
    endian: str
    header: bytes
    shape_xyz: tuple[int, int, int]
    spacing_xyz_mm: tuple[float, float, float]


def sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _header_bytes(path: Path, *, limit: int = 1024 * 1024) -> tuple[bytes, int]:
    data = bytearray()
    with path.open("rb") as stream:
        while len(data) < limit:
            block = stream.read(min(65536, limit - len(data)))
            if not block:
                break
            data.extend(block)
            candidates = [
                (data.find(marker), len(marker))
                for marker in (b"\n\n", b"\r\n\r\n")
                if data.find(marker) >= 0
            ]
            if candidates:
                index, marker_size = min(candidates)
                return bytes(data[: index + marker_size]), index + marker_size
    raise ValueError(f"NRRD header terminator missing: {path}")


def parse_nrrd_header(data: bytes) -> dict[str, str]:
    blob = data.split(b"\n\n", 1)[0].split(b"\r\n\r\n", 1)[0]
    lines = blob.decode("utf-8").splitlines()
    if not lines or not lines[0].startswith("NRRD"):
        raise ValueError("file does not start with an NRRD magic line")
    fields: dict[str, str] = {"_magic": lines[0].strip()}
    for raw in lines[1:]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":=" in line:
            key, value = line.split(":=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            raise ValueError(f"malformed NRRD header line: {line!r}")
        fields[key.strip().casefold()] = value.strip()
    return fields


def _nrrd_dtype(header: Mapping[str, str]) -> np.dtype[Any]:
    aliases: dict[str, np.dtype[Any]] = {
        "signed char": np.dtype(np.int8),
        "int8": np.dtype(np.int8),
        "int8_t": np.dtype(np.int8),
        "unsigned char": np.dtype(np.uint8),
        "uchar": np.dtype(np.uint8),
        "uint8": np.dtype(np.uint8),
        "uint8_t": np.dtype(np.uint8),
        "unsigned short": np.dtype(np.uint16),
        "ushort": np.dtype(np.uint16),
        "uint16": np.dtype(np.uint16),
        "uint16_t": np.dtype(np.uint16),
    }
    type_name = " ".join(header.get("type", "").casefold().split())
    dtype = aliases.get(type_name)
    if dtype is None:
        raise ValueError(f"unsupported NRRD value type {type_name!r}")
    if dtype.itemsize > 1:
        endian = header.get("endian", "").casefold()
        if endian not in {"little", "big"}:
            raise ValueError("multi-byte NRRD is missing valid endian metadata")
        dtype = dtype.newbyteorder("<" if endian == "little" else ">")
    return dtype


def _parse_segments(header: Mapping[str, str], layer_count: int) -> tuple[Segment, ...]:
    grouped: dict[int, dict[str, str]] = {}
    for key, value in header.items():
        match = _SEGMENT_FIELD_RE.fullmatch(key)
        if match:
            grouped.setdefault(int(match.group(1)), {})[
                match.group(2).casefold()
            ] = value
    result: list[Segment] = []
    seen_keys: set[tuple[int, int]] = set()
    seen_names: set[str] = set()
    for index in sorted(grouped):
        fields = grouped[index]
        try:
            name = fields["name"].strip()
            layer = int(fields["layer"])
            label = int(fields["labelvalue"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Segment{index} has invalid identity metadata") from error
        if not name:
            raise ValueError(f"Segment{index} has an empty name")
        if not 0 <= layer < layer_count or label <= 0:
            raise ValueError(f"Segment{index} has an invalid layer/label pair")
        if (layer, label) in seen_keys:
            raise ValueError(f"duplicate NRRD layer/label pair {(layer, label)}")
        if name in seen_names:
            raise ValueError(f"duplicate NRRD segment name {name!r}")
        seen_keys.add((layer, label))
        seen_names.add(name)
        extent_value = fields.get("extent")
        extent: tuple[int, int, int, int, int, int] | None = None
        if extent_value is not None:
            parsed = tuple(int(item) for item in extent_value.split())
            if len(parsed) != 6:
                raise ValueError(f"Segment{index} has an invalid extent")
            extent = parsed  # type: ignore[assignment]
        result.append(Segment(index, name, layer, label, extent))
    if not result:
        raise ValueError("NRRD contains no declared Slicer segments")
    return tuple(result)


def inspect_nrrd(path: Path) -> NrrdLayout:
    path = path.expanduser().resolve(strict=True)
    raw_header, payload_offset = _header_bytes(path)
    header = parse_nrrd_header(raw_header)
    try:
        dimension = int(header["dimension"])
        sizes = tuple(int(item) for item in header["sizes"].split())
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid NRRD size metadata: {path}") from error
    if dimension not in {3, 4} or len(sizes) != dimension:
        raise ValueError("continuity data supports 3-D or list-axis-first 4-D NRRD")
    if any(size <= 0 for size in sizes):
        raise ValueError("NRRD dimensions must be positive")

    if dimension == 3:
        layer_count = 1
        spatial_shape = sizes
    else:
        tokens = _DIRECTION_TOKEN_RE.findall(header.get("space directions", ""))
        none_axes = [
            index for index, token in enumerate(tokens) if token.casefold() == "none"
        ]
        if none_axes != [0]:
            raise ValueError("4-D Slicer NRRD must use list axis 0")
        layer_count = sizes[0]
        spatial_shape = sizes[1:]
    if len(spatial_shape) != 3:
        raise ValueError("NRRD does not contain exactly three spatial axes")

    encoding_value = header.get("encoding", "").casefold()
    if encoding_value == "raw":
        encoding = "raw"
    elif encoding_value in {"gzip", "gz"}:
        encoding = "gzip"
    else:
        raise ValueError(f"unsupported NRRD encoding {encoding_value!r}")
    if "data file" in header or "datafile" in header:
        raise ValueError("detached NRRD payloads are not supported")

    segments = _parse_segments(header, layer_count)
    return NrrdLayout(
        path=path,
        header=header,
        payload_offset=payload_offset,
        encoding=encoding,
        dtype=_nrrd_dtype(header),
        layer_count=layer_count,
        spatial_shape_xyz=spatial_shape,  # type: ignore[arg-type]
        segments=segments,
    )


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    blocks: list[bytes] = []
    remaining = count
    while remaining:
        block = stream.read(remaining)
        if not block:
            raise ValueError("NRRD payload is shorter than declared")
        blocks.append(block)
        remaining -= len(block)
    return b"".join(blocks)


def iter_spatial_vectors(
    layout: NrrdLayout, *, chunk_spatial_voxels: int = 500_000
) -> Iterator[tuple[int, np.ndarray[Any, Any]]]:
    """Yield ``(flat_start, values[voxel, layer])`` in x-fastest order."""

    if chunk_spatial_voxels <= 0:
        raise ValueError("chunk_spatial_voxels must be positive")
    total_spatial = math.prod(layout.spatial_shape_xyz)
    with layout.path.open("rb") as source:
        source.seek(layout.payload_offset)
        payload: BinaryIO
        if layout.encoding == "gzip":
            payload = gzip.GzipFile(fileobj=source, mode="rb")
        else:
            payload = source
        try:
            start = 0
            while start < total_spatial:
                count = min(chunk_spatial_voxels, total_spatial - start)
                byte_count = count * layout.layer_count * layout.dtype.itemsize
                raw = _read_exact(payload, byte_count)
                values = np.frombuffer(raw, dtype=layout.dtype).reshape(
                    count, layout.layer_count
                )
                yield start, values
                start += count
            if payload.read(1):
                raise ValueError("NRRD payload is longer than declared")
        finally:
            if payload is not source:
                payload.close()


def inspect_nifti(path: Path) -> NiftiGeometry:
    path = path.expanduser().resolve(strict=True)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as stream:
        header = stream.read(348)
    if len(header) != 348:
        raise ValueError(f"short NIfTI header: {path}")
    if struct.unpack("<I", header[:4])[0] == 348:
        endian = "<"
    elif struct.unpack(">I", header[:4])[0] == 348:
        endian = ">"
    else:
        raise ValueError(f"invalid NIfTI-1 sizeof_hdr: {path}")
    dimensions = struct.unpack(endian + "8h", header[40:56])
    if dimensions[0] < 3:
        raise ValueError("continuity data requires a 3-D NIfTI")
    shape = tuple(int(item) for item in dimensions[1:4])
    pixdim = struct.unpack(endian + "8f", header[76:108])
    spacing = tuple(abs(float(item)) for item in pixdim[1:4])
    if any(item <= 0 for item in shape) or any(item <= 0 for item in spacing):
        raise ValueError("NIfTI has invalid shape or spacing")
    return NiftiGeometry(
        path=path,
        endian=endian,
        header=header,
        shape_xyz=shape,  # type: ignore[arg-type]
        spacing_xyz_mm=spacing,  # type: ignore[arg-type]
    )


def read_nifti_array(path: Path) -> np.ndarray[Any, Any]:
    """Read a scalar 3-D NIfTI-1 array for audits and unit tests."""

    geometry = inspect_nifti(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as stream:
        header = stream.read(348)
        datatype = struct.unpack(geometry.endian + "h", header[70:72])[0]
        vox_offset = int(round(struct.unpack(geometry.endian + "f", header[108:112])[0]))
        types = {
            2: np.dtype(np.uint8),
            4: np.dtype(np.int16),
        }
        dtype = types.get(datatype)
        if dtype is None:
            raise ValueError(f"unsupported audit NIfTI datatype code {datatype}")
        dtype = dtype.newbyteorder(geometry.endian)
        stream.seek(vox_offset)
        count = math.prod(geometry.shape_xyz)
        raw = _read_exact(stream, count * dtype.itemsize)
        if stream.read(1):
            raise ValueError("NIfTI payload is longer than its declared 3-D shape")
    return np.frombuffer(raw, dtype=dtype).reshape(
        geometry.shape_xyz, order="F"
    )


_NIFTI_DTYPES: dict[str, tuple[int, int, np.dtype[Any]]] = {
    "uint8": (2, 8, np.dtype(np.uint8)),
    "int16": (4, 16, np.dtype(np.int16)),
}


def write_nifti_crop_like(
    reference: NiftiGeometry,
    destination: Path,
    crop: np.ndarray[Any, Any],
    bbox_xyz: Sequence[Sequence[int]],
    *,
    dtype_name: str,
    fill_value: int = 0,
) -> None:
    """Write a full-grid deterministic ``.nii.gz`` from one spatial crop."""

    if dtype_name not in _NIFTI_DTYPES:
        raise ValueError(f"unsupported output NIfTI dtype {dtype_name}")
    datatype, bitpix, dtype = _NIFTI_DTYPES[dtype_name]
    if len(bbox_xyz) != 3 or any(len(axis) != 2 for axis in bbox_xyz):
        raise ValueError("bbox_xyz must contain three [start, stop] pairs")
    starts = tuple(int(axis[0]) for axis in bbox_xyz)
    stops = tuple(int(axis[1]) for axis in bbox_xyz)
    if any(
        start < 0 or stop > size or stop <= start
        for start, stop, size in zip(starts, stops, reference.shape_xyz)
    ):
        raise ValueError("crop bbox is outside the reference grid")
    expected_crop_shape = tuple(stop - start for start, stop in zip(starts, stops))
    if tuple(crop.shape) != expected_crop_shape:
        raise ValueError(
            f"crop shape {crop.shape} differs from bbox shape {expected_crop_shape}"
        )
    info = np.iinfo(dtype)
    if int(np.min(crop, initial=fill_value)) < info.min or int(
        np.max(crop, initial=fill_value)
    ) > info.max:
        raise ValueError(f"target values do not fit {dtype_name}")
    if not info.min <= fill_value <= info.max:
        raise ValueError(f"fill value does not fit {dtype_name}")

    header = bytearray(reference.header)
    endian = reference.endian
    struct.pack_into(endian + "h", header, 70, datatype)
    struct.pack_into(endian + "h", header, 72, bitpix)
    struct.pack_into(endian + "f", header, 108, 352.0)
    struct.pack_into(endian + "f", header, 112, 1.0)
    struct.pack_into(endian + "f", header, 116, 0.0)
    struct.pack_into(endian + "f", header, 124, float(max(fill_value, int(crop.max(initial=fill_value)))))
    struct.pack_into(endian + "f", header, 128, float(min(fill_value, int(crop.min(initial=fill_value)))))
    struct.pack_into(endian + "i", header, 140, int(max(fill_value, int(crop.max(initial=fill_value)))))
    struct.pack_into(endian + "i", header, 144, int(min(fill_value, int(crop.min(initial=fill_value)))))
    header[344:348] = b"n+1\0"

    output_dtype = dtype.newbyteorder(endian)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_output, compresslevel=6, mtime=0
        ) as output:
            output.write(header)
            output.write(b"\0\0\0\0")
            row = np.full(reference.shape_xyz[0], fill_value, dtype=output_dtype)
            x0, y0, z0 = starts
            x1, y1, z1 = stops
            for z in range(reference.shape_xyz[2]):
                in_z = z0 <= z < z1
                for y in range(reference.shape_xyz[1]):
                    row.fill(fill_value)
                    if in_z and y0 <= y < y1:
                        row[x0:x1] = crop[:, y - y0, z - z0]
                    output.write(row.tobytes(order="C"))


def materialize_image(source: Path, destination: Path, *, mode: str) -> str:
    """Materialize an input image without ever opening the source for writing."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    if mode == "symlink":
        destination.symlink_to(source.resolve(strict=True))
        return "symlink"
    if mode != "hardlink":
        raise ValueError("image mode must be hardlink, copy, or symlink")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"
