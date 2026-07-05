"""
Verify safetensors checkpoint integrity.

Each .safetensors file = 8-byte header length + JSON header + tensor data.
This checks that the file on disk is at least as large as the header claims,
catching truncated / partially-downloaded weight files (the cause of
"Error while deserializing header: incomplete metadata, file not fully covered").

Usage:  python check_checkpoints.py [experiments_dir]
Exit code 0 = all good, 1 = at least one bad file.
"""
import glob
import json
import os
import struct
import sys


def check(path: str) -> tuple[bool, str]:
    size = os.path.getsize(path)
    try:
        with open(path, "rb") as fh:
            hdr_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hdr_len))
    except Exception as ex:  # noqa: BLE001
        return False, f"unreadable header: {type(ex).__name__}: {ex}"

    max_end = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        max_end = max(max_end, meta["data_offsets"][1])

    expected = 8 + hdr_len + max_end
    if expected != size:
        short_by = expected - size
        return False, f"truncated: have {size:,} B, need {expected:,} B (short {short_by:,} B)"
    return True, f"ok ({size:,} B)"


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "./experiments"
    files = sorted(glob.glob(os.path.join(root, "**", "model.safetensors"), recursive=True))
    if not files:
        print(f"No model.safetensors files found under {root!r}")
        return 1

    bad = 0
    for f in files:
        ok, msg = check(f)
        print(f"{'OK ' if ok else 'BAD'}  {f}  -> {msg}")
        bad += not ok

    print(f"\n{len(files) - bad}/{len(files)} valid, {bad} bad")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
