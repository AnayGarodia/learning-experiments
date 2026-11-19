"""enwik8, byte-level. One-time download (~36MB zip), cached as raw bytes.
Train = first 20MB, val = next 2MB."""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

CACHE = Path(__file__).parent / "enwik8.npy"
URL = "https://mattmahoney.net/dc/enwik8.zip"
TRAIN_BYTES, VAL_BYTES = 20_000_000, 2_000_000


def load():
    if not CACHE.exists():
        raw = urllib.request.urlopen(URL, timeout=120).read()
        data = zipfile.ZipFile(io.BytesIO(raw)).read("enwik8")
        np.save(CACHE, np.frombuffer(data[: TRAIN_BYTES + VAL_BYTES], dtype=np.uint8))
    arr = np.load(CACHE)
    return arr[:TRAIN_BYTES], arr[TRAIN_BYTES:]


def batches(arr, batch, seq, seed):
    """Random 256-byte windows, endless."""
    rng = np.random.default_rng(seed)
    while True:
        idx = rng.integers(0, len(arr) - seq - 1, size=batch)
        x = np.stack([arr[i : i + seq + 1] for i in idx]).astype(np.int64)
        yield x  # (batch, seq+1); input x[:, :-1], target x[:, 1:]


def val_windows(arr, n=512, seq=256, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr) - seq - 1, size=n)
    return np.stack([arr[i : i + seq + 1] for i in idx]).astype(np.int64)


if __name__ == "__main__":
    tr, va = load()
    assert len(tr) == TRAIN_BYTES and len(va) == VAL_BYTES
    b = next(batches(tr, 4, 256, 0))
    assert b.shape == (4, 257) and b.max() < 256
    print("data ok:", len(tr), "train bytes; sample:", bytes(tr[1000:1040]).decode("utf-8", "replace"))
