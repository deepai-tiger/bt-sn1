"""Round-trip check for the distilled-embedding blob format.

The packer lives in the harness and the unpacker lives in the submission, so
nothing catches a disagreement between them except this. A silent mismatch
would not raise -- `load_distilled()` swallows exceptions by design -- it would
just quietly return garbage weights, which is the worst possible failure.

    /tmp/venv/bin/python harness/test_blob.py
"""

from __future__ import annotations

import importlib.util
import lzma
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import distill_fit as fit  # noqa: E402

SUBMISSION = Path(__file__).parent.parent / "champion-code" / "code_submission_v1.py"


def load_submission():
    spec = importlib.util.spec_from_file_location("submission_blob_test", SUBMISSION)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_blob_test"] = module
    spec.loader.exec_module(module)
    return module


def check_char_packing() -> None:
    rng = np.random.default_rng(1)
    module = load_submission()
    for size in (1, 2, 3, 7, 1000, 20001):
        blob = rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
        text = fit.encode_chars(blob)
        assert all(0 <= ord(c) - 19968 < 32768 for c in text), "codepoint out of range"
        got = module._unpack_blob(text)
        assert got[:size] == blob, f"char packing broken at {size} bytes"
    print("char packing: ok")


def check_trailing_pad_tolerance() -> None:
    """Packing rounds up to whole codepoints, so the decoder is handed padding.

    Every compressed length whose bit count is not a multiple of 15 -- fourteen
    out of fifteen of them -- produces a trailing pad byte. `lzma.decompress`
    treats that as a corrupt stream and the failure is silent, so this is
    checked across sizes rather than trusting whichever one got baked in.
    """
    module = load_submission()
    rng = np.random.default_rng(3)
    for size in (100, 1000, 9999, 24596):
        raw = rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
        packed = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
        recovered = module._unpack_blob(fit.encode_chars(packed))
        assert lzma.LZMADecompressor().decompress(recovered) == raw, \
            f"pad byte broke decompression at {size} bytes"
    print("trailing pad tolerance: ok")


def check_matrix_round_trip(sparse: bool = False) -> None:
    """Full path: a random matrix through quantize -> serialize -> decode."""
    rng = np.random.default_rng(2)
    word, char, dims, blocks, protos = 512, 256, 16, 4, 64
    rows = word + char
    # Row norms spanning orders of magnitude, like a real fit.
    W = rng.normal(size=(rows, dims)).astype(np.float32)
    W *= np.exp(rng.normal(0, 2.0, size=rows)).astype(np.float32)[:, None]
    # Most rows empty, as after pruning, and deliberately including both the
    # first and last row: the sparse layout deltas indices against a sentinel,
    # and an off-by-one there only shows up when an edge row survives.
    W[rng.choice(rows, int(rows * 0.6), replace=False)] = 0.0
    W[0] = rng.normal(size=dims)
    W[-1] = rng.normal(size=dims)

    payload, approx = fit.quantize_pq(W, blocks, protos, seed=0)
    payload["sparse"] = sparse
    blob = fit.serialize(payload, dims, word, char)
    text = fit.encode_chars(lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME))

    module = load_submission()
    module.DISTILLED_BLOB = text
    module._DISTILLED_CACHE = None
    loaded = module.load_distilled()
    assert loaded is not None, "submission failed to decode the blob"
    matrix, got_word, got_char = loaded
    assert (got_word, got_char) == (word, char), "bucket counts lost"
    assert matrix.shape == (rows, dims), f"shape {matrix.shape} != {(rows, dims)}"

    # The decoder must reproduce the quantized matrix exactly, bar int8 rounding
    # of the prototypes, which is applied on the packing side only.
    scale = np.abs(approx).max()
    error = np.abs(matrix - approx).max() / scale
    assert error < 0.02, f"decode mismatch: max relative error {error:.4f}"
    layout = "sparse" if sparse else "dense"
    print(f"matrix round trip ({layout}): ok "
          f"(max relative error {error:.5f}, {len(text):,} chars)")

    zero_rows = np.abs(approx).sum(1) == 0
    assert np.abs(matrix[zero_rows]).sum() == 0, "zero rows did not survive"
    print("zero rows: ok")


def check_absent_blob() -> None:
    module = load_submission()
    module.DISTILLED_BLOB = ""
    module._DISTILLED_CACHE = None
    assert module.load_distilled() is None
    assert module.distilled_features(["hello world"] * 5) is None
    module.DISTILLED_BLOB = "not a valid blob"
    module._DISTILLED_CACHE = None
    assert module.load_distilled() is None, "corrupt blob must decode to None"
    print("missing/corrupt blob: ok")


if __name__ == "__main__":
    check_char_packing()
    check_trailing_pad_tolerance()
    check_matrix_round_trip(sparse=False)
    check_matrix_round_trip(sparse=True)
    check_absent_blob()
    print("all blob format checks passed")
