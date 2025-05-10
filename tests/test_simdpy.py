import importlib, pathlib, sys, time

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

# ------------------------------------------------------------------ #
#  util: import freshly built module from ./build                     #
# ------------------------------------------------------------------ #
root = pathlib.Path(__file__).resolve().parents[1]
build_dir = root / "build"
sys.path.insert(0, str(build_dir))

simdpy = importlib.import_module("simdpy")   # .so produced by CMake

# ------------------------------------------------------------------ #
#  helpers                                                           #
# ------------------------------------------------------------------ #
def rand_array(size: int, dtype):
    return np.random.default_rng().random(size, dtype=dtype)


# ------------------------------------------------------------------ #
#  1. property‑based correctness (hypothesis)                        #
# ------------------------------------------------------------------ #
@given(
    size=st.integers(min_value=0, max_value=1_000_000),
    dtype=st.sampled_from([np.float32, np.float64]),
)
@settings(max_examples=50, deadline=None)
def test_add_correct_random(size, dtype):
    a, b = rand_array(size, dtype), rand_array(size, dtype)
    expected = a + b
    result = simdpy.add(a, b)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=0.0)
    # ensure new allocation (no alias)
    assert result.ctypes.data != a.ctypes.data
    assert result.ctypes.data != b.ctypes.data


# ------------------------------------------------------------------ #
#  2. edge cases                                                     #
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_empty_and_singleton(dtype):
    for size in (0, 1):
        arr = np.ones(size, dtype=dtype)
        out = simdpy.add(arr, arr)
        np.testing.assert_array_equal(out, arr * 2)


def test_large_array():
    n = 2 ** 21  # 2M elements (~16 MiB for float64)
    a = np.ones(n, dtype=np.float64)
    out = simdpy.add(a, a)
    assert out[0] == 2.0 and out[-1] == 2.0


# ------------------------------------------------------------------ #
#  3. error paths                                                    #
# ------------------------------------------------------------------ #
def test_length_mismatch():
    a = np.ones(8, dtype=np.float32)
    b = np.ones(7, dtype=np.float32)
    with pytest.raises(RuntimeError):
        simdpy.add(a, b)


def test_dtype_mismatch():
    a = np.ones(4, dtype=np.float32)
    b = np.ones(4, dtype=np.float64)
    with pytest.raises(RuntimeError):
        simdpy.add(a, b)


# ------------------------------------------------------------------ #
#  4. (optional) performance smoke test                              #
#     skip on CI by default – enable locally with:                   #
#     pytest -m perf                                                 #
# ------------------------------------------------------------------ #
@pytest.mark.perf
def test_speed_vs_numpy(benchmark):
    n = 10_000_000
    a = rand_array(n, np.float32)
    b = rand_array(n, np.float32)

    # NumPy baseline
    numpy_stats = benchmark(lambda: a + b)
    numpy_mean  = numpy_stats.mean()   # seconds

    # simdpy timing
    start = time.perf_counter()
    simdpy.add(a, b)
    simd_time = time.perf_counter() - start

    # Expect at least 2× faster than pure NumPy on Apple Silicon
    assert simd_time < numpy_mean * 0.5
