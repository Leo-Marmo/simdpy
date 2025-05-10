//
// Created by Leo Straccia on 5/10/25.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Accelerate/Accelerate.h>

namespace py = pybind11;

/* ------------------------------------------------------------------ */
/*  ── internal helpers ────────────────────────────────────────────── */

template<typename T>
struct vadd_impl;

// float32 specialization
template<>
struct vadd_impl<float> {
    static void call(const float *a, const float *b, float *result, std::size_t n) {
        vDSP_vadd(a, (vDSP_Stride) 1, b, (vDSP_Stride) 1, result, (vDSP_Stride) 1, (vDSP_Length) n);
    }
};

// float64 specialization
template<>
struct vadd_impl<double> {
    static void call(const double *a, const double *b, double *result, std::size_t n) {
        vDSP_vaddD(a, (vDSP_Stride) 1, b, (vDSP_Stride) 1, result, (vDSP_Stride) 1, (vDSP_Length) n);
    }
};

/* ------------------------------------------------------------------ */
/*  ── generic wrapper ─────────────────────────────────────────────── */

template<typename T>
py::array_t<T> add_with_accelerate(const py::array_t<T, py::array::c_style | py::array::forcecast> &a,
                                   const py::array_t<T, py::array::c_style | py::array::forcecast> &b) {
    auto A = a.template unchecked<1>();
    auto B = b.template unchecked<1>();

    if (A.size() != B.size()) {
        throw std::runtime_error("Input lists must have the same length.");
    }

    // allocation output with same dtype
    py::array_t<T> out(A.size());
    auto O = out.template mutable_unchecked<1>(); {
        // release GIL during heavy compute
        py::gil_scoped_release release;
        vadd_impl<T>::call(A.data(0), B.data(0), O.mutable_data(0), A.size());
    }
    return out;
}

/* ------------------------------------------------------------------ */
/*  ── Python‑visible dispatcher ───────────────────────────────────── */

py::array add_accel(const py::array &a, const py::array &b) {
    // Decide based on dtype
    if (py::isinstance<py::array_t<float> >(a) && py::isinstance<py::array_t<float> >(b)) {
        return add_with_accelerate<float>(a, b);
    }

    if (py::isinstance<py::array_t<double> >(a) && py::isinstance<py::array_t<double> >(b)) {
        return add_with_accelerate<double>(a, b);
    }

    throw std::runtime_error("Accelerated add only supports float32 and float64 arrays of the same length");
}

PYBIND11_MODULE(simdpy, m) {
    m.def("add", &add_accel, "Element-wise addition using Apple Accelerate SIMD");
}
