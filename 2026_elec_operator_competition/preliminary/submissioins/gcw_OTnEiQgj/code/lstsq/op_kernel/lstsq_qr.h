/*
 * Copyright (c) 2026 gcw_OTnEiQgj.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "lstsq_common.h"

namespace elec {

ELEC_FN float lu_abs(float x) {
    return x < 0.0f ? -x : x;
}

ELEC_FN bool lu_finite(float x) {
    return is_finite(x);
}

ELEC_FN void lu_add(float value, float& sum, float& c) {
    float y = value - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

ELEC_FN void lstsq_two_sum(float a, float b, float& high, float& low) {
    volatile float sum = a + b;
    volatile float virtual_b = sum - a;
    volatile float recovered_a = sum - virtual_b;
    volatile float error_a = a - recovered_a;
    volatile float error_b = b - virtual_b;
    volatile float error = error_a + error_b;
    high = sum;
    low = error;
}

ELEC_FN void lstsq_dot_add(float a, float b, float& high, float& low) {
    volatile float product = a * b;
    const float ah = bits_float(float_bits(a) & 0xfffff000u);
    const float bh = bits_float(float_bits(b) & 0xfffff000u);
    volatile float al = a - ah, bl = b - bh;
    volatile float hh = ah * bh;
    volatile float first = hh - product;
    volatile float hl = ah * bl, lh = al * bh, ll = al * bl;
    volatile float second = first + hl;
    volatile float third = second + lh;
    volatile float product_error = third + ll;

    float h, he, l, le, cross, ce, next_h, next_l;
    lstsq_two_sum(high, product, h, he);
    lstsq_two_sum(low, product_error, l, le);
    lstsq_two_sum(he, l, cross, ce);
    lstsq_two_sum(h, cross, next_h, next_l);
    volatile float tail_first = next_l + ce;
    volatile float tail = tail_first + le;
    lstsq_two_sum(next_h, tail, high, low);
}

ELEC_FN void lstsq_difference_pair(float a, float high, float low,
                                   float& result_high, float& result_low) {
    float first, first_low, second, second_low;
    lstsq_two_sum(a, -high, first, first_low);
    lstsq_two_sum(first, -low, second, second_low);
    volatile float tail = first_low + second_low;
    lstsq_two_sum(second, tail, result_high, result_low);
}

ELEC_FN int lstsq_residual(Tensor a, Tensor b, Tensor x, Tensor high, Tensor low,
                           uint64_t m, uint64_t n, uint64_t k) {
    for (uint64_t row = 0; row < m; ++row) {
        for (uint64_t target = 0; target < k; ++target) {
            float dot = 0.0f, error = 0.0f;
            for (uint64_t col = 0; col < n; ++col) {
                lstsq_dot_add(a.get(row * n + col), x.get(col * k + target), dot, error);
            }
            float rh, rl;
            lstsq_difference_pair(b.get(row * k + target), dot, error, rh, rl);
            if (!lu_finite(rh) || !lu_finite(rl)) {
                return NUMERIC_ERROR;
            }
            high.set(row * k + target, rh);
            low.set(row * k + target, rl);
        }
    }
    return SUCCESS;
}

ELEC_FN int lstsq_qr_apply_reflector(
    uint64_t col, uint64_t m, uint64_t n, uint64_t k, float beta,
    Tensor& qr, Tensor& y, const Tensor& v) {
    for (uint64_t target = col + 1; target < n; ++target) {
        float dot = 0.0f, dot_correction = 0.0f;
        for (uint64_t row = col; row < m; ++row) {
            lu_add(v.get(row) * qr.get(target * m + row), dot, dot_correction);
        }
        const float multiplier = beta * dot;
        for (uint64_t row = col; row < m; ++row) {
            const float value = qr.get(target * m + row) - v.get(row) * multiplier;
            if (!lu_finite(value)) {
                return NUMERIC_ERROR;
            }
            qr.set(target * m + row, value);
        }
    }
    for (uint64_t target = 0; target < k; ++target) {
        float dot = 0.0f, dot_correction = 0.0f;
        for (uint64_t row = col; row < m; ++row) {
            lu_add(v.get(row) * y.get(row * k + target), dot, dot_correction);
        }
        const float multiplier = beta * dot;
        for (uint64_t row = col; row < m; ++row) {
            const float value = y.get(row * k + target) - v.get(row) * multiplier;
            if (!lu_finite(value)) {
                return NUMERIC_ERROR;
            }
            y.set(row * k + target, value);
        }
    }
    return SUCCESS;
}

ELEC_FN int lstsq_qr_factor_col(
    uint64_t col, uint64_t m, uint64_t n, uint64_t k,
    Tensor& qr, Tensor& y, Tensor& v, Math& math) {
    float scale = 0.0f;
    for (uint64_t row = col; row < m; ++row) {
        const float magnitude = lu_abs(qr.get(col * m + row));
        if (magnitude > scale) { scale = magnitude; }
    }
    if (scale == 0.0f) { return SINGULAR; }
    float squares = 0.0f, correction = 0.0f;
    for (uint64_t row = col; row < m; ++row) {
        const float ratio = qr.get(col * m + row) / scale;
        lu_add(ratio * ratio, squares, correction);
    }
    const float norm = scale * math.sqrt(squares);
    if (!lu_finite(norm) || norm == 0.0f) { return NUMERIC_ERROR; }
    const float sign = qr.get(col * m + col) < 0.0f ? -1.0f : 1.0f;
    for (uint64_t row = col; row < m; ++row) { v.set(row, qr.get(col * m + row) / norm); }
    v.set(col, v.get(col) + sign);
    squares = 0.0f;
    correction = 0.0f;
    for (uint64_t row = col; row < m; ++row) {
        const float value = v.get(row);
        lu_add(value * value, squares, correction);
    }
    if (squares == 0.0f) { return SINGULAR; }
    const int status = lstsq_qr_apply_reflector(col, m, n, k, 2.0f / squares, qr, y, v);
    if (status != SUCCESS) { return status; }
    qr.set(col * m + col, -sign * norm);
    for (uint64_t row = col + 1; row < m; ++row) { qr.set(col * m + row, 0.0f); }
    return SUCCESS;
}

ELEC_FN int lstsq_qr_back_sub(uint64_t m, uint64_t n, uint64_t k, const Tensor& qr, const Tensor& y, Tensor& x) {
    for (uint64_t ii = n; ii > 0; --ii) {
        const uint64_t row = ii - 1;
        const float diagonal = qr.get(row * m + row);
        if (diagonal == 0.0f) { return SINGULAR; }
        for (uint64_t target = 0; target < k; ++target) {
            float dot = 0.0f, correction = 0.0f;
            for (uint64_t col = row + 1; col < n; ++col) {
                lu_add(qr.get(col * m + row) * x.get(col * k + target), dot, correction);
            }
            const float value = (y.get(row * k + target) - dot) / diagonal;
            if (!lu_finite(value)) { return NUMERIC_ERROR; }
            x.set(row * k + target, value);
        }
    }
    return SUCCESS;
}

ELEC_FN int lstsq_qr_solve_rtr(uint64_t m, uint64_t n, const Tensor& qr, Tensor& v) {
    for (uint64_t row = 0; row < n; ++row) {
        const float diag = qr.get(row * m + row);
        if (diag == 0.0f) { return SINGULAR; }
        float dot = 0.0f, error = 0.0f;
        for (uint64_t col = 0; col < row; ++col) { lstsq_dot_add(qr.get(row * m + col), v.get(col), dot, error); }
        const float value = ((v.get(row) - dot) - error) / diag;
        if (!lu_finite(value)) { return NUMERIC_ERROR; }
        v.set(row, value);
    }
    for (uint64_t ii = n; ii > 0; --ii) {
        const uint64_t r_idx = ii - 1;
        const float r_diag = qr.get(r_idx * m + r_idx);
        if (r_diag == 0.0f) { return SINGULAR; }
        float acc = 0.0f, err = 0.0f;
        for (uint64_t c_idx = r_idx + 1; c_idx < n; ++c_idx) { lstsq_dot_add(qr.get(c_idx * m + r_idx), v.get(c_idx), acc, err); }
        const float next_v = ((v.get(r_idx) - acc) - err) / r_diag;
        if (!lu_finite(next_v)) { return NUMERIC_ERROR; }
        v.set(r_idx, next_v);
    }
    return SUCCESS;
}

ELEC_FN int lstsq_qr_refine_target(
    uint64_t target, uint64_t m, uint64_t n, uint64_t k,
    const Tensor& a, const Tensor& qr, const Tensor& y, const Tensor& y_low, Tensor& v, Tensor& x) {
    for (uint64_t col = 0; col < n; ++col) {
        float dot = 0.0f, error = 0.0f;
        for (uint64_t row = 0; row < m; ++row) {
            const float value = a.get(row * n + col);
            lstsq_dot_add(value, y.get(row * k + target), dot, error);
            lstsq_dot_add(value, y_low.get(row * k + target), dot, error);
        }
        v.set(col, dot + error);
    }
    const int status = lstsq_qr_solve_rtr(m, n, qr, v);
    if (status != SUCCESS) {
        return status;
    }
    for (uint64_t row = 0; row < n; ++row) {
        const float value = x.get(row * k + target) + v.get(row);
        if (!lu_finite(value)) {
            return NUMERIC_ERROR;
        }
        x.set(row * k + target, value);
    }
    return SUCCESS;
}

ELEC_FN int lstsq_qr_write_residuals(
    uint64_t batch, uint64_t m, uint64_t n, uint64_t k,
    Tensor a, Tensor b, Tensor x, Tensor y, Tensor y_low, Tensor& out_r) {
    const int status = lstsq_residual(a, b, x, y, y_low, m, n, k);
    if (status != SUCCESS) {
        return status;
    }
    for (uint64_t target = 0; target < k; ++target) {
        float sum = 0.0f, correction = 0.0f;
        for (uint64_t row = 0; row < m; ++row) {
            const float value = y.get(row * k + target);
            lstsq_dot_add(value, value, sum, correction);
            lstsq_dot_add(2.0f * value, y_low.get(row * k + target), sum, correction);
        }
        sum += correction;
        if (!lu_finite(sum)) {
            return NUMERIC_ERROR;
        }
        out_r.set(batch * k + target, sum);
    }
    return SUCCESS;
}

ELEC_FN bool lstsq_qr_valid_args(const Config& c, const Tensor* in, const Tensor* out, const Tensor& work) {
    if (c.batch == 0 || c.m == 0 || c.n == 0 || c.m < c.n || c.k == 0 || c.dtype != F32) {
        return false;
    }
    if (!in[0].valid() || !in[1].valid() || !out[0].valid() || !work.valid()) {
        return false;
    }
    const bool need_r = (c.m > c.n) || ((c.flags & LSTSQ_SQUARE_RESIDUAL) != 0);
    return !need_r || out[1].valid();
}

ELEC_FN int lstsq_qr_load_batch(uint64_t m, uint64_t n, uint64_t k, const Tensor& a, const Tensor& b, Tensor& qr, Tensor& y) {
    if (n == 0) {
        return INVALID;
    }
    for (uint64_t i = 0; i < m * n; ++i) {
        const float value = a.get(i);
        if (!lu_finite(value)) {
            return NUMERIC_ERROR;
        }
        qr.set((i % n) * m + i / n, value);
    }
    for (uint64_t i = 0; i < m * k; ++i) {
        const float value = b.get(i);
        if (!lu_finite(value)) {
            return NUMERIC_ERROR;
        }
        y.set(i, value);
    }
    return SUCCESS;
}

ELEC_FN int compute_lstsq(const Config& c, Tensor* in, Tensor* out, Tensor work, Math& math) {
    if (!lstsq_qr_valid_args(c, in, out, work)) { return INVALID; }
    const uint64_t m = c.m, n = c.n, k = c.k;
    Tensor qr = work, y = work.slice(m * n), y_low = work.slice(m * n + m * k), v = work.slice(m * n + 2 * m * k);
    for (uint64_t batch = 0; batch < c.batch; ++batch) {
        Tensor a = in[0].slice(batch * m * n);
        Tensor b = in[1].slice(batch * c.rhs_batch_stride);
        Tensor x = out[0].slice(batch * n * k);
        int status = lstsq_qr_load_batch(m, n, k, a, b, qr, y);
        if (status != SUCCESS) { return status; }
        for (uint64_t col = 0; col < n; ++col) {
            status = lstsq_qr_factor_col(col, m, n, k, qr, y, v, math);
            if (status != SUCCESS) { return status; }
        }
        status = lstsq_qr_back_sub(m, n, k, qr, y, x);
        if (status != SUCCESS) { return status; }
        for (uint32_t iteration = 0; iteration < 3; ++iteration) {
            status = lstsq_residual(a, b, x, y, y_low, m, n, k);
            if (status != SUCCESS) { return status; }
            for (uint64_t target = 0; target < k; ++target) {
                status = lstsq_qr_refine_target(target, m, n, k, a, qr, y, y_low, v, x);
                if (status != SUCCESS) { return status; }
            }
        }
        if (m > n || (c.flags & LSTSQ_SQUARE_RESIDUAL)) {
            status = lstsq_qr_write_residuals(batch, m, n, k, a, b, x, y, y_low, out[1]);
            if (status != SUCCESS) { return status; }
        }
    }
    return SUCCESS;
}

}  // namespace elec
