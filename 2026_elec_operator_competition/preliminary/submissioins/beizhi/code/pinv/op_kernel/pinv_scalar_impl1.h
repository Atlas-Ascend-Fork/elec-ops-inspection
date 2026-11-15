/*
 * Copyright (c) 2026 Beizhi contributors.
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

template <class T> __aicore__ inline float PinvScalar<T>::Fabs(float x)
{
    return x < 0 ? -x : x;
}

template <class T> __aicore__ inline float PinvScalar<T>::Root(float x)
{
    red.SetValue(0, x > 0 ? x : 0);
    PipeBarrier<PIPE_ALL>();
    Sqrt(red, red, 1);
    PipeBarrier<PIPE_ALL>();
    return red.GetValue(0);
}

template <class T> __aicore__ inline float PinvScalar<T>::Hyp(float x, float y)
{
    x = Fabs(x);
    y = Fabs(y);
    float z = x > y ? x : y;
    if (z == 0)
        return 0;
    x /= z;
    y /= z;
    return z * Root(x * x + y * y);
}

template <class T> __aicore__ inline float PinvScalar<T>::Signed(float x, float y)
{
    return y < 0 ? -Fabs(x) : Fabs(x);
}

template <class T> __aicore__ inline void PinvScalar<T>::Fence()
{
    PipeBarrier<PIPE_ALL>();
}

template <class T> __aicore__ inline float PinvScalar<T>::Dot(LocalTensor<float> x, LocalTensor<float> y, uint32_t len)
{
    Mul(tmp, x, y, len);
    PipeBarrier<PIPE_V>();
    ReduceSum(red, tmp, tmp2, len);
    Fence();
    return red.GetValue(0);
}

template <class T> __aicore__ inline float PinvScalar<T>::MaxAbs(LocalTensor<float> x, uint32_t len)
{
    Abs(tmp, x, len);
    PipeBarrier<PIPE_V>();
    uint32_t full = len / 64, tail = len % 64;
    if (full)
        WholeReduceMax(red, tmp, 64, full, 1, 1, 8);
    if (tail)
        WholeReduceMax(red[full * 2], tmp[full * 64], tail, 1, 1, 1, 8);
    Fence();
    float largest = 0;
    for (uint32_t i = 0; i < full + (tail ? 1 : 0); i++) {
        float z = red.GetValue(i * 2);
        if (z > largest)
            largest = z;
    }
    return largest;
}

template <class T> __aicore__ inline void PinvScalar<T>::ReadInput(LocalTensor<float> dst, uint32_t col)
{
    PinvReadInput(input, raw, offsets, bfOffsets, mem[36864], t, base, col, dst);
}

template <class T> __aicore__ inline void PinvScalar<T>::ReadCol(bool right, uint32_t col, LocalTensor<float> dst)
{
    uint32_t len = right ? t.pp : t.qp;
    Fence();
    if (t.cached)
        Adds(dst, (right ? vc : wc)[col * len], 0.0f, len);
    else
        DataCopy(dst, (right ? vg : work)[col * len], len);
    Fence();
}

template <class T> __aicore__ inline void PinvScalar<T>::WriteCol(bool right, uint32_t col, LocalTensor<float> src)
{
    uint32_t len = right ? t.pp : t.qp;
    Fence();
    if (t.cached)
        Adds((right ? vc : wc)[col * len], src, 0.0f, len);
    else
        DataCopy((right ? vg : work)[col * len], src, len);
    Fence();
}

template <class T> __aicore__ inline void PinvScalar<T>::Rotate(bool right, uint32_t j, uint32_t k, float c, float s)
{
    uint32_t len = right ? t.pp : t.qp;
    ReadCol(right, j, a);
    ReadCol(right, k, b);
    Muls(out, a, c, len);
    Muls(tmp2, b, s, len);
    PipeBarrier<PIPE_V>();
    Sub(out, out, tmp2, len);
    Muls(b, b, c, len);
    Muls(tmp2, a, s, len);
    PipeBarrier<PIPE_V>();
    Add(b, b, tmp2, len);
    Fence();
    WriteCol(right, j, out);
    WriteCol(right, k, b);
}

template <class T> __aicore__ inline void PinvScalar<T>::WriteOutput(LocalTensor<float> src, uint32_t at, uint32_t len)
{
    PinvWriteOutput(output, raw, src, at, len);
}

template <class T>
template <typename ExtractType102_0, typename ExtractType102_1, typename ExtractType102_2, typename ExtractType102_3,
          typename ExtractType102_4, typename ExtractType102_5, typename ExtractType102_6, typename ExtractType102_7,
          typename ExtractType102_8, typename ExtractType102_9>
__aicore__ inline bool
PinvScalar<T>::FastInverseBlock102(ExtractType102_0 &n, ExtractType102_1 &span, ExtractType102_2 &width,
                                   ExtractType102_3 &mat, ExtractType102_4 &u, ExtractType102_5 &v, ExtractType102_6 &h,
                                   ExtractType102_7 &tmp, ExtractType102_8 &red, ExtractType102_9 &anorm)
{
    return PinvTinyEliminate<T>(n, span, width, mat, u, v, h, tmp, red, anorm, offsets);
}

template <class T>
template <typename ExtractType106_0, typename ExtractType106_1, typename ExtractType106_2, typename ExtractType106_3>
__aicore__ inline void PinvScalar<T>::FastInverseBlock106(ExtractType106_0 &n, ExtractType106_1 &np,
                                                          ExtractType106_2 &width, ExtractType106_3 &mat)
{
    PinvLoadTiny(input, raw, base, n, np, width, mat);
}

template <class T> template <uint32_t N = 0> __aicore__ inline bool PinvScalar<T>::FastInverse()
{
    if (t.m != t.n || t.p > 64)
        return false;
    uint32_t n = N ? N : t.p, np = N ? (N + 7) / 8 * 8 : t.pp, span = np * 2, width = span + 8;
    auto mat = mem;
    auto u = mem[38408], v = mem[38928], h = mem[39448], tmp = mem[41496], red = mem[42528];
    Duplicate(mat, 0.0f, n * width);
    Fence();
    FastInverseBlock106(n, np, width, mat);
    Fence();
    Mul(mem[16384], mat, mat, n * width);
    PipeBarrier<PIPE_V>();
    ReduceSum(red, mem[16384], mem[24576], n * width);
    LocalSync<HardEvent::V_S>();
    float anorm = red.GetValue(0);
    if (anorm == 0)
        return false;
    for (uint32_t i = 0; i < n; i++) {
        mat.SetValue(i * width + np + i, 1.0f);
        offsets.SetValue(i, i * width * 4);
    }
    Fence();
    if (!FastInverseBlock102(n, span, width, mat, u, v, h, tmp, red, anorm))
        return false;
    Fence(); // Augmented Frobenius norm is a conservative inverse-norm bound.
    Mul(mem[16384], mat, mat, n * width);
    PipeBarrier<PIPE_V>();
    ReduceSum(red, mem[16384], mem[24576], n * width);
    LocalSync<HardEvent::V_S>();
    float ix = red.GetValue(0);
    float threshold = t.atol * t.atol > anorm * t.rtol * t.rtol ? t.atol * t.atol : anorm * t.rtol * t.rtol;
    if (!(threshold * ix < 0.25f))
        return false;
    PinvStoreTiny(output, raw, mat, base, n, np, width);
    return true;
}

template <class T>
template <typename ExtractType100_0, typename ExtractType100_1>
__aicore__ inline void PinvScalar<T>::BidiagonalBlock100(ExtractType100_0 &k, ExtractType100_1 &scale)
{
    if (scale > 0) {
        Muls(v, v, 1.0f / scale, t.pp);
        Fence();
        float ss = Dot(v, v, t.pp), f = v.GetValue(k + 1), g = -Signed(Root(ss), f), beta = f * g - ss;
        v.SetValue(k + 1, f - g);
        e.SetValue(k + 1, g * scale);
        Duplicate(h, 0.0f, t.qp);
        Duplicate(hv, 0.0f, t.pp);
        Fence();
        for (uint32_t j = k + 1; j < t.p; j++) {
            float factor = v.GetValue(j);
            ReadCol(false, j, a);
            ReadCol(true, j, b);
            Axpy(h, a, factor, t.qp);
            Axpy(hv, b, factor, t.pp);
            Fence();
        }
        if (k + 1)
            Duplicate(h, 0.0f, k + 1);
        Fence();
        Muls(h, h, 1.0f / beta, t.qp);
        Muls(hv, hv, 1.0f / beta, t.pp);
        Fence();
        for (uint32_t j = k + 1; j < t.p; j++) {
            float factor = v.GetValue(j);
            ReadCol(false, j, a);
            ReadCol(true, j, b);
            Axpy(a, h, factor, t.qp);
            Axpy(b, hv, factor, t.pp);
            Fence();
            WriteCol(false, j, a);
            WriteCol(true, j, b);
        }
    }
}

template <class T> __aicore__ inline void PinvScalar<T>::Bidiagonal()
{
    Duplicate(d, 0.0f, t.pp);
    Duplicate(e, 0.0f, t.pp);
    Fence();
    for (uint32_t k = 0; k < t.p; k++) {
        ReadCol(false, k, u);
        if (k)
            Duplicate(u, 0.0f, k);
        Fence();
        float scale = MaxAbs(u, t.qp);
        if (scale > 0) {
            Muls(u, u, 1.0f / scale, t.qp);
            Fence();
            float ss = Dot(u, u, t.qp);
            float f = u.GetValue(k), g = -Signed(Root(ss), f), beta = f * g - ss;
            u.SetValue(k, f - g);
            d.SetValue(k, g * scale);
            Fence();
            for (uint32_t j = k + 1; j < t.p; j++) {
                ReadCol(false, j, a);
                float factor = Dot(u, a, t.qp) / beta;
                Axpy(a, u, factor, t.qp);
                Fence();
                WriteCol(false, j, a);
            }
        }
        if (k + 1 == t.p)
            continue;
        Duplicate(v, 0.0f, t.pp);
        Fence();
        for (uint32_t j = k + 1; j < t.p; j++) {
            ReadCol(false, j, a);
            v.SetValue(j, a.GetValue(k));
        }
        Fence();
        scale = MaxAbs(v, t.pp);
        BidiagonalBlock100(k, scale);
    }
}

template <class T> __aicore__ inline float PinvScalar<T>::QRShift(float x, float y, float z, float g, float hh)
{
    float denominator = 2 * hh * y;
    // A zero diagonal or an underflowed denominator requires an unshifted QR step.
    if (x == 0.0f || denominator == 0.0f)
        return x;
    float f = ((y - z) * (y + z) + (g - hh) * (g + hh)) / denominator;
    g = Hyp(f, 1);
    float signedNorm = f + Signed(g, f);
    if (signedNorm == 0.0f)
        return x;
    return ((x - z) * (x + z) + hh * (y / signedNorm - hh)) / x;
}

template <class T>
template <typename ExtractType103_0, typename ExtractType103_1, typename ExtractType103_2, typename ExtractType103_3,
          typename ExtractType103_4, typename ExtractType103_5, typename ExtractType103_6, typename ExtractType103_7,
          typename ExtractType103_8, typename ExtractType103_9>
__aicore__ inline void PinvScalar<T>::DiagonalQRBlock103(ExtractType103_0 &k, ExtractType103_1 &l, ExtractType103_2 &z,
                                                         ExtractType103_3 &x, ExtractType103_4 &y, ExtractType103_5 &g,
                                                         ExtractType103_6 &hh, ExtractType103_7 &f, ExtractType103_8 &c,
                                                         ExtractType103_9 &s)
{
    for (int j = l; j < k; j++) {
        int i = j + 1;
        g = e.GetValue(i);
        y = d.GetValue(i);
        hh = s * g;
        g = c * g;
        z = Hyp(f, hh);
        e.SetValue(j, z);
        c = z ? f / z : 1;
        s = z ? hh / z : 0;
        f = x * c + g * s;
        g = g * c - x * s;
        hh = y * s;
        y *= c;
        Rotate(true, j, i, c, -s);
        z = Hyp(f, hh);
        d.SetValue(j, z);
        if (z) {
            c = f / z;
            s = hh / z;
        }
        f = c * g + s * y;
        x = c * y - s * g;
    }
}

template <class T>
template <typename ExtractType105_0, typename ExtractType105_1, typename ExtractType105_2, typename ExtractType105_3>
__aicore__ inline void PinvScalar<T>::DiagonalQRBlock104Block105(ExtractType105_0 &tol, ExtractType105_1 &k,
                                                                 ExtractType105_2 &flag, ExtractType105_3 &l)
{
    if (flag) {
        float c = 0, s = 1;
        for (int i = l; i <= k; i++) {
            float f = s * e.GetValue(i);
            e.SetValue(i, c * e.GetValue(i));
            if (Fabs(f) <= tol)
                break;
            float g = d.GetValue(i), z = Hyp(f, g);
            d.SetValue(i, z);
            if (z == 0.0f)
                break;
            c = g / z;
            s = -f / z;
        }
    }
}

template <class T>
template <typename ExtractType104_0>
__aicore__ inline void PinvScalar<T>::DiagonalQRBlock104(ExtractType104_0 &tol)
{
    for (int k = int(t.p) - 1; k >= 0; k--) {
        for (uint32_t it = 0; it < 80; it++) {
            bool flag = true;
            int l = k;
            for (; l >= 0; l--) {
                if (Fabs(e.GetValue(l)) <= tol) {
                    flag = false;
                    break;
                }
                if (l == 0 || Fabs(d.GetValue(l - 1)) <= tol)
                    break;
            }
            DiagonalQRBlock104Block105(tol, k, flag, l);
            float z = d.GetValue(k);
            if (l == k) {
                if (z < 0) {
                    d.SetValue(k, -z);
                    ReadCol(true, k, a);
                    Muls(a, a, -1.0f, t.pp);
                    Fence();
                    WriteCol(true, k, a);
                }
                break;
            }
            float x = d.GetValue(l), y = d.GetValue(k - 1), g = e.GetValue(k - 1), hh = e.GetValue(k);
            float f = QRShift(x, y, z, g, hh);
            float c = 1, s = 1;
            DiagonalQRBlock103(k, l, z, x, y, g, hh, f, c, s);
            e.SetValue(l, 0.0f);
            e.SetValue(k, f);
            d.SetValue(k, x);
        }
    }
}

template <class T> __aicore__ inline void PinvScalar<T>::DiagonalQR()
{
    float anorm = 0;
    for (uint32_t i = 0; i < t.p; i++) {
        float z = Fabs(d.GetValue(i)) + Fabs(e.GetValue(i));
        if (z > anorm)
            anorm = z;
    }
    float tol = anorm * 1.1920928955078125e-7f;
    DiagonalQRBlock104(tol);
}

template <class T> __aicore__ inline void PinvScalar<T>::RebuildAndPolish()
{
    for (uint32_t j = 0; j < t.p; j++) {
        ReadCol(true, j, v);
        Duplicate(h, 0.0f, t.qp);
        Fence();
        for (uint32_t k = 0; k < t.p; k++) {
            ReadInput(a, k);
            Axpy(h, a, v.GetValue(k), t.qp);
            Fence();
        }
        WriteCol(false, j, h);
    }
    for (uint32_t j = 0; j < t.p; j++)
        for (uint32_t k = j + 1; k < t.p; k++) {
            ReadCol(false, j, u);
            ReadCol(false, k, v);
            float aa = Dot(u, u, t.qp), bb = Dot(v, v, t.qp), ab = Dot(u, v, t.qp);
            if (aa <= 0 || bb <= 0 || Fabs(ab) <= 2e-7f * Root(aa * bb))
                continue;
            float tau = (bb - aa) / (2 * ab), tt = Signed(1.0f, tau) / (Fabs(tau) + Hyp(1.0f, tau)),
                  c = 1.0f / Root(1 + tt * tt), s = tt * c;
            Rotate(false, j, k, c, s);
            Rotate(true, j, k, c, s);
        }
}
