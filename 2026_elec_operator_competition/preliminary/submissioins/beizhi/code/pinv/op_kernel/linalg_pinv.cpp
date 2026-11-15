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

#include "pinv_common.h"
#include "pinv_io.h"
#include "pinv_scalar.h"
#include "pinv_tiny.h"
#include "pinv_inverse_state.h"
#include "pinv_parallel_inverse.h"
#include "pinv_inverse199.h"
#include "pinv_deferred256.h"
#include "pinv_jacobi.h"

// One row or column has a single singular value: ||A||_2.
template <class T, typename ExtractType7_0, typename ExtractType7_1, typename ExtractType7_2, typename ExtractType7_3,
          typename ExtractType7_4>
__aicore__ inline void VectorPinvBlock3Block7(ExtractType7_0 &t, ExtractType7_1 &input, ExtractType7_2 &a,
                                              ExtractType7_3 &raw, ExtractType7_4 &base)
{
    if constexpr (sizeof(T) == 4) {
        DataCopyPad(a, input[base], {1, t.q * 4, 0, 0, 0}, {false, 0, 0, 0});
        LocalSync<HardEvent::MTE2_V>();
    } else {
        DataCopyPad(raw, input[base], {1, t.q * 2, 0, 0, 0}, {false, 0, 0, 0});
        LocalSync<HardEvent::MTE2_V>();
        Cast(a, raw, RoundMode::CAST_NONE, t.q);
        PipeBarrier<PIPE_V>();
    }
}

template <class T, typename ExtractType8_0, typename ExtractType8_1, typename ExtractType8_2>
__aicore__ inline void VectorPinvBlock3Block8(ExtractType8_0 &a, ExtractType8_1 &raw, ExtractType8_2 &x)
{
    if constexpr (sizeof(T) == 4) {
        LocalSync<HardEvent::MTE2_S>();
        x = raw.GetValue(0);
    } else {
        LocalSync<HardEvent::MTE2_V>();
        Cast(a, raw, RoundMode::CAST_NONE, 1);
        LocalSync<HardEvent::V_S>();
        x = a.GetValue(0);
    }
}

template <class T, typename ExtractType9_0, typename ExtractType9_1, typename ExtractType9_2>
__aicore__ inline void VectorPinvBlock3Block9(ExtractType9_0 &a, ExtractType9_1 &raw, ExtractType9_2 &y)
{
    if constexpr (sizeof(T) == 4) {
        raw.SetValue(0, y);
        LocalSync<HardEvent::S_MTE3>();
    } else {
        a.SetValue(0, y);
        LocalSync<HardEvent::S_V>();
        Cast(raw, a, RoundMode::CAST_RINT, 1);
        LocalSync<HardEvent::V_MTE3>();
    }
}

template <class T, typename ExtractType3_0, typename ExtractType3_1, typename ExtractType3_2, typename ExtractType3_3,
          typename ExtractType3_4, typename ExtractType3_5, typename ExtractType3_6, typename ExtractType3_7>
__aicore__ inline void VectorPinvBlock3(ExtractType3_0 &t, ExtractType3_1 &input, ExtractType3_2 &output,
                                        ExtractType3_3 &a, ExtractType3_4 &sq, ExtractType3_5 &red,
                                        ExtractType3_6 &scratch, ExtractType3_7 &raw)
{
    for (uint32_t batch = GetBlockIdx(); batch < t.batch; batch += GetBlockNum()) {
        uint32_t base = batch * t.q;
        if (t.q == 1) {
            DataCopyPad(raw, input[base], {1, (uint32_t)sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
            float x;
            VectorPinvBlock3Block8<T>(a, raw, x);
            float ax = x < 0 ? -x : x;
            float threshold = t.atol > ax * t.rtol ? t.atol : ax * t.rtol;
            float y = ax > threshold ? 1.0f / x : 0.0f;
            VectorPinvBlock3Block9<T>(a, raw, y);
            DataCopyPad(output[base], raw, {1, (uint32_t)sizeof(T), 0, 0, 0});
            LocalSync<HardEvent::MTE3_MTE2>();
            continue;
        }
        VectorPinvBlock3Block7<T>(t, input, a, raw, base);
        Mul(sq, a, a, t.q);
        PipeBarrier<PIPE_V>();
        uint32_t full = t.q / 64, tail = t.q % 64;
        if (full)
            WholeReduceSum(scratch, sq, 64, full, 1, 1, 8);
        if (tail)
            WholeReduceSum(scratch[full], sq[full * 64], tail, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(red, scratch, full + (tail ? 1 : 0), 1, 1, 1, 8);
        LocalSync<HardEvent::V_S>();
        float norm = red.GetValue(0), scale = (norm > t.atol * t.atol && t.rtol < 1.0f) ? 1.0f / norm : 0.0f;
        Muls(a, a, scale, t.q);
        if constexpr (sizeof(T) == 4) {
            LocalSync<HardEvent::V_MTE3>();
            DataCopyPad(output[base], a, {1, t.q * 4, 0, 0, 0});
        } else {
            PipeBarrier<PIPE_V>();
            Cast(raw, a, RoundMode::CAST_RINT, t.q);
            LocalSync<HardEvent::V_MTE3>();
            DataCopyPad(output[base], raw, {1, t.q * 2, 0, 0, 0});
        }
        LocalSync<HardEvent::MTE3_MTE2>();
    }
}

template <class T> __aicore__ inline void VectorPinv(GM_ADDR ai, GM_ADDR ao, LinalgPinvTilingData t)
{
    InitSocState();
    GlobalTensor<T> input, output;
    input.SetGlobalBuffer((__gm__ T *)ai);
    output.SetGlobalBuffer((__gm__ T *)ao);
    LocalTensor<float> mem(TPosition::VECCALC, 0, 2048);
    auto a = mem;
    auto sq = mem[512];
    auto red = mem[1024];
    auto scratch = mem[1088];
    auto raw = mem[1536].template ReinterpretCast<T>();
    VectorPinvBlock3<T>(t, input, output, a, sq, red, scratch, raw);
}

template <class T, bool SINGLE>
__aicore__ __attribute__((noinline)) void CachedFallback(GM_ADDR A, GM_ADDR out, GM_ADDR user, LinalgPinvTilingData t)
{
    PinvScalar<T> op;
    op.Init(A, out, user, t);
    if constexpr (SINGLE)
        op.Run(false, 0);
    else
        op.Run();
}

template <class T>
__aicore__ __attribute__((noinline, aligned(128))) bool PrepareDeferred256(GM_ADDR A, GM_ADDR out, GM_ADDR ws,
                                                                           LinalgPinvTilingData t, bool &need)
{
    PinvDeferred256<float, 256> fast;
    fast.Init(A, out, ws, t);
    bool okay = fast.Run();
    need = fast.NeedsRefine();
    return okay;
}

template <class T>
__aicore__ inline void CubeInverse256(GM_ADDR A, GM_ADDR out, GM_ADDR ws, LinalgPinvTilingData t, GM_ADDR systemWs,
                                      AscendC::tiling::TCubeTiling cubeTiling, bool &okay)
{
    if ASCEND_IS_AIC {
        SetHF32Mode(false);
        SetHF32TransMode(false);
    }
    bool need = false;
    okay = true;
    if ASCEND_IS_AIV {
        okay = PrepareDeferred256<T>(A, out, ws, t, need);
    }
    ParallelInverse<float, 256> fast;
    FP32Cube mm;
    REGIST_MATMUL_OBJ(fast.Pipe(), systemWs, mm, &cubeTiling);
    if ASCEND_IS_AIV {
        if (okay && need) {
            fast.Init(A, out, ws, t);
            fast.SetCube(&mm);
            fast.RefineOnly();
        }
    }
}

template <class T>
__aicore__ inline void CubePinv256(GM_ADDR A, GM_ADDR out, GM_ADDR ws, LinalgPinvTilingData t, GM_ADDR systemWs,
                                   AscendC::tiling::TCubeTiling cubeTiling)
{
    bool okay = true;
    CubeInverse256<T>(A, out, ws, t, systemWs, cubeTiling, okay);
    if ASCEND_IS_AIV {
        if (!okay) {
            PinvJacobi<float> fallback;
            fallback.Init(A, out, ws, t);
            fallback.Run();
        }
    }
}

template <typename DT_A, typename ExtractType6_0, typename ExtractType6_1, typename ExtractType6_2,
          typename ExtractType6_3>
__aicore__ inline void linalg_pinvBlock2Block4Block6(ExtractType6_0 &A, ExtractType6_1 &A_pinv, ExtractType6_2 &user,
                                                     ExtractType6_3 &ok)
{
    if constexpr (sizeof(DT_A) == 4) {
        if (t.p == 199)
            ok = SpecializedInverse<DT_A, 199>(A, A_pinv, user, t);
        else if (t.p == 256)
            ok = SpecializedInverse<DT_A, 256>(A, A_pinv, user, t);
        else
            ok = SpecializedInverse<DT_A, 0>(A, A_pinv, user, t);
    } else
        ok = SpecializedInverse<DT_A, 0>(A, A_pinv, user, t);
}

template <typename DT_A, typename ExtractType4_0, typename ExtractType4_1, typename ExtractType4_2,
          typename ExtractType4_3>
__aicore__ inline void linalg_pinvBlock2Block4(ExtractType4_0 &A, ExtractType4_1 &A_pinv, ExtractType4_2 &user,
                                               ExtractType4_3 &ok)
{
    if (t.m == t.n) {
        linalg_pinvBlock2Block4Block6<DT_A>(A, A_pinv, user, ok);
    } else {
        uint64_t oldSize =
            (uint64_t)t.batch * t.p * (2 * t.qp + t.pp + 8) * 4 + (uint64_t)(t.group * t.groups + t.groups + 1) * 32;
        uint64_t fastSize = (uint64_t)t.groups * 2 * t.cols * (t.pp + 8) * 4 + (uint64_t)t.batch * t.pp * t.pp * 4 +
                            (uint64_t)(t.group * t.groups + 1) * 32;
        uint64_t offset = ((oldSize > fastSize ? oldSize : fastSize) + 255) / 256 * 256;
        GM_ADDR gram = user + offset;
        GM_ADDR inv = gram + (uint64_t)t.batch * t.p * t.p * 4;
        GM_ADDR scratch = inv + (uint64_t)t.batch * t.p * t.p * 4;
        {
            PinvJacobi<DT_A> pre;
            pre.Init(A, A_pinv, user, t);
            pre.FormGram(gram);
        }
        auto gt = t;
        gt.m = t.p;
        gt.n = t.p;
        gt.q = t.p;
        gt.qp = t.pp;
        gt.atol = t.atol * t.atol;
        gt.rtol = t.rtol * t.rtol;
        if (gt.rtol < 1e-6f)
            gt.rtol = 1e-6f;
        {
            ParallelInverse<float> fast;
            fast.Init(gram, inv, scratch, gt);
            ok = fast.Run();
        }
        if (ok) {
            PinvJacobi<DT_A> post;
            post.Init(A, A_pinv, user, t);
            post.GramOutput(inv);
        }
    }
}

template <typename DT_A, uint32_t WIDTH, typename ExtractType5_0, typename ExtractType5_1, typename ExtractType5_2>
__aicore__ inline bool linalg_pinvBlock2Block5(ExtractType5_0 &A, ExtractType5_1 &A_pinv, ExtractType5_2 &workspace)
{
    if constexpr (WIDTH > 64) {
        constexpr uint32_t order = WIDTH - 64;
        if (GetBlockIdx() >= t.batch * 4) {
            SyncAll();
            return true;
        }
        auto user = GetUserWorkspace(workspace);
        if (!SmallInverse<DT_A, order, 4>(A, A_pinv, user, t) && GetBlockIdx() % 4 == 0) {
            auto one = t;
            one.batch = 1;
            uint64_t offset = uint64_t(GetBlockIdx() / 4) * order * order * sizeof(DT_A);
            CachedFallback<DT_A, true>(A + offset, A_pinv + offset, user, one);
        }
        return true;
    } else if constexpr (WIDTH > 0 && sizeof(DT_A) == 4) {
        if (GetBlockIdx() >= t.batch)
            return true;
        if constexpr (WIDTH <= 16) {
            if (AlignedTinyInverse<DT_A, WIDTH, 1>(A, A_pinv, nullptr, t))
                return true;
        } else {
            if (SmallInverse<DT_A, WIDTH, 1>(A, A_pinv, nullptr, t))
                return true;
        }
    } else if constexpr (WIDTH > 0) {
        TinyPinv<DT_A, WIDTH> fast;
        fast.Init(A, A_pinv, t);
        if (fast.Run())
            return true;
    }
    return false;
}

template <typename DT_A, bool LARGE, uint32_t WIDTH, typename ExtractType2_0, typename ExtractType2_1,
          typename ExtractType2_2>
__aicore__ inline bool linalg_pinvBlock2(ExtractType2_0 &A, ExtractType2_1 &A_pinv, ExtractType2_2 &workspace)
{
    if constexpr (LARGE) {
        auto user = GetUserWorkspace(workspace);
        bool ok = false;
        linalg_pinvBlock2Block4<DT_A>(A, A_pinv, user, ok);
        if (!ok) {
            PinvJacobi<DT_A> op;
            op.Init(A, A_pinv, user, t);
            op.Run();
        }
    } else {
        if constexpr (WIDTH <= 1) {
            if (t.p == 1) {
                VectorPinv<DT_A>(A, A_pinv, t);
                return true;
            }
        }
        if (linalg_pinvBlock2Block5<DT_A, WIDTH>(A, A_pinv, workspace))
            return true;
        CachedFallback<DT_A, false>(A, A_pinv, GetUserWorkspace(workspace), t);
    }
    return false;
}

template <typename DT_A, bool LARGE, uint32_t WIDTH>
__global__ __aicore__ void linalg_pinv(GM_ADDR A, GM_ADDR A_pinv, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LinalgPinvCubeTilingData);
    REGISTER_TILING_FOR_TILINGKEY("LARGE != 1 || WIDTH != 2", LinalgPinvTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData, t, tiling);
    if constexpr (LARGE && WIDTH == 3) {
        auto user = GetUserWorkspace(workspace);
        bool ok = t.p == 199 ? SpecializedInverse<DT_A, 199>(A, A_pinv, user, t)
                             : WideInPlaceInverse<DT_A>(A, A_pinv, user, t);
        if (!ok) {
            PinvJacobi<DT_A> fallback;
            fallback.Init(A, A_pinv, user, t);
            fallback.Run();
        }
        return;
    }
    if constexpr (LARGE && WIDTH == 2) {
        GET_TILING_DATA_WITH_STRUCT(LinalgPinvCubeTilingData, ct, tiling);
        CubePinv256<DT_A>(A, A_pinv, GetUserWorkspace(workspace), t, GetSysWorkSpacePtr(), ct.cube);
        return;
    }
    if (linalg_pinvBlock2<DT_A, LARGE, WIDTH>(A, A_pinv, workspace))
        return;
}
