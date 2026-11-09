/*
 * Copyright (c) 2026 dihidao.
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
// Included in the KernelLinalgDet class; preserves its member scope and access.
    __aicore__ inline void N16TailFourAcceptedProdSmallCommonPhase2(int32_t &ex, bool &fastProd, float &prod,
                                                                    uint32_t &selected, float &sign, float &t00,
                                                                    float &t01, float &t02, float &t03, float &t20,
                                                                    float &t21, float &t22, float &t23, float &t30,
                                                                    float &t31, float &t32, float &t33) {

        if (selected == 2U) {
            sign = -sign;
            float swap0 = t00;
            t00 = t20;
            t20 = swap0;
            float swap1 = t01;
            t01 = t21;
            t21 = swap1;
            float swap2 = t02;
            t02 = t22;
            t22 = swap2;
            float swap3 = t03;
            t03 = t23;
            t23 = swap3;
        }
        if (selected == 3U) {
            sign = -sign;
            float swap0 = t00;
            t00 = t30;
            t30 = swap0;
            float swap1 = t01;
            t01 = t31;
            t31 = swap1;
            float swap2 = t02;
            t02 = t32;
            t32 = swap2;
            float swap3 = t03;
            t03 = t33;
            t33 = swap3;
        }
        ProdSmallCommon(fastProd, prod, ex, t00);
    }

    __aicore__ inline void N16TailFourAcceptedStatePhase3(int32_t &ex, bool &fastProd, float &prod, uint32_t &selected,
                                                          float &sign, float &t00, float &t01, float &t02, float &t03,
                                                          float &t10, float &t11, float &t12, float &t13, float &t20,
                                                          float &t21, float &t22, float &t23, float &t30, float &t31,
                                                          float &t32, float &t33) {

        if (selected == 1U) {
            sign = -sign;
            float swap0 = t00;
            t00 = t10;
            t10 = swap0;
            float swap1 = t01;
            t01 = t11;
            t11 = swap1;
            float swap2 = t02;
            t02 = t12;
            t12 = swap2;
            float swap3 = t03;
            t03 = t13;
            t13 = swap3;
        }
        N16TailFourAcceptedProdSmallCommonPhase2(ex, fastProd, prod, selected, sign, t00, t01, t02, t03, t20, t21, t22,
                                                 t23, t30, t31, t32, t33);
        float inverse = LdetSharedOperations::ReciprocalWithZero58(t00, -1.0f);
        float normalized1 = t01 * inverse;
        float normalized2 = t02 * inverse;
        float normalized3 = t03 * inverse;
        t11 = t11 + t10 * normalized1;
        t12 = t12 + t10 * normalized2;
        t13 = t13 + t10 * normalized3;
        t21 = t21 + t20 * normalized1;
        t22 = t22 + t20 * normalized2;
        t23 = t23 + t20 * normalized3;
        t31 = t31 + t30 * normalized1;
        t32 = t32 + t30 * normalized2;
        t33 = t33 + t30 * normalized3;
    }

    __aicore__ inline void N16TailFourAcceptedStatePhase4(float &maximum, uint32_t &selected, float &t10, float &t20,
                                                          float &t30) {

        float abs1 = t10 < 0.0f ? -t10 : t10;
        if (abs1 > maximum) {
            maximum = abs1;
            selected = 1U;
        }
        float abs2 = t20 < 0.0f ? -t20 : t20;
        if (abs2 > maximum) {
            maximum = abs2;
            selected = 2U;
        }
        float abs3 = t30 < 0.0f ? -t30 : t30;
        if (abs3 > maximum) {
            maximum = abs3;
            selected = 3U;
        }
    }

    __aicore__ inline void N16TailFourAcceptedStatePhase5(float &maximum, uint32_t &selected, float &t21, float &t31) {

        float abs2 = t21 < 0.0f ? -t21 : t21;
        if (abs2 > maximum) {
            maximum = abs2;
            selected = 2U;
        }
        float abs3 = t31 < 0.0f ? -t31 : t31;
        if (abs3 > maximum) {
            maximum = abs3;
            selected = 3U;
        }
    }

    __aicore__ inline void N16TailFourAcceptedProdSmallCommonPhase6(int32_t &ex, bool &fastProd, float &prod,
                                                                    uint32_t &selected, float &sign, float &t22,
                                                                    float &t23, float &t32, float &t33) {

        if (selected == 3U) {
            sign = -sign;
            float swap2 = t22;
            t22 = t32;
            t32 = swap2;
            float swap3 = t23;
            t23 = t33;
            t33 = swap3;
        }
        ProdSmallCommon(fastProd, prod, ex, t22);
        float inverse = LdetSharedOperations::ReciprocalWithZero58(t22, -1.0f);
        float normalized3 = t23 * inverse;
        t33 = t33 + t32 * normalized3;
    }

    __aicore__ inline void N16TailFourAcceptedProdSmallCommonPhase7(int32_t &ex, bool &fastProd, float &prod,
                                                                    float &sign, bool &singular, float &t11, float &t12,
                                                                    float &t13, float &t21, float &t22, float &t23,
                                                                    float &t31, float &t32, float &t33) {

        {
            float maximum = t11 < 0.0f ? -t11 : t11;
            uint32_t selected = 1U;
            N16TailFourAcceptedStatePhase5(maximum, selected, t21, t31);
            if (!(maximum > 0.0f)) {
                singular = 1U;
                return;
            }
            N16TailFourAcceptedProdSmallCommonPhase1(ex, fastProd, prod, selected, sign, t11, t12, t13, t21, t22, t23,
                                                     t31, t32, t33);
        }
        {
            float maximum = t22 < 0.0f ? -t22 : t22;
            uint32_t selected = 2U;
            float abs3 = t32 < 0.0f ? -t32 : t32;
            if (abs3 > maximum) {
                maximum = abs3;
                selected = 3U;
            }
            if (!(maximum > 0.0f)) {
                singular = 1U;
                return;
            }
            N16TailFourAcceptedProdSmallCommonPhase6(ex, fastProd, prod, selected, sign, t22, t23, t32, t33);
        }
        {
            float maximum = t33 < 0.0f ? -t33 : t33;
            if (!(maximum > 0.0f)) {
                singular = 1U;
                return;
            }
            ProdSmallCommon(fastProd, prod, ex, t33);
        }
        return;
    }
    template <typename SingularT>
    __aicore__ inline void N16TailFourAccepted(float &sign, float &prod, int32_t &ex, bool &fastProd,
                                               SingularT &singular) {
        float t00 = mat_.GetValue(300U);
        float t01 = mat_.GetValue(301U);
        float t02 = mat_.GetValue(302U);
        float t03 = mat_.GetValue(303U);
        float t10 = mat_.GetValue(324U);
        float t11 = mat_.GetValue(325U);
        float t12 = mat_.GetValue(326U);
        float t13 = mat_.GetValue(327U);
        float t20 = mat_.GetValue(348U);
        float t21 = mat_.GetValue(349U);
        float t22 = mat_.GetValue(350U);
        float t23 = mat_.GetValue(351U);
        float t30 = mat_.GetValue(372U);
        float t31 = mat_.GetValue(373U);
        float t32 = mat_.GetValue(374U);
        float t33 = mat_.GetValue(375U);
        {
            float maximum = t00 < 0.0f ? -t00 : t00;
            uint32_t selected = 0U;
            N16TailFourAcceptedStatePhase4(maximum, selected, t10, t20, t30);
            if (!(maximum > 0.0f)) {
                singular = 1U;
                return;
            }
            N16TailFourAcceptedStatePhase3(ex, fastProd, prod, selected, sign, t00, t01, t02, t03, t10, t11, t12, t13,
                                           t20, t21, t22, t23, t30, t31, t32, t33);
        }
        return N16TailFourAcceptedProdSmallCommonPhase7(ex, fastProd, prod, sign, singular, t11, t12, t13, t21, t22,
                                                        t23, t31, t32, t33);
    }
    __aicore__ inline static void N16PlaneReload(GM_ADDR input, GM_ADDR output) {
        constexpr LinalgDetTilingData fixed16{1U, 16U, 24U, 1U, 0U, 1U};
        KernelLinalgDet fallback;
        fallback.Init(input, output, nullptr, fixed16, false, false, false, true);
        fallback.template ProcessSmallFixed<16U, false, false, true, true, true, true>();
    }
    __aicore__ inline void N16PlaneOriginal() {
        PipeBarrier<PIPE_ALL>();
        N16PlaneReload(reinterpret_cast<GM_ADDR>(const_cast<__gm__ DT_A *>(aGm_.GetPhyAddr())),
                       reinterpret_cast<GM_ADDR>(const_cast<__gm__ float *>(detGm_.GetPhyAddr())));
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
    }

    __aicore__ inline void N16BlockFourSetFlagPhase1(__ubuf__ float *&broad, const uint32_t &kb, __ubuf__ float *&mat,
                                                     __ubuf__ uint32_t *&mp, uint32_t &phase, float &productScalar,
                                                     __ubuf__ float *&rp, __ubuf__ float *&up) {

        const uint32_t p = kb * 40U + kb;
        const uint64_t diagonal = reinterpret_cast<uint64_t>(mat) + kb * 164U;
        // Signal completion of the previous Schur update (initially MTE2_V).
        // R Gather is read-only and may run while Scalar obtains the fixed2x2 panel.
        SetFlag<HardEvent::V_S>(2);
        set_vector_mask(0ULL, 0xffffffffULL);
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(rp), mp + 32U + phase * 32U, diagonal, 8U, 1U);
        WaitFlag<HardEvent::V_S>(2);
        float a = mat_.GetValue(p);
        float b = mat_.GetValue(p + 1U);
        float c = mat_.GetValue(p + 40U);
        float d = mat_.GetValue(p + 41U);
        float delta = a * d - b * c;
        float inverse = 1.0f / delta;
        productScalar *= delta;
        PipeBarrier<PIPE_V>();
        set_vector_mask(0ULL, 0xffffffffULL);
        vmuls(rp, rp, inverse, 1U, 1U, 1U, 8U, 8U);
        auto qp = mat + kb * 40U;
        set_vector_mask(0ULL, 0xffffULL);
        vmuls(up, qp, -d, 1U, 1U, 1U, 8U, 8U);
        vmuls(up + 16U, qp, c, 1U, 1U, 1U, 8U, 8U);
        PipeBarrier<PIPE_V>();
        ResetMask();
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(broad), reinterpret_cast<__ubuf__ uint32_t *>(rp), 1U, 8U, 4U);
        set_vector_mask(0ULL, 0xffffULL);
        vaxpy(up, qp + 40U, b, 1U, 1U, 1U, 8U, 8U);
        vaxpy(up + 16U, qp + 40U, -a, 1U, 1U, 1U, 8U, 8U);
        PipeBarrier<PIPE_V>();
        auto dst = mat + (kb + 2U) * 40U;
        vmla(dst, broad, up, static_cast<uint8_t>(14U - kb), 1U, 0U, 1U, 5U, 1U, 0U);
        PipeBarrier<PIPE_V>();
        vmla(dst, broad + 128U, up + 16U, static_cast<uint8_t>(14U - kb), 1U, 0U, 1U, 5U, 1U, 0U);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool N16BlockFourvgatherPhase2(__ubuf__ float *&broad, __ubuf__ float *&ds, __ubuf__ float *&lhs,
                                                     __ubuf__ float *&mat, __ubuf__ uint32_t *&mp,
                                                     __ubuf__ float *&pairs, float &productScalar,
                                                     __ubuf__ float *&result, __ubuf__ float *&rp,
                                                     __ubuf__ float *&terms, __ubuf__ float *&up) {

#pragma unroll
        for (uint32_t phase = 0U; phase < 6U; ++phase) {
            const uint32_t kb = phase * 2U;
            N16BlockFourSetFlagPhase1(broad, kb, mat, mp, phase, productScalar, rp, up);
        }
        // Six signed complementary-minor products. Gather produces both Mul operands.
        const uint64_t diagonal = reinterpret_cast<uint64_t>(mat) + 12U * 164U;
        set_vector_mask(0ULL, 0xffffffffffffffffULL);
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(lhs), mp + 224U, diagonal, 8U, 1U);
        PipeBarrier<PIPE_V>();
        set_vector_mask(0ULL, 0xffffffffULL);
        vmul(terms, lhs, lhs + 32U, 1U, 1U, 1U, 1U, 8U, 8U, 8U);
        PipeBarrier<PIPE_V>();
        set_vector_mask(0ULL, 0xffULL);
        vsub(pairs, terms, terms + 8U, 2U, 1U, 1U, 1U, 1U, 2U, 2U);
        PipeBarrier<PIPE_V>();
        set_vector_mask(0ULL, 0x3fULL);
        vmul(terms, pairs, pairs + 8U, 1U, 1U, 1U, 1U, 8U, 8U, 8U);
        PipeBarrier<PIPE_V>();
        vcadd(ds, terms, 1U, 1U, 1U, 8U, 0);
        PipeBarrier<PIPE_V>();
        set_vector_mask(0ULL, 1ULL);
        vmuls(result, ds, productScalar, 1U, 1U, 1U, 8U, 8U);
        ResetMask();
        return true;
    }
    __aicore__ inline bool N16BlockFour(GM_ADDR tiling) {
        static_assert(LD_BATCH == 9U, "private n16/B1 native block2 with direct tail4 performance research");
        LocalTensor<uint32_t> maps(TPosition::VECCALC, 4096U, 288U);
        GlobalTensor<uint32_t> tables;
        tables.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling + 576U), 288U);
        DataCopy(maps, tables, 288U);
        auto mp = reinterpret_cast<__ubuf__ uint32_t *>(maps.GetPhyAddr());
        auto pk = reinterpret_cast<__ubuf__ float *>(6144U);
        auto pairs = reinterpret_cast<__ubuf__ float *>(6272U);
        auto pb = reinterpret_cast<__ubuf__ float *>(6784U);
        auto lhs = reinterpret_cast<__ubuf__ float *>(7808U);
        auto rhs = reinterpret_cast<__ubuf__ float *>(8320U);
        auto terms = reinterpret_cast<__ubuf__ float *>(8832U);
        auto up = reinterpret_cast<__ubuf__ float *>(9152U);
        auto rp = reinterpret_cast<__ubuf__ float *>(9408U);
        auto broad = reinterpret_cast<__ubuf__ float *>(9664U);
        auto ds = reinterpret_cast<__ubuf__ float *>(12224U);
        auto product = reinterpret_cast<__ubuf__ float *>(12256U);
        auto mat = reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr());
        auto result = reinterpret_cast<__ubuf__ float *>(res_.GetPhyAddr());
        SetMaskNorm();
        float productScalar = 1.0f;
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        return N16BlockFourvgatherPhase2(broad, ds, lhs, mat, mp, pairs, productScalar, result, rp, terms, up);
    }
    template <class PhaseArg2>
    __aicore__ inline void N16DualPlanesvmulsPhase1(__ubuf__ float *&column, uint32_t &count, PhaseArg2 &currentRow,
                                                    uint64_t &diagonalAddress, __ubuf__ float *&factors,
                                                    uint32_t &index, uint32_t &p, float &product, __ubuf__ float *&row,
                                                    uint32_t &signMask) {

        vdiv(row, currentRow + p * 40U, factors + 128U + (index & 15U) * 8U, 1, 1, 1, 0, 8, 8, 0);
        if (p != 0U) {
            signMask ^= 0x80000000U;
            vmuls(currentRow + p * 40U, currentRow, 1.0f, 1, 1, 1, 8, 8);
            SmallIdentityCopy(factors + p * 8U, factors, 8U);
        }
        PipeBarrier<PIPE_V>();
        AscendCUtils::SetMask<float>(0ULL, count > 9U ? 0xffffffffULL : 0xff00ff00ULL);
        vmla(currentRow + 40U, factors + 8U, row, static_cast<uint8_t>(count - 1U), 1, 0, 1, 5, 1, 0);
        const float pivot = reinterpret_cast<__ubuf__ volatile float *>(column)[index & 15U];
        product *= pivot;
        currentRow += 40U;
        diagonalAddress += 164U;
    }

    __aicore__ inline void N16DualPlanesvgatherPhase2(__ubuf__ float *&column, uint32_t &count,
                                                      uint64_t &diagonalAddress, __ubuf__ float *&factors,
                                                      __ubuf__ uint32_t *&offsets) {

        uint64_t live = (1ULL << count) - 1ULL;
        AscendCUtils::SetMask<float>(0ULL, live | (live << 16U));
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(column), offsets, diagonalAddress, 8U, 1U);
        PipeBarrier<PIPE_V>();
        LdetVcmax128(tmp_, colbuf_, 32U);
        SetFlag<HardEvent::V_S>(::kEvVS);
        ResetMask();
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors), reinterpret_cast<__ubuf__ uint32_t *>(column), 1U, 8U,
              static_cast<uint8_t>((count + 23U) / 8U));
        PipeBarrier<PIPE_V>();
        AscendCUtils::SetMask<float>(0ULL, count > 9U ? 0xffffffffULL : 0xff00ff00ULL);
        WaitFlag<HardEvent::V_S>(::kEvVS);
    }
