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
// Included in the KernelUB216 class; preserves its member scope and access.
    __aicore__ inline KernelUB216() {}

    __aicore__ inline void RunPipeBarrierPhase1(float &inv, uint32_t &k, uint32_t &kA, uint32_t &p, uint32_t &wA) {

        __ubuf__ float *mp = reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr());
        __ubuf__ float *rq = reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr());
        __ubuf__ uint16_t *maskP = reinterpret_cast<__ubuf__ uint16_t *>(absbuf_.GetPhyAddr()) + 128U;
        LDET_MASK_SET(64U);
        vector_dup(reinterpret_cast<__ubuf__ uint32_t *>(maskP), 0xFFFFF000U, 1, 1, 1, 8, 0);
        PipeBarrier<PIPE_V>();
        __ubuf__ float *H = k < 2U ? reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr()) : mp;
        __ubuf__ float *L = k < 2U ? H + 32U : mp + S;
        uint32_t tile = k < 2U ? 32U : wA;
        for (uint32_t off = 0; off < wA; off += tile) {
            uint32_t ct = wA - off;
            if (ct > tile)
                ct = tile;
            __ubuf__ float *aP = mp + (p * S + kA + off), *qP = rq + off;
            LDET_MASK_SET(2U * ct);
            vand(reinterpret_cast<__ubuf__ uint16_t *>(H), reinterpret_cast<__ubuf__ uint16_t *>(aP), maskP, 1, 1, 1, 1,
                 8, 8, 0);
            PipeBarrier<PIPE_V>();
            LDET_MASK_SET(ct);
            vsub(L, aP, H, 1, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
            vmuls(qP, H, inv, 1, 1, 1, 8, 8);
            PipeBarrier<PIPE_V>();
            vmuls(L, L, inv, 1, 1, 1, 8, 8);
            PipeBarrier<PIPE_V>();
            vadd(qP, qP, L, 1, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
        LDET_MASK_SET(wA);
        if (p != k) {
            vmuls(mp + (p * S + kA), mp + (k * S + kA), 1.0f, 1, 1, 1, 8, 8);
        }
    }
    template <class PhaseArg1, class PhaseArg3>
    __aicore__ inline void RunSetFlagPhase2(GM_ADDR &a, PhaseArg1 &ag, GM_ADDR &det, PhaseArg3 &dg) {

        ag.SetGlobalBuffer(reinterpret_cast<__gm__ DT_A *>(a));
        dg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(det));
        for (uint32_t i = 0; i < S; ++i)
            tabI_.SetValue(i, static_cast<int32_t>(i * S * sizeof(float)));
        SetFlag<HardEvent::S_V>(::kEvSV);
        WaitFlag<HardEvent::S_V>(::kEvSV);
        LDET_MASK_SET(S);
        for (uint32_t r = 0; r < 8U; ++r)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(tabRI_[r * S].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(tabI_.GetPhyAddr()), static_cast<int32_t>(r * sizeof(float)), 1,
                  1, 1, 8, 8);
        LDET_MASK_RST();
        SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
        if constexpr (N == S) {
            DataCopy(mat_, ag, N * N);
        } else {
            DataCopyExtParams load{static_cast<uint16_t>(N), N * sizeof(float), 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{true, 0U, static_cast<uint8_t>(S - N), 0.0f};
            DataCopyPad(mat_, ag, load, pad);
        }
        SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
        WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
    }

    __aicore__ inline void RunvaxpyPhase3(uint32_t &c0, uint32_t &i, uint32_t &k, uint32_t &kA, uint32_t &w) {

        LDET_MASK_SET(w);
        __ubuf__ float *dstP = reinterpret_cast<__ubuf__ float *>(mat_[(k + 1U) * S + c0].GetPhyAddr());
        __ubuf__ float *rkP = reinterpret_cast<__ubuf__ float *>(rowkbuf_[c0 - kA].GetPhyAddr());
        __ubuf__ float *lP = reinterpret_cast<__ubuf__ float *>(colbuf_[1].GetPhyAddr());
        for (; i + 7U < N; i += 8U) {
            float l0 = lP[0], l1 = lP[1], l2 = lP[2], l3 = lP[3], l4 = lP[4], l5 = lP[5], l6 = lP[6], l7 = lP[7];
            vaxpy(dstP, rkP, l0, 1, 1, 1, 8, 8);
            vaxpy(dstP + S, rkP, l1, 1, 1, 1, 8, 8);
            vaxpy(dstP + 2U * S, rkP, l2, 1, 1, 1, 8, 8);
            vaxpy(dstP + 3U * S, rkP, l3, 1, 1, 1, 8, 8);
            vaxpy(dstP + 4U * S, rkP, l4, 1, 1, 1, 8, 8);
            vaxpy(dstP + 5U * S, rkP, l5, 1, 1, 1, 8, 8);
            vaxpy(dstP + 6U * S, rkP, l6, 1, 1, 1, 8, 8);
            vaxpy(dstP + 7U * S, rkP, l7, 1, 1, 1, 8, 8);
            dstP += 8U * S;
            lP += 8U;
        }
        for (; i < N; ++i) {
            float lv = *lP;
            vaxpy(dstP, rkP, lv, 1, 1, 1, 8, 8);
            dstP += S;
            lP += 1U;
        }
    }

    __aicore__ inline void RunSetFlagPhase4(uint32_t &cnt, int32_t &ex, uint32_t &k, uint32_t &kA, uint32_t &p,
                                            float &prod, float &sign, uint32_t &wA) {

        if (p != k) {
            float tv = colbuf_.GetValue(0);
            colbuf_.SetValue(0, colbuf_.GetValue(p - k));
            colbuf_.SetValue(p - k, tv);
            sign = -sign;
        }
        float piv = colbuf_.GetValue(0);
        float inv = LdetSharedOperations::ReciprocalWithZero58(piv, -1.0f);
        SetFlag<HardEvent::S_V>(::kEvSV);
        WaitFlag<HardEvent::S_V>(::kEvSV);
        RunPipeBarrierPhase1(inv, k, kA, p, wA);
        LDET_MASK_RST();
        uint32_t c0 = (k + 1U) & ~(::kAlign - 1U), w = S - c0, i = k + 1U;
        if (wA > 64U) {
            RunvaxpyPhase3(c0, i, k, kA, w);
        } else if (cnt > 1U) {
            LocalTensor<float> factors(TPosition::VECCALC, Mat, 64U * 8U);
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(colbuf_.GetPhyAddr()), 1, 8,
                  static_cast<uint8_t>((cnt + 7U) / 8U));
            PipeBarrier<PIPE_V>();
            for (uint32_t col = c0; col < S; col += 64U) {
                uint32_t width = S - col;
                if (width > 64U)
                    width = 64U;
                AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
                vmla(reinterpret_cast<__ubuf__ float *>(mat_[(k + 1U) * S + col].GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(rowkbuf_[col - kA].GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(factors[8].GetPhyAddr()), static_cast<uint8_t>(cnt - 1U), 1, 1,
                     0, static_cast<uint8_t>(S / 8U), 0, 1);
            }
        }
        LDET_MASK_RST();
        ProdAccum(prod, ex, piv);
    }
    __aicore__ inline void RunSetFlagPhase5(uint32_t &cnt) {

        LDET_MASK_RST();
        LDET_MASK_SET(cnt);
        for (uint32_t c = 0, coff = 0; c < (cnt + 127U) >> 7; ++c, coff += 128U) {
            uint32_t cc = cnt - coff;
            if (cc > 128U)
                cc = 128U;
            AscendCUtils::SetMask<float>(0, cc);
            LdetVcmax128(tmp_[c * 8U], absbuf_[coff], cc);
        }
        LDET_MASK_RST();
        SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
    }
    template <class PhaseArg1, class PhaseArg3>
    __aicore__ inline void RunvgatherPhase6(GM_ADDR &a, PhaseArg1 &ag, GM_ADDR &det, PhaseArg3 &dg) {

        RunSetFlagPhase2(a, ag, det, dg);
        float sign = 1.0f, prod = 1.0f;
        int32_t ex = 0;
        uint32_t singular = 0;
        for (uint32_t k = 0; k < N; ++k) {
            uint32_t kA = k & ~(::kAlign - 1U), cnt = N - k, be = k * (S + 1U);
            LDET_MASK_SET(cnt);
            vgather(reinterpret_cast<__ubuf__ uint32_t *>(colbuf_.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t *>(tabRU_[(be & 7U) * S].GetPhyAddr()),
                    reinterpret_cast<uint64_t>(mat_.GetPhyAddr()) + (be & ~7U) * sizeof(float), 8, 1);
            PipeBarrier<PIPE_V>();
            vabs(reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(colbuf_.GetPhyAddr()), 1, 1, 1, 8, 8);
            uint32_t segN = (cnt + 63U) / 64U;
            RunSetFlagPhase5(cnt);
            float maxv = 0.0f;
            uint32_t p = k;
            for (uint32_t s = 0; s < segN; ++s) {
                uint32_t so = LDET_RM_PAIR_OFF(s);
                float v = tmp_.GetValue(so);
                if (v > maxv) {
                    float idxF = tmp_.GetValue(so + 1U);
                    maxv = v;
                    p = k + s * 64U + *reinterpret_cast<uint32_t *>(&idxF);
                }
            }
            if (maxv == 0.0f) {
                singular = 1U;
                break;
            }
            uint32_t wA = S - kA;
            RunSetFlagPhase4(cnt, ex, k, kA, p, prod, sign, wA);
        }
        res_.SetValue(0, singular != 0U ? 0.0f : sign * ProdFinal(prod, ex));
    }
    template <bool ANY_CORE = false> __aicore__ inline void Run(GM_ADDR a, GM_ADDR det, const LinalgDetTilingData &td) {
        if ((!ANY_CORE && GetBlockIdx() != 0U) || td.n != N || td.stride != N || td.batchCount != 1U ||
            td.matPerCore != 1U || td.isLarge != 0U || td.groupSize != 1U)
            return;
        mat_ = LocalTensor<float>(TPosition::VECCALC, Mat, S * S);
        colbuf_ = LocalTensor<float>(TPosition::VECCALC, Col, S);
        absbuf_ = LocalTensor<float>(TPosition::VECCALC, Abs, S);
        rowkbuf_ = LocalTensor<float>(TPosition::VECCALC, Row, S);
        tabI_ = LocalTensor<int32_t>(TPosition::VECCALC, Col, S);
        tabRI_ = LocalTensor<int32_t>(TPosition::VECCALC, Tabs, 8U * S);
        tabRU_ = LocalTensor<uint32_t>(TPosition::VECCALC, Tabs, 8U * S);
        tmp_ = LocalTensor<float>(TPosition::VECCALC, Tmp, 16U);
        res_ = LocalTensor<float>(TPosition::VECCALC, Res, 8U);
        GlobalTensor<DT_A> ag;
        GlobalTensor<float> dg;
        RunvgatherPhase6(a, ag, det, dg);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(dg, res_, {1, 4U, 0, 0, 0});
    }
