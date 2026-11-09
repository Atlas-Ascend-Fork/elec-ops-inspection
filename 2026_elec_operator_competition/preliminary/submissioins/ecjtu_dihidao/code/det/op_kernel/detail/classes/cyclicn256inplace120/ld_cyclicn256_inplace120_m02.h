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
// Included in the CyclicN256Inplace120 class; preserves its member scope and access.
    __aicore__ inline void ApplyPanelUpdates(uint32_t kb, uint32_t q) {
        uint32_t first = LdetSharedOperations::FirstOwnedColumn54<CyclicN256Inplace120, CONTIGUOUS_BANDS>(*this, q);
        const uint32_t begin = first * B, width = owned * B > begin ? owned * B - begin : 0;
        if (!width)
            return;
        if constexpr (HETERO3) {
            if (width > 64U) {
                return ApplyPanelUpdatesPipeBarrierPhase4(begin, kb, width);
            }
        }
        if constexpr (PANEL_MASK) {
            auto mp = reinterpret_cast<__ubuf__ float *>(mat.GetPhyAddr()),
                 rp = reinterpret_cast<__ubuf__ float *>(row.GetPhyAddr());
            SetMaskNorm();
            AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
            for (uint32_t f = 0; f < B; ++f) {
                uint32_t k = kb + f, h = n - k, p = mu.GetValue(8 + f);
                float inv = value(mu.GetValue(8U + B + f));
                if (f)
                    PipeBarrier<PIPE_V>();
                LdetSharedOperations::SharedOwnerVmulsStage20<CyclicN256Inplace120>(*this, begin, f, h, inv, k, kb, mp,
                                                                                    p, rp);
            }
            PipeBarrier<PIPE_V>();
            ResetMask();
            SetMaskNorm();
            if constexpr (!PRUNE_STAGE_WAITS)
                VS();
            return;
        }
        ApplyPanelUpdatesvmulsPhase5(begin, kb, width);
    }
    __aicore__ inline void PrepareTail64Indices() {
        return LdetSharedOperations::PrepareTail64IndicesShared30<CyclicN256Inplace120>(*this);
    }

    template <auto T, auto TS>
    __aicore__ inline void Finish64vmlaPhase1(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f,
                                              const uint32_t &first, uint32_t &h, __ubuf__ uint32_t *&ip,
                                              __ubuf__ float *&mp, bool &pendingFar, float &piv, bool &prepared,
                                              __ubuf__ float *&rp) {

        if (h > 1U && first < T) {
            SetMaskNorm();
            AscendCUtils::SetMask<float>(64);
            vmla(mp + (f + 1U) * TS + first, rp + first, reinterpret_cast<__ubuf__ float *>(brc[8U].GetPhyAddr()),
                 static_cast<uint8_t>((h - 1U + 7U) / 8U), TS / 8U, 0, 1, TS, 0, 8);
            PipeBarrier<PIPE_V>();
            const uint32_t nf = f + 1U, nh = h - 1U;
            LdetSharedOperations::SharedOwnerVgatherStage33<CyclicN256Inplace120, T, TS>(*this, ap, cp, ip, mp, nf, nh);
            prepared = true;
            if (first + 8U < T) {
                LdetSharedOperations::SharedOwner26VmlaStage6<CyclicN256Inplace120, T, TS>(*this, f, first, h, mp, rp);
                pendingFar = true;
            }
        }
        acc(piv);
        if constexpr (PANEL_ACC) {
            if ((f & 31U) == 31U)
                normalizeProd();
        }
    }
    template <auto T, auto TS>
    __aicore__ inline void Finish64PipeBarrierPhase2(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f, uint32_t &h,
                                                     float &inv, __ubuf__ uint32_t *&ip, __ubuf__ float *&mp,
                                                     const uint32_t &p, bool &pendingFar, float &piv, bool &prepared,
                                                     __ubuf__ float *&rp) {

        LdetSharedOperations::SharedOwnerPipeBarrierStage15<CyclicN256Inplace120, T, TS>(*this, f, h, inv, mp, p,
                                                                                         pendingFar, rp);
        const uint32_t first = (f + 1U) & ~7U;
        prepared = false;
        Finish64vmlaPhase1<T, TS>(ap, cp, f, first, h, ip, mp, pendingFar, piv, prepared, rp);
    }
    template <auto T, auto TS, class PhaseArg0> __aicore__ inline void Finish64DataCopyPhase3(PhaseArg0 &tm) {

        DataCopy(tm, mat[192U * S], {static_cast<uint16_t>(T), T / 8U, (S - T) / 8U, (TS - T) / 8U});
        PipeBarrier<PIPE_V>();
        if constexpr (!EARLY_TAIL_INDEX) {
            LdetSharedOperations::PrepareTailRowIndices46<TS, 32>(*this);
            LDET_MASK_SET(T);
            for (uint32_t j = 0; j < 8U; ++j)
                vadds(reinterpret_cast<__ubuf__ int32_t *>(lanes[j * T].GetPhyAddr()),
                      reinterpret_cast<__ubuf__ int32_t *>(tab.GetPhyAddr()), static_cast<int32_t>(j * 4U), 1, 1, 1, 8,
                      8);
            LDET_MASK_RST();
        }
        PipeBarrier<PIPE_V>();
    }
    template <auto T, auto TS>
    __aicore__ inline void Finish64vgatherPhase4(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f, uint32_t &h,
                                                 __ubuf__ uint32_t *&ip, __ubuf__ float *&mp, bool &prepared) {
        return LdetSharedOperations::Finish64vgatherPhase4SharedPhase10<CyclicN256Inplace120, T, TS>(
            *this, ap, cp, f, h, ip, mp, prepared);
    }
    __aicore__ inline void Finish64() {
        constexpr uint32_t T = 64U, TS = 72U;
        static_assert((T + 8U) * TS <= P * (NMAX + GUARD_ROWS), "retired PANEL tail+guard fits");
        auto tm = LocalTensor<float>(TPosition::VECCALC, PANEL, (T + 8U) * TS);
        Finish64DataCopyPhase3<T, TS>(tm);
        auto views = LdetSharedOperations::GetTailRawViews51(*this, tm);
        bool prepared = false, pendingFar = false;
        for (uint32_t f = 0; f < T; ++f) {
            uint32_t h = T - f;
            Finish64vgatherPhase4<T, TS>(views.ap, views.cp, f, h, views.ip, views.mp, prepared);
            const uint32_t ri = word(views.tp[1]);
            if (ri >= h) {
                singular = 1;
                break;
            }
            const uint32_t p = f + ri;
            float piv = views.cp[ri];
            const uint32_t pivotExponent = NORMAL_INVERSE_GUARD ? ((word(piv) >> 23U) & 255U) : 0U;
            if (piv == 0.0f || (NORMAL_INVERSE_GUARD ? pivotExponent : ((word(piv) >> 23U) & 255U)) == 255U) {
                acc(piv);
                break;
            }
            float inv = -1.0f / piv;
            if ((!NORMAL_INVERSE_GUARD || pivotExponent == 0U) && ((word(inv) >> 23U) & 255U) == 255U) {
                acc(piv);
                singular = 2;
                break;
            }
            Finish64PipeBarrierPhase2<T, TS>(views.ap, views.cp, f, h, inv, views.ip, views.mp, p, pendingFar, piv,
                                             prepared, views.rp);
        }
        if constexpr (PANEL_ACC)
            normalizeProd();
        VS();
        meta.SetValue(0, singular ? 0.0f : sign * final());
        SM();
        DataCopyPad(out, meta, DataCopyExtParams{1, 4, 0, 0, 0});
        MS();
    }
