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
// Included in the CyclicN256PacketSafe26624 class; preserves its member scope and access.
    template <bool EXPORT = true> __aicore__ inline void factor(uint32_t kb, uint32_t q) {
        factorDataCopyPhase34(kb, q);
        auto factorViews = LdetSharedOperations::GetFactorRawViews60(*this);
        bool pivotPrepared = false;
        for (uint32_t f = 0; f < B; ++f) {
            uint32_t k = kb + f, h = n - k, be = k * P + f;
            factorvgatherPhase30<EXPORT>(factorViews.ap, be, factorViews.cp, f, h, factorViews.ip, k, kb, pivotPrepared,
                                         factorViews.pp);
            if constexpr (GUARDED || ClassPlan46::PIVOT8_AHEAD)
                WaitFlag<HardEvent::V_S>(2);
            else
                VS();
            float maxv = 0;
            uint32_t p = k;
            factorconstexprPhase33(h, k, maxv, p, factorViews.tp);
            float piv, inv;
            if (!LdetSharedOperations::PrepareSignedPivot51<ClassPlan46::DEFER_BOOKKEEP>(*this, maxv, factorViews.cp, k,
                                                                                         p, piv, inv))
                break;
            factorconstexprPhase35<EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, factorViews.pp, factorViews.rp);
        }
        factorconstexprPhase36<EXPORT>(kb);
    }
    template <uint32_t FIRST, uint32_t LAST>
    __aicore__ inline void ApplyPanelUpdatesvmulsPhase8(const uint32_t &begin, uint32_t &kb, const uint32_t &width) {

        auto mp = reinterpret_cast<__ubuf__ float *>(mat.GetPhyAddr());
        auto rp = reinterpret_cast<__ubuf__ float *>(row.GetPhyAddr());
        for (uint32_t f = FIRST; f < LAST; ++f) {
            uint32_t k = kb + f, h = n - k, p = loadPivot(f, kb);
            float inv = loadInverse(f, kb);
            LdetSharedOperations::SharedOwnerVmulsStage29<CyclicN256PacketSafe26624, ClassPlan46::NORMAL_COPY>(
                *this, begin, inv, k, mp, p, rp, width);
            if (h > 1)
                vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(brc.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(fac[rawOffset(f, kb)].GetPhyAddr()), 1, 8,
                      static_cast<uint8_t>((h + 7) / 8));
            PipeBarrier<PIPE_V>();
            LdetSharedOperations::RepairAndUpdateRows64<CyclicN256PacketSafe26624, ClassPlan46::FAC_REPAIRED>(
                *this, p, k, begin, h, width);
        }
        if constexpr (!ClassPlan46::PRUNE_STAGE_WAITS)
            VS();
    }
    template <uint32_t FIRST = 0U, uint32_t LAST = B>
    __aicore__ inline void ApplyPanelUpdates(uint32_t kb, uint32_t q) {
        uint32_t first = q < rank ? 0 : (q - rank) / G + 1;
        if constexpr (ClassPlan46::CONTIGUOUS_BANDS) {
            const uint32_t base = bandFirst(rank);
            first = q < base ? 0U : q - base + 1U;
        }
        const uint32_t begin = first * B, width = owned * B > begin ? owned * B - begin : 0;
        if (!width)
            return;
        if constexpr (ClassPlan46::PANEL_MASK) {
            auto mp = reinterpret_cast<__ubuf__ float *>(mat.GetPhyAddr()),
                 rp = reinterpret_cast<__ubuf__ float *>(row.GetPhyAddr());
            auto consumerFactors = LocalTensor<float>(TPosition::VECCALC, CONSUMER_COEFF, (NMAX + GUARD_ROWS) * S);
            auto bp = reinterpret_cast<__ubuf__ float *>(consumerFactors.GetPhyAddr());
            SetMaskNorm();
            AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
            for (uint32_t f = FIRST; f < LAST; ++f) {
                uint32_t k = kb + f, h = n - k, p = loadPivot(f, kb);
                float inv = loadInverse(f, kb);
                vmuls(rp, mp + p * S + begin, inv, 1, 1, 1, 8, 8);
                if (p != k)
                    vmuls(mp + p * S + begin, mp + k * S + begin, 1.0f, 1, 1, 1, 8, 8);
                if (h > 1U)
                    vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(bp + k * S),
                          reinterpret_cast<__ubuf__ uint32_t *>(fac[rawOffset(f, kb)].GetPhyAddr()), S / 8U, S,
                          static_cast<uint8_t>((h + 7U) / 8U));
                PipeBarrier<PIPE_V>();
                if (h > 1U)
                    vmla(mp + (k + 1U) * S + begin, rp, bp + (k + 1U) * S, static_cast<uint8_t>(h - 1U), 1, 1, 0,
                         S / 8U, 0, S / 8U);
                PipeBarrier<PIPE_V>();
            }
            ResetMask();
            SetMaskNorm();
            if constexpr (!ClassPlan46::PRUNE_STAGE_WAITS) {
                if (rank != q + 1U)
                    VS();
            }
            return;
        }
        ApplyPanelUpdatesvmulsPhase8<FIRST, LAST>(begin, kb, width);
    }

    template <uint32_t J> __aicore__ inline void consumePacket(uint32_t kb, uint32_t q, bool &aborted) {
        static_assert(J < 3U, "prefix packet");
        if (aborted)
            return;
        if (!waitPacket(q, 8U * (J + 1U))) {
            aborted = true;
            return;
        }
        receivePacket<J>(kb);
        // Packet authorization, NOT stale header singular.
        ApplyPanelUpdates<8U * J, 8U * (J + 1U)>(kb, q);
    }
    template <auto T, auto TS>
    __aicore__ inline void Finish64GeppvmlaPhase1(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f,
                                                  const uint32_t &first, uint32_t &h, __ubuf__ uint32_t *&ip,
                                                  __ubuf__ float *&mp, bool &pendingFar, float &piv, bool &prepared,
                                                  __ubuf__ float *&rp) {
        return LdetSharedOperations::Finish64GeppvmlaPhase1SharedPhase22<CyclicN256PacketSafe26624,
                                                                         ClassPlan46::PANEL_ACC, T, TS>(
            *this, ap, cp, f, first, h, ip, mp, pendingFar, piv, prepared, rp);
    }
    template <auto T, auto TS>
    __aicore__ inline void Finish64GeppPipeBarrierPhase2(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f,
                                                         uint32_t &h, float &inv, __ubuf__ uint32_t *&ip,
                                                         __ubuf__ float *&mp, const uint32_t &p, bool &pendingFar,
                                                         float &piv, bool &prepared, __ubuf__ float *&rp) {

        LdetSharedOperations::SharedOwner15PipeBarrierStage2<CyclicN256PacketSafe26624, T, TS>(*this, f, h, inv, mp, p,
                                                                                               pendingFar, rp);
        const uint32_t first = (f + 1U) & ~7U;
        prepared = false;
        Finish64GeppvmlaPhase1<T, TS>(ap, cp, f, first, h, ip, mp, pendingFar, piv, prepared, rp);
    }
    template <auto T, auto TS>
    __aicore__ inline void Finish64GeppvgatherPhase3(__ubuf__ float *&ap, __ubuf__ float *&cp, __ubuf__ uint32_t *&ip,
                                                     __ubuf__ float *&mp, bool &pendingFar, bool &prepared,
                                                     __ubuf__ float *&rp, __ubuf__ float *&tp) {

        for (uint32_t f = 0; f < T; ++f) {
            uint32_t h = T - f;
            LdetSharedOperations::SharedOwnerVgatherStage10<CyclicN256PacketSafe26624, T, TS>(*this, ap, cp, f, h, ip,
                                                                                              mp, prepared);
            uint32_t p = 0U;
            float piv = 0.0f, inv = 0.0f;
            if (!LdetSharedOperations::PrepareTailPivot60(*this, tp, cp, h, f, p, piv, inv))
                break;
            Finish64GeppPipeBarrierPhase2<T, TS>(ap, cp, f, h, inv, ip, mp, p, pendingFar, piv, prepared, rp);
        }
    }
    template <auto T, auto TS, class Capture0> __aicore__ inline void SharedDataCopyStage3(Capture0 &tm) {
        DataCopy(tm, mat[192U * S], {static_cast<uint16_t>(T), T / 8U, (S - T) / 8U, (TS - T) / 8U});
        LdetSharedOperations::SharedOwnerPipeBarrierStage25<CyclicN256PacketSafe26624, TS>(*this);
        LDET_MASK_SET(T);
        for (uint32_t j = 0; j < 8U; ++j)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(lanes[j * T].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(tab.GetPhyAddr()), static_cast<int32_t>(j * 4U), 1, 1, 1, 8, 8);
        LDET_MASK_RST();
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Finish64Gepp() {
        constexpr uint32_t T = 64U, TS = 72U;
        static_assert((T + 8U) * TS <= P * (NMAX + GUARD_ROWS), "retired PANEL tail+guard fits");
        auto tm = LocalTensor<float>(TPosition::VECCALC, PANEL, (T + 8U) * TS);
        SharedDataCopyStage3<T, TS>(tm);
        auto views = LdetSharedOperations::GetTailRawViews51(*this, tm);
        bool prepared = false, pendingFar = false;
        Finish64GeppvgatherPhase3<T, TS>(views.ap, views.cp, views.ip, views.mp, pendingFar, prepared, views.rp,
                                         views.tp);
        if constexpr (ClassPlan46::PANEL_ACC)
            normalizeProd();
        VS();
        meta.SetValue(0, singular ? 0.0f : sign * final());
        SM();
        DataCopyPad(out, meta, DataCopyExtParams{1, 4, 0, 0, 0});
        MS();
    }

    template <auto T, auto TS, class PhaseArg4, class PhaseArg14, class PhaseArg15, class PhaseArg16, class PhaseArg17>
    __aicore__ inline void Finish64Block2vgatherPhase1(const uint32_t &address0, const uint32_t &address1,
                                                       __ubuf__ float *&ap, __ubuf__ float *&b0, PhaseArg4 &b1,
                                                       __ubuf__ float *&cp, uint32_t &f, const uint32_t &h2,
                                                       __ubuf__ uint32_t *&ip, float &m00, float &m01, float &m10,
                                                       float &m11, __ubuf__ float *&mp, PhaseArg14 &r0, PhaseArg15 &r1,
                                                       PhaseArg16 &t0, PhaseArg17 &t1) {
        return LdetSharedOperations::FinishLocalTailBlock2Shared57vgatherPhase1<T, TS, PhaseArg4, PhaseArg14,
                                                                                PhaseArg15, PhaseArg16, PhaseArg17>(
            address0, address1, ap, b0, b1, cp, f, h2, ip, m00, m01, m10, m11, mp, r0, r1, t0, t1);
    }
    template <auto T, auto TS, class PhaseArg2, class PhaseArg6, class PhaseArg7, class PhaseArg8, class PhaseArg9>
    __aicore__ inline void Finish64Block2PipeBarrierPhase2(__ubuf__ float *&ap, __ubuf__ float *&b0, PhaseArg2 &b1,
                                                           __ubuf__ float *&cp, __ubuf__ uint32_t *&ip,
                                                           __ubuf__ float *&mp, PhaseArg6 &r0, PhaseArg7 &r1,
                                                           PhaseArg8 &t0, PhaseArg9 &t1) {

        VS();
        for (uint32_t f = 0U; f < T; f += 2U) {
            if (f != 0U)
                WaitFlag<HardEvent::V_S>(4);
            const uint32_t h2 = T - f - 2U;
            float a = mp[f * TS + f], b = mp[f * TS + f + 1U], c = mp[(f + 1U) * TS + f],
                  d = mp[(f + 1U) * TS + f + 1U];
            float ad = a * d, bc = b * c, delta = ad - bc;
            float da = delta < 0.0f ? -delta : delta, pa = ad < 0.0f ? -ad : ad, pb = bc < 0.0f ? -bc : bc;
            const uint32_t exponent = (word(delta) >> 23U) & 255U;
            if (exponent == 0U || exponent == 255U || !(da >= 0.25f * (pa + pb))) {
                singular = 3U;
                break;
            }
            float inverse = 1.0f / delta;
            float m00 = -d * inverse, m01 = b * inverse, m10 = c * inverse, m11 = -a * inverse;
            if (((word(m00) >> 23U) & 255U) == 255U || ((word(m01) >> 23U) & 255U) == 255U ||
                ((word(m10) >> 23U) & 255U) == 255U || ((word(m11) >> 23U) & 255U) == 255U) {
                singular = 3U;
                break;
            }
            if (!h2) {
                acc(delta);
                break;
            }
            const uint32_t address0 = (f + 2U) * TS + f, address1 = address0 + 1U;
            LDET_MASK_SET(h2);
            Finish64Block2vgatherPhase1<T, TS>(address0, address1, ap, b0, b1, cp, f, h2, ip, m00, m01, m10, m11, mp,
                                               r0, r1, t0, t1);
            ResetMask();
            SetMaskNorm();
            acc(delta);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Finish64Block2() {
        constexpr uint32_t T = 64U, TS = 72U;
        static_assert((T + 8U) * TS <= P * (NMAX + GUARD_ROWS), "retired PANEL tail+guard fits");
        auto tm = LocalTensor<float>(TPosition::VECCALC, PANEL, (T + 8U) * TS);
        SharedDataCopyStage3<T, TS>(tm);
        auto views = LdetSharedOperations::GetTailRawViews51(*this, tm);
        auto r0 = views.rp, r1 = views.tp, t0 = views.tp + 64U, t1 = views.tp + 128U;
        auto b0 = reinterpret_cast<__ubuf__ float *>(brc.GetPhyAddr()), b1 = b0 + T * 8U;
        Finish64Block2PipeBarrierPhase2<T, TS>(views.ap, b0, b1, views.cp, views.ip, views.mp, r0, r1, t0, t1);
        VS();
        meta.SetValue(0, singular ? 0.0f : sign * final());
        SM();
        DataCopyPad(out, meta, DataCopyExtParams{1, 4, 0, 0, 0});
        MS();
    }

    __aicore__ inline void Finish64() {
        Finish64Gepp();
    }
