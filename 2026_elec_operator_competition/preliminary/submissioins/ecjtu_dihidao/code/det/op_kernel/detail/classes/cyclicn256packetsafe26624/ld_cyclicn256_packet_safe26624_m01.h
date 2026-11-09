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
    __aicore__ inline void VS() {
        return LdetSharedOperations::VSShared0<CyclicN256PacketSafe26624>(*this);
    }
    __aicore__ inline void SV() {
        return LdetSharedOperations::SVShared1<CyclicN256PacketSafe26624>(*this);
    }
    __aicore__ inline void MV() {
        return LdetSharedOperations::MVShared2<CyclicN256PacketSafe26624>(*this);
    }
    __aicore__ inline void VM() {
        return LdetSharedOperations::VMShared3<CyclicN256PacketSafe26624>(*this);
    }
    __aicore__ inline void SM() {
        return LdetSharedOperations::SMShared4<CyclicN256PacketSafe26624>(*this);
    }
    __aicore__ inline void MS() {
        return LdetSharedOperations::MSShared5<CyclicN256PacketSafe26624>(*this);
    }
    __aicore__ inline uint32_t bandFirst(uint32_t r) {
        return LdetSharedOperations::bandFirstShared34<CyclicN256PacketSafe26624>(*this, r);
    }
    __aicore__ inline uint32_t word(float f) {
        return LdetSharedOperations::wordShared6<CyclicN256PacketSafe26624>(*this, f);
    }
    __aicore__ inline float value(uint32_t u) {
        return LdetSharedOperations::valueShared7<CyclicN256PacketSafe26624>(*this, u);
    }
    __aicore__ inline void normalizeProd() {
        return LdetSharedOperations::normalizeProdShared8<CyclicN256PacketSafe26624>(*this);
    }
    __aicore__ inline void acc(float piv) {
        return LdetSharedOperations::accShared9<CyclicN256PacketSafe26624>(*this, piv);
    }

    __aicore__ inline float final() {
        return LdetSharedOperations::finalShared10<CyclicN256PacketSafe26624>(*this);
    }
    __aicore__ inline void state() {
        return LdetSharedOperations::stateShared11<CyclicN256PacketSafe26624>(*this);
    }
    __aicore__ inline uint32_t facPitch(uint32_t kb) {
        return LdetSharedOperations::facPitchShared12<CyclicN256PacketSafe26624>(*this, kb);
    }
    __aicore__ inline GM_ADDR prefixTicket(uint32_t q) {
        return LdetSharedOperations::prefixTicketShared32<CyclicN256PacketSafe26624>(*this, q);
    }
    __aicore__ inline uint32_t packetWords(uint32_t kb) {
        return 8U * facPitch(kb) + 16U;
    }
    __aicore__ inline uint32_t packetOffset(uint32_t j, uint32_t kb) {
        return j * packetWords(kb);
    }
    __aicore__ inline uint32_t rawOffset(uint32_t f, uint32_t kb) {
        // Algebraically identical to packetBase + lane*p.
        return f * facPitch(kb) + 16U * (f >> 3U);
    }
    __aicore__ inline uint32_t indexOffset(uint32_t f, uint32_t kb) {
        return packetOffset(f >> 3U, kb) + 8U * facPitch(kb) + (f & 7U);
    }
    __aicore__ inline void storePivot(uint32_t f, uint32_t kb, uint32_t p, float inv) {
        const uint32_t a = indexOffset(f, kb);
        fu.SetValue(a, p);
        fu.SetValue(a + 8U, word(inv));
    }
    __aicore__ inline uint32_t loadPivot(uint32_t f, uint32_t kb) {
        return fu.GetValue(indexOffset(f, kb));
    }
    __aicore__ inline float loadInverse(uint32_t f, uint32_t kb) {
        return value(fu.GetValue(indexOffset(f, kb) + 8U));
    }
    __aicore__ inline void bindPayloadMeta(uint32_t kb) {
        meta = LocalTensor<float>(TPosition::VECCALC, ACTIVE_FAC + (32U * facPitch(kb) + 64U) * 4U, 8U);
        mu = meta.template ReinterpretCast<uint32_t>();
    }
    __aicore__ inline void importState() {
        sign = value(mu.GetValue(0));
        prod = value(mu.GetValue(1));
        ex = int32_t(mu.GetValue(2));
        singular = mu.GetValue(3);
    }
    __aicore__ inline void receive(uint32_t kb) {
        const uint32_t slot = ((kb / B) & 1U) * GM_WORDS;
        bindPayloadMeta(kb);
        DataCopy(fac, ws[slot], B * facPitch(kb) + META_WORDS);
        MV();
        VS();
        importState();
    }

    template <uint32_t J> __aicore__ inline void receivePacket(uint32_t kb) {
        static_assert(J < 3U, "intermediate packet");
        const uint32_t slot = ((kb / B) & 1U) * GM_WORDS;
        const uint32_t a = packetOffset(J, kb);
        DataCopy(fac[a], ws[slot + a], packetWords(kb));

        // Separate dependencies: Scalar metadata and Vector FAC.
        SetFlag<HardEvent::MTE2_S>(5);
        SetFlag<HardEvent::MTE2_V>(2U + J);
        WaitFlag<HardEvent::MTE2_S>(5);
        WaitFlag<HardEvent::MTE2_V>(2U + J);
    }
    __aicore__ inline void receiveFinalPacket(uint32_t kb) {
        const uint32_t slot = ((kb / B) & 1U) * GM_WORDS;
        const uint32_t a = packetOffset(3U, kb);
        bindPayloadMeta(kb);
        DataCopy(fac[a], ws[slot + a], packetWords(kb) + 8U);
        MV();
        VS(); // retain original full-header dependency
        importState();
    }

    // Existing prefixTicket(q) is retained verbatim.
    __aicore__ inline void sendImmutableTicket(uint32_t which, uint32_t kb) {
        if constexpr (ClassPlan46::THIN_H6) {
            GlobalTensor<uint32_t> readyGm;
            readyGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(prefixTicket(kb / B)), 8U);
            DataCopy(readyGm, prefixReady[which * 8U], 8U);
        }
    }
    __aicore__ inline void publishPacket(uint32_t j, uint32_t kb) {
        if constexpr (ClassPlan46::THIN_H6) {
            const uint32_t slot = ((kb / B) & 1U) * GM_WORDS;
            const uint32_t a = packetOffset(j, kb);
            SM(); // packet p/inv and immutable ticket stores
            VM(); // gather and patchFac writers
            DataCopy(ws[slot + a], fac[a], packetWords(kb));
            PipeBarrier<PIPE_MTE3>();
            sendImmutableTicket(j, kb);
            // Deliberately no MTE3_S set/wait here.
        }
    }
    __aicore__ inline void publishAbort(uint32_t kb) {
        if constexpr (ClassPlan46::THIN_H6) {
            SM();                     // immutable abort source initialized by S
            PipeBarrier<PIPE_MTE3>(); // after every preceding payload/ticket
            sendImmutableTicket(3U, kb);
        }
    }
    __aicore__ inline void publish(uint32_t kb) {
        const uint32_t slot = ((kb / B) & 1U) * GM_WORDS;
        const uint32_t a = packetOffset(3U, kb);
        bindPayloadMeta(kb);
        state();
        SM();
        VM();
        DataCopy(ws[slot + a], fac[a], packetWords(kb) + 8U);
        MS(); // original final retirement
    }

    __aicore__ inline bool waitPacket(uint32_t q, uint32_t required) {
        bool ready = false;
        do {
            LuFlushFlag(prefixTicket(q));
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            const uint32_t t = *reinterpret_cast<volatile __gm__ uint32_t *>(prefixTicket(q));
            if (t == 0x6A1F8000U)
                return false;
            ready = (t == 0x6A1F0008U || t == 0x6A1F0010U || t == 0x6A1F0018U) && (t & 0xffffU) >= required;
        } while (!ready);
        return true;
    }
    __aicore__ inline void repair(uint32_t pRel) {
        return LdetSharedOperations::repairShared15<CyclicN256PacketSafe26624>(*this, pRel);
    }
    __aicore__ inline void patchFac(uint32_t f, uint32_t kb, uint32_t pRel) {
        if (pRel) {
            SetVectorMask<float, MaskMode::NORMAL>(0ULL, 1ULL << (pRel & 7U));
            vmuls(reinterpret_cast<__ubuf__ float *>(fac[rawOffset(f, kb) + (pRel & ~7U)].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ float *>(brc.GetPhyAddr()), 1.0f, 1, 1, 1, 8, 8);
            ResetMask();
        }
    }
    __aicore__ inline void updateRows(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow, uint32_t h,
                                      uint32_t stride, uint32_t width) {
        return LdetSharedOperations::updateRowsShared17<CyclicN256PacketSafe26624>(*this, dst, pivotRow, h, stride,
                                                                                   width);
    }

    __aicore__ inline void updatePanelRows8x8(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow,
                                              uint32_t h, uint32_t width) {
        return LdetSharedOperations::updatePanelRows8x8Shared36<CyclicN256PacketSafe26624>(*this, dst, pivotRow, h,
                                                                                           width);
    }

    template <bool EXPORT = true> __aicore__ inline void issuePreparedPivot(uint32_t k, uint32_t f) {
        const uint32_t h = n - k, be = k * P + f;
        auto cp = reinterpret_cast<__ubuf__ float *>(col.GetPhyAddr());
        if constexpr (ClassPlan46::DIRECT_FAC && EXPORT)
            cp = reinterpret_cast<__ubuf__ float *>(fac[rawOffset(f, k - f)].GetPhyAddr());
        auto ap = reinterpret_cast<__ubuf__ float *>(ab.GetPhyAddr());
        auto pp = reinterpret_cast<__ubuf__ float *>(panel.GetPhyAddr());
        auto ip = reinterpret_cast<__ubuf__ uint32_t *>(lanes.GetPhyAddr());

        LDET_MASK_SET(h);
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(cp), ip + (f & 7U) * NMAX,
                reinterpret_cast<uint64_t>(pp) + (be & ~7U) * 4U, 8, 1);
        PipeBarrier<PIPE_V>();
        vabs(ap, cp, 1, 1, 1, 8, 8);
        LDET_MASK_RST();
        LDET_MASK_SET(h);
        if constexpr (GUARDED) {
            LdetVcmax128(tmp, ab, h);
        } else {
            const uint32_t head = h < 128U ? h : 128U;
            AscendCUtils::SetMask<float>(0, head);
            LdetVcmax128(tmp, ab, head);
            if (h > 128U) {
                const uint32_t tail = h - 128U;
                AscendCUtils::SetMask<float>(0, tail);
                LdetVcmax128(tmp[8], ab[128], tail);
            }
        }
        LDET_MASK_RST();
        SetFlag<HardEvent::V_S>(2);
    }
    template <bool EXPORT = true>
    __aicore__ inline bool updatePanelAndPrepare(uint32_t kb, uint32_t f, uint32_t h, uint32_t c0, uint32_t width) {
        return LdetSharedOperations::updatePanelAndPrepareShared37<CyclicN256PacketSafe26624, EXPORT>(*this, kb, f, h,
                                                                                                      c0, width);
    }

    template <bool EXPORT>
    __aicore__ inline void factorvmulsPhase29(uint32_t &f, uint32_t &h, float &inv, uint32_t &k, uint32_t &kb,
                                              uint32_t &p, float &piv, bool &pivotPrepared, __ubuf__ float *&pp,
                                              __ubuf__ float *&rp) {

        LdetSharedOperations::SharedOwnerVmulsStage11<CyclicN256PacketSafe26624, ClassPlan46::FAC_REPAIRED,
                                                      ClassPlan46::NORMAL_COPY, ClassPlan46::PIVOT8_AHEAD, EXPORT>(
            *this, f, h, inv, k, kb, p, pivotPrepared, pp, rp);
        if constexpr (ClassPlan46::DEFER_BOOKKEEP) {
            acc(piv);
            if constexpr (EXPORT) {
                storePivot(f, kb, p, inv);
            }
        }
        PipeBarrier<PIPE_V>();
    }
    template <bool EXPORT>
    __aicore__ inline void factorvgatherPhase30(__ubuf__ float *&ap, uint32_t &be, __ubuf__ float *&cp, uint32_t &f,
                                                uint32_t &h, __ubuf__ uint32_t *&ip, uint32_t &k, uint32_t &kb,
                                                bool &pivotPrepared, __ubuf__ float *&pp) {

        if constexpr (ClassPlan46::DIRECT_FAC && EXPORT)
            cp = reinterpret_cast<__ubuf__ float *>(fac[rawOffset(f, kb)].GetPhyAddr());
        if constexpr (ClassPlan46::PIVOT8_AHEAD) {
            if (!pivotPrepared)
                issuePreparedPivot<EXPORT>(k, f);
        } else {
            LdetSharedOperations::SharedOwnerVgatherStage16<CyclicN256PacketSafe26624>(*this, ap, be, cp, f, h, ip, pp);
        }
        if constexpr (GUARDED && !ClassPlan46::PIVOT8_AHEAD)
            SetFlag<HardEvent::V_S>(2);
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(brc.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t *>(cp), 1, 8,
              static_cast<uint8_t>((h + 7) / 8));
        if constexpr (EXPORT && !ClassPlan46::DIRECT_FAC)
            DataCopy(fac[rawOffset(f, kb)], col, ((h + 7U) / 8U) * 8U);
    }

    __aicore__ inline void factorGetPhyAddrPhase31(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p) {

        auto records = reinterpret_cast<__ubuf__ uint64_t *>(tmp.GetPhyAddr());
        const uint64_t r0 = records[0], r1 = records[1];
        const uint64_t r2 = h > 128U ? records[2] : 0ULL;
        const uint64_t r3 = h > 192U ? records[3] : 0ULL;
        uint64_t selected = 0ULL;
        uint32_t segment = 0U;
        float v = value(static_cast<uint32_t>(r0));
        if (v > maxv) {
            maxv = v;
            selected = r0;
        }
        v = value(static_cast<uint32_t>(r1));
        if (v > maxv) {
            maxv = v;
            selected = r1;
            segment = 64U;
        }
        if (h > 128U) {
            v = value(static_cast<uint32_t>(r2));
            if (v > maxv) {
                maxv = v;
                selected = r2;
                segment = 128U;
            }
        }
        if (h > 192U) {
            v = value(static_cast<uint32_t>(r3));
            if (v > maxv) {
                maxv = v;
                selected = r3;
                segment = 192U;
            }
        }
        p = k + segment + static_cast<uint32_t>(selected >> 32U);
    }

    __aicore__ inline void factorwordPhase32(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p, __ubuf__ float *&tp) {
        return LdetSharedOperations::factorwordPhase24SharedPhase16<CyclicN256PacketSafe26624>(*this, h, k, maxv, p,
                                                                                               tp);
    }

    __aicore__ inline void factorconstexprPhase33(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                  __ubuf__ float *&tp) {

        if constexpr (GUARDED && ClassPlan46::HETERO6) {
            factorGetPhyAddrPhase31(h, k, maxv, p);
        } else if constexpr (GUARDED) {
            factorwordPhase32(h, k, maxv, p, tp);
        } else {
            LdetSharedOperations::SharedOwnerScalarUpdateStage24<CyclicN256PacketSafe26624>(*this, h, k, maxv, p, tp);
        }
    }

    __aicore__ inline void factorDataCopyPhase34(uint32_t &kb, uint32_t &q) {
        return LdetSharedOperations::factorDataCopyPhase18SharedPhase11<
            CyclicN256PacketSafe26624, ClassPlan46::ACTIVE_PANEL_ROWS, ClassPlan46::CONTIGUOUS_BANDS,
            ClassPlan46::GROUP, ClassPlan46::HETERO6, ClassPlan46::PRUNE_STAGE_WAITS, ClassPlan46::THIN_H6>(*this, kb,
                                                                                                            q);
    }
    template <bool EXPORT>
    __aicore__ inline void factorconstexprPhase35(uint32_t &f, uint32_t &h, float &inv, uint32_t &k, uint32_t &kb,
                                                  uint32_t &p, float &piv, bool &pivotPrepared, __ubuf__ float *&pp,
                                                  __ubuf__ float *&rp) {

        if (p != k)
            sign = -sign;
        if constexpr (EXPORT && !ClassPlan46::DEFER_BOOKKEEP) {
            storePivot(f, kb, p, inv);
        }
        factorvmulsPhase29<EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, pp, rp);
        if constexpr (PREFIX_STREAM && EXPORT) {
            if (!singular && f < 24U && (f & 7U) == 7U)
                publishPacket(f >> 3U, kb);
        }
    }
    template <bool EXPORT> __aicore__ inline void factorconstexprPhase36(uint32_t &kb) {

        if constexpr (ClassPlan46::PANEL_ACC)
            normalizeProd();
        VS();
        if constexpr (PREFIX_STREAM && EXPORT) {
            if (singular)
                publishAbort(kb);
        }
    }
