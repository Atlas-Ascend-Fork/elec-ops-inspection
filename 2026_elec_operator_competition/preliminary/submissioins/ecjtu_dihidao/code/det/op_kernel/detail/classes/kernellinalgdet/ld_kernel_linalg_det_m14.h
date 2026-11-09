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
    template <uint32_t TN> __aicore__ inline void ProcessTinyFixed() {
        constexpr uint32_t nn = TN * TN;
        constexpr uint32_t capacity = (::kSmallMaxN * ::kSmallMaxN) / nn;
        uint32_t processed = 0U;
        while (processed < cnt_) {
            uint32_t chunk = cnt_ - processed;
            if (chunk > capacity)
                chunk = capacity;
            if (processed != 0U) {
                SetFlag<HardEvent::S_MTE2>(::kEvSMte2);
                WaitFlag<HardEvent::S_MTE2>(::kEvSMte2);
            }
            uint32_t total = chunk * nn;
            for (uint32_t off = 0; off < total; off += 4096U) {
                uint32_t length = total - off;
                if (length > 4096U)
                    length = 4096U;
                DataCopyPad(mat_[off], aGm_[static_cast<uint64_t>(start_ + processed) * nn + off],
                            {1, static_cast<uint32_t>(length * sizeof(float)), 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
            SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
            WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
            for (uint32_t m = 0U; m < chunk; ++m) {
                res_.SetValue(processed + m, TinyLUReg<TN, true>(m * nn));
            }
            processed += chunk;
        }
    }
    __aicore__ inline void SmallMaskSet(uint32_t count) {
        if constexpr (LD_BATCH == 3U || LD_BATCH == 6U || (LD_BATCH >= 9U && LD_BATCH <= 14U)) {
            AscendCUtils::SetMask<float>(static_cast<int32_t>(count));
        } else {
            LDET_MASK_SET(count);
        }
    }
    __aicore__ inline void SmallMaskReset() {
        if constexpr (LD_BATCH != 3U && LD_BATCH != 6U && (LD_BATCH < 9U || LD_BATCH == 15U)) {
            LDET_MASK_RST();
        }
    }
    __aicore__ inline float TinyAbs(float value) {
        return value < 0.0f ? -value : value;
    }
    __aicore__ inline float TinyMadd(float a, float b, float c) {
        return a * b + c;
    }
    template <uint32_t TN> __aicore__ inline float TinyDetGuarded(uint32_t base) {
        float a[TN * TN];
        uint32_t unsafe = 0U;
#pragma unroll
        for (uint32_t i = 0; i < TN * TN; ++i) {
            float value = mat_.GetValue(base + i);
            a[i] = value;
            uint32_t magnitude = *reinterpret_cast<uint32_t *>(&value) & 0x7fffffffU;
            unsafe |= magnitude > 0x477fffffU;
        }
        if (unsafe != 0U)
            return TinyLUReg<TN>(base);
        float determinant;
        float magnitudeSum;
        if constexpr (TN == 2U) {
            float q = a[1] * a[2];
            determinant = TinyMadd(a[0], a[3], -q);
            magnitudeSum = TinyAbs(a[0] * a[3]) + TinyAbs(q);
        } else {
            float c0 = TinyMadd(a[4], a[8], -(a[5] * a[7]));
            float c1 = TinyMadd(a[3], a[8], -(a[5] * a[6]));
            float c2 = TinyMadd(a[3], a[7], -(a[4] * a[6]));
            determinant = TinyMadd(a[0], c0, TinyMadd(-a[1], c1, a[2] * c2));
            magnitudeSum = TinyAbs(a[0] * (a[4] * a[8])) + TinyAbs(a[0] * (a[5] * a[7])) +
                           TinyAbs(a[1] * (a[3] * a[8])) + TinyAbs(a[1] * (a[5] * a[6])) +
                           TinyAbs(a[2] * (a[3] * a[7])) + TinyAbs(a[2] * (a[4] * a[6]));
        }
        if (magnitudeSum >= 1.0e-18f && TinyAbs(determinant) >= 0.02f * magnitudeSum)
            return determinant;
        return TinyLUReg<TN>(base);
    }
    __aicore__ inline void ProcessTinyDirect() {
        uint32_t nn = n_ * n_;
        uint32_t total = cnt_ * nn;
        if (total > ::kSmallMaxN * ::kSmallMaxN) {
            ProcessSmall();
            return;
        }
        for (uint32_t off = 0; off < total; off += 4096U) {
            uint32_t length = total - off;
            if (length > 4096U)
                length = 4096U;
            DataCopyPad(mat_[off], aGm_[static_cast<uint64_t>(start_) * nn_ + off],
                        {1, static_cast<uint32_t>(length * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
        WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
        for (uint32_t m = 0; m < cnt_; ++m) {
            float determinant = n_ == 2U ? TinyDetGuarded<2>(m * nn) : TinyDetGuarded<3>(m * nn);
            res_.SetValue(m, determinant);
        }
    }
    __aicore__ inline void CopyIn2D(const LocalTensor<float> &dst, uint64_t srcGmElem, uint32_t rows, uint32_t colsElem,
                                    uint32_t srcStrideElem, uint32_t dstStrideElem) {
        for (uint32_t r0 = 0; r0 < rows; r0 += 32) {
            uint32_t rc = rows - r0;
            if (rc > 32U) {
                rc = 32U;
            }
            DataCopyExtParams cp{static_cast<uint16_t>(rc), colsElem * static_cast<uint32_t>(sizeof(float)),
                                 (srcStrideElem - colsElem) * static_cast<uint32_t>(sizeof(float)),
                                 (dstStrideElem - colsElem) / ::kAlign, 0};
            DataCopyPad(dst[r0 * dstStrideElem], wsGm_[srcGmElem + static_cast<uint64_t>(r0) * srcStrideElem], cp,
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
    }
    __aicore__ inline void CopyOut2D(uint64_t dstGmElem, const LocalTensor<float> &src, uint32_t rows,
                                     uint32_t colsElem, uint32_t srcStrideElem, uint32_t dstStrideElem) {
        for (uint32_t r0 = 0; r0 < rows; r0 += 32) {
            uint32_t rc = rows - r0;
            if (rc > 32U) {
                rc = 32U;
            }
            DataCopyExtParams cp{static_cast<uint16_t>(rc), colsElem * static_cast<uint32_t>(sizeof(float)),
                                 (srcStrideElem - colsElem) / ::kAlign,
                                 (dstStrideElem - colsElem) * static_cast<uint32_t>(sizeof(float)), 0};
            DataCopyPad(wsGm_[dstGmElem + static_cast<uint64_t>(r0) * dstStrideElem], src[r0 * srcStrideElem], cp);
        }
    }
    __aicore__ inline void WsInit(uint64_t g) {
        uint64_t srcBase = g * nn_;
        uint64_t dstBase = g * nnS_;
        for (uint32_t r0 = 0; r0 < n_; r0 += 2 * ::kTileRows) {
            uint32_t rows = n_ - r0;
            if (rows > 2 * ::kTileRows) {
                rows = 2 * ::kTileRows;
            }
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopyPad(tile_[i * ::kUbW], aGm_[srcBase + static_cast<uint64_t>(r0 + i) * n_],
                            {1, static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
            SetFlag<HardEvent::MTE2_MTE3>(::kEvMte2Mte3);
            WaitFlag<HardEvent::MTE2_MTE3>(::kEvMte2Mte3);
            CopyOut2D(dstBase + static_cast<uint64_t>(r0) * s_, tile_, rows, s_, ::kUbW, s_);
            SetFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
            WaitFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
        }
    }
    __aicore__ inline void WsInitRange(uint64_t g, uint32_t rBegin, uint32_t rEnd, uint64_t dstBase) {
        uint64_t srcBase = g * nn_;
        for (uint32_t r0 = rBegin; r0 < rEnd; r0 += 4) {
            uint32_t rows = rEnd - r0;
            if (rows > 4) {
                rows = 4;
            }
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopyPad(cpbuf_[i * s_], aGm_[srcBase + static_cast<uint64_t>(r0 + i) * n_],
                            {1, static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
            SetFlag<HardEvent::MTE2_MTE3>(::kEvMte2Mte3);
            WaitFlag<HardEvent::MTE2_MTE3>(::kEvMte2Mte3);
            CopyOut2D(dstBase + static_cast<uint64_t>(r0) * s_, cpbuf_, rows, s_, s_, s_);
            SetFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
            WaitFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
        }
    }
    template <auto kCmStride, auto kPanel>
    __aicore__ inline void
    ProcessLargeParconstexprPhase1(uint32_t &cur, uint32_t &i, float &l0, float &l1, float &l10, float &l11, float &l12,
                                   float &l13, float &l14, float &l15, float &l16, float &l17, float &l18, float &l19,
                                   float &l2, float &l20, float &l21, float &l22, float &l23, float &l3, float &l4,
                                   float &l5, float &l6, float &l7, float &l8, float &l9, uint32_t &lb, uint32_t &w) {

        if constexpr (kPanel > 23U)
            l23 = lbuf_.GetValue(23 * kCmStride + lb);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[0 * ::kUbW], -l0, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[1 * ::kUbW], -l1, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[2 * ::kUbW], -l2, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[3 * ::kUbW], -l3, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[4 * ::kUbW], -l4, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[5 * ::kUbW], -l5, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[6 * ::kUbW], -l6, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[7 * ::kUbW], -l7, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[8 * ::kUbW], -l8, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[9 * ::kUbW], -l9, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[10 * ::kUbW], -l10, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[11 * ::kUbW], -l11, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[12 * ::kUbW], -l12, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[13 * ::kUbW], -l13, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[14 * ::kUbW], -l14, w);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[15 * ::kUbW], -l15, w);
        if constexpr (kPanel > 16U)
            LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[16 * ::kUbW], -l16, w);
        if constexpr (kPanel > 17U)
            LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[17 * ::kUbW], -l17, w);
        if constexpr (kPanel > 18U)
            LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[18 * ::kUbW], -l18, w);
        if constexpr (kPanel > 19U)
            LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[19 * ::kUbW], -l19, w);
        if constexpr (kPanel > 20U)
            LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[20 * ::kUbW], -l20, w);
        if constexpr (kPanel > 21U)
            LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[21 * ::kUbW], -l21, w);
        if constexpr (kPanel > 22U)
            LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[22 * ::kUbW], -l22, w);
        if constexpr (kPanel > 23U)
            LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[23 * ::kUbW], -l23, w);
    }
    template <bool LAZY, auto kCmStride>
    __aicore__ inline void ProcessLargeParSetFlagPhase2(uint32_t &bw, uint32_t &kb, uint32_t &l21, uint64_t &srcReg,
                                                        uint32_t &tw, uint32_t &w) {

        if (l21 != 0U) {
            LoadPivotedUpper60<::kUbW>(bw, kb, srcReg, w);
            SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
            WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
            LDET_MASK_SET(w);
            for (uint32_t t = 1; t < bw; ++t) {
                for (uint32_t s2 = 0; s2 < t; ++s2) {
                    float l = lbuf_.GetValue(s2 * kCmStride + t);
                    LDET_AXPY_TRK(ubuf_[t * ::kUbW], ubuf_[s2 * ::kUbW], -l, w);
                }
            }
            LDET_MASK_RST();
        }
        if constexpr (CUBE_LU && LAZY) {
            if (tw > 128U) {
                if (rank_ == 0U) {
                    LDET_VFENCE();
                    SetFlag<HardEvent::V_MTE3>(::kEvVMte3);
                    WaitFlag<HardEvent::V_MTE3>(::kEvVMte3);
                    DataCopyPad(this->cubeU_, ubuf_,
                                DataCopyExtParams{16, w * 4U, (::kUbW - w) / 8U, (512U - w) * 4U, 0});
                }
            }
        }
    }
    template <bool LAZY, auto kCmStride> __aicore__ inline void SharedSetFlagStage25(uint32_t &bw, uint32_t &h) {
        if constexpr (LAZY) {
            uint32_t hp = (h + 7U) & ~7U;
            SetFlag<HardEvent::S_V>(::kEvSV);
            WaitFlag<HardEvent::S_V>(::kEvSV);
            DataCopy(coloffU_, posU_, hp);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < bw; ++j) {
                Gather(panel_[j * hp], panelCM_[j * kCmStride], coloffU_, 0U, h);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < bw; ++j) {
                DataCopy(panelCM_[j * kCmStride], panel_[j * hp], hp);
            }
            SetFlag<HardEvent::V_S>(::kEvVS);
            WaitFlag<HardEvent::V_S>(::kEvVS);
        }
    }
    template <bool LAZY, auto evS2M3, auto kCmStride>
    __aicore__ inline void SharedSetFlagStage26(uint64_t &base, uint32_t &bw, uint32_t &h, uint32_t &l21nz) {
        LDET_VFENCE();
        SetFlag<HardEvent::V_MTE3>(::kEvVMte3);
        WaitFlag<HardEvent::V_MTE3>(::kEvVMte3);
        SetFlag<HardEvent::S_MTE3>(evS2M3);
        WaitFlag<HardEvent::S_MTE3>(evS2M3);
        if (l21nz != 0U) {
            if constexpr (FAST_MC && LAZY) {
                uint32_t hp = (h + 7U) & ~7U;
                CopyOut2D(base + 2U * nnS_, panelCM_, bw, hp, kCmStride, kCmStride);
            } else {
                for (uint32_t j = 0; j < bw; ++j) {
                    DataCopyPad(wsGm_[base + 2U * nnS_ + j * kCmStride], panelCM_[j * kCmStride], {1, h * 4, 0, 0, 0});
                }
            }
        }
    }
    template <bool LAZY, auto evS2M3, auto kCmStride>
    __aicore__ inline void ProcessLargeParSetFlagPhase3(uint64_t &base, uint32_t &bw, uint32_t &h, uint32_t &l21nz,
                                                        uint32_t &singular) {

        SharedSetFlagStage25<LAZY, kCmStride>(bw, h);
        metaU_.SetValue(0, singular);
        metaU_.SetValue(25, l21nz);
        SharedSetFlagStage26<LAZY, evS2M3, kCmStride>(base, bw, h, l21nz);
    }
    template <auto kCmStride, class PhaseArg0>
    __aicore__ inline void ProcessLargeParPipeBarrierPhase4(PhaseArg0 &alive, uint32_t &bw, LocalTensor<float> &factors,
                                                            uint32_t &h, uint32_t &nCh, uint32_t &t) {

        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < h; row += 64U) {
            uint32_t width = h - row;
            if (width > 64U)
                width = 64U;
            AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
            LdetPanelMla(panelCM_[(t + 1U) * kCmStride + row], factors[(t + 1U) * 8U], panelCM_[t * kCmStride + row],
                         1U, static_cast<uint8_t>(kCmStride / 8U));
        }
        LDET_MASK_RST();
        LDET_MASK_SET(h);
        LdetPanelAbs(absbuf_, panelCM_[(t + 1U) * kCmStride]);
        LdetPanelMul(absbuf_, absbuf_, alive);
        for (uint32_t c = 0; c < nCh; ++c) {
            uint32_t cc = h - c * 128U;
            if (cc > 128U)
                cc = 128U;
            AscendCUtils::SetMask<float>(0, cc);
            LdetVcmax128(tmp_[c * 8], absbuf_[c * 128], cc);
        }
        LDET_MASK_RST();
        SetFlag<HardEvent::V_S>(::kEvVMte3);
        if (t + 2U < bw) {
            for (uint32_t row = 0; row < h; row += 64U) {
                uint32_t width = h - row;
                if (width > 64U)
                    width = 64U;
                AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
                LdetPanelMla(panelCM_[(t + 2U) * kCmStride + row], factors[(t + 2U) * 8U],
                             panelCM_[t * kCmStride + row], static_cast<uint8_t>(bw - t - 2U),
                             static_cast<uint8_t>(kCmStride / 8U));
            }
            LDET_MASK_RST();
        }
    }
    template <auto kCmStride, auto kPanel>
    __aicore__ inline void ProcessLargeParGetValuePhase5(uint32_t &cur, uint32_t &i, float &l0, float &l1, float &l2,
                                                         float &l3, float &l4, float &l5, float &l6, uint32_t &lb,
                                                         uint32_t &w) {

        float l7 = lbuf_.GetValue(7 * kCmStride + lb);
        float l8 = lbuf_.GetValue(8 * kCmStride + lb);
        float l9 = lbuf_.GetValue(9 * kCmStride + lb);
        float l10 = lbuf_.GetValue(10 * kCmStride + lb);
        float l11 = lbuf_.GetValue(11 * kCmStride + lb);
        float l12 = lbuf_.GetValue(12 * kCmStride + lb);
        float l13 = lbuf_.GetValue(13 * kCmStride + lb);
        float l14 = lbuf_.GetValue(14 * kCmStride + lb);
        float l15 = lbuf_.GetValue(15 * kCmStride + lb);
        float l16 = 0.0f;
        if constexpr (kPanel > 16U)
            l16 = lbuf_.GetValue(16 * kCmStride + lb);
        float l17 = 0.0f;
        if constexpr (kPanel > 17U)
            l17 = lbuf_.GetValue(17 * kCmStride + lb);
        float l18 = 0.0f;
        if constexpr (kPanel > 18U)
            l18 = lbuf_.GetValue(18 * kCmStride + lb);
        float l19 = 0.0f;
        if constexpr (kPanel > 19U)
            l19 = lbuf_.GetValue(19 * kCmStride + lb);
        float l20 = 0.0f;
        if constexpr (kPanel > 20U)
            l20 = lbuf_.GetValue(20 * kCmStride + lb);
        float l21 = 0.0f;
        if constexpr (kPanel > 21U)
            l21 = lbuf_.GetValue(21 * kCmStride + lb);
        float l22 = 0.0f;
        if constexpr (kPanel > 22U)
            l22 = lbuf_.GetValue(22 * kCmStride + lb);
        float l23 = 0.0f;
        ProcessLargeParconstexprPhase1<kCmStride, kPanel>(cur, i, l0, l1, l10, l11, l12, l13, l14, l15, l16, l17, l18,
                                                          l19, l2, l20, l21, l22, l23, l3, l4, l5, l6, l7, l8, l9, lb,
                                                          w);
    }
