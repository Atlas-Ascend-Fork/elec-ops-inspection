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
    template <uint32_t TN> __aicore__ inline void LoadTinyBatch60() {
        DataCopyPad(mat_, aGm_[static_cast<uint64_t>(start_) * TN * TN],
                    {static_cast<uint16_t>(cnt_ * TN), TN * 4U, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
    }

    template <auto EVENT, auto CM, auto PANEL, auto BYTES>
    __aicore__ inline void PublishPanelMetadata60(uint64_t base, uint64_t barBase, uint32_t panelIdx) {
        DataCopyPad(wsGmU_[base + 2U * nnS_ + CM * PANEL], metaU_, {1, BYTES, 0, 0, 0});
        SetFlag<HardEvent::MTE3_S>(::kEvVMte2);
        WaitFlag<HardEvent::MTE3_S>(::kEvVMte2);
        barU_.SetValue(0, ::TagLo(panelIdx));
        barU_.SetValue(1, ::TagHi(panelIdx));
        SetFlag<HardEvent::S_MTE3>(EVENT);
        WaitFlag<HardEvent::S_MTE3>(EVENT);
        DataCopyPad(wsGmU_[barBase], barU_, {1, 32, 0, 0, 0});
    }
    __aicore__ inline uint32_t PanelPositionChanged60(uint32_t bw) {
        for (uint32_t t = 0; t < bw; ++t)
            if (metaU_.GetValue(1 + t) != t)
                return 1U;
        return 0U;
    }

    template <auto STRIDE>
    __aicore__ inline void LoadPivotedUpper60(uint32_t bw, uint32_t kb, uint64_t srcReg, uint32_t w) {
        if (rank_ == 0U) {
            LDET_VFENCE();
            SetFlag<HardEvent::V_MTE2>(::kEvVMte2);
            WaitFlag<HardEvent::V_MTE2>(::kEvVMte2);
        }
        for (uint32_t t = 0; t < bw; ++t) {
            uint64_t rs = srcReg + static_cast<uint64_t>(kb + posU_.GetValue(t)) * s_ + kb + bw;
            DataCopyPad(ubuf_[t * STRIDE], wsGm_[rs], {1, w * 4, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
    }

    template <bool NEGATE>
    __aicore__ inline void UpdateRowsAxpy61(uint32_t limit, uint32_t pitch, uint32_t &i, __ubuf__ float *&dstP,
                                            __ubuf__ float *&rkP, __ubuf__ float *&lP) {
        for (; i + 7U < limit; i += 8U) {
            float l0 = lP[0];
            float l1 = lP[1];
            float l2 = lP[2];
            float l3 = lP[3];
            float l4 = lP[4];
            float l5 = lP[5];
            float l6 = lP[6];
            float l7 = lP[7];
            vaxpy(dstP, rkP, (NEGATE ? -(l0) : (l0)), 1, 1, 1, 8, 8);
            vaxpy(dstP + pitch, rkP, (NEGATE ? -(l1) : (l1)), 1, 1, 1, 8, 8);
            vaxpy(dstP + 2U * pitch, rkP, (NEGATE ? -(l2) : (l2)), 1, 1, 1, 8, 8);
            vaxpy(dstP + 3U * pitch, rkP, (NEGATE ? -(l3) : (l3)), 1, 1, 1, 8, 8);
            vaxpy(dstP + 4U * pitch, rkP, (NEGATE ? -(l4) : (l4)), 1, 1, 1, 8, 8);
            vaxpy(dstP + 5U * pitch, rkP, (NEGATE ? -(l5) : (l5)), 1, 1, 1, 8, 8);
            vaxpy(dstP + 6U * pitch, rkP, (NEGATE ? -(l6) : (l6)), 1, 1, 1, 8, 8);
            vaxpy(dstP + 7U * pitch, rkP, (NEGATE ? -(l7) : (l7)), 1, 1, 1, 8, 8);
            dstP += 8U * pitch;
            lP += 8U;
        }
        for (; i < limit; ++i) {
            float lv = *lP;
            vaxpy(dstP, rkP, (NEGATE ? -(lv) : (lv)), 1, 1, 1, 8, 8);
            dstP += pitch;
            lP += 1U;
        }
    }
