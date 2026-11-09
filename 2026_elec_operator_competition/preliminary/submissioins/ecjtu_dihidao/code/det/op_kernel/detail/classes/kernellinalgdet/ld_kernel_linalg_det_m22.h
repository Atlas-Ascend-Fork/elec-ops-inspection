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
    template <bool LAZY> __aicore__ inline PanelStepGeometry61 GetPanelStepGeometry61(uint32_t t, uint32_t h) {
        uint32_t tA = LAZY ? 0U : t & ~(::kAlign - 1);
        uint32_t j0 = LAZY ? 0U : t - tA;
        uint32_t cnt = h - tA;
        uint32_t segN = (cnt + 63U) / 64U;
        uint32_t nCh = (cnt + 127U) >> 7;
        return {tA, j0, cnt, segN, nCh};
    }

    template <bool RAW>
    __aicore__ inline bool FindScalarTinyPivot64(__ubuf__ float *matrix, uint32_t limit, uint32_t pitch, uint32_t k,
                                                 float &maxv, uint32_t &p, uint32_t &singular) {
        for (uint32_t i = k; i < limit; ++i) {
            float v = ReadScalarMatrix56<RAW>(matrix, i * pitch + k);
            float av = v < 0.0f ? -v : v;
            if (av > maxv) {
                maxv = av;
                p = i;
            }
        }
        if (maxv == 0.0f) {
            singular = 1;
            return false;
        }
        return true;
    }

    template <bool RAW> __aicore__ inline float ReadReductionLane64(__ubuf__ float *values, uint32_t offset) {
        if constexpr (RAW)
            return values[offset];
        else
            return tmp_.GetValue(offset);
    }
    template <bool RAW>
    __aicore__ inline void MergeMaxSegments64(__ubuf__ float *values, uint32_t segN, uint32_t k, float &maxv,
                                              uint32_t &p) {
        for (uint32_t s = 0; s < segN; ++s) {
            uint32_t so = LDET_RM_PAIR_OFF(s);
            float v = ReadReductionLane64<RAW>(values, so);
            if (v > maxv) {
                float idxF = ReadReductionLane64<RAW>(values, so + 1);
                maxv = v;
                p = k + s * 64U + *reinterpret_cast<uint32_t *>(&idxF);
            }
        }
    }

    __aicore__ inline void WaitSmallInputReuse64(uint32_t m, bool tiny, uint32_t &pending) {
        if (tiny) {
            if (m != 0U)
                WaitFlag<HardEvent::S_MTE2>(::kEvSMte2);
        } else if (pending != 0U) {
            WaitFlag<HardEvent::V_MTE2>(::kEvVMte2);
            pending = 0;
        }
    }
    __aicore__ inline void StoreSmallResult64(uint32_t m, bool tiny, uint32_t &pending, uint32_t singular, float sign,
                                              float prod, int32_t ex) {
        if (m + 1 < cnt_) {
            if (tiny)
                SetFlag<HardEvent::S_MTE2>(::kEvSMte2);
            else {
                SetFlag<HardEvent::V_MTE2>(::kEvVMte2);
                pending = 1;
            }
        }
        float detVal = singular != 0U ? 0.0f : sign * ProdFinal(prod, ex);
        res_.SetValue(m, detVal);
    }
    template <auto EARLY, auto NORMAL>
    __aicore__ inline void CompleteEarlyProduct64(bool &fastProd, float &prod, int32_t &ex, float piv) {
        SmallMaskReset();
        if constexpr (EARLY) {
            if constexpr (NORMAL)
                ProdSmallCommon(fastProd, prod, ex, piv);
            else
                ProdAccum(prod, ex, piv);
        }
    }
    template <bool LAZY> __aicore__ inline bool RejectZeroPanelPivot64(float maxv, bool &pending, uint32_t &singular) {
        if (maxv != 0.0f)
            return false;
        if constexpr (LAZY) {
            if (pending) {
                WaitFlag<HardEvent::V_S>(::kEvVS);
                pending = false;
            }
        }
        singular = 1;
        return true;
    }
    __aicore__ inline void ReduceColumnChunks64(uint32_t nCh, uint32_t cnt) {
        for (uint32_t c = 0; c < nCh; ++c) {
            uint32_t cc = cnt - c * 128U;
            if (cc > 128U)
                cc = 128U;
            AscendCUtils::SetMask<float>(0, cc);
            LdetVcmax128(tmp_[c * 8], absbuf_[c * 128], cc);
        }
        LDET_MASK_RST();
    }
    __aicore__ inline void PrefetchFollowingSlot64(LocalTensor<float> &slots, uint32_t m, uint32_t slotElems) {
        if (m + 1U < cnt_) {
            CopyPrefetchSlot(slots[((m + 1U) & 1U) * slotElems], start_ + m + 1U);
            SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
            SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        }
    }
