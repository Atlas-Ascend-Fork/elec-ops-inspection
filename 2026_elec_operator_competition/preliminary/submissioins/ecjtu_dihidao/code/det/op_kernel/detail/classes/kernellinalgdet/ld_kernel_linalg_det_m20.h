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
    __aicore__ inline SmallRawViews51 GetSmallRawViews51() {
        return {reinterpret_cast<__ubuf__ float *>(colbuf_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(tabRU_.GetPhyAddr())};
    }

    __aicore__ inline void StartPrefetchSlots54(LocalTensor<float> &slots) {
        CopyPrefetchSlot(slots, start_);
        SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
    }
    __aicore__ inline void BindPrefetchSlot54(LocalTensor<float> &slots, uint32_t m, uint32_t slotElems) {
        WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        mat_ = slots[(m & 1U) * slotElems];
    }

    template <uint32_t TN> __aicore__ inline void CopyFixedMatrix56(uint64_t g, uint32_t nn32, bool tiny) {
        if (TN == (TN + 8U)) {
            DataCopy(mat_, aGm_[g * (static_cast<uint64_t>(TN) * TN)], nn32);
        } else if (tiny) {
            DataCopyPad(mat_[0], aGm_[g * (static_cast<uint64_t>(TN) * TN)],
                        {1, static_cast<uint32_t>(nn32 * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        } else {
            SharedDataCopyPadStage8<TN>(g);
        }
    }

    __aicore__ inline void CopyAndRunTiny56_1(uint64_t matrixElements, uint32_t nn32) {

        uint32_t tot = cnt_ * nn32;
        for (uint32_t off = 0; off < tot; off += 4096) {
            uint32_t len = tot - off;
            if (len > 4096) {
                len = 4096;
            }
            DataCopyPad(mat_[off], aGm_[static_cast<uint64_t>(start_) * matrixElements + off],
                        {1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
        WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
        for (uint32_t m = 0; m < cnt_; ++m) {
            res_.SetValue(m, TinyLUDispatch<LD_BATCH == 4U>(m * nn32));
        }
        return;
    }

    __aicore__ inline void GatherAbsColumn56(__ubuf__ float *column, __ubuf__ uint32_t *offsets, __ubuf__ float *matrix,
                                             __ubuf__ float *absolute, uint32_t begin, uint32_t count) {
        SmallMaskSet(count);
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(column), offsets + (begin & 7U) * ::kSmallMaxN,
                reinterpret_cast<uint64_t>(matrix) + (begin & ~7U) * sizeof(float), 8, 1);
        PipeBarrier<PIPE_V>();
        vabs(absolute, column, 1, 1, 1, 8, 8);
    }

    template <bool RAW> __aicore__ inline float ReadScalarMatrix56(__ubuf__ float *pointer, uint32_t index) {
        if constexpr (RAW)
            return pointer[index];
        else
            return mat_.GetValue(index);
    }
    template <auto normalMask, bool RAW>
    __aicore__ inline void ScalarTinyStep56_1(uint32_t limit, uint32_t pitch, uint32_t k, uint32_t p,
                                              __ubuf__ float *matrixPointer, int32_t &ex, bool &fastProd, float &prod,
                                              float &sign) {

        if (p != k) {
            for (uint32_t j = k; j < limit; ++j) {
                float va = ReadScalarMatrix56<RAW>(matrixPointer, k * pitch + j);
                float vb = ReadScalarMatrix56<RAW>(matrixPointer, p * pitch + j);
                mat_.SetValue(k * pitch + j, vb);
                mat_.SetValue(p * pitch + j, va);
            }
            sign = -sign;
        }
        float piv = ReadScalarMatrix56<RAW>(matrixPointer, k * pitch + k);
        if constexpr (normalMask) {
            ProdSmallCommon(fastProd, prod, ex, piv);
        } else {
            ProdAccum(prod, ex, piv);
        }
        float inv = LdetSharedOperations::ReciprocalWithZero58(piv, 1.0f);
        float rowk[16];
        for (uint32_t j = k + 1; j < limit; ++j) {
            rowk[j - k - 1] = ReadScalarMatrix56<RAW>(matrixPointer, k * pitch + j);
        }
        for (uint32_t i = k + 1; i < limit; ++i) {
            float l = ReadScalarMatrix56<RAW>(matrixPointer, i * pitch + k) * inv;
            for (uint32_t j = k + 1; j < limit; ++j) {
                float u = ReadScalarMatrix56<RAW>(matrixPointer, i * pitch + j) - l * rowk[j - k - 1];
                mat_.SetValue(i * pitch + j, u);
            }
        }
    }
