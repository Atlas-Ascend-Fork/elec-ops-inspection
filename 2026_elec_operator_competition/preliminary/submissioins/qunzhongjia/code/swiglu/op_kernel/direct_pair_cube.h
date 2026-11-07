/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 黎何.
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

namespace optiling {
template <typename T, uint32_t FixedM = 0, uint32_t FixedD = 0, bool Unaligned = false, bool Resident = false,
          bool FullTile = false, uint32_t FixedN = 0>
class DirectPairCube {
public:
    __aicore__ inline void Init(uint32_t k, uint32_t m, uint32_t n);
    __aicore__ inline void PrepareWeights(const GlobalTensor<T>& w1, const GlobalTensor<T>& w2,
                                          uint32_t cols);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
    __aicore__ inline __attribute__((always_inline)) void
    ComputeStage1a5debfb(A0& w1, A1& w2, A2& a1, A3& b1, A4& a2, A5& b2, A6& c, A7& alignedM, A8& bp);
    __aicore__ inline void Compute(const GlobalTensor<T>& x, const GlobalTensor<T>& w1,
                                   const GlobalTensor<T>& w2, const GlobalTensor<T>& dst, uint32_t rows,
                                   uint32_t cols);

private:
    uint32_t k_ = 0;
    uint32_t realK_ = 0;
    uint32_t m_ = 0;
    uint32_t n_ = 0;
};
}

namespace optiling {
template <typename T, uint32_t FixedM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile,
          uint32_t FixedN>
__aicore__ inline void
DirectPairCube<T, FixedM, FixedD, Unaligned, Resident, FullTile, FixedN>::Init(uint32_t k, uint32_t m,
                                                                               uint32_t n)
{
    realK_ = FixedD != 0 ? FixedD : k;
    k_ = Unaligned ? (realK_ + 15U) / 16U * 16U : realK_;
    m_ = FixedM != 0 ? FixedM : m;
    n_ = FixedN != 0 ? FixedN : (FixedD != 0 ? (FixedD < 64U ? FixedD : 64U) : n);
}

template <typename T, uint32_t FixedM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile,
          uint32_t FixedN>
__aicore__ inline void
DirectPairCube<T, FixedM, FixedD, Unaligned, Resident, FullTile, FixedN>::PrepareWeights(
    const GlobalTensor<T>& w1, const GlobalTensor<T>& w2, uint32_t cols)
{
    LocalTensor<T> b1(TPosition::B1, m_ * k_ * sizeof(T), 2U * n_ * k_);
    LocalTensor<T> b2(TPosition::B2, 0, 2U * n_ * k_);
    Nd2NzParams bp{1,
                   static_cast<uint16_t>(cols),
                   static_cast<uint16_t>(realK_),
                   0,
                   static_cast<uint16_t>(realK_),
                   static_cast<uint16_t>(2U * n_),
                   1,
                   0};
    DataCopy(b1, w1, bp);
    DataCopy(b1[n_ * 16U], w2, bp);
    ::SharedSyncb55bd560<HardEvent::MTE2_MTE1>();
    LoadData2dParams bl;
    bl.repeatTimes = static_cast<uint8_t>(2U * n_ * k_ / 256U);
    bl.srcStride = 1;
    LoadData(b2, b1, bl);
    ::SharedSyncb55bd560<HardEvent::MTE1_M>();
}

template <typename T, uint32_t FixedM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile,
          uint32_t FixedN>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
__aicore__ inline __attribute__((always_inline)) void
DirectPairCube<T, FixedM, FixedD, Unaligned, Resident, FullTile, FixedN>::ComputeStage1a5debfb(
    A0& w1, A1& w2, A2& a1, A3& b1, A4& a2, A5& b2, A6& c, A7& alignedM, A8& bp)
{
    if constexpr (!Resident) {
        DataCopy(b1, w1, bp);
        DataCopy(b1[n_ * 16U], w2, bp);
    }
    ::SharedSyncb55bd560<HardEvent::MTE2_MTE1>();
    LoadData2dParams al;
    al.repeatTimes = static_cast<uint8_t>(k_ / 16U);
    al.srcStride = static_cast<uint16_t>(alignedM / 16U);
    if constexpr (FullTile) {
#pragma unroll
        for (uint32_t mb = 0; mb < FixedM / 16U; ++mb) {
            LoadData(a2[mb * k_ * 16U], a1[mb * 256U], al);
        }
    } else {
        for (uint32_t mb = 0; mb < alignedM / 16U; ++mb) {
            LoadData(a2[mb * k_ * 16U], a1[mb * 256U], al);
        }
    }
    LoadData2dParams bl;
    bl.repeatTimes = static_cast<uint8_t>(2U * n_ * k_ / 256U);
    bl.srcStride = 1;
    if constexpr (!Resident) {
        LoadData(b2, b1, bl);
    }
    if constexpr (Resident) {
        ::SharedSyncb55bd560<HardEvent::MTE1_MTE2>();
    }
    ::SharedSyncb55bd560<HardEvent::MTE1_M>();
    MmadParams mad;
    mad.m = static_cast<uint16_t>(alignedM);
    mad.n = static_cast<uint16_t>(2U * n_);
    mad.k = static_cast<uint16_t>(k_);
    mad.cmatrixInitVal = true;
    Mmad(c, a2, b2, mad);
}

template <typename T, uint32_t FixedM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile,
          uint32_t FixedN>
__aicore__ inline void DirectPairCube<T, FixedM, FixedD, Unaligned, Resident, FullTile, FixedN>::Compute(
    const GlobalTensor<T>& x, const GlobalTensor<T>& w1, const GlobalTensor<T>& w2,
    const GlobalTensor<T>& dst, uint32_t rows, uint32_t cols)
{
    LocalTensor<T> a1(TPosition::A1, 0, m_ * k_);
    LocalTensor<T> b1(TPosition::B1, m_ * k_ * sizeof(T), 2U * n_ * k_);
    LocalTensor<T> a2(TPosition::A2, 0, m_ * k_);
    LocalTensor<T> b2(TPosition::B2, 0, 2U * n_ * k_);
    LocalTensor<float> c(TPosition::CO1, 0, 2U * m_ * n_);
    const uint32_t alignedM = FullTile ? FixedM : (rows + 15U) / 16U * 16U;
    Nd2NzParams ap{1,
                   static_cast<uint16_t>(rows),
                   static_cast<uint16_t>(realK_),
                   0,
                   static_cast<uint16_t>(realK_),
                   static_cast<uint16_t>(alignedM),
                   1,
                   0};
    Nd2NzParams bp{1,
                   static_cast<uint16_t>(cols),
                   static_cast<uint16_t>(realK_),
                   0,
                   static_cast<uint16_t>(realK_),
                   static_cast<uint16_t>(2U * n_),
                   1,
                   0};
    DataCopy(a1, x, ap);
    ComputeStage1a5debfb(w1, w2, a1, b1, a2, b2, c, alignedM, bp);
    if constexpr (Resident) {
        ::SharedSyncb55bd560<HardEvent::M_MTE1>();
    }
    ::SharedSyncb55bd560<HardEvent::M_FIX>();
    FixpipeParamsV220 fix;
    fix.nSize = static_cast<uint16_t>(2U * n_);
    fix.mSize = static_cast<uint16_t>(rows);
    fix.srcStride = static_cast<uint16_t>(alignedM);
    fix.dstStride = 2U * n_;
    fix.quantPre = QuantMode_t::F322F16;
    Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c, fix);
    if constexpr (Resident) {
        ::SharedSyncb55bd560<HardEvent::FIX_M>();
    } else {
        PipeBarrier<PIPE_ALL>();
    }
}

}
