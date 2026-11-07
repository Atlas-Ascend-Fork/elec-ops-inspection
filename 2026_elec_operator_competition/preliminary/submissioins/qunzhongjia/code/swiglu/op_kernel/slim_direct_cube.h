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
template <typename T, uint32_t TileM, uint32_t FixedD = 0, bool Unaligned = false, bool Resident = false,
          bool FullTile = false>
class SlimDirectCube {
public:
    template <class A0, class A1, class A2, class A3, class A4>
    __aicore__ inline __attribute__((always_inline)) void RunStageaea053d6(A0& td, A1& cores, A2& core,
                                                                           A3& plane, A4& tasksN);
    template <class A0, class A1, class A2, class A3, class A4>
    __aicore__ inline __attribute__((always_inline)) void RunStagef73b4171(A0& y, A1& workspace, A2& td,
                                                                           A3& cores, A4& core);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               GM_ADDR workspace, const SwiGluDirectTilingData* td);

private:
    __aicore__ inline uint32_t Min(uint32_t a, uint64_t b);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10>
    __aicore__ inline __attribute__((always_inline)) void
    EpilogueStagee4f4b231(A0& row, A1& col, A2& cols, A3& begin, A4& nr, A5& g, A6& u, A7& b1, A8& b2, A9& s,
                          A10& out);
    __aicore__ inline void Epilogue(uint64_t off, uint64_t row, uint32_t col, uint32_t rows, uint32_t cols);
    __aicore__ inline void WaitReady(uint32_t s);
    __aicore__ inline void Ready(uint32_t s);
    __aicore__ inline void WaitConsumed(uint32_t s);
    __aicore__ inline void Consumed(uint32_t s);
    DirectPairCube<T, TileM, FixedD, Unaligned, Resident, FullTile> cube_;
    GlobalTensor<T> x_;
    GlobalTensor<T> w1_;
    GlobalTensor<T> w2_;
    GlobalTensor<T> b1_;
    GlobalTensor<T> b2_;
    GlobalTensor<T> y_;
    GlobalTensor<T> slots_;
    static constexpr uint32_t tm_ = TileM;
    uint32_t d_ = 0;
    uint32_t n_ = 0;
};
}

namespace optiling {
template <typename T, uint32_t TileM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile>
template <class A0, class A1, class A2, class A3, class A4>
__aicore__ inline __attribute__((always_inline)) void
SlimDirectCube<T, TileM, FixedD, Unaligned, Resident, FullTile>::RunStageaea053d6(A0& td, A1& cores, A2& core,
                                                                                  A3& plane, A4& tasksN)
{
    uint32_t local = 0;
    for (uint32_t task = core; task < td->taskNum; task += cores, ++local) {
        const uint32_t slot = local % 4U;
        const uint64_t row = static_cast<uint64_t>(task / tasksN) * tm_;
        const uint32_t col = task % tasksN * n_;
        const uint32_t rows = FullTile ? tm_ : Min(tm_, td->m - row);
        const uint32_t cols = FullTile ? n_ : Min(n_, d_ - col);
        const uint64_t off = (static_cast<uint64_t>(core) * 4U + slot) * 2U * plane;
        if ASCEND_IS_AIC {
            if (local >= 4U) {
                WaitConsumed(slot);
            }
            cube_.Compute(x_[row * d_], w1_[static_cast<uint64_t>(col) * d_],
                          w2_[static_cast<uint64_t>(col) * d_], slots_[off], rows, cols);
            Ready(slot);
        }
        if ASCEND_IS_AIV {
            WaitReady(slot);
            Epilogue(off, row, col, rows, cols);
            Consumed(slot);
        }
    }
    if ASCEND_IS_AIC {
        for (uint32_t slot = 0; slot < Min(local, 4U); ++slot) {
            WaitConsumed(slot);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T, uint32_t TileM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile>
template <class A0, class A1, class A2, class A3, class A4>
__aicore__ inline __attribute__((always_inline)) void
SlimDirectCube<T, TileM, FixedD, Unaligned, Resident, FullTile>::RunStagef73b4171(A0& y, A1& workspace,
                                                                                  A2& td, A3& cores, A4& core)
{
    y_.SetGlobalBuffer((__gm__ T*)y, td->m * d_);
    const uint32_t plane = tm_ * n_;
    slots_.SetGlobalBuffer((__gm__ T*)workspace, static_cast<uint64_t>(2U * plane) * 4U * cores);
    if ASCEND_IS_AIC {
        cube_.Init(d_, tm_, n_);
    }
    SetAtomicNone();
    if ASCEND_IS_AIV {
        SetMaskNorm();
        SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    }
    const uint32_t tasksN = (d_ + n_ - 1U) / n_;
    if constexpr (Resident) {
        const uint32_t col = core % tasksN * n_;
        const uint32_t cols = FullTile ? n_ : Min(n_, d_ - col);
        if ASCEND_IS_AIC {
            cube_.PrepareWeights(w1_[static_cast<uint64_t>(col) * d_], w2_[static_cast<uint64_t>(col) * d_],
                                 cols);
        }
        if ASCEND_IS_AIV {
            LocalTensor<T> bias(TPosition::VECIN, tm_ * n_ * sizeof(T), 2U * n_);
            if constexpr (Unaligned) {
                DataCopyExtParams bp{1, cols * 2U, 0, 0, 0};
                DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>((cols + 15U) / 16U * 16U - cols),
                                            static_cast<T>(0)};
                DataCopyPad(bias, b1_[col], bp, pad);
                DataCopyPad(bias[n_], b2_[col], bp, pad);
            } else {
                DataCopy(bias, b1_[col], cols);
                DataCopy(bias[n_], b2_[col], cols);
            }
        }
    }
    RunStageaea053d6(td, cores, core, plane, tasksN);
}

template <typename T, uint32_t TileM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile>
__aicore__ inline void SlimDirectCube<T, TileM, FixedD, Unaligned, Resident, FullTile>::Run(
    GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y, GM_ADDR workspace,
    const SwiGluDirectTilingData* td)
{
    d_ = FixedD != 0 ? FixedD : static_cast<uint32_t>(td->d);
    n_ = FixedD != 0 ? (FixedD < 64U ? FixedD : 64U) : td->taskN;
    const uint32_t cores = td->usedCoreNum;
    uint32_t core = GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= 2U;
    }
    x_.SetGlobalBuffer((__gm__ T*)x, td->m * d_);
    w1_.SetGlobalBuffer((__gm__ T*)w1, static_cast<uint64_t>(d_) * d_);
    w2_.SetGlobalBuffer((__gm__ T*)w2, static_cast<uint64_t>(d_) * d_);
    b1_.SetGlobalBuffer((__gm__ T*)b1, d_);
    b2_.SetGlobalBuffer((__gm__ T*)b2, d_);
    RunStagef73b4171(y, workspace, td, cores, core);
}

template <typename T, uint32_t TileM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile>
__aicore__ inline uint32_t SlimDirectCube<T, TileM, FixedD, Unaligned, Resident, FullTile>::Min(uint32_t a,
                                                                                                uint64_t b)
{
    return a < b ? a : b;
}

template <typename T, uint32_t TileM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10>
__aicore__ inline __attribute__((always_inline)) void
SlimDirectCube<T, TileM, FixedD, Unaligned, Resident, FullTile>::EpilogueStagee4f4b231(
    A0& row, A1& col, A2& cols, A3& begin, A4& nr, A5& g, A6& u, A7& b1, A8& b2, A9& s, A10& out)
{
    ::SharedSync64ce5475<HardEvent::MTE2_V>();
    const uint8_t stride = static_cast<uint8_t>(n_ / 16U);
    const BinaryRepeatParams repeat{1, 1, 1, stride, stride, 0};
    Add(g, g, b1, static_cast<uint64_t>(cols), static_cast<uint8_t>(nr), repeat);
    Add(u, u, b2, static_cast<uint64_t>(cols), static_cast<uint8_t>(nr), repeat);
    PipeBarrier<PIPE_V>();
    const uint32_t count = nr * n_;
    GateCounter<T, true>(g, u, s, out, out, count);
    ::SharedSync64ce5475<HardEvent::V_MTE3>();
    if constexpr (Unaligned) {
        const uint32_t pitch = (cols + 15U) / 16U * 16U;
        DataCopyExtParams output{static_cast<uint16_t>(nr), cols * 2U, (n_ - pitch) / 16U, (d_ - cols) * 2U,
                                 0};
        DataCopyPad(y_[(row + begin) * d_ + col], out, output);
    } else {
        DataCopyParams output{static_cast<uint16_t>(nr), static_cast<uint16_t>(cols / 16U),
                              static_cast<uint16_t>((n_ - cols) / 16U),
                              static_cast<uint16_t>((d_ - cols) / 16U)};
        DataCopy(y_[(row + begin) * d_ + col], out, output);
    }
    ::SharedSync64ce5475<HardEvent::MTE3_MTE2>();
}

template <typename T, uint32_t TileM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile>
__aicore__ inline void SlimDirectCube<T, TileM, FixedD, Unaligned, Resident, FullTile>::Epilogue(
    uint64_t off, uint64_t row, uint32_t col, uint32_t rows, uint32_t cols)
{
    const uint32_t half = FullTile ? tm_ / 2U : (rows + 1U) / 2U;
    const uint32_t begin = GetSubBlockIdx() * half;
    if constexpr (!FullTile) {
        if (begin >= rows) {
            return;
        }
    }
    const uint32_t nr = FullTile ? tm_ / 2U : Min(half, rows - begin);
    const uint32_t capacity = tm_ / 2U * n_;
    LocalTensor<T> input(TPosition::VECIN, 0, (tm_ + 2U) * n_);
    auto g = input;
    auto u = input[capacity];
    auto b1 = input[2U * capacity];
    auto b2 = input[2U * capacity + n_];
    LocalTensor<T> work(TPosition::VECOUT, (tm_ + 2U) * n_ * sizeof(T), tm_ * n_);
    auto s = work;
    auto out = work[capacity];
    const uint64_t start = off + begin * 2U * n_;
    if constexpr (Unaligned) {
        const uint32_t pitch = (cols + 15U) / 16U * 16U;
        DataCopyExtParams cp{static_cast<uint16_t>(nr), cols * 2U, (2U * n_ - cols) * 2U, (n_ - pitch) / 16U,
                             0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(pitch - cols), static_cast<T>(0)};
        DataCopyPad(g, slots_[start], cp, pad);
        DataCopyPad(u, slots_[start + n_], cp, pad);
        DataCopyExtParams bp{1, cols * 2U, 0, 0, 0};
        if constexpr (!Resident) {
            DataCopyPad(b1, b1_[col], bp, pad);
            DataCopyPad(b2, b2_[col], bp, pad);
        }
    } else {
        DataCopyParams cp{static_cast<uint16_t>(nr), static_cast<uint16_t>(cols / 16U),
                          static_cast<uint16_t>((2U * n_ - cols) / 16U),
                          static_cast<uint16_t>((n_ - cols) / 16U)};
        DataCopy(g, slots_[start], cp);
        DataCopy(u, slots_[start + n_], cp);
        if constexpr (!Resident) {
            DataCopy(b1, b1_[col], cols);
            DataCopy(b2, b2_[col], cols);
        }
    }
    EpilogueStagee4f4b231(row, col, cols, begin, nr, g, u, b1, b2, s, out);
}

template <typename T, uint32_t TileM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile>
__aicore__ inline void SlimDirectCube<T, TileM, FixedD, Unaligned, Resident, FullTile>::WaitReady(uint32_t s)
{
    CubeSlotSync::WaitReady(s);
}

template <typename T, uint32_t TileM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile>
__aicore__ inline void SlimDirectCube<T, TileM, FixedD, Unaligned, Resident, FullTile>::Ready(uint32_t s)
{
    CubeSlotSync::Ready(s);
}

template <typename T, uint32_t TileM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile>
__aicore__ inline void
SlimDirectCube<T, TileM, FixedD, Unaligned, Resident, FullTile>::WaitConsumed(uint32_t s)
{
    CubeSlotSync::WaitConsumed(s);
}

template <typename T, uint32_t TileM, uint32_t FixedD, bool Unaligned, bool Resident, bool FullTile>
__aicore__ inline void SlimDirectCube<T, TileM, FixedD, Unaligned, Resident, FullTile>::Consumed(uint32_t s)
{
    CubeSlotSync::Consumed(s);
}

}
