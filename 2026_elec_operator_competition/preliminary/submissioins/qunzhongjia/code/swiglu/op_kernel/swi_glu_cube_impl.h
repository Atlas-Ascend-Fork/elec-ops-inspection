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
template <typename xType, const auto& Config = CFG_NORM, bool Direct = false, bool Padded = false,
          bool Merged = false>
class SwiGluCubeImpl {
public:
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
    __aicore__ inline __attribute__((always_inline)) void
    InitStagebd4e262f(A0& x, A1& w1, A2& b1, A3& w2, A4& b2, A5& y, A6& userWorkspace, A7& pipe, A8& td);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                                GM_ADDR userWorkspace, TPipe* pipe, const SwiGluTilingData* td);
    template <class A0, class A1, class A2>
    __aicore__ inline __attribute__((always_inline)) void ProcessStage3460b559(A0& local, A1& task, A2& slot);
    __aicore__ inline void Process();
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, xType, false>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, xType, true>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, xType>;
    typename MatmulStorageSelector<Direct, AType, BType, CType, Config>::Type mm;
    DirectPairCube<xType> direct;

private:
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b);
    __aicore__ inline void PackMatrix(const GlobalTensor<xType>& dst, const GlobalTensor<xType>& src,
                                      uint64_t realRows, uint64_t totalRows);
    __aicore__ inline void Projection(const GlobalTensor<xType>& weight, uint64_t row, uint32_t col,
                                      uint32_t rows, uint32_t cols, uint64_t off);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9>
    __aicore__ inline __attribute__((always_inline)) void
    EpilogueStage75cb58ea(A0& row, A1& col, A2& cols, A3& begin, A4& count, A5& pitch, A6& gate, A7& up,
                          A8& b1, A9& b2);
    __aicore__ inline void Epilogue(uint64_t off, uint64_t row, uint32_t col, uint32_t rows, uint32_t cols);
    __aicore__ inline void WaitReady(uint32_t s);
    __aicore__ inline void SignalReady(uint32_t s);
    __aicore__ inline void WaitConsumed(uint32_t s);
    __aicore__ inline void SignalConsumed(uint32_t s);
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> gateQ_;
    TQue<QuePosition::VECIN, 1> upQ_;
    TQue<QuePosition::VECIN, 1> bias1Q_;
    TQue<QuePosition::VECIN, 1> bias2Q_;
    TQue<QuePosition::VECOUT, 1> outQ_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> packQ_;
    TBuf<TPosition::VECCALC> gateBuf_;
    TBuf<TPosition::VECCALC> upBuf_;
    TBuf<TPosition::VECCALC> negBuf_;
    TBuf<TPosition::VECCALC> expBuf_;
    TBuf<TPosition::VECCALC> onesBuf_;
    TBuf<TPosition::VECCALC> sigmoidBuf_;
    GlobalTensor<xType> x_;
    GlobalTensor<xType> w1_;
    GlobalTensor<xType> w2_;
    GlobalTensor<xType> b1_;
    GlobalTensor<xType> b2_;
    GlobalTensor<xType> y_;
    GlobalTensor<xType> slots_;
    GlobalTensor<xType> paddedX_;
    GlobalTensor<xType> paddedW1_;
    GlobalTensor<xType> paddedW2_;
    uint64_t m_ = 0;
    uint32_t d_ = 0;
    uint32_t taskM_ = 0;
    uint32_t taskN_ = 0;
    uint32_t taskNum_ = 0;
    uint32_t tasksN_ = 0;
    uint32_t core_ = 0;
    uint32_t sub_ = 0;
    uint32_t cores_ = 0;
    uint32_t plane_ = 0;
    uint32_t computeD_ = 0;
    uint32_t packRows_ = 0;
    static constexpr bool padded_ = Padded;
    static constexpr bool merged_ = Merged || Direct;
};
}

namespace optiling {
template <typename xType, const auto& Config, bool Direct, bool Padded, bool Merged>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
__aicore__ inline __attribute__((always_inline)) void
SwiGluCubeImpl<xType, Config, Direct, Padded, Merged>::InitStagebd4e262f(A0& x, A1& w1, A2& b1, A3& w2,
                                                                         A4& b2, A5& y, A6& userWorkspace,
                                                                         A7& pipe, A8& td)
{
    pipe_ = pipe;
    core_ = GetBlockIdx();
    sub_ = GetSubBlockIdx();
    if ASCEND_IS_AIV {
        core_ /= 2U;
    }
    m_ = td->dimensions.m;
    d_ = static_cast<uint32_t>(td->dimensions.d);
    computeD_ = td->dimensions.dAlign;
    taskM_ = td->dimensions.taskM;
    taskN_ = td->dimensions.taskN;
    taskNum_ = td->dimensions.taskNum;
    cores_ = td->dimensions.usedCoreNum;
    tasksN_ = (d_ + taskN_ - 1U) / taskN_;
    plane_ = taskM_ * taskN_;
    x_.SetGlobalBuffer((__gm__ xType*)x, m_ * d_);
    w1_.SetGlobalBuffer((__gm__ xType*)w1, static_cast<uint64_t>(d_) * d_);
    w2_.SetGlobalBuffer((__gm__ xType*)w2, static_cast<uint64_t>(d_) * d_);
    b1_.SetGlobalBuffer((__gm__ xType*)b1, d_);
    b2_.SetGlobalBuffer((__gm__ xType*)b2, d_);
    y_.SetGlobalBuffer((__gm__ xType*)y, m_ * d_);
    slots_.SetGlobalBuffer((__gm__ xType*)userWorkspace,
                           static_cast<uint64_t>(plane_) * 2 * kSlotCount * cores_);
    if constexpr (padded_) {
        const uint64_t base = static_cast<uint64_t>(plane_) * 2 * kSlotCount * cores_;
        const uint64_t wBase = base + m_ * computeD_;
        paddedX_.SetGlobalBuffer((__gm__ xType*)userWorkspace + base, m_ * computeD_);
        paddedW1_.SetGlobalBuffer((__gm__ xType*)userWorkspace + wBase,
                                  static_cast<uint64_t>(computeD_) * computeD_ * (merged_ ? 2U : 1U));
        paddedW2_.SetGlobalBuffer((__gm__ xType*)userWorkspace + wBase +
                                      static_cast<uint64_t>(computeD_) * computeD_,
                                  static_cast<uint64_t>(computeD_) * computeD_);
        packRows_ = 4096U / computeD_;
    }
}

template <typename xType, const auto& Config, bool Direct, bool Padded, bool Merged>
__aicore__ inline void
SwiGluCubeImpl<xType, Config, Direct, Padded, Merged>::Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
                                                            GM_ADDR b2, GM_ADDR y, GM_ADDR userWorkspace,
                                                            TPipe* pipe, const SwiGluTilingData* td)
{
    InitStagebd4e262f(x, w1, b1, w2, b2, y, userWorkspace, pipe, td);
    if ASCEND_IS_AIC {
        if constexpr (Direct) {
            direct.Init(computeD_, taskM_, taskN_);
        } else {
            mm.Init(&td->cubeTilingData, pipe_);
            mm.SetHF32(false, 0);
            const uint32_t matmulN = merged_ ? 2U * taskN_ : taskN_;
            mm.SetSingleShape(taskM_, matmulN, computeD_);
            mm.SetOrgShape(taskM_, matmulN, computeD_, computeD_);
        }
    }
    if ASCEND_IS_AIV {
        const uint32_t cBytes = (taskM_ / 2U) * taskN_ * sizeof(xType);
        const uint32_t vBytes = kBlockRows * taskN_ * sizeof(xType);
        pipe_->InitBuffer(gateQ_, 1, cBytes);
        pipe_->InitBuffer(upQ_, 1, cBytes);
        pipe_->InitBuffer(bias1Q_, 1, taskN_ * sizeof(xType));
        pipe_->InitBuffer(bias2Q_, 1, taskN_ * sizeof(xType));
        pipe_->InitBuffer(outQ_, 1, vBytes);
        pipe_->InitBuffer(gateBuf_, vBytes);
        pipe_->InitBuffer(upBuf_, vBytes);
        pipe_->InitBuffer(negBuf_, vBytes);
        pipe_->InitBuffer(expBuf_, vBytes);
        pipe_->InitBuffer(onesBuf_, vBytes);
        pipe_->InitBuffer(sigmoidBuf_, vBytes);
        if constexpr (padded_) {
            pipe_->InitBuffer(packQ_, 1, packRows_ * computeD_ * sizeof(xType));
        }
    }
}

template <typename xType, const auto& Config, bool Direct, bool Padded, bool Merged>
template <class A0, class A1, class A2>
__aicore__ inline __attribute__((always_inline)) void
SwiGluCubeImpl<xType, Config, Direct, Padded, Merged>::ProcessStage3460b559(A0& local, A1& task, A2& slot)
{
    const uint64_t row = static_cast<uint64_t>(task / tasksN_) * taskM_;
    const uint32_t col = (task % tasksN_) * taskN_;
    const uint32_t rows = Min(taskM_, static_cast<uint32_t>(m_ - row));
    const uint32_t cols = Min(taskN_, d_ - col);
    const uint64_t off = (static_cast<uint64_t>(core_) * kSlotCount + slot) * 2 * plane_;
    if ASCEND_IS_AIC {
        if (local >= kSlotCount) {
            WaitConsumed(slot);
        }
        if constexpr (Direct) {
            direct.Compute(x_[row * d_], w1_[static_cast<uint64_t>(col) * d_],
                           w2_[static_cast<uint64_t>(col) * d_], slots_[off], rows, cols);
        } else {
            if constexpr (padded_) {
                const uint32_t computeCols = Min(taskN_, computeD_ - col);
                if constexpr (merged_) {
                    Projection(paddedW1_, row, 0, rows, 2U * computeCols, off);
                } else {
                    Projection(paddedW1_, row, col, rows, computeCols, off);
                    Projection(paddedW2_, row, col, rows, computeCols, off + plane_);
                }
            } else {
                Projection(w1_, row, col, rows, cols, off);
                Projection(w2_, row, col, rows, cols, off + plane_);
            }
        }
        SignalReady(slot);
    }
    if ASCEND_IS_AIV {
        WaitReady(slot);
        Epilogue(off, row, col, rows, cols);
        SignalConsumed(slot);
    }
}

template <typename xType, const auto& Config, bool Direct, bool Padded, bool Merged>
__aicore__ inline void SwiGluCubeImpl<xType, Config, Direct, Padded, Merged>::Process()
{
    if (core_ >= cores_) {
        return;
    }
    if constexpr (padded_) {
        if ASCEND_IS_AIV {
            PackMatrix(paddedX_, x_, m_, m_);
            PackMatrix(paddedW1_, w1_, d_, computeD_);
            PackMatrix(paddedW2_, w2_, d_, computeD_);
            SyncAll();
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(0x4);
        }
    }
    uint32_t local = 0;
    for (uint32_t task = core_; task < taskNum_; task += cores_, ++local) {
        const uint32_t slot = local % kSlotCount;
        ProcessStage3460b559(local, task, slot);
    }
    if ASCEND_IS_AIC {
        for (uint32_t slot = 0; slot < Min(local, kSlotCount); ++slot) {
            WaitConsumed(slot);
        }
    }
}

template <typename xType, const auto& Config, bool Direct, bool Padded, bool Merged>
__aicore__ inline uint32_t SwiGluCubeImpl<xType, Config, Direct, Padded, Merged>::Min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

template <typename xType, const auto& Config, bool Direct, bool Padded, bool Merged>
__aicore__ inline void SwiGluCubeImpl<xType, Config, Direct, Padded, Merged>::PackMatrix(
    const GlobalTensor<xType>& dst, const GlobalTensor<xType>& src, uint64_t realRows, uint64_t totalRows)
{
    const uint64_t chunks = (totalRows + packRows_ - 1U) / packRows_;
    for (uint64_t block = GetBlockIdx(); block < chunks; block += cores_ * 2U) {
        const uint64_t row = block * packRows_;
        const uint32_t nr = Min(packRows_, static_cast<uint32_t>(totalRows - row));
        const uint32_t valid = row < realRows ? Min(nr, static_cast<uint32_t>(realRows - row)) : 0;
        LocalTensor<xType> tile = packQ_.AllocTensor<xType>();
        if (valid != 0) {
            DataCopyExtParams copy{static_cast<uint16_t>(valid), d_ * 2U, 0, 0, 0};
            DataCopyPadExtParams<xType> pad{true, 0, static_cast<uint8_t>(computeD_ - d_), 0};
            DataCopyPad(tile, src[row * d_], copy, pad);
        }
        if (valid < nr) {
            Duplicate(tile[valid * computeD_], static_cast<xType>(0), (nr - valid) * computeD_);
            const event_t event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(event);
            WaitFlag<HardEvent::V_MTE3>(event);
        }
        packQ_.EnQue(tile);
        tile = packQ_.DeQue<xType>();
        DataCopy(dst[row * computeD_], tile, nr * computeD_);
        packQ_.FreeTensor(tile);
    }
}

template <typename xType, const auto& Config, bool Direct, bool Padded, bool Merged>
__aicore__ inline void SwiGluCubeImpl<xType, Config, Direct, Padded, Merged>::Projection(
    const GlobalTensor<xType>& weight, uint64_t row, uint32_t col, uint32_t rows, uint32_t cols, uint64_t off)
{
    mm.SetTail(rows, cols, computeD_);
    if constexpr (padded_) {
        mm.SetTensorA(paddedX_[row * computeD_], false);
    } else {
        mm.SetTensorA(x_[row * d_], false);
    }
    mm.SetTensorB(weight[static_cast<uint64_t>(col) * computeD_], true);
    mm.IterateAll(slots_[off], false);
    mm.End();
    PipeBarrier<PIPE_ALL>();
}

template <typename xType, const auto& Config, bool Direct, bool Padded, bool Merged>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
__aicore__ inline __attribute__((always_inline)) void
SwiGluCubeImpl<xType, Config, Direct, Padded, Merged>::EpilogueStage75cb58ea(A0& row, A1& col, A2& cols,
                                                                             A3& begin, A4& count, A5& pitch,
                                                                             A6& gate, A7& up, A8& b1, A9& b2)
{
    bias1Q_.EnQue(b1);
    bias2Q_.EnQue(b2);
    gate = gateQ_.template DeQue<xType>();
    up = upQ_.template DeQue<xType>();
    b1 = bias1Q_.template DeQue<xType>();
    b2 = bias2Q_.template DeQue<xType>();
    LocalTensor<xType> a = gateBuf_.template Get<xType>();
    LocalTensor<xType> b = upBuf_.template Get<xType>();
    LocalTensor<xType> neg = negBuf_.template Get<xType>();
    LocalTensor<xType> exp = expBuf_.template Get<xType>();
    LocalTensor<xType> ones = onesBuf_.template Get<xType>();
    LocalTensor<xType> sig = sigmoidBuf_.template Get<xType>();
    for (uint32_t group = 0; group < count; group += kBlockRows) {
        const uint32_t nr = Min(kBlockRows, count - group);
        const uint32_t n = nr * pitch;
        const uint8_t rowStride = static_cast<uint8_t>(pitch / 16U);
        const BinaryRepeatParams biasRepeat{1, 1, 1, rowStride, rowStride, 0};
        for (uint32_t k = 0; k < cols; k += 128U) {
            const uint32_t width = Min(128U, cols - k);
            Add(a[k], gate[group * pitch + k], b1[k], static_cast<uint64_t>(width), static_cast<uint8_t>(nr),
                biasRepeat);
            Add(b[k], up[group * pitch + k], b2[k], static_cast<uint64_t>(width), static_cast<uint8_t>(nr),
                biasRepeat);
        }
        LocalTensor<xType> out = outQ_.template AllocTensor<xType>();
        GateCounter<xType, true>(a, b, exp, out, ones, n);
        outQ_.EnQue(out);
        out = outQ_.template DeQue<xType>();
        DataCopyExtParams outParams{static_cast<uint16_t>(nr), cols * static_cast<uint32_t>(sizeof(xType)), 0,
                                    (d_ - cols) * static_cast<uint32_t>(sizeof(xType)), 0};
        DataCopyPad(y_[(row + begin + group) * d_ + col], out, outParams);
        outQ_.FreeTensor(out);
    }
}

template <typename xType, const auto& Config, bool Direct, bool Padded, bool Merged>
__aicore__ inline void
SwiGluCubeImpl<xType, Config, Direct, Padded, Merged>::Epilogue(uint64_t off, uint64_t row, uint32_t col,
                                                                uint32_t rows, uint32_t cols)
{
    const uint32_t halfRows = (rows + 1U) / 2U;
    const uint32_t begin = sub_ * halfRows;
    if (begin >= rows) {
        return;
    }
    const uint32_t count = Min(halfRows, rows - begin);
    const uint32_t pitch = AlignUp16(cols);
    const uint32_t cPitch = merged_ ? 2U * taskN_ : taskN_;
    LocalTensor<xType> gate = gateQ_.AllocTensor<xType>();
    LocalTensor<xType> up = upQ_.AllocTensor<xType>();
    DataCopyExtParams cParams{static_cast<uint16_t>(count), cols * static_cast<uint32_t>(sizeof(xType)),
                              (cPitch - cols) * static_cast<uint32_t>(sizeof(xType)), 0, 0};
    DataCopyPadExtParams<xType> pad{false, 0, 0, 0};
    const uint64_t cOff = off + static_cast<uint64_t>(begin) * cPitch;
    DataCopyPad(gate, slots_[cOff], cParams, pad);
    DataCopyPad(up, slots_[cOff + (merged_ ? taskN_ : plane_)], cParams, pad);
    gateQ_.EnQue(gate);
    upQ_.EnQue(up);
    LocalTensor<xType> b1 = bias1Q_.AllocTensor<xType>();
    LocalTensor<xType> b2 = bias2Q_.AllocTensor<xType>();
    DataCopyExtParams biasParams{1, cols * static_cast<uint32_t>(sizeof(xType)), 0, 0, 0};
    DataCopyPad(b1, b1_[col], biasParams, pad);
    DataCopyPad(b2, b2_[col], biasParams, pad);
    EpilogueStage75cb58ea(row, col, cols, begin, count, pitch, gate, up, b1, b2);
    gateQ_.FreeTensor(gate);
    upQ_.FreeTensor(up);
    bias1Q_.FreeTensor(b1);
    bias2Q_.FreeTensor(b2);
}

template <typename xType, const auto& Config, bool Direct, bool Padded, bool Merged>
__aicore__ inline void SwiGluCubeImpl<xType, Config, Direct, Padded, Merged>::WaitReady(uint32_t s)
{
    CubeSlotSync::WaitReady(s);
}

template <typename xType, const auto& Config, bool Direct, bool Padded, bool Merged>
__aicore__ inline void SwiGluCubeImpl<xType, Config, Direct, Padded, Merged>::SignalReady(uint32_t s)
{
    CubeSlotSync::Ready(s);
}

template <typename xType, const auto& Config, bool Direct, bool Padded, bool Merged>
__aicore__ inline void SwiGluCubeImpl<xType, Config, Direct, Padded, Merged>::WaitConsumed(uint32_t s)
{
    CubeSlotSync::WaitConsumed(s);
}

template <typename xType, const auto& Config, bool Direct, bool Padded, bool Merged>
__aicore__ inline void SwiGluCubeImpl<xType, Config, Direct, Padded, Merged>::SignalConsumed(uint32_t s)
{
    CubeSlotSync::Consumed(s);
}

}
