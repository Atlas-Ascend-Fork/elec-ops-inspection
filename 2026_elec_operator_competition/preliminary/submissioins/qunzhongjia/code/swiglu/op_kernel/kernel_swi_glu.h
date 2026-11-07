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

template <class DT_INPUT_X, int Mode> class KernelSwiGlu {
public:
    __aicore__ inline KernelSwiGlu();
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                GM_ADDR bias_2, GM_ADDR output, const SwiGluTilingData& tiling);
    __aicore__ inline void Process();

private:
    template <HardEvent event> __aicore__ inline void SyncPipe();
    __aicore__ inline void ProcessRow(uint64_t row);
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8,
              class B9, class B10>
    __aicore__ inline __attribute__((always_inline)) void
    SharedStage0f62bc2d85(B0& accG16, B1& accU16, B2& b2Tile, B3& c0, B4& chunk, B5& d, B6& eBuf, B7& negBuf,
                          B8& onesBuf, B9& row, B10& sBuf);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11>
    __aicore__ inline __attribute__((always_inline)) void
    ProcessColTileStagea7eea475(A0& row, A1& c0, A2& chunk, A3& d, A4& accG, A5& accU, A6& accG16, A7& accU16,
                                A8& negBuf, A9& eBuf, A10& sBuf, A11& onesBuf);
    __aicore__ inline void ProcessColTile(uint64_t row, uint32_t c0, uint32_t chunk, uint32_t d,
                                          uint32_t dAlign, const LocalTensor<float>& xf);
    __aicore__ inline void PrepareOneWeight(const GlobalTensor<DT_INPUT_X>& weight,
                                            const LocalTensor<float>& cached, uint32_t c0, uint32_t chunk,
                                            uint32_t d);
    __aicore__ inline void PrepareWt(uint32_t c0, uint32_t chunk, uint32_t d);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
    __aicore__ inline __attribute__((always_inline)) void
    ProcessColTileTStage89d99c96(A0& c0, A1& chunk, A2& dAlign, A3& xf, A4& wf, A5& accG, A6& accU,
                                 A7& accG16, A8& accU16, A9& w1t, A10& w2t, A11& prodG, A12& prodU,
                                 A13& b1Tile, A14& biasCopyParams, A15& bPad, A16& b2Tile);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
    __aicore__ inline __attribute__((always_inline)) void
    ProcessColTileTStagebeb32067(A0& row, A1& c0, A2& chunk, A3& d, A4& dAlign, A5& xf, A6& wf, A7& accG,
                                 A8& accU, A9& accG16, A10& accU16, A11& negBuf, A12& eBuf, A13& sBuf,
                                 A14& onesBuf, A15& w1t, A16& w2t);
    __aicore__ inline void ProcessColTileT(uint64_t row, uint32_t c0, uint32_t chunk, uint32_t d);
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> stageA_;
    TQue<TPosition::VECIN, 1> stageB_;
    TQue<TPosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> xfBuf_;
    TBuf<TPosition::VECCALC> wfBuf_;
    TBuf<TPosition::VECCALC> prodGBuf_;
    TBuf<TPosition::VECCALC> prodUBuf_;
    TBuf<TPosition::VECCALC> accGBuf_;
    TBuf<TPosition::VECCALC> accUBuf_;
    TBuf<TPosition::VECCALC> accG16Buf_;
    TBuf<TPosition::VECCALC> accU16Buf_;
    TBuf<TPosition::VECCALC> negBuf_;
    TBuf<TPosition::VECCALC> eBuf_;
    TBuf<TPosition::VECCALC> sBuf_;
    TBuf<TPosition::VECCALC> onesBuf_;
    TBuf<TPosition::VECCALC> w1tBuf_;
    TBuf<TPosition::VECCALC> w2tBuf_;
    TQue<QuePosition::VECIN, 1> wblkQueue_;
    GlobalTensor<DT_INPUT_X> xGm_;
    GlobalTensor<DT_INPUT_X> w1Gm_;
    GlobalTensor<DT_INPUT_X> b1Gm_;
    GlobalTensor<DT_INPUT_X> w2Gm_;
    GlobalTensor<DT_INPUT_X> b2Gm_;
    GlobalTensor<DT_INPUT_X> outGm_;
    const SwiGluTilingData* td_ = nullptr;
};

template <class DT_INPUT_X, int Mode> __aicore__ inline KernelSwiGlu<DT_INPUT_X, Mode>::KernelSwiGlu()
{
}

template <class DT_INPUT_X, int Mode>
__aicore__ inline void KernelSwiGlu<DT_INPUT_X, Mode>::Init(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                                            GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                                            const SwiGluTilingData& tiling)
{
    td_ = &tiling;
    const uint64_t xElems = td_->dimensions.m * td_->dimensions.d;
    const uint64_t wElems = td_->dimensions.d * td_->dimensions.d;
    xGm_.SetGlobalBuffer((__gm__ DT_INPUT_X*)input_x, xElems);
    w1Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X*)weight_1, wElems);
    b1Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X*)bias_1, td_->dimensions.d);
    w2Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X*)weight_2, wElems);
    b2Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X*)bias_2, td_->dimensions.d);
    outGm_.SetGlobalBuffer((__gm__ DT_INPUT_X*)output, xElems);
    pipe_.InitBuffer(stageA_, 1, AlignUpBytes(td_->dimensions.dAlign * sizeof(DT_INPUT_X)));
    pipe_.InitBuffer(stageB_, 1, AlignUpBytes(td_->dimensions.dAlign * sizeof(DT_INPUT_X)));
    pipe_.InitBuffer(outQueue_, 1, AlignUpBytes(td_->dimensions.tileN * sizeof(DT_INPUT_X)));
    pipe_.InitBuffer(xfBuf_, AlignUpBytes(td_->dimensions.dAlign * sizeof(float)));
    pipe_.InitBuffer(wfBuf_, AlignUpBytes(td_->dimensions.dAlign * sizeof(float)));
    const uint32_t productElems =
        td_->dimensions.dAlign * (Mode != 0 && td_->dimensions.dAlign <= 256 ? kChunk16 : 1U);
    pipe_.InitBuffer(prodGBuf_, AlignUpBytes(productElems * sizeof(float)));
    pipe_.InitBuffer(prodUBuf_, AlignUpBytes(productElems * sizeof(float)));
    pipe_.InitBuffer(accGBuf_, AlignUpBytes(td_->dimensions.tileN * sizeof(float)));
    pipe_.InitBuffer(accUBuf_, AlignUpBytes(td_->dimensions.tileN * sizeof(float)));
    pipe_.InitBuffer(accG16Buf_, AlignUpBytes(td_->dimensions.tileN * sizeof(DT_INPUT_X)));
    pipe_.InitBuffer(accU16Buf_, AlignUpBytes(td_->dimensions.tileN * sizeof(DT_INPUT_X)));
    pipe_.InitBuffer(negBuf_, AlignUpBytes(td_->dimensions.tileN * sizeof(DT_INPUT_X)));
    pipe_.InitBuffer(eBuf_, AlignUpBytes(td_->dimensions.tileN * sizeof(DT_INPUT_X)));
    pipe_.InitBuffer(sBuf_, AlignUpBytes(td_->dimensions.tileN * sizeof(DT_INPUT_X)));
    pipe_.InitBuffer(onesBuf_, AlignUpBytes(td_->dimensions.tileN * sizeof(DT_INPUT_X)));
    if constexpr (Mode != 0) {
        pipe_.InitBuffer(w1tBuf_, AlignUpBytes(td_->dimensions.dAlign * kChunk16 * sizeof(float)));
        pipe_.InitBuffer(w2tBuf_, AlignUpBytes(td_->dimensions.dAlign * kChunk16 * sizeof(float)));
        pipe_.InitBuffer(wblkQueue_, 1, AlignUpBytes(kChunk16 * td_->dimensions.dAlign * sizeof(DT_INPUT_X)));
    }
}

template <class DT_INPUT_X, int Mode> __aicore__ inline void KernelSwiGlu<DT_INPUT_X, Mode>::Process()
{
    const uint64_t core = GetBlockIdx();
    if (core >= td_->dimensions.usedCoreNum) {
        return;
    }
    if constexpr (Mode == 2) {
        const uint32_t d = static_cast<uint32_t>(td_->dimensions.d);
        const uint32_t columns = Ceiling(d, kChunk16);
        if (columns == 0U) {
            return;
        }
        const uint64_t tasks = td_->dimensions.m * columns;
        for (uint64_t task = core; task < tasks; task += td_->dimensions.usedCoreNum) {
            const uint64_t row = task / columns;
            const uint32_t c0 = static_cast<uint32_t>(task % columns) * kChunk16;
            const uint32_t chunk = d - c0 < kChunk16 ? d - c0 : kChunk16;
            PrepareWt(c0, chunk, d);
            ProcessColTileT(row, c0, chunk, d);
        }
    } else if constexpr (Mode == 1) {
        const uint32_t d = static_cast<uint32_t>(td_->dimensions.d);
        for (uint32_t c0 = 0; c0 < d; c0 += kChunk16) {
            const uint32_t chunk = d - c0 < kChunk16 ? d - c0 : kChunk16;
            PrepareWt(c0, chunk, d);
            for (uint64_t row = core; row < td_->dimensions.m; row += td_->dimensions.usedCoreNum) {
                ProcessColTileT(row, c0, chunk, d);
            }
        }
    } else {
        for (uint64_t row = core; row < td_->dimensions.m; row += td_->dimensions.usedCoreNum) {
            ProcessRow(row);
        }
    }
}

template <class DT_INPUT_X, int Mode>
template <HardEvent event>
__aicore__ inline void KernelSwiGlu<DT_INPUT_X, Mode>::SyncPipe()
{
    const event_t id = static_cast<event_t>(pipe_.FetchEventID(event));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

template <class DT_INPUT_X, int Mode>
__aicore__ inline void KernelSwiGlu<DT_INPUT_X, Mode>::ProcessRow(uint64_t row)
{
    const uint32_t d = static_cast<uint32_t>(td_->dimensions.d);
    const uint32_t dAlign = td_->dimensions.dAlign;
    const uint32_t tileN = td_->dimensions.tileN;
    LocalTensor<DT_INPUT_X> xTile = stageA_.AllocTensor<DT_INPUT_X>();
    DataCopyExtParams xCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
    DataCopyPadExtParams<DT_INPUT_X> xPad{true, 0, static_cast<uint8_t>(dAlign - d),
                                          static_cast<DT_INPUT_X>(0)};
    DataCopyPad(xTile, xGm_[row * d], xCopy, xPad);
    stageA_.EnQue(xTile);
    xTile = stageA_.DeQue<DT_INPUT_X>();
    LocalTensor<float> xf = xfBuf_.Get<float>();
    Cast(xf, xTile, RoundMode::CAST_NONE, dAlign);
    stageA_.FreeTensor(xTile);
    for (uint32_t c0 = 0; c0 < d; c0 += tileN) {
        const uint32_t chunk = d - c0 < tileN ? d - c0 : tileN;
        ProcessColTile(row, c0, chunk, d, dAlign, xf);
    }
}

template <class DT_INPUT_X, int Mode>
template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8, class B9,
          class B10>
__aicore__ inline __attribute__((always_inline)) void
KernelSwiGlu<DT_INPUT_X, Mode>::SharedStage0f62bc2d85(B0& accG16, B1& accU16, B2& b2Tile, B3& c0, B4& chunk,
                                                      B5& d, B6& eBuf, B7& negBuf, B8& onesBuf, B9& row,
                                                      B10& sBuf)
{
    stageB_.FreeTensor(b2Tile);
    Duplicate(onesBuf, static_cast<DT_INPUT_X>(1.0f), chunk);
    Muls(negBuf, accG16, (DT_INPUT_X)(-1.0), chunk);
    Exp(eBuf, negBuf, chunk);
    Adds(eBuf, eBuf, (DT_INPUT_X)(1.0), chunk);
    Div(sBuf, onesBuf, eBuf, chunk);
    Mul(accG16, accG16, sBuf, chunk);
    LocalTensor<DT_INPUT_X> outTile = outQueue_.template AllocTensor<DT_INPUT_X>();
    Mul(outTile, accG16, accU16, chunk);
    outQueue_.EnQue(outTile);
    outTile = outQueue_.template DeQue<DT_INPUT_X>();
    DataCopyExtParams oCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
    DataCopyPad(outGm_[row * d + c0], outTile, oCopy);
    outQueue_.FreeTensor(outTile);
}

template <class DT_INPUT_X, int Mode>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11>
__aicore__ inline __attribute__((always_inline)) void
KernelSwiGlu<DT_INPUT_X, Mode>::ProcessColTileStagea7eea475(A0& row, A1& c0, A2& chunk, A3& d, A4& accG,
                                                            A5& accU, A6& accG16, A7& accU16, A8& negBuf,
                                                            A9& eBuf, A10& sBuf, A11& onesBuf)
{
    DataCopyExtParams biasCopyParams{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
    DataCopyPadExtParams<DT_INPUT_X> bPad{true, 0, static_cast<uint8_t>(AlignUp16(chunk) - chunk),
                                          static_cast<DT_INPUT_X>(0)};
    LocalTensor<DT_INPUT_X> b1Tile = stageA_.template AllocTensor<DT_INPUT_X>();
    LocalTensor<DT_INPUT_X> b2Tile = stageB_.template AllocTensor<DT_INPUT_X>();
    DataCopyPad(b1Tile, b1Gm_[c0], biasCopyParams, bPad);
    DataCopyPad(b2Tile, b2Gm_[c0], biasCopyParams, bPad);
    stageA_.EnQue(b1Tile);
    stageB_.EnQue(b2Tile);
    b1Tile = stageA_.template DeQue<DT_INPUT_X>();
    b2Tile = stageB_.template DeQue<DT_INPUT_X>();
    Cast(accG16, accG, RoundMode::CAST_RINT, chunk);
    Cast(accU16, accU, RoundMode::CAST_RINT, chunk);
    Add(accG16, accG16, b1Tile, chunk);
    stageA_.FreeTensor(b1Tile);
    Add(accU16, accU16, b2Tile, chunk);
    SharedStage0f62bc2d85(accG16, accU16, b2Tile, c0, chunk, d, eBuf, negBuf, onesBuf, row, sBuf);
}

template <class DT_INPUT_X, int Mode>
__aicore__ inline void
KernelSwiGlu<DT_INPUT_X, Mode>::ProcessColTile(uint64_t row, uint32_t c0, uint32_t chunk, uint32_t d,
                                               uint32_t dAlign, const LocalTensor<float>& xf)
{
    LocalTensor<float> wf = wfBuf_.Get<float>();
    LocalTensor<float> prodG = prodGBuf_.Get<float>();
    LocalTensor<float> prodU = prodUBuf_.Get<float>();
    LocalTensor<float> accG = accGBuf_.Get<float>();
    LocalTensor<float> accU = accUBuf_.Get<float>();
    LocalTensor<DT_INPUT_X> accG16 = accG16Buf_.Get<DT_INPUT_X>();
    LocalTensor<DT_INPUT_X> accU16 = accU16Buf_.Get<DT_INPUT_X>();
    LocalTensor<DT_INPUT_X> negBuf = negBuf_.Get<DT_INPUT_X>();
    LocalTensor<DT_INPUT_X> eBuf = eBuf_.Get<DT_INPUT_X>();
    LocalTensor<DT_INPUT_X> sBuf = sBuf_.Get<DT_INPUT_X>();
    LocalTensor<DT_INPUT_X> onesBuf = onesBuf_.Get<DT_INPUT_X>();
    DataCopyExtParams wCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
    DataCopyPadExtParams<DT_INPUT_X> wPad{true, 0, static_cast<uint8_t>(dAlign - d),
                                          static_cast<DT_INPUT_X>(0)};
    for (uint32_t j = 0; j < chunk; ++j) {
        const uint64_t wBase = static_cast<uint64_t>(c0 + j) * d;
        LocalTensor<DT_INPUT_X> w1 = stageA_.AllocTensor<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> w2 = stageB_.AllocTensor<DT_INPUT_X>();
        DataCopyPad(w1, w1Gm_[wBase], wCopy, wPad);
        DataCopyPad(w2, w2Gm_[wBase], wCopy, wPad);
        stageA_.EnQue(w1);
        stageB_.EnQue(w2);
        w1 = stageA_.DeQue<DT_INPUT_X>();
        w2 = stageB_.DeQue<DT_INPUT_X>();
        Cast(wf, w1, RoundMode::CAST_NONE, dAlign);
        Mul(prodG, xf, wf, dAlign);
        Cast(wf, w2, RoundMode::CAST_NONE, dAlign);
        Mul(prodU, xf, wf, dAlign);
        stageA_.FreeTensor(w1);
        stageB_.FreeTensor(w2);
        ReduceSum(accG[j], prodG, wf, dAlign);
        ReduceSum(accU[j], prodU, wf, dAlign);
    }
    ProcessColTileStagea7eea475(row, c0, chunk, d, accG, accU, accG16, accU16, negBuf, eBuf, sBuf, onesBuf);
}

template <class DT_INPUT_X, int Mode>
__aicore__ inline void
KernelSwiGlu<DT_INPUT_X, Mode>::PrepareOneWeight(const GlobalTensor<DT_INPUT_X>& weight,
                                                 const LocalTensor<float>& cached, uint32_t c0,
                                                 uint32_t chunk, uint32_t d)
{
    LocalTensor<DT_INPUT_X> wblk = wblkQueue_.AllocTensor<DT_INPUT_X>();
    DataCopyExtParams blkCopy{static_cast<uint16_t>(chunk), static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0,
                              0, 0};
    DataCopyPadExtParams<DT_INPUT_X> blkPad{true, 0, static_cast<uint8_t>(td_->dimensions.dAlign - d),
                                            static_cast<DT_INPUT_X>(0)};
    DataCopyPad(wblk, weight[static_cast<uint64_t>(c0) * d], blkCopy, blkPad);
    wblkQueue_.EnQue(wblk);
    wblk = wblkQueue_.DeQue<DT_INPUT_X>();
    Cast(cached, wblk, RoundMode::CAST_NONE, chunk * td_->dimensions.dAlign);
    wblkQueue_.FreeTensor(wblk);
}

template <class DT_INPUT_X, int Mode>
__aicore__ inline void KernelSwiGlu<DT_INPUT_X, Mode>::PrepareWt(uint32_t c0, uint32_t chunk, uint32_t d)
{
    PrepareOneWeight(w1Gm_, w1tBuf_.Get<float>(), c0, chunk, d);
    PrepareOneWeight(w2Gm_, w2tBuf_.Get<float>(), c0, chunk, d);
}

template <class DT_INPUT_X, int Mode>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16>
__aicore__ inline __attribute__((always_inline)) void
KernelSwiGlu<DT_INPUT_X, Mode>::ProcessColTileTStage89d99c96(A0& c0, A1& chunk, A2& dAlign, A3& xf, A4& wf,
                                                             A5& accG, A6& accU, A7& accG16, A8& accU16,
                                                             A9& w1t, A10& w2t, A11& prodG, A12& prodU,
                                                             A13& b1Tile, A14& biasCopyParams, A15& bPad,
                                                             A16& b2Tile)
{
    DataCopyPad(b2Tile, b2Gm_[c0], biasCopyParams, bPad);
    stageB_.EnQue(b2Tile);
    b2Tile = stageB_.template DeQue<DT_INPUT_X>();
    if (dAlign <= 256U) {
        const uint8_t rowStride = static_cast<uint8_t>(dAlign / 8U);
        const BinaryRepeatParams mulParams{1, 1, 1, rowStride, 0, rowStride};
        for (uint32_t k = 0; k < dAlign; k += 64U) {
            const uint32_t width = dAlign - k < 64U ? dAlign - k : 64U;
            Mul(prodG[k], xf[k], w1t[k], static_cast<uint64_t>(width), static_cast<uint8_t>(chunk),
                mulParams);
            Mul(prodU[k], xf[k], w2t[k], static_cast<uint64_t>(width), static_cast<uint8_t>(chunk),
                mulParams);
            if (k == 0) {
                WholeReduceSum(accG, prodG[k], width, chunk, 1, 1, rowStride);
                WholeReduceSum(accU, prodU[k], width, chunk, 1, 1, rowStride);
            } else {
                WholeReduceSum(wf, prodG[k], width, chunk, 1, 1, rowStride);
                Add(accG, accG, wf, chunk);
                WholeReduceSum(wf, prodU[k], width, chunk, 1, 1, rowStride);
                Add(accU, accU, wf, chunk);
            }
        }
    } else {
        for (uint32_t j = 0; j < chunk; ++j) {
            Mul(prodG, xf, w1t[j * dAlign], dAlign);
            Mul(prodU, xf, w2t[j * dAlign], dAlign);
            ReduceSum(accG[j], prodG, wf, dAlign);
            ReduceSum(accU[j], prodU, wf, dAlign);
        }
    }
    Cast(accG16, accG, RoundMode::CAST_RINT, chunk);
    Cast(accU16, accU, RoundMode::CAST_RINT, chunk);
    Add(accG16, accG16, b1Tile, chunk);
    stageA_.FreeTensor(b1Tile);
    Add(accU16, accU16, b2Tile, chunk);
}

template <class DT_INPUT_X, int Mode>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16>
__aicore__ inline __attribute__((always_inline)) void
KernelSwiGlu<DT_INPUT_X, Mode>::ProcessColTileTStagebeb32067(A0& row, A1& c0, A2& chunk, A3& d, A4& dAlign,
                                                             A5& xf, A6& wf, A7& accG, A8& accU, A9& accG16,
                                                             A10& accU16, A11& negBuf, A12& eBuf, A13& sBuf,
                                                             A14& onesBuf, A15& w1t, A16& w2t)
{
    LocalTensor<float> prodG = prodGBuf_.template Get<float>();
    LocalTensor<float> prodU = prodUBuf_.template Get<float>();
    LocalTensor<DT_INPUT_X> xTile = stageA_.template AllocTensor<DT_INPUT_X>();
    DataCopyExtParams xCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
    DataCopyPadExtParams<DT_INPUT_X> xPad{true, 0, static_cast<uint8_t>(dAlign - d),
                                          static_cast<DT_INPUT_X>(0)};
    DataCopyPad(xTile, xGm_[row * d], xCopy, xPad);
    stageA_.EnQue(xTile);
    xTile = stageA_.template DeQue<DT_INPUT_X>();
    Cast(xf, xTile, RoundMode::CAST_NONE, dAlign);
    stageA_.FreeTensor(xTile);
    LocalTensor<DT_INPUT_X> b1Tile = stageA_.template AllocTensor<DT_INPUT_X>();
    DataCopyExtParams biasCopyParams{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
    DataCopyPadExtParams<DT_INPUT_X> bPad{true, 0, static_cast<uint8_t>(AlignUp16(chunk) - chunk),
                                          static_cast<DT_INPUT_X>(0)};
    DataCopyPad(b1Tile, b1Gm_[c0], biasCopyParams, bPad);
    stageA_.EnQue(b1Tile);
    b1Tile = stageA_.template DeQue<DT_INPUT_X>();
    LocalTensor<DT_INPUT_X> b2Tile = stageB_.template AllocTensor<DT_INPUT_X>();
    ProcessColTileTStage89d99c96(c0, chunk, dAlign, xf, wf, accG, accU, accG16, accU16, w1t, w2t, prodG,
                                 prodU, b1Tile, biasCopyParams, bPad, b2Tile);
    SharedStage0f62bc2d85(accG16, accU16, b2Tile, c0, chunk, d, eBuf, negBuf, onesBuf, row, sBuf);
}

template <class DT_INPUT_X, int Mode>
__aicore__ inline void KernelSwiGlu<DT_INPUT_X, Mode>::ProcessColTileT(uint64_t row, uint32_t c0,
                                                                       uint32_t chunk, uint32_t d)
{
    const uint32_t dAlign = td_->dimensions.dAlign;
    LocalTensor<float> xf = xfBuf_.Get<float>();
    LocalTensor<float> wf = wfBuf_.Get<float>();
    LocalTensor<float> accG = accGBuf_.Get<float>();
    LocalTensor<float> accU = accUBuf_.Get<float>();
    LocalTensor<DT_INPUT_X> accG16 = accG16Buf_.Get<DT_INPUT_X>();
    LocalTensor<DT_INPUT_X> accU16 = accU16Buf_.Get<DT_INPUT_X>();
    LocalTensor<DT_INPUT_X> negBuf = negBuf_.Get<DT_INPUT_X>();
    LocalTensor<DT_INPUT_X> eBuf = eBuf_.Get<DT_INPUT_X>();
    LocalTensor<DT_INPUT_X> sBuf = sBuf_.Get<DT_INPUT_X>();
    LocalTensor<DT_INPUT_X> onesBuf = onesBuf_.Get<DT_INPUT_X>();
    LocalTensor<float> w1t = w1tBuf_.Get<float>();
    LocalTensor<float> w2t = w2tBuf_.Get<float>();
    ProcessColTileTStagebeb32067(row, c0, chunk, d, dAlign, xf, wf, accG, accU, accG16, accU16, negBuf, eBuf,
                                 sBuf, onesBuf, w1t, w2t);
}
