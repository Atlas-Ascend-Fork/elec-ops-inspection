/*
 * Copyright (c) 2026 Lesem.
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

namespace AscendC::SwiGluKernel {

__aicore__ inline void BindSwiGluInputs(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GlobalTensor<half> &xTensor, GlobalTensor<half> &w1Tensor, GlobalTensor<half> &b1Tensor,
    GlobalTensor<half> &w2Tensor, GlobalTensor<half> &b2Tensor, GlobalTensor<half> &yTensor) {
    xTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
    w1Tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w1));
    w2Tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w2));
    b1Tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b1));
    b2Tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b2));
    yTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
}

__aicore__ inline void CopyCubeTileNz(uint32_t srcStride, LocalTensor<half> dst, GlobalTensor<half> src,
    uint32_t rows, uint32_t cols) {
    Nd2NzParams p{}; p.ndNum=1; p.nValue=rows; p.dValue=cols;
    p.srcDValue=srcStride; p.dstNzC0Stride=(rows+15U)/16U*16U;
    p.dstNzNStride=1; DataCopy(dst,src,p);
}

__aicore__ inline void LoadCubeA(uint32_t k, LocalTensor<half> dst,LocalTensor<half> src,uint32_t mc) {
    const uint32_t am=(mc+15U)/16U*16U;
    LoadData2DParams p{}; p.repeatTimes=k/16U;
    p.srcStride=am/16U; p.ifTranspose=false;
    for(uint32_t mb=0;mb<am/16U;++mb)
        LoadData(dst[mb*k*16U],src[mb*256U],p);
}

__aicore__ inline void LoadCubeB(uint32_t n, uint32_t k, LocalTensor<half> dst,LocalTensor<half> src) {
    // Physical W is (output channel,K); its NZ rows are N, not K.
    LoadData2DParams p{}; p.repeatTimes=n/16U;
    p.srcStride=1; p.ifTranspose=false;
    for(uint32_t kb=0;kb<k/16U;++kb) {
        uint32_t off=kb*n*16U; LoadData(dst[off],src[off],p);
    }
}

__aicore__ inline void CopyCubeBias(const SwiGluWorkTiling &t, LocalTensor<half> b1,LocalTensor<half> b2,uint32_t n0, GlobalTensor<half> b1Src, GlobalTensor<half> b2Src) {
    const uint32_t nc=t.dim-n0<t.tileN ? t.dim-n0 : t.tileN;
        if(nc==t.tileN) {
            DataCopy(b1,b1Src[n0],t.tileN); DataCopy(b2,b2Src[n0],t.tileN);
        } else {
            DataCopyExtParams bi{1,nc*2U,0,0,0};
            DataCopyPadExtParams<half> bp{true,0,
                static_cast<uint8_t>(t.tileN-nc),static_cast<half>(0)};
            DataCopyPad(b1,b1Src[n0],bi,bp); DataCopyPad(b2,b2Src[n0],bi,bp);
        }
}


__aicore__ inline void InitDoubleInputCubeArena(uint32_t mk, uint32_t nk, uint32_t area,
    LocalTensor<half> &a1, LocalTensor<half> &b11, LocalTensor<half> &b12,
    LocalTensor<half> &a0, LocalTensor<half> &b0, LocalTensor<float> &c0) {
    a1 = LocalTensor<half>(TPosition::A1, 0, 2U * mk);
    b11 = LocalTensor<half>(TPosition::B1, 4U * mk, nk);
    b12 = LocalTensor<half>(TPosition::B1, 4U * mk + 2U * nk, nk);
    a0 = LocalTensor<half>(TPosition::A2, 0, mk);
    b0 = LocalTensor<half>(TPosition::B2, 0, 2U * nk);
    c0 = LocalTensor<float>(TPosition::CO1, 0, 2U * area);
}

__aicore__ inline void InitCubeVectorArena(uint32_t area, uint32_t tileN,
    LocalTensor<half> &c1, LocalTensor<half> &c2, LocalTensor<float> &z,
    LocalTensor<half> &b1, LocalTensor<half> &b2) {
    c1 = LocalTensor<half>(TPosition::VECCALC, 0, 2U * area);
    c2 = LocalTensor<half>(TPosition::VECCALC, 4U * area, area);
    z = LocalTensor<float>(TPosition::VECCALC, 6U * area, area);
    b1 = LocalTensor<half>(TPosition::VECCALC, 10U * area, tileN);
    b2 = LocalTensor<half>(TPosition::VECCALC, 10U * area + 2U * tileN, tileN);
}

__aicore__ inline void CopyPaddedCubeInput(LocalTensor<half> dst, GlobalTensor<half> src,
    uint32_t rows, uint32_t tileM, uint32_t dim, uint32_t k) {
    if (rows < tileM) {
        InitConstValue(dst, InitConstValueParams<half>{
            1, static_cast<uint16_t>(tileM * k / 16U), 0, static_cast<half>(0)});
    }
    CopyCubeTileNz(dim, dst, src, rows, dim);
}

__aicore__ inline void ProjectResidentCube(LocalTensor<half> a, LocalTensor<half> b,
    LocalTensor<half> right, LocalTensor<float> c, GlobalTensor<half> out,
    uint32_t rows, uint32_t n, uint32_t k, uint32_t branch, uint32_t job,
    TEventID loadReady, TEventID inputFree, uint32_t readyFlag) {
    if (job == 0U) LoadCubeB(n, k, b, right);
    // Each new A0 needs the forward event. Resident B0 is loaded only once.
    if (branch == 0U || job == 0U) {
        SetFlag<HardEvent::MTE1_M>(loadReady);
        WaitFlag<HardEvent::MTE1_M>(loadReady);
    }
    MmadParams p{};
    p.m = rows;
    p.n = n;
    p.k = k;
    p.cmatrixInitVal = true;
    p.unitFlag = 3;
    SetMMRowMajor();
    Mmad(c, a, b, p);
    FixpipeParamsV220 f{};
    f.nSize = n;
    f.mSize = rows;
    f.unitFlag = 3;
    f.srcStride = rows;
    f.dstStride = n;
    f.ndNum = 1;
    f.quantPre = QuantMode_t::F322F16;
    Fixpipe<half, float>(out, c, f);
    if (branch == 1U) {
        SetFlag<HardEvent::M_MTE1>(inputFree);
        WaitFlag<HardEvent::M_MTE1>(inputFree);
    }
    CrossCoreSetFlag<2, PIPE_FIX>(readyFlag);
}

} // namespace AscendC::SwiGluKernel
