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
    __aicore__ inline void ProcessLargeSetFlagPhase9(uint64_t &base, uint32_t &bw, uint32_t &bwp, uint32_t &h,
                                                     uint32_t &kb, uint32_t &vMte2Pend) {

        if (vMte2Pend != 0U) {
            WaitFlag<HardEvent::V_MTE2>(::kEvVMte2);
            vMte2Pend = 0;
        }
        CopyIn2D(panel_, base + static_cast<uint64_t>(kb) * s_ + kb, h, bwp, s_, kPanel);
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        SetFlag<HardEvent::S_V>(::kEvSV);
        WaitFlag<HardEvent::S_V>(::kEvSV);
        for (uint32_t j = 0; j < bw; ++j) {
            Adds(coloffI_, tabI_[0], static_cast<int32_t>(j * sizeof(float)), h);
            Gather(panelCM_[j * ::kMaxN], panel_, coloffU_, 0U, h);
        }
    }
    __aicore__ inline void ProcessLargeSetMaskPhase10(uint32_t &cnt) {

        uint32_t nCh = (cnt + 127U) >> 7;
        ReduceColumnChunks64(nCh, cnt);
    }

    __aicore__ inline void ProcessLargevmulsPhase11(uint32_t &W, uint64_t &base, uint32_t &bw, int32_t &ex, uint32_t &h,
                                                    uint32_t &kb, uint32_t &l21nz, float &prod, float &sign,
                                                    uint32_t &singular, uint32_t &swapPend, uint32_t &tb,
                                                    uint32_t &wpad) {

        for (uint32_t t = 0; t < bw; ++t) {
            uint32_t tA = t & ~(::kAlign - 1);
            uint32_t j0 = t - tA;
            uint32_t cnt = h - tA;
            LDET_MASK_SET(cnt);
            vabs(reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(panelCM_[t * ::kMaxN + tA].GetPhyAddr()), 1, 1, 1, 8, 8);
            if (j0 != 0U) {
                AscendCUtils::SetMask<float>(0, j0);
                vmuls(reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr()), 0.0f, 1, 1, 1, 8, 8);
            }
            uint32_t segN = (cnt + 63U) / 64U;
            ProcessLargeSetMaskPhase10(cnt);
            uint32_t sExact = segN;
            if (bw < h) {
                uint32_t sCross = (bw - tA) >> 6;
                sExact = ((bw - tA) & 63U) != 0U ? sCross + 1U : sCross;
            }
            SetFlag<HardEvent::V_S>(::kEvVS);
            WaitFlag<HardEvent::V_S>(::kEvVS);
            float maxv = 0.0f;
            uint32_t p = tA;
            ProcessLargeLDET_RM_PAIR_OFFPhase1(bw, h, l21nz, maxv, p, sExact, segN, t, tA);
            if (maxv == 0.0f) {
                singular = 1;
                break;
            }
            ProcessLargeSetFlagPhase8(W, base, bw, cnt, ex, kb, p, prod, sign, swapPend, t, tA, tb, wpad);
        }
        if (swapPend != 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
            swapPend = 0;
        }
    }
    __aicore__ inline void ProcessLarge() {
        uint32_t vMte2Pend = 0;
        for (uint32_t m = 0; m < cnt_; ++m) {
            uint64_t g = static_cast<uint64_t>(start_ + m);
            WsInit(g);
            uint64_t base = g * nnS_;
            float sign = 1.0f;
            float prod = 1.0f;
            int32_t ex = 0;
            uint32_t singular = 0;
            for (uint32_t kb = 0; kb < n_ && singular == 0U; kb += kPanel) {
                uint32_t h = n_ - kb;
                uint32_t bw = h < kPanel ? h : kPanel;
                uint32_t bwp = (bw + ::kAlign - 1) / ::kAlign * ::kAlign;
                uint32_t tb = bw & ~(::kAlign - 1);
                uint32_t W = n_ - kb - tb;
                uint32_t wpad = (W + ::kAlign - 1) / ::kAlign * ::kAlign;
                uint32_t l21nz = 0;
                uint32_t swapPend = 0;
                ProcessLargeSetFlagPhase9(base, bw, bwp, h, kb, vMte2Pend);
                ProcessLargevmulsPhase11(W, base, bw, ex, h, kb, l21nz, prod, sign, singular, swapPend, tb, wpad);
                uint32_t rBase = kb + bw;
                uint32_t nTiles = 0;
                ProcessLargeSetFlagPhase7(W, base, bw, kb, l21nz, m, nTiles, rBase, singular, tb, vMte2Pend, wpad);
            }
            float detVal = (singular != 0U) ? 0.0f : sign * ProdFinal(prod, ex);
            res_.SetValue(m, detVal);
        }
    }
