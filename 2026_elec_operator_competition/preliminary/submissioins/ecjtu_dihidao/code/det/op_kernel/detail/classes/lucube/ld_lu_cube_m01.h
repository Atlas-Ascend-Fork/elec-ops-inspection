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
// Included in the LuCube class; preserves its member scope and access.
    __aicore__ inline void Run(GM_ADDR workspace, GM_ADDR neg, GM_ADDR u, GM_ADDR flags, GM_ADDR t, uint32_t n,
                               uint32_t epoch) {
        RunStandard46<false, false>(workspace, neg, u, flags, t, n, n, epoch);
    }

    __aicore__ inline void Shared24ScalarUpdateStage9(uint32_t &stride, GM_ADDR &t, GM_ADDR &u) {
        auto src = reinterpret_cast<const __gm__ int32_t *>(t);
        auto dst = reinterpret_cast<int32_t *>(&tiling);
        for (uint32_t i = 0; i < sizeof(tiling) / 4; ++i)
            dst[i] = src[i];
        mm.Init(&tiling, &pipe);
        mm.SetOrgShape(256, 512, 32, 32, stride);
        mm.SetHF32(false, 0);
        U.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(u));
    }
    __aicore__ inline void RunRange(GM_ADDR workspace, GM_ADDR neg, GM_ADDR u, GM_ADDR flags, GM_ADDR t, uint32_t n,
                                    uint32_t stride, uint32_t epoch) {
        RunStandard46<true, false>(workspace, neg, u, flags, t, n, stride, epoch);
    }
    __aicore__ inline void RunTail208(GM_ADDR workspace, GM_ADDR neg, GM_ADDR u, GM_ADDR flags, GM_ADDR t, uint32_t n,
                                      uint32_t stride, uint32_t epoch) {
        RunStandard46<true, true>(workspace, neg, u, flags, t, n, stride, epoch);
    }
    template <auto stride>
    __aicore__ inline void SharedSetFlagStage6(uint32_t &cores, uint32_t &curReg, uint32_t &kb, uint32_t &n,
                                               GM_ADDR &neg, uint32_t &rank, uint32_t &singular, uint32_t &tw,
                                               GM_ADDR &workspace) {
        uint32_t bw = n - kb - tw;
        uint32_t band = LuN256Span(tw, cores), r0 = rank * band, rows = r0 < tw ? tw - r0 : 0U;
        if (rows > band)
            rows = band;
        if (singular == 0U && tw > 0U && rows > 0U) {
            for (uint32_t offset = 0; offset < rows; offset += 64U) {
                uint32_t count = rows - offset;
                if (count > 64U)
                    count = 64U;
                L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(neg) + r0 + offset);
                D.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) +
                                  static_cast<uint64_t>(curReg) * n * stride +
                                  static_cast<uint64_t>(kb + bw + r0 + offset) * stride + kb + bw);
                mm.SetTail(count, tw, bw);
                mm.SetTensorA(L, true);
                mm.SetTensorB(U, false);
                mm.IterateAll(D, 1);
                mm.End();
                auto e = pipe.FetchEventID(HardEvent::FIX_S);
                SetFlag<HardEvent::FIX_S>(e);
                WaitFlag<HardEvent::FIX_S>(e);
            }
        }
    }

    __aicore__ inline void RunN256Group(GM_ADDR workspace, GM_ADDR neg, GM_ADDR u, GM_ADDR flags, GM_ADDR t, uint32_t n,
                                        uint32_t epoch, uint32_t rank, uint32_t cores) {
        RunGroup54<false>(workspace, neg, u, flags, t, n, epoch, rank, cores);
    }

    __aicore__ inline void RunWideGroup(GM_ADDR workspace, GM_ADDR neg, GM_ADDR u, GM_ADDR flags, GM_ADDR t, uint32_t n,
                                        uint32_t epoch, uint32_t rank, uint32_t cores) {
        RunGroup54<true>(workspace, neg, u, flags, t, n, epoch, rank, cores);
    }

    template <bool RANGE, bool TAIL>
    __aicore__ inline void RunStandard46(GM_ADDR workspace, GM_ADDR neg, GM_ADDR u, GM_ADDR flags, GM_ADDR t,
                                         uint32_t n, uint32_t stride, uint32_t epoch) {
        uint32_t rank = GetBlockIdx();
        LuControlWait(flags + 12U * 64U);
        auto control = reinterpret_cast<volatile __gm__ uint32_t *>(flags + 12U * 64U);
        if (control[1] == 0U)
            return;
        Shared24ScalarUpdateStage9(stride, t, u);
        for (uint32_t kb = 0; kb + (TAIL ? 208U : 160U) < n;
             kb += (TAIL ? (n == 256U && kb == 32U ? 16U : 32U) : 32U)) {
            uint32_t token = epoch * 32U + kb / 32U + 1U;
            LuPoll(flags + 2U * rank * 64U, token);
            LuPoll(flags + (2U * rank + 1U) * 64U, token);
            LuPoll(flags, token);
            auto metadata = reinterpret_cast<volatile __gm__ uint32_t *>(flags);
            uint32_t singular = metadata[1], curReg = metadata[2], tw = metadata[3];
            uint32_t bw = TAIL ? n - kb - tw : 32U;
            uint32_t band = LuSpan(tw), r0 = rank * band, rows = r0 < tw ? tw - r0 : 0U;
            if (rows > band)
                rows = band;
            if (singular == 0U && tw > 0U && rows > 0U) {
                L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(neg) + r0);
                if constexpr (RANGE) {
                    D.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) +
                                      static_cast<uint64_t>(curReg) * n * stride +
                                      static_cast<uint64_t>(kb + bw + r0) * stride + kb + bw);
                } else {
                    D.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + curReg * n * n +
                                      (kb + 32U + r0) * n + kb + 32U);
                }
                RunStandardUpdate46(rows, tw, bw);
            }
            LuPublish(flags + (8U + rank) * 64U, token);
            if (singular != 0U)
                break;
        }
    }
    __aicore__ inline void RunStandardUpdate46(uint32_t rows, uint32_t tw, uint32_t bw) {
        mm.SetTail(rows, tw, bw);
        mm.SetTensorA(L, true);
        mm.SetTensorB(U, false);
        mm.IterateAll(D, 1);
        mm.End();
        auto e = pipe.FetchEventID(HardEvent::FIX_S);
        SetFlag<HardEvent::FIX_S>(e);
        WaitFlag<HardEvent::FIX_S>(e);
    }

    template <bool WIDE>
    __aicore__ inline void RunGroup54(GM_ADDR workspace, GM_ADDR neg, GM_ADDR u, GM_ADDR flags, GM_ADDR t, uint32_t n,
                                      uint32_t epoch, uint32_t rank, uint32_t cores) {
        constexpr uint32_t stride = WIDE ? 512U : 256U;
        LuControlWait(flags + 12U * 64U, epoch);
        auto control = reinterpret_cast<volatile __gm__ uint32_t *>(flags + 12U * 64U);
        if (control[1] == 0U)
            return;
        auto source = reinterpret_cast<const __gm__ int32_t *>(t);
        auto target = reinterpret_cast<int32_t *>(&tiling);
        for (uint32_t i = 0; i < sizeof(tiling) / 4; ++i)
            target[i] = source[i];
        mm.Init(&tiling, &pipe);
        mm.SetOrgShape(WIDE ? 512 : 256, 512, 32, 32, stride);
        mm.SetHF32(false, 0);
        U.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(u));
        for (uint32_t kb = 0; kb + 208U < n;
             kb += (WIDE ? (n - kb <= 224U ? 16U : 32U) : (n == 256U && kb == 32U ? 16U : 32U))) {
            uint32_t token = epoch * 32U + kb / 32U + 1U;
            LuPoll(flags + 2U * rank * 64U, token);
            LuPoll(flags + (2U * rank + 1U) * 64U, token);
            LuPoll(flags, token);
            auto metadata = reinterpret_cast<volatile __gm__ uint32_t *>(flags);
            uint32_t singular = metadata[1], curReg = metadata[2], tw = metadata[3];
            SharedSetFlagStage6<stride>(cores, curReg, kb, n, neg, rank, singular, tw, workspace);
            LuPublish(flags + (8U + rank) * 64U, token);
            if (singular != 0U)
                break;
        }
    }
