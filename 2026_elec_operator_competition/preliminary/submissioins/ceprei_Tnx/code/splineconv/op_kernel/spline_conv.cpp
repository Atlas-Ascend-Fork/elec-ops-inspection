/*
 * Copyright (c) 2026 ceprei_Tnx.
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
/* 910B: target-owned CSR; adaptive compensated coefficient aggregation. */
#include "kernel_operator.h"
#include "spline_conv_tiling.h"
#include "tiling_key_spline_conv.h"
#include <type_traits>
#include "lib/matmul_intf.h"
using namespace matmul;
template <bool CUBE> struct SplineMatmul {
    using Operand = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false>;
    using Type = Matmul<Operand, Operand, Operand, Operand, CFG_MDL>;
};
template <> struct SplineMatmul<false> {
    struct Type {};
};
using namespace AscendC;

template <class T, bool USE_CUBE> class KernelSplineConv {
  public:
    typename SplineMatmul<USE_CUBE>::Type mm;
    __aicore__ inline uint32_t Core() {
        if constexpr (USE_CUBE)
            return GetBlockIdx() / GetTaskRation();
        return GetBlockIdx();
    }
    __aicore__ inline void InitBuffers() {
        const uint32_t nodeBytes = ((t_.nodesPerCore + 8) / 8) * 32;
        /* 行向量暂存容量取 max(C_in, C_out)，另加 8 个元素的向量
           指令越界余量。 */
        const uint32_t widest = t_.cIn > t_.cOut ? t_.cIn : t_.cOut;
        const uint32_t elemCap = ((widest + 7) / 8) * 8 + 8;
        const uint32_t nslot = t_.weightSlots > 0 ? t_.weightSlots : 1;

        pipe_->InitBuffer(offsetBuf_, nodeBytes);
        pipe_->InitBuffer(cursorBuf_, nodeBytes);
        pipe_->InitBuffer(dstBuf_, SPLINE_EDGE_TILE * sizeof(int64_t));
        pipe_->InitBuffer(srcBuf_, SPLINE_EDGE_TILE * sizeof(int64_t));
        pipe_->InitBuffer(recBuf_, SPLINE_EDGE_TILE * SPLINE_RECORD_WORDS * sizeof(float));
        pipe_->InitBuffer(stageBuf_, SPLINE_EDGE_TILE * SPLINE_STAGE_STRIDE * sizeof(float));
        pipe_->InitBuffer(recOffBuf_, SPLINE_EDGE_TILE * sizeof(uint32_t));
        pipe_->InitBuffer(prawBuf_, SPLINE_EDGE_TILE * 3 * sizeof(T));
        pipe_->InitBuffer(pendBuf_, SPLINE_EDGE_TILE * 3 * sizeof(float));
        pipe_->InitBuffer(pfBuf_, 32);
        pipe_->InitBuffer(xBuf_, elemCap * sizeof(float));
        pipe_->InitBuffer(rawVecBuf_, ((elemCap * sizeof(T) + 31) / 32) * 32);
        pipe_->InitBuffer(vecBuf_, 6 * t_.pitch * sizeof(float));
        pipe_->InitBuffer(idxBuf_, SPLINE_MAX_SLOTS * sizeof(uint32_t));
        pipe_->InitBuffer(bListBuf_, SPLINE_MAX_SLOTS * sizeof(float));
        if constexpr (std::is_same<T, float>::value) {
            /* fp32：批量槽本身就是 float 布局，搬运直接写进去。 */
            pipe_->InitBuffer(wfBuf_, nslot * t_.cIn * t_.pitch * sizeof(float));
        } else {
            /* fp16/bf16：原始布局按批预取，float 侧也按批开槽 —— 这样
               一个批次里的 cnt 个矩阵能同时以 float 形态存在，内层循环
               就可以「通道在外、槽在内」，x 的标量只读 C_in 次。 */
            pipe_->InitBuffer(wrawBuf_, nslot * t_.cIn * t_.pitch * sizeof(T));
            pipe_->InitBuffer(wfBuf_, nslot * t_.cIn * t_.pitch * sizeof(float));
        }
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                GM_ADDR workspace, const SplineConvTilingData &t, TPipe *pipeIn) {
        t_ = t;
        pipe_ = pipeIn;
        start_ = 0;
        rows_ = 0;
        lastSrc_ = kNoSrc;
        x_.SetGlobalBuffer((__gm__ T *)x);
        edge_.SetGlobalBuffer((__gm__ int64_t *)edges);
        pseudo_.SetGlobalBuffer((__gm__ T *)pseudo);
        w_.SetGlobalBuffer((__gm__ T *)weight);
        out_.SetGlobalBuffer((__gm__ T *)out);
        map_.SetGlobalBuffer((__gm__ float *)GetUserWorkspace(workspace) +
                             uint64_t(Core()) * (uint64_t(t_.mapStride) + t_.cubeWsElems));

        InitBuffers();

        off_ = offsetBuf_.Get<uint32_t>();
        cur_ = cursorBuf_.Get<uint32_t>();
        dst_ = dstBuf_.Get<int64_t>();
        src_ = srcBuf_.Get<int64_t>();
        rec_ = recBuf_.Get<float>();
        stage_ = stageBuf_.Get<float>();
        recOff_ = recOffBuf_.Get<uint32_t>();
        pend_ = pendBuf_.Get<float>();
        rawVec_ = rawVecBuf_.Get<T>();
        xf_ = xBuf_.Get<float>();
        pf_ = pfBuf_.Get<float>();
        idxList_ = idxBuf_.Get<uint32_t>();
        bList_ = bListBuf_.Get<float>();
        auto v = vecBuf_.Get<float>();
        sum_ = v;
        sc_ = v[t_.pitch];
        msg_ = v[2 * t_.pitch];
        y_ = v[3 * t_.pitch];
        z_ = v[4 * t_.pitch];
        wSrc_ = wfBuf_.Get<float>();
        if constexpr (!std::is_same<T, float>::value) {
            wRaw_ = wrawBuf_.Get<T>();
            praw_ = prawBuf_.Get<T>();
        }
        BuildSlotTables();
    }

    __aicore__ inline void Process() {
        /* 权重矩阵预热到 UB（只在 K 个矩阵全部放得下时由 Host 打
           开）。 */
        if (t_.cachedK) {
            if constexpr (std::is_same<T, float>::value) {
                Sync<HardEvent::V_MTE2>();
                Sync<HardEvent::S_MTE2>();
                DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(t_.pitch - t_.cOut), {}};
                const uint32_t totalRows = t_.cachedK * t_.cIn;
                for (uint32_t row = 0; row < totalRows; row += 4095) {
                    DataCopyExtParams cp{static_cast<uint16_t>(Min(4095, totalRows - row)),
                                         static_cast<uint32_t>(t_.cOut * sizeof(float)), 0, 0, 0};
                    DataCopyPad(cache_[uint64_t(row) * t_.pitch], w_[uint64_t(row) * t_.cOut], cp, pad);
                }
                Sync<HardEvent::MTE2_V>();
                Sync<HardEvent::MTE2_S>();
            } else {
                for (uint32_t k = 0; k < t_.cachedK; ++k) {
                    Sync<HardEvent::V_MTE2>();
                    Sync<HardEvent::S_MTE2>();
                    IssueWeight(w_, uint64_t(k) * t_.cIn * t_.cOut, wRaw_);
                    Sync<HardEvent::MTE2_V>();
                    Cast(cache_[uint64_t(k) * t_.cIn * t_.pitch], wRaw_, RoundMode::CAST_NONE, t_.cIn * t_.pitch);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
        const uint32_t step = t_.cores * t_.nodesPerCore;
        if (step == 0)
            return;
        for (uint32_t base = 0; base < t_.n; base += step) {
            start_ = base + Core() * t_.nodesPerCore;
            rows_ = start_ < t_.n ? Min(t_.nodesPerCore, t_.n - start_) : 0;
            /* start_ 只增不减，后续轮次同样没有节点 */
            if (rows_ == 0)
                break;
            /* 上一轮 compute 对 map_ 的读（MTE2）必须先于本轮
               scatter 的写（MTE3）。 */
            Sync<HardEvent::MTE2_MTE3>();
            BuildIncoming();
            ProcessRange();
        }
    }

  private:
    static constexpr uint32_t kNoSrc = 0xFFFFFFFFu;

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) {
        return a < b ? a : b;
    }
    template <HardEvent event> __aicore__ inline void Sync() {
        SetFlag<event>(EVENT_ID0);
        WaitFlag<event>(EVENT_ID0);
    }
    template <class U>
    __aicore__ inline void Read(LocalTensor<U> dst, GlobalTensor<U> gm, uint64_t offset, uint32_t count) {
        /* 显式跨流水事件：同流水内的 barrier 不足以同步标量与
           MTE2。 */
        Sync<HardEvent::V_MTE2>();
        Sync<HardEvent::S_MTE2>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(U)), 0, 0, 0};
        DataCopyPadExtParams<U> pad{false, 0, 0, {}};
        DataCopyPad(dst, gm[offset], cp, pad);
        Sync<HardEvent::MTE2_S>();
        Sync<HardEvent::MTE2_V>();
    }
    /* 把本 tile 的伪坐标搬进 UB 并统一成 float（dim <= 3）。 */
    __aicore__ inline void ReadPseudo(uint32_t b, uint32_t count) {
        if constexpr (std::is_same<T, float>::value) {
            Read(pend_, pseudo_, uint64_t(b) * t_.dim, count * t_.dim);
        } else {
            Read(praw_, pseudo_, uint64_t(b) * t_.dim, count * t_.dim);
            Cast(pend_, praw_, RoundMode::CAST_NONE, count * t_.dim);
            PipeBarrier<PIPE_V>();
            Sync<HardEvent::V_S>();
        }
    }

    // Independent input buffers: issue both DMA descriptors before waiting.
    __aicore__ inline void ReadSourcePseudo(uint32_t b, uint32_t count) {
        Sync<HardEvent::V_MTE2>();
        Sync<HardEvent::S_MTE2>();
        DataCopyExtParams sourceCopy{1, static_cast<uint32_t>(count * sizeof(int64_t)), 0, 0, 0};
        DataCopyPadExtParams<int64_t> sourcePad{false, 0, 0, {}};
        DataCopyPad(src_, edge_[b], sourceCopy, sourcePad);
        DataCopyExtParams pseudoCopy{1, static_cast<uint32_t>(count * t_.dim * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pseudoPad{false, 0, 0, {}};
        if constexpr (std::is_same<T, float>::value)
            DataCopyPad(pend_, pseudo_[uint64_t(b) * t_.dim], pseudoCopy, pseudoPad);
        else
            DataCopyPad(praw_, pseudo_[uint64_t(b) * t_.dim], pseudoCopy, pseudoPad);
        Sync<HardEvent::MTE2_S>();
        Sync<HardEvent::MTE2_V>();
        if constexpr (!std::is_same<T, float>::value) {
            Cast(pend_, praw_, RoundMode::CAST_NONE, count * t_.dim);
            PipeBarrier<PIPE_V>();
            Sync<HardEvent::V_S>();
        }
    }

    /* 为本核节点区间建立 CSR 索引：(source, 伪坐标) 记录按目
       标节点连续排布。 */
    __aicore__ inline void CountIncoming() {
        for (uint32_t i = 0; i <= rows_; ++i)
            off_.SetValue(i, 0);
        /* 第一趟：统计每个目标节点的入边数。 */
        for (uint32_t b = 0; b < t_.e; b += SPLINE_EDGE_TILE) {
            const uint32_t count = Min(SPLINE_EDGE_TILE, t_.e - b);
            Read(dst_, edge_, uint64_t(t_.e) + b, count);
            for (uint32_t j = 0; j < count; ++j) {
                const int64_t d = dst_.GetValue(j);
                if (d >= start_ && d < int64_t(start_) + rows_) {
                    const uint32_t i = static_cast<uint32_t>(d) - start_;
                    off_.SetValue(i, off_.GetValue(i) + 1);
                }
            }
        }
        uint32_t total = 0;
        for (uint32_t i = 0; i < rows_; ++i) {
            const uint32_t c = off_.GetValue(i);
            off_.SetValue(i, total);
            cur_.SetValue(i, total);
            total += c;
        }
        off_.SetValue(rows_, total);
    }

    __aicore__ inline void ScatterIncoming() { /* 第二趟：写记录。整 tile 的记录先在 UB 攒好，再一次性下
    发一批 DMA， 同步只做一对，而不是每条边一对。 */
        for (uint32_t b = 0; b < t_.e; b += SPLINE_EDGE_TILE) {
            const uint32_t count = Min(SPLINE_EDGE_TILE, t_.e - b);
            Read(dst_, edge_, uint64_t(t_.e) + b, count);
            ReadSourcePseudo(b, count);
            uint32_t nfound = 0;
            for (uint32_t j = 0; j < count; ++j) {
                const int64_t d = dst_.GetValue(j);
                if (d < start_ || d >= int64_t(start_) + rows_)
                    continue;
                const uint32_t i = static_cast<uint32_t>(d) - start_;
                const uint32_t pos = cur_.GetValue(i);
                cur_.SetValue(i, pos + 1);
                recOff_.SetValue(nfound, pos);
                const uint32_t w = SPLINE_STAGE_STRIDE * nfound;
                stage_.SetValue(w + 0, static_cast<float>(static_cast<int32_t>(src_.GetValue(j))));
                stage_.SetValue(w + 1, pend_.GetValue(j * t_.dim + 0));
                stage_.SetValue(w + 2, t_.dim > 1 ? pend_.GetValue(j * t_.dim + 1) : 0.0f);
                stage_.SetValue(w + 3, t_.dim > 2 ? pend_.GetValue(j * t_.dim + 2) : 0.0f);
                ++nfound;
            }
            if (nfound) {
                Sync<HardEvent::S_MTE3>();
                for (uint32_t k = 0; k < nfound; ++k) {
                    DataCopyExtParams cp{1, static_cast<uint32_t>(SPLINE_RECORD_WORDS * sizeof(float)), 0, 0, 0};
                    DataCopyPad(map_[uint64_t(recOff_.GetValue(k)) * SPLINE_RECORD_WORDS],
                                stage_[uint64_t(k) * SPLINE_STAGE_STRIDE], cp);
                }
                /* 本 tile 的搬运读完暂存后，下一个 tile 才能复写 stage_
                   。 */
                Sync<HardEvent::MTE3_S>();
            }
        }
        Sync<HardEvent::MTE3_MTE2>();
        /* 每核只读自己构建的区域，无跨核依赖。 */
    }

    __aicore__ inline void BuildIncoming() {
        CountIncoming();
        ScatterIncoming();
    }

    __aicore__ inline void SelectSource(uint32_t source) {
        if (source == lastSrc_)
            return;
        if (t_.cachedX)
            xf_ = allX_[uint64_t(source) * t_.cIn];
        else
            LoadVector(x_, uint64_t(source) * t_.cIn, t_.cIn, xf_);
        lastSrc_ = source;
    }

    // Small graph: construct the same target-ordered CSR entirely in UB.
    // No per-edge GM writes or subsequent GM record reads are required.

    // Brcb + strided Mul is adapted from the working local det row-scaling
    // path. Broadcast one coefficient block across every output block.

    // A small target range: retain edge order inside each target, but issue
    // one DMA for the complete target's tile rather than one per record.

    // One target: no histogram or scatter is needed. Pack all matching
    // records into one strided DMA per edge tile (32B UB -> 16B GM).

    // Algebraic reassociation, without changing the spline polynomials:
    // A[k,c] = sum_e basis[e,k] * x[src[e],c]; output = sum_k A[k] @ W[k].
    // Both accumulation stages use compensation; all intermediates are FP32.

    // Four unique tensor-product slots fill four contiguous coefficient rows.
    // Preserve separately rounded products and the original 16-edge blocks;
    // combine row additions and compensated merges into wider vector commands.

    // Each core builds a private coefficient panel. No inter-core dependency.

    // Few targets per core: all target coefficient rows remain live in UB.
    // Build each tile's incoming records locally, consume immediately, and
    // retain compensation state between tiles. No global CSR is materialized.

    __aicore__ inline uint32_t BuildSlotList(uint32_t *idxReg, float *bReg, bool needMod) {
        uint32_t ns = 0U;
        if (t_.dim == 2U && t_.degree == 1U && !needMod) {
            const uint32_t a0 = idxTab_[0][baseOff_[0]], a1 = idxTab_[0][baseOff_[0] + 1U];
            const uint32_t b0 = idxTab_[1][baseOff_[1]], b1 = idxTab_[1][baseOff_[1] + 1U];
            const float u0 = basis_[0][0], v0 = basis_[0][1];
            const float u1 = basis_[1][0], v1 = basis_[1][1];
            const float f00 = u0 * u1, f10 = v0 * u1, f01 = u0 * v1, f11 = v0 * v1;
            if (f00 != 0.0f) {
                idxReg[ns] = a0 + b0;
                bReg[ns++] = f00;
            }
            if (f10 != 0.0f) {
                idxReg[ns] = a1 + b0;
                bReg[ns++] = f10;
            }
            if (f01 != 0.0f) {
                idxReg[ns] = a0 + b1;
                bReg[ns++] = f01;
            }
            if (f11 != 0.0f) {
                idxReg[ns] = a1 + b1;
                bReg[ns++] = f11;
            }
            return ns;
        }
        for (uint32_t s = 0U; s < t_.slots; ++s) {
            uint32_t index = 0U;
            float bv = 1.0f;
            for (uint32_t d = 0U; d < t_.dim; ++d) {
                const uint32_t k = slotTab_[s * 3U + d];
                index += idxTab_[d][baseOff_[d] + k];
                bv *= basis_[d][k];
            }
            if (bv == 0.0f)
                continue;
            if (needMod && index >= t_.weightK)
                index %= t_.weightK;
            idxReg[ns] = index;
            bReg[ns++] = bv;
        }
        return ns;
    }

    __aicore__ inline void ApplyWeights(const uint32_t *idxReg, const float *bReg, uint32_t ns) {
        const uint32_t nbuf = t_.weightSlots ? t_.weightSlots : 1U;
        const uint64_t cacheStride = uint64_t(t_.cIn) * t_.pitch;
        uint64_t dstOff[SPLINE_MAX_SLOTS];
        for (uint32_t k = 0U; k < nbuf; ++k) {
            dstOff[k] = uint64_t(k) * cacheStride;
        }
        for (uint32_t p = 0U; p < ns; p += nbuf) {
            const uint32_t cnt = Min(nbuf, ns - p);
            Sync<HardEvent::V_MTE2>();
            Sync<HardEvent::S_MTE2>();
            for (uint32_t k = 0U; k < cnt; ++k) {
                IssueWeight(w_, uint64_t(idxReg[p + k]) * t_.cIn * t_.cOut, BatchRaw(k));
            }
            Sync<HardEvent::MTE2_V>();
            if constexpr (!std::is_same<T, float>::value) {
                for (uint32_t k = 0U; k < cnt; ++k) {
                    Cast(wSrc_[dstOff[k]], BatchRaw(k), RoundMode::CAST_NONE, t_.cIn * t_.pitch);
                }
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t c = 0U; c < t_.cIn; ++c) {
                const float xc = xf_.GetValue(c);
                const uint64_t rowBase = uint64_t(c) * t_.pitch;
                for (uint32_t k = 0U; k < cnt; ++k) {
                    Axpy(msg_, wSrc_[dstOff[k] + rowBase], xc * bReg[p + k], t_.cOut);
                }
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeMessage() {
        const bool needMod = t_.weightK > 0U && t_.weightK < t_.kernelProduct;
        uint32_t idxReg[SPLINE_MAX_SLOTS];
        float bReg[SPLINE_MAX_SLOTS];
        const uint32_t ns = BuildSlotList(idxReg, bReg, needMod);
        Zero(msg_);
        ApplyWeights(idxReg, bReg, ns);
    }

    // Reassociate only the finite tensor-product sum: W_edge = sum_s b_s W_s.
    // One channel reduction instead of one reduction per spline slot. The
    // unused staging matrix and broadcast panel are reused, with no extra UB.

    // Sparse target range: keep all outputs and corrections in the old CSR
    // scatter buffer. A single edge scan; no CSR GM write/read or node sorting.
    // Filtering preserves the original incoming-edge order for every target.

    __aicore__ inline void ProcessRange() {
        for (uint32_t i = 0; i < rows_; ++i) {
            Zero(sum_);
            Zero(sc_);
            const uint32_t begin = off_.GetValue(i), end = off_.GetValue(i + 1);
            lastSrc_ = kNoSrc;
            for (uint32_t pos = begin; pos < end; pos += SPLINE_EDGE_TILE) {
                const uint32_t count = Min(SPLINE_EDGE_TILE, end - pos);
                if (t_.e > SPLINE_EDGE_TILE)
                    Read(rec_, map_, uint64_t(pos) * SPLINE_RECORD_WORDS, count * SPLINE_RECORD_WORDS);
                for (uint32_t j = 0; j < count; ++j) {
                    const uint32_t base = SPLINE_RECORD_WORDS * (j + (t_.e <= SPLINE_EDGE_TILE ? pos : 0));
                    const uint32_t source = static_cast<uint32_t>(static_cast<int32_t>(rec_.GetValue(base)));
                    const float p0 = rec_.GetValue(base + 1);
                    const float p1 = rec_.GetValue(base + 2);
                    const float p2 = rec_.GetValue(base + 3);
                    SelectSource(source);
                    PrepareBasis(p0, p1, p2);
                    ComputeMessage();
                    if (pos == begin && j == 0) {
                        DataCopy(sum_, msg_, t_.pitch);
                        PipeBarrier<PIPE_V>();
                    } else
                        KAdd(sum_, sc_, msg_);
                }
            }
            if (t_.mean && end != begin) {
                Muls(sum_, sum_, 1.0f / static_cast<float>(static_cast<int32_t>(end - begin)), t_.cOut);
                PipeBarrier<PIPE_V>();
            }
            WriteOutput(start_ + i);
        }
    }

    template <class V>
    __aicore__ inline void LoadVector(GlobalTensor<V> gm, uint64_t off, uint32_t count, LocalTensor<float> dst) {
        auto raw = rawVecBuf_.Get<V>();
        Read(raw, gm, off, count);
        if constexpr (std::is_same<V, float>::value) {
            DataCopy(dst, raw, (count + 7) / 8 * 8);
        } else {
            Cast(dst, raw, RoundMode::CAST_NONE, count);
        }
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_S>();
    }
    /* 权重按 [C_in, pitch] 的行块搬进 UB（行尾补零到 32B 对
       齐）。 */
    template <class V> __aicore__ inline void IssueWeight(GlobalTensor<V> gm, uint64_t off, LocalTensor<V> dst) {
        DataCopyExtParams cp{static_cast<uint16_t>(t_.cIn), static_cast<uint32_t>(t_.cOut * sizeof(V)), 0, 0, 0};
        const uint32_t rowElements = ((t_.cOut * sizeof(V) + 31) / 32) * 32 / sizeof(V);
        DataCopyPadExtParams<V> pad{true, 0, static_cast<uint8_t>(rowElements - t_.cOut), {}};
        DataCopyPad(dst, gm[off], cp, pad);
    }
    /* 第 k 个批量槽的 DMA 落点。fp32 直接落 float 槽；fp16/bf16
       落原始布局槽，之后统一整批转 float。 */
    __aicore__ inline LocalTensor<T> BatchRaw(uint32_t k) {
        if constexpr (std::is_same<T, float>::value) {
            return wSrc_[uint64_t(k) * t_.cIn * t_.pitch];
        } else {
            return wRaw_[uint64_t(k) * t_.cIn * t_.pitch];
        }
    }
    /* 第 k 个批量槽的 float 视图。整批一起转换，因此批次内所有槽
       可以同时参与向量计算（内层循环才能「通道在外、槽在内」）。 */

    /* 单独取一个权重矩阵（预填充缓存用）。 */

    /* 预计算两张查表，把热循环里的整数除法全部消掉：
       slotTab_[s*3+d] —— 槽 s 在第 d 维上的基函数下标（混合进制分解）
       idxTab_[d][b*radix+k] —— ((b+k) mod kernel_size[d]) * stride_d
       两者都只依赖 tiling 常量，每核初始化时算一次即可。 */
    __aicore__ inline void BuildSlotTables() {
        uint32_t radix = t_.degree + 1;
        if (radix < 2)
            radix = 2;
        if (radix > 4)
            radix = 4;
        radix_ = radix;
        for (uint32_t s = 0; s < t_.slots; ++s) {
            uint32_t q = s;
            for (uint32_t d = 0; d < t_.dim; ++d) {
                slotTab_[s * 3 + d] = static_cast<uint8_t>(q % radix);
                q /= radix;
            }
        }
        uint32_t stride = 1;
        for (uint32_t d = 0; d < t_.dim; ++d) {
            const uint32_t ks = (d < t_.dim && t_.kernelSize[d] > 0) ? t_.kernelSize[d] : 1U;
            for (uint32_t b = 0; b < ks; ++b) {
                for (uint32_t k = 0; k < radix; ++k) {
                    idxTab_[d][b * radix + k] = ((b + k) % ks) * stride;
                }
            }
            stride *= ks;
        }
    }
    /* 伪坐标由 CSR 记录直接以标量传入，不再经 UB 张量中转。 */
    __aicore__ inline void PrepareLinear(uint32_t d, float pseudo) {
        const int32_t size = static_cast<int32_t>(t_.kernelSize[d]);
        const float scaled = pseudo * static_cast<float>(size - static_cast<int32_t>(t_.open[d]));
        int32_t floorValue = static_cast<int32_t>(scaled);
        if (scaled < static_cast<float>(floorValue))
            --floorValue;
        int32_t wrapped = floorValue;
        if (wrapped >= 0 && wrapped <= size) {
            if (wrapped == size)
                wrapped = 0;
        } else {
            wrapped %= size;
            if (wrapped < 0)
                wrapped += size;
        }
        baseOff_[d] = static_cast<uint32_t>(wrapped) * 2;
        const float v = scaled - static_cast<float>(floorValue);
        basis_[d][0] = 1.0f - v;
        basis_[d][1] = v;
    }

    __aicore__ inline void PrepareBasis(float p0, float p1, float p2) {
        if (t_.dim == 2 && t_.degree == 1) {
            PrepareLinear(0, p0);
            PrepareLinear(1, p1);
            return;
        }
        float pv[3] = {p0, p1, p2};
        for (uint32_t d = 0; d < t_.dim; ++d) {
            const int32_t size = static_cast<int32_t>(t_.kernelSize[d]);
            const int32_t scale = size - static_cast<int32_t>(t_.degree * t_.open[d]);
            const float scaled = pv[d] * static_cast<float>(scale);
            int32_t floorValue = static_cast<int32_t>(scaled);
            if (scaled < static_cast<float>(floorValue))
                --floorValue;
            /* 先把有符号取模归一到非负，再回到无符号下标。 */
            int32_t wrapped = floorValue;
            // Valid pseudo lies in [0,1]: floor(scale*p) is in [0,size].
            // The only wrap needed is the closed-spline p=1 endpoint.
            if (wrapped >= 0 && wrapped <= size) {
                if (wrapped == size)
                    wrapped = 0;
            } else
                wrapped %= size;
            if (wrapped < 0)
                wrapped += size;
            base_[d] = static_cast<uint32_t>(wrapped);
            baseOff_[d] = static_cast<uint32_t>(wrapped) * radix_;
            const float v = scaled - static_cast<float>(floorValue), u = 1.0f - v;
            if (t_.degree == 1) {
                basis_[d][0] = u;
                basis_[d][1] = v;
            } else if (t_.degree == 2) {
                basis_[d][0] = 0.5f * v * v - v + 0.5f;
                basis_[d][1] = -v * v + v + 0.5f;
                basis_[d][2] = 0.5f * v * v;
            } else {
                const float v2 = v * v, v3 = v2 * v;
                basis_[d][0] = u * u * u / 6.0f;
                basis_[d][1] = (3.0f * v3 - 6.0f * v2 + 4.0f) / 6.0f;
                basis_[d][2] = (-3.0f * v3 + 3.0f * v2 + 3.0f * v + 1.0f) / 6.0f;
                basis_[d][3] = v3 / 6.0f;
            }
        }
    }
    __aicore__ inline void Zero(LocalTensor<float> dst) {
        Duplicate(dst, 0.0f, t_.pitch);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void KAdd(LocalTensor<float> acc, LocalTensor<float> correction, LocalTensor<float> value) {
        /* Kahan 补偿求和，只保留在「同一节点的多条入边」这一层长
           求和上。 */
        Sub(y_, value, correction, t_.cOut);
        PipeBarrier<PIPE_V>();
        Add(z_, acc, y_, t_.cOut);
        PipeBarrier<PIPE_V>();
        Sub(correction, z_, acc, t_.cOut);
        PipeBarrier<PIPE_V>();
        Sub(correction, correction, y_, t_.cOut);
        PipeBarrier<PIPE_V>();
        Adds(acc, z_, 0.0f, t_.cOut);
        PipeBarrier<PIPE_V>();
    }
    /* acc += Σ_c (x[c]·b) · W[c][:]。单条 Axpy 链（FMA，乘加
       各只舍入一次）， 不需要 row_ 暂存、不需要先算 dot 再缩
       放、也不需要树形归约。 基函数权 b 直接折进标量，所以每
       个槽只花 C_in 条向量指令。 */
    __aicore__ inline void RankAdd(LocalTensor<float> acc, LocalTensor<float> w, float b) {
        for (uint32_t c = 0; c < t_.cIn; ++c) {
            Axpy(acc, w[uint64_t(c) * t_.pitch], xf_.GetValue(c) * b, t_.cOut);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void WriteOutput(uint32_t node) {
        DataCopyExtParams cp{1, static_cast<uint32_t>(t_.cOut * sizeof(T)), 0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            Sync<HardEvent::V_MTE3>();
            DataCopyPad(out_[uint64_t(node) * t_.cOut], sum_, cp);
        } else {
            Cast(rawVec_, sum_, RoundMode::CAST_RINT, t_.cOut);
            PipeBarrier<PIPE_V>();
            Sync<HardEvent::V_MTE3>();
            DataCopyPad(out_[uint64_t(node) * t_.cOut], rawVec_, cp);
        }
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::MTE3_S>();
        Sync<HardEvent::MTE3_MTE2>();
    }

    TBuf<TPosition::VECCALC> allXBuf_, dotBuf_;
    TBuf<TPosition::VECCALC> batchXBuf_, batchRawBuf_;
    LocalTensor<float> batchX_;
    LocalTensor<T> batchRaw_;
    LocalTensor<float> allX_, broadcast_, products_;
    TBuf<TPosition::VECCALC> partialBuf_;
    LocalTensor<float> partial_;
    TBuf<TPosition::VECCALC> coefBuf_, coefTmpBuf_;
    LocalTensor<float> coef_, coefComp_, coefValue_, coefY_, coefZ_;
    TPipe *pipe_ = nullptr;
    SplineConvTilingData t_;
    uint32_t start_, rows_, lastSrc_, base_[3];
    float basis_[3][4];
    GlobalTensor<T> x_, out_;
    GlobalTensor<T> w_;
    GlobalTensor<T> pseudo_;
    GlobalTensor<int64_t> edge_;
    GlobalTensor<float> map_;
    TBuf<TPosition::VECCALC> offsetBuf_, cursorBuf_, dstBuf_, srcBuf_;
    TBuf<TPosition::VECCALC> recBuf_, stageBuf_, recOffBuf_, prawBuf_, pendBuf_, pfBuf_;
    TBuf<TPosition::VECCALC> xBuf_, rawVecBuf_, vecBuf_, idxBuf_, bListBuf_;
    TBuf<TPosition::VECCALC> wfBuf_, wrawBuf_, cacheBuf_;
    LocalTensor<uint32_t> off_, cur_, recOff_, idxList_;
    LocalTensor<int64_t> dst_, src_;
    LocalTensor<float> rec_, stage_, pend_, xf_, pf_, bList_, wSrc_;
    LocalTensor<T> rawVec_, wRaw_, praw_;
    LocalTensor<float> sum_, sc_, msg_, y_, z_;
    LocalTensor<float> cache_;
    /* 槽位分解与权重下标查表（只依赖 tiling 常量，Init 里算一次）。
       idxTab_ 每维最多 kernel_size(8) * radix(4) = 32 项。 */
    GlobalTensor<float> cubeA_, cubeB_, cubeC_;
    uint8_t slotTab_[SPLINE_MAX_SLOTS * 3];
    uint32_t idxTab_[3][SPLINE_MAX_KERNEL_SIZE * 4U];
    uint32_t baseOff_[3];
    uint32_t radix_;
};

/* 入口模板参数名 DT_X 必须与 tiling_key_spline_conv.h 中
   ASCENDC_TPL_DATATYPE_DECL 的参数名完全一致，编译期才会
   按 dtype 展开成 float / half / bfloat16_t 三份 kernel
   。 */
template <typename DT_X, bool USE_CUBE = false>
__global__ __aicore__ void spline_conv(GM_ADDR x, GM_ADDR edge_index, GM_ADDR edge_attr, GM_ADDR weight, GM_ADDR out,
                                       GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SplineConvTilingData);
    GET_TILING_DATA_WITH_STRUCT(SplineConvTilingData, t, tiling);
    /* TPipe 必须在算子类之外创建（官方优化指南：放在类内会阻
       断标量折叠优化）。 */
    TPipe pipe;
    KernelSplineConv<DT_X, USE_CUBE> op;
    if constexpr (USE_CUBE) {
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm, &t.cubeTiling);
        if ASCEND_IS_AIC {
            return;
        }
        if (GetSubBlockIdx() != 0)
            return;
    }
    op.Init(x, edge_index, edge_attr, weight, out, workspace, t, &pipe);
    op.Process();
}
