/*
 * Copyright (c) 2026 李冠杰.
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

// FireCandidateDecodeFusion: fused FP32 sigmoid, box decoding and confidence product.
// One implementation retains the original compile-time layout, scalar-only and
// batch-split paths. Buffer sizes, vector arithmetic order and queue lifetimes
// match the validated implementation.
#include "kernel_operator.h"

#include <type_traits>

#include "fire_candidate_decode_fusion_tiling.h"

// ---------------------------------------------------------------- Round 7 开关
// FCF_PLANE_BASE / FCF_PLANE_PAD : 两种平面 pitch（float 元素数）。512 = R6 布局
//   （每平面恰好 2048B）；520 = 2080B，用于打散"所有平面在 2048B 模下同余"的
//   UB bank 冲突。两者由 kernel 模板参数 USE_PAD 在编译期选定，一次调用只实例化
//   并运行其中一种，不存在两套 UB 同时驻留。
// FCF_R72_NOCOPY : 1 => FP32 不再先把 raw 复制进 sig，直接 Muls(stmp, raw, -1, n6)
// FCF_R73_ONES   : 1 => 常量 ones 只在 Init 填一次，逐 tile 不再 Duplicate
#ifndef FCF_PLANE_BASE
#define FCF_PLANE_BASE 512
#endif
#ifndef FCF_PLANE_PAD
#define FCF_PLANE_PAD 520
#endif
#ifndef FCF_R72_NOCOPY
#define FCF_R72_NOCOPY 1
#endif
#ifndef FCF_R73_ONES
#define FCF_R73_ONES 1
#endif
#ifndef FCF_DIRECT_LAUNCH
// tiling key 头文件依赖 tiling 框架的 C_DT_* 常量，只在 registry-invoke 构建里
// 可用；直调工程不需要 tiling key，跳过。
#include "tiling_key_fire_candidate_decode_fusion.h"
#endif

namespace {
constexpr uint32_t CHANNELS = 6;
constexpr uint32_t PRIOR_CHANNELS = 5;
constexpr uint32_t OUTPUT_CHANNELS = 5;
constexpr uint32_t MAX_TILE = 512;
constexpr uint32_t OFFSET_FIELDS = CHANNELS + PRIOR_CHANNELS;
constexpr uint32_t CALC_SLICES = 8;

// TQue 的第二个模板参数就是队列内 buffer 块数，必须与 InitBuffer 的 num 严格
// 相等。二者不一致时队列的编译期环形存储会越界，AllocTensor/FreeTensor 记账
// 失效：每核恰好只有一个 tile 时看不出问题，一旦某个核需要在队列里同时挂两个
// tile（例如核的区间跨越 batch 边界、切出两个 tile 并预取），该核就会在
// AllocTensor 上永久自旋，整条 stream 不再返回（表现为 kernel 超时）。
constexpr uint32_t QUEUE_DEPTH = 2;
static_assert(QUEUE_DEPTH >= 2, "two live buffers per queue are required by the pack path");

// DataCopyExtParams 的部分字段是 16 位，用花括号直接初始化会把 uint32_t 表达式
// 判成窄化错误；这里逐个字段赋值，取值范围由调用点保证（blockCount <= 512）。
__aicore__ inline AscendC::DataCopyExtParams MakeCopyExt(uint32_t blockCount, uint32_t blockLen,
                                                         uint32_t srcStride, uint32_t dstStride)
{
    AscendC::DataCopyExtParams p{};
    p.blockCount = static_cast<uint16_t>(blockCount);
    p.blockLen = blockLen;
    p.srcStride = srcStride;
    p.dstStride = dstStride;
    return p;
}
}  // namespace

namespace fcf {


template <class DT_RAW_PRED, bool USE_PAD, bool SCALAR_ONLY, bool BATCH_SPLIT>
class KernelFireCandidateDecodeFusion {
    // Preserve scalar, regular and batch paths as compile-time specializations.
    static_assert(!SCALAR_ONLY || (std::is_same_v<DT_RAW_PRED, float> && !USE_PAD && !BATCH_SPLIT),
                  "SCALAR_ONLY requires FP32 flat without batch split");
    // R14：按 batch 分核只对 FP32 flat 常规路径成立（组合表里也只有这一项）。
    static_assert(!BATCH_SPLIT || (std::is_same_v<DT_RAW_PRED, float> && !USE_PAD),
                  "BATCH_SPLIT is only valid on the FP32 flat normal instance");

public:
    // 编译期布局：一次实例化只用一种 pitch，不存在两套 UB 同时驻留
    static constexpr uint32_t PLANE = USE_PAD ? FCF_PLANE_PAD : FCF_PLANE_BASE;

    __aicore__ inline KernelFireCandidateDecodeFusion() {}

    // TD 既可以是 registry-invoke 路径下 GET_TILING_DATA 得到的本地结构体，
    // 也可以是直调路径下指向 GM 的 __gm__ 指针：只读几个标量，两种都成立。
    template <typename TD>
    __aicore__ inline void InitRange(const TD* td)
    {
        const uint32_t block = AscendC::GetBlockIdx();
        if constexpr (BATCH_SPLIT) {
            // R14：按 batch 分核。每个 batch 用独立的 coresPerBatch 个核，
            // 每核只处理本 batch 内一段连续候选，因此不再有核跨越 batch 边界、
            // 也消除"少数核串行做第二个 tile"。
            //   coresPerBatch = ceil(N / blockFactor)          （由已有 tiling 字段导出）
            //   batchIdx      = blockIdx / coresPerBatch
            //   localCore     = blockIdx % coresPerBatch
            //   start = batchIdx*N + localCore*blockFactor
            //   end   = batchIdx*N + min(N, localCore*blockFactor + blockFactor)
            // host 已保证 B>1、B<=C、C%B==0、16 <= blockFactor < 512，
            // 且 usedCoreNum = B*coresPerBatch <= C（证明见 host 注释）。
            const uint32_t coresPerBatch =
                static_cast<uint32_t>((td->candidates + td->blockFactor - 1u) / td->blockFactor);
            const uint32_t batchIdx = block / coresPerBatch;
            const uint32_t localCore = block % coresPerBatch;
            const uint32_t base = batchIdx * td->candidates;
            const uint32_t candStart = localCore * td->blockFactor;
            uint32_t candEnd = candStart + td->blockFactor;
            if (candEnd > td->candidates) candEnd = td->candidates;
            start_ = base + candStart;
            end_ = base + candEnd;
            candStart_ = candStart;   // R15-P1A：缓存候选内偏移，Process 不再做 global%N
        } else {
            start_ = block * td->blockFactor;
            end_ = start_ + td->blockFactor;
            if (end_ > td->total) end_ = td->total;
        }
    }

    __aicore__ inline void InitBuffers()
    {
        // PAD UB totals: FP32 160320 B, FP16 158272 B; capacity 196608 B.
        // Keep every allocation and its order unchanged.
        pipe_.InitBuffer(rawQ_, QUEUE_DEPTH, MAX_TILE * CHANNELS * sizeof(DT_RAW_PRED));
        pipe_.InitBuffer(priorQ_, QUEUE_DEPTH, MAX_TILE * PRIOR_CHANNELS * sizeof(float));
        pipe_.InitBuffer(outQ_, QUEUE_DEPTH, PLANE * OUTPUT_CHANNELS * sizeof(float));
        // FP32 的 pack 缓冲只需要 float 区（half 区仅 FP16 的 Cast 输出用），
        // 按 dtype 取值可省 10 KiB，给 R7.3 的常量 ones 腾出空间。
        constexpr uint32_t PACK_ELEM_BYTES =
            std::is_same_v<DT_RAW_PRED, float> ? sizeof(float) : (sizeof(float) + sizeof(half));
        pipe_.InitBuffer(packQ_, QUEUE_DEPTH, MAX_TILE * OUTPUT_CHANNELS * PACK_ELEM_BYTES);
        // R6-B：唯一新增的缓冲，长度 6*MAX_TILE 个 float。
        pipe_.InitBuffer(sigBuf_, MAX_TILE * CHANNELS * sizeof(float));
#if FCF_R73_ONES
        // R7.3：常量 ones 只在这里填一次，逐 tile 不再 Duplicate
        pipe_.InitBuffer(onesBuf_, MAX_TILE * CHANNELS * sizeof(float));
#endif
        pipe_.InitBuffer(calcBuf_, PLANE * CALC_SLICES * sizeof(float));
        pipe_.InitBuffer(offsetBuf_, MAX_TILE * OFFSET_FIELDS * sizeof(uint32_t));
        pipe_.InitBuffer(packOffsetBuf_, MAX_TILE * OUTPUT_CHANNELS * sizeof(uint32_t));
        // ---- P1b：按本核实际最大 tile 长度裁剪初始化 ----
        // 单 tile 的候选数上界 = min(MAX_TILE, candidates, end-start)，对齐到 8。
        // UB 容量/地址/申请顺序不变，只是少生成用不到的表项。
        {
            uint32_t span = end_ - start_;
            if (span > candidates_) span = candidates_;
            if (span > MAX_TILE) span = MAX_TILE;
            span = (span + 7u) & ~7u;
            initLen_ = (span == 0u) ? 8u : span;
        }
    }

    template <typename TD>
    __aicore__ inline void Init(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded, const TD* td)
    {
        InitRange(td);
        candidates_ = td->candidates;

        rawGm_.SetGlobalBuffer((__gm__ DT_RAW_PRED*)raw_pred);
        priorGm_.SetGlobalBuffer((__gm__ float*)priors);
        decodedGm_.SetGlobalBuffer((__gm__ DT_RAW_PRED*)decoded);

        InitBuffers();
        InitOffsetTables();
#if FCF_R73_ONES
        // P1b：ones 只填 6*L；任意 tile 的 6*count <= 6*L（count <= L）
        AscendC::Duplicate(onesBuf_.Get<float>(), 1.0f, CHANNELS * initLen_);
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
    }

    __aicore__ inline void Process()
    {
        if constexpr (BATCH_SPLIT) {
            // R15-P1A：host 已保证本实例「每核只属于一个 batch、span = end - start <= blockFactor < 512」，
            //   所以本核只会有一个 tile：直接一次 CopyIn -> Compute -> CopyOut，
            //   省掉 GetTileCount 的两次取模/比较与循环回边。
            //   候选内偏移用 Init 缓存的 candStart_，不再做 global % candidates_。
            //   队列操作与同步完全复用原 CopyIn/Compute/CopyOut，未删任何 EnQue/DeQue/FreeTensor。
            const uint32_t count = end_ - start_;
            if (count == 0u) return;
            CopyIn(start_, candStart_, count);
            Compute(count);
            CopyOut(start_, count);
            return;
        }
        uint32_t global = start_;
        if (global >= end_) return;

        uint32_t count = GetTileCount(global);
        CopyIn(global, global % candidates_, count);
        while (global < end_) {
            Compute(count);
            CopyOut(global, count);
            const uint32_t nextGlobal = global + count;
            if (nextGlobal >= end_) break;
            global = nextGlobal;
            count = GetTileCount(global);
            CopyIn(global, global % candidates_, count);
        }
    }

private:
    __aicore__ inline uint32_t GetTileCount(uint32_t global) const
    {
        // tile 不能越过 batch 边界：priors 在 batch 维共享，越界后 prior 索引
        // 会回绕，切 tile 是最省事的正确做法。
        uint32_t count = end_ - global;
        if (count > MAX_TILE) count = MAX_TILE;
        const uint32_t untilBatchEnd = candidates_ - (global % candidates_);
        if (count > untilBatchEnd) count = untilBatchEnd;
        return count;
    }

    __aicore__ inline void BuildPackTable(uint32_t len)
    {

            auto packOffs = packOffsetBuf_.Get<int32_t>();
            // tab[k] = (k%5)*MAX_TILE + k/5，单位先按元素算，最后乘 sizeof(float)。
            // k/5 用乘加移位精确实现（3277/2^14 在 k<2560 时逐值验证无误差），
            // 再借 512k - 2559q 恒等式避免整数除法与第三个缓冲：
            //   512*(k - 5q) + q = 512k - 2559q
            // 整表 7 条向量指令，取代原先 2560 次标量 SetValue（实测约 20 us，
            // 是 gate shape 上最大的一笔固定开销）。
            constexpr uint32_t kCap = MAX_TILE * OUTPUT_CHANNELS;
            // 表长 5*L，另加 8 项余量覆盖尾块 Gather 的向上取整
            uint32_t kTableLen = len * OUTPUT_CHANNELS + 8u;
            if (kTableLen > kCap) kTableLen = kCap;
            auto q = calcBuf_.Get<float>().template ReinterpretCast<int32_t>();
            AscendC::ArithProgression<int32_t>(packOffs, 0, 1, kTableLen);                    // k
            AscendC::Muls(q, packOffs, static_cast<int32_t>(3277), kTableLen);                 // 3277k
            AscendC::ShiftRight(q, q, static_cast<int32_t>(14), kTableLen);                    // q = k/5
            // tab[k] = (k%5)*PLANE + k/5 = PLANE*k - (5*PLANE-1)*(k/5)，与 PLANE 无关的恒等式
            AscendC::Muls(packOffs, packOffs, static_cast<int32_t>(PLANE), kTableLen);
            AscendC::Muls(q, q, static_cast<int32_t>(PLANE * OUTPUT_CHANNELS - 1), kTableLen);
            AscendC::Sub(packOffs, packOffs, q, kTableLen);
            AscendC::Muls(packOffs, packOffs, static_cast<int32_t>(sizeof(float)), kTableLen);  // 字节偏移
            // 上表是标量写、之后由向量 Gather 读，必须跨 pipe 同步一次；
            // raw/prior 两组表由 ArithProgression（向量写）生成才不需要。
            AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitOffsetTables()
    {
        auto offsets = offsetBuf_.Get<int32_t>();
        const uint32_t len = initLen_;
        // P1b 与 v1 的关键差别：
        //   v1 把 11 次 ArithProgression 的 count 都改成运行期变量，编译器为每次
        //   调用各内联一套带两段循环的完整实现，目标文件 +26%，Init 反而更慢。
        //   本版只发一次 ArithProgression 生成 0..L-1，其余 11 张表是它的仿射
        //   变换（tab_f[k] = 4f + stride*k），用 Muls/Adds 平移即可：
        //   向量指令对任意 count 都是统一的分块处理，不存在"小 count 走慢路径"。
        //   临时索引放在 calcBuf_ 的 pack 表区之后，二者生命周期不重叠。
        auto idx = calcBuf_.Get<float>().template ReinterpretCast<int32_t>()[MAX_TILE * OUTPUT_CHANNELS];
        // R10-P0：只对 FP32 + USE_PAD 的编译实例换用 CreateVecIndex；
        // 全部 FP16 与 FP32 flat 保留 R8 的 ArithProgression。
        // 依据：R9 的无条件替换在 FP16 B8N16384（+11.7%）与 FP32 B4N513（+12.5%）
        // 上是稳定退化，而 FP32 pad 的七例配对中位均改善。这是按 dtype/layout 的
        // 通用路由，不识别 case 名或 benchmark tag。
        if constexpr (std::is_same_v<DT_RAW_PRED, float> && USE_PAD) {
            AscendC::CreateVecIndex<int32_t>(idx, static_cast<int32_t>(0), static_cast<int32_t>(len));
        } else {
            AscendC::ArithProgression<int32_t>(idx, 0, 1, static_cast<int32_t>(len));
        }
        for (uint32_t field = 0; field < CHANNELS; ++field) {
            auto dst = offsets[field * MAX_TILE];
            AscendC::Muls(dst, idx, static_cast<int32_t>(CHANNELS * sizeof(float)), len);
            AscendC::Adds(dst, dst, static_cast<int32_t>(field * sizeof(float)), len);
        }
        const uint32_t priorBase = CHANNELS * MAX_TILE;
        for (uint32_t field = 0; field < PRIOR_CHANNELS; ++field) {
            auto dst = offsets[priorBase + field * MAX_TILE];
            AscendC::Muls(dst, idx, static_cast<int32_t>(PRIOR_CHANNELS * sizeof(float)), len);
            AscendC::Adds(dst, dst, static_cast<int32_t>(field * sizeof(float)), len);
        }
        // 写回重排表：offset[k] = (k%5)*MAX_TILE + k/5，单位为字节，不是等差数列，
        // 只能逐元素生成。建整表是 O(MAX_TILE*5) 次标量写，只有在每核 tile 数 >= 2
        // 时才比标量 pack（O(count)）便宜，因此按每核候选数决定是否启用。
        // 每核候选数 >= 16 时用 Gather 重排：向量建表约 1230 cycle，标量 pack 约
        // 14 cycle/对，二者在 18 个候选附近持平；低于门槛的极小 shape（N=1/33/257）
        // 继续走标量 pack，避免为几十个元素付整张表。
        // R10-P2：pad 实例的 packGather 恒为真（blockFactor>=512 ⇒ 每核 >=465），
        // 用编译期常量表达；flat 实例保留原运行时判据。
        if constexpr (SCALAR_ONLY) {
            packGather_ = false;
        } else if constexpr (USE_PAD) {
            packGather_ = true;
        } else {
            packGather_ = (end_ - start_) >= 16;
        }
        if constexpr (!SCALAR_ONLY) {
            if (packGather_) {
                BuildPackTable(len);
            }
        }

    }

    __aicore__ inline void CopyIn(uint32_t global, uint32_t priorOffset, uint32_t count)
    {
        if (count == 0) return;
        auto raw = rawQ_.AllocTensor<DT_RAW_PRED>();
        AscendC::DataCopyExtParams rawCopy =
            MakeCopyExt(1, count * CHANNELS * sizeof(DT_RAW_PRED), 0, 0);
        AscendC::DataCopyPadExtParams<DT_RAW_PRED> rawPad{false, 0, 0, static_cast<DT_RAW_PRED>(0)};
        AscendC::DataCopyPad(raw, rawGm_[static_cast<uint64_t>(global) * CHANNELS], rawCopy, rawPad);
        rawQ_.EnQue(raw);

        auto prior = priorQ_.AllocTensor<float>();
        AscendC::DataCopyExtParams priorCopy = MakeCopyExt(1, count * PRIOR_CHANNELS * sizeof(float), 0, 0);
        AscendC::DataCopyPadExtParams<float> priorPad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(prior, priorGm_[static_cast<uint64_t>(priorOffset) * PRIOR_CHANNELS], priorCopy, priorPad);
        priorQ_.EnQue(prior);
    }

    __aicore__ inline void Sigmoid(const AscendC::LocalTensor<float>& dst,
                                   const AscendC::LocalTensor<float>& tmp, uint32_t count)
    {
        AscendC::Muls(tmp, dst, -1.0f, count);
        AscendC::Exp(tmp, tmp, static_cast<int32_t>(count));
        AscendC::Adds(tmp, tmp, 1.0f, count);
        AscendC::Div(dst, calcBuf_.Get<float>(), tmp, count);
    }

    // 取 raw_pred 的第 field 个通道（stride = CHANNELS 个元素）到 dst。
    // fp32 直接 Gather 到 dst；fp16 先 Gather 成 half 再 Cast 到 fp32。
    __aicore__ inline void GatherRaw(const AscendC::LocalTensor<DT_RAW_PRED>& raw,
                                     const AscendC::LocalTensor<float>& dst,
                                     const AscendC::LocalTensor<float>& tmp,
                                     uint32_t field, uint32_t count)
    {
        auto offsetsI32 = offsetBuf_.Get<int32_t>()[field * MAX_TILE];
        auto offsets = offsetsI32.template ReinterpretCast<uint32_t>();
        if constexpr (std::is_same_v<DT_RAW_PRED, float>) {
            AscendC::Gather(dst, raw, offsets, 0, count);
        } else {
            auto rawField = tmp.template ReinterpretCast<DT_RAW_PRED>();
            AscendC::Gather(rawField, raw, offsets, 0, count);
            AscendC::Cast(dst, rawField, AscendC::RoundMode::CAST_NONE, count);
        }
    }

    __aicore__ inline void GatherPrior(const AscendC::LocalTensor<float>& prior,
                                       const AscendC::LocalTensor<float>& dst,
                                       uint32_t field, uint32_t count)
    {
        const uint32_t priorBase = CHANNELS * MAX_TILE;
        auto offsetsI32 = offsetBuf_.Get<int32_t>()[priorBase + field * MAX_TILE];
        auto offsets = offsetsI32.template ReinterpretCast<uint32_t>();
        AscendC::Gather(dst, prior, offsets, 0, count);
    }

    __aicore__ inline void ComputeSigmoid(const AscendC::LocalTensor<DT_RAW_PRED>& raw, uint32_t count)
    {
        // ---- 连续 AoS 上的一次 sigmoid（长度 6*count） ----
        // 临时区借用 calc[0..5]（共 6*MAX_TILE 个 float），此阶段 GatherPrior
        // 还没写 calc，生命周期不重叠；下面 GatherPrior 开始写 calc[2..6] 时
        // 这段临时数据已经没用了。
        const uint32_t n6 = count * CHANNELS;
        auto sig = sigBuf_.Get<float>();
        auto stmp = calcBuf_.Get<float>();
#if FCF_R73_ONES
        auto sones = onesBuf_.Get<float>();
#else
        AscendC::LocalTensor<float> sones;   // 由下面的 Duplicate(sig, 1.0f) 就地生成
#endif
#if FCF_R72_NOCOPY
        if constexpr (std::is_same_v<DT_RAW_PRED, float>) {
            // R7.2：FP32 直接从 raw 取负，省掉一趟 6*count 的复制
            AscendC::Muls(stmp, raw, -1.0f, n6);
        } else {
            AscendC::Cast(sig, raw, AscendC::RoundMode::CAST_NONE, n6);
            AscendC::Muls(stmp, sig, -1.0f, n6);
        }
#else
        if constexpr (std::is_same_v<DT_RAW_PRED, float>) {
            AscendC::Adds(sig, raw, 0.0f, n6);                            // 连续 FP32 整段搬一次
        } else {
            AscendC::Cast(sig, raw, AscendC::RoundMode::CAST_NONE, n6);   // 6*count 一次 Cast
        }
        AscendC::Muls(stmp, sig, -1.0f, n6);
#endif
        AscendC::Exp(stmp, stmp, static_cast<int32_t>(n6));
        AscendC::Adds(stmp, stmp, 1.0f, n6);
#if FCF_R73_ONES
        AscendC::Div(sig, sones, stmp, n6);         // 读常量 ones
#else
        AscendC::Duplicate(sig, 1.0f, n6);          // sig 原值已用完，直接当 ones
        AscendC::Div(sig, sig, stmp, n6);
#endif

    }

    __aicore__ inline void DecodeCoordinates(const AscendC::LocalTensor<float>& out, uint32_t count)
    {
        auto calc = calcBuf_.Get<float>();
        auto s5 = calc[PLANE];
        auto gx = calc[2 * PLANE];
        auto gy = calc[3 * PLANE];
        auto stride = calc[4 * PLANE];
        auto anchorW = calc[5 * PLANE];
        auto anchorH = calc[6 * PLANE];
        // x = (2*sigmoid(tx) - 0.5 + gx) * stride
        AscendC::Muls(out[0 * PLANE], out[0 * PLANE], 2.0f, count);
        AscendC::Adds(out[0 * PLANE], out[0 * PLANE], -0.5f, count);
        AscendC::Add(out[0 * PLANE], out[0 * PLANE], gx, count);
        AscendC::Mul(out[0 * PLANE], out[0 * PLANE], stride, count);

        // y = (2*sigmoid(ty) - 0.5 + gy) * stride
        AscendC::Muls(out[1 * PLANE], out[1 * PLANE], 2.0f, count);
        AscendC::Adds(out[1 * PLANE], out[1 * PLANE], -0.5f, count);
        AscendC::Add(out[1 * PLANE], out[1 * PLANE], gy, count);
        AscendC::Mul(out[1 * PLANE], out[1 * PLANE], stride, count);

        // w = (2*sigmoid(tw))^2 * anchor_w * stride
        AscendC::Muls(out[2 * PLANE], out[2 * PLANE], 2.0f, count);
        AscendC::Mul(out[2 * PLANE], out[2 * PLANE], out[2 * PLANE], count);
        AscendC::Mul(out[2 * PLANE], out[2 * PLANE], anchorW, count);
        AscendC::Mul(out[2 * PLANE], out[2 * PLANE], stride, count);

        // h = (2*sigmoid(th))^2 * anchor_h * stride
        AscendC::Muls(out[3 * PLANE], out[3 * PLANE], 2.0f, count);
        AscendC::Mul(out[3 * PLANE], out[3 * PLANE], out[3 * PLANE], count);
        AscendC::Mul(out[3 * PLANE], out[3 * PLANE], anchorH, count);
        AscendC::Mul(out[3 * PLANE], out[3 * PLANE], stride, count);

        // fire_score = sigmoid(obj_logit) * sigmoid(fire_logit)
        AscendC::Mul(out[4 * PLANE], out[4 * PLANE], s5, count);

    }

    __aicore__ inline void Compute(uint32_t count)
    {
        auto raw = rawQ_.DeQue<DT_RAW_PRED>();
        auto prior = priorQ_.DeQue<float>();
        auto out = outQ_.AllocTensor<float>();
        auto calc = calcBuf_.Get<float>();
        auto s5 = calc[PLANE];            // 第 6 个 sigmoid 字段（R4 的 tmp 槽位）

        ComputeSigmoid(raw, count);
        auto sig = sigBuf_.Get<float>();

        // ---- 解交错六个 sigmoid 字段（4 字节 FP32 偏移） ----
        auto offs = offsetBuf_.Get<int32_t>().template ReinterpretCast<uint32_t>();
        AscendC::Gather(out[0 * PLANE], sig, offs[0 * MAX_TILE], 0, count);
        AscendC::Gather(out[1 * PLANE], sig, offs[1 * MAX_TILE], 0, count);
        AscendC::Gather(out[2 * PLANE], sig, offs[2 * MAX_TILE], 0, count);
        AscendC::Gather(out[3 * PLANE], sig, offs[3 * MAX_TILE], 0, count);
        AscendC::Gather(out[4 * PLANE], sig, offs[4 * MAX_TILE], 0, count);
        AscendC::Gather(s5, sig, offs[5 * MAX_TILE], 0, count);

        // ---- 以下与 R4 逐行同序 ----
        auto gx = calc[2 * PLANE];
        auto gy = calc[3 * PLANE];
        auto stride = calc[4 * PLANE];
        auto anchorW = calc[5 * PLANE];
        auto anchorH = calc[6 * PLANE];

        GatherPrior(prior, gx, 0, count);
        GatherPrior(prior, gy, 1, count);
        GatherPrior(prior, stride, 2, count);
        GatherPrior(prior, anchorW, 3, count);
        GatherPrior(prior, anchorH, 4, count);

        DecodeCoordinates(out, count);

        rawQ_.FreeTensor(raw);
        priorQ_.FreeTensor(prior);
        outQ_.EnQue(out);
    }

    __aicore__ inline void ScalarPack(const AscendC::LocalTensor<float>& packed,
                                      const AscendC::LocalTensor<float>& out, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t c = 0; c < OUTPUT_CHANNELS; ++c) {
                packed.SetValue(i * OUTPUT_CHANNELS + c, out.GetValue(c * PLANE + i));
            }
        }
    }

    __aicore__ inline void CopyOut(uint32_t global, uint32_t count)
    {
        auto out = outQ_.DeQue<float>();
        const uint32_t packedCount = count * OUTPUT_CHANNELS;
        auto packT = packQ_.AllocTensor<float>();
        if constexpr (SCALAR_ONLY) {
            ScalarPack(packT, out, count);
        } else if constexpr (USE_PAD) {
            // R10-P2：pad 路由保证 blockFactor>=512，最后一个工作核长度
            // >= blockFactor-47 >= 465，全部 pad 工作核都满足 (end-start)>=16，
            // packGather 恒真。这里直接生成向量 pack，pad 实例不含 ScalarPack。
            auto offs = packOffsetBuf_.Get<int32_t>().template ReinterpretCast<uint32_t>();
            AscendC::Gather(packT, out, offs, static_cast<uint32_t>(0), packedCount);
        } else {
            if (packGather_) {
                auto offs = packOffsetBuf_.Get<int32_t>().template ReinterpretCast<uint32_t>();
                AscendC::Gather(packT, out, offs, static_cast<uint32_t>(0), packedCount);
            } else {
                ScalarPack(packT, out, count);
            }
        }
        if constexpr (!std::is_same_v<DT_RAW_PRED, float>) {
            // FP16 的 Cast 放在 EnQue 之前，让 V->MTE3 同步点只留一处。
            // 注意不能与 float 版同址：逐元素 Cast 会覆盖尚未读到的源。
            auto halfRegion = packT.template ReinterpretCast<half>()[MAX_TILE * OUTPUT_CHANNELS * 2];
            AscendC::Cast(halfRegion, packT, AscendC::RoundMode::CAST_RINT, packedCount);
        }
        packQ_.EnQue(packT);
        auto packed = packQ_.DeQue<float>();
        WriteBack(global, packed, packedCount);
        packQ_.FreeTensor(packed);
        outQ_.FreeTensor(out);
    }

    __aicore__ inline void WriteBack(uint32_t global, const AscendC::LocalTensor<float>& packed,
                                     uint32_t packedCount)
    {
        if constexpr (std::is_same_v<DT_RAW_PRED, float>) {
            AscendC::DataCopyExtParams params = MakeCopyExt(1, packedCount * sizeof(float), 0, 0);
            AscendC::DataCopyPad(decodedGm_[static_cast<uint64_t>(global) * OUTPUT_CHANNELS], packed, params);
        } else {
            auto halfRegion = packed.template ReinterpretCast<half>()[MAX_TILE * OUTPUT_CHANNELS * 2];
            AscendC::DataCopyExtParams params = MakeCopyExt(1, packedCount * sizeof(half), 0, 0);
            AscendC::DataCopyPad(decodedGm_[static_cast<uint64_t>(global) * OUTPUT_CHANNELS], halfRegion, params);
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<DT_RAW_PRED> rawGm_;
    AscendC::GlobalTensor<float> priorGm_;
    AscendC::GlobalTensor<DT_RAW_PRED> decodedGm_;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> rawQ_, priorQ_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH> outQ_, packQ_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_, offsetBuf_, packOffsetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sigBuf_;
#if FCF_R73_ONES
    AscendC::TBuf<AscendC::TPosition::VECCALC> onesBuf_;
#endif
    uint32_t start_ = 0;
    uint32_t end_ = 0;
    uint32_t candidates_ = 1;
    uint32_t candStart_ = 0;   // R15-P1A：本核在本 batch 内的候选起点（仅 BATCH_SPLIT 使用）
    // P1b：本核实际最大 tile 长度（8 的倍数，<= MAX_TILE），只影响 Init 工作量
    uint32_t initLen_ = 8;
    bool packGather_ = false;
};

}  // namespace fcf

template <typename DT_RAW_PRED, bool USE_PAD, bool SCALAR_ONLY, bool BATCH_SPLIT>
__global__ __aicore__ void fire_candidate_decode_fusion(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
#ifdef FCF_DIRECT_LAUNCH
    const auto* td = reinterpret_cast<const __gm__ FireCandidateDecodeFusionTilingData*>(tiling);
#else
    REGISTER_TILING_DEFAULT(FireCandidateDecodeFusionTilingData);
    GET_TILING_DATA_WITH_STRUCT(FireCandidateDecodeFusionTilingData, tiling_data, tiling);
    const auto* td = &tiling_data;
#endif
    fcf::KernelFireCandidateDecodeFusion<DT_RAW_PRED, USE_PAD, SCALAR_ONLY, BATCH_SPLIT> op;
    op.Init(raw_pred, priors, decoded, td);
    op.Process();
}
