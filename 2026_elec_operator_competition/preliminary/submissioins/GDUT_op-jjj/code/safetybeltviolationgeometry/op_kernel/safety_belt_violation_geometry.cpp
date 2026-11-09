/*
 * Copyright (c) 2026 JJJie999.
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

// Kernel侧核函数实现
#include "kernel_operator.h"

#include "safety_belt_violation_geometry_tiling.h"
#include "tiling_key_safety_belt_violation_geometry.h"

using namespace AscendC;

/**
 * ============================================================================
 *  安全带低挂高用几何判定融合算子（Ascend C）
 * ============================================================================
 *
 *  语义（题目 3.4，不得改动）：
 *      Y_ref           = y1 + waist_ratio * (y2 - y1)
 *      violation_score = 1.0   if  cy > Y_ref + threshold
 *                        0.0   otherwise        （相等判 0，不引入 epsilon）
 *
 *  ---------------------------------------------------------------------------
 *  单模板类 KernelSbvVec<InT>（按输入 dtype 在编译期分派，数值结果逐位一致）
 *  ---------------------------------------------------------------------------
 *  (1) 全向量路径：fp32 与 fp16 共用一份源码，仅「是否先整块 Cast」由 if constexpr 分派。
 *      搬运保持「整块连续」（每 tile 只有 2 条描述符），列抽取用 3 次 vgather：
 *
 *          y1[i] = person_boxes[4i + 1]
 *          y2[i] = person_boxes[4i + 3]
 *          cy[i] = anchor_points[2i + 1]
 *
 *      下标向量在 Init 里用 CreateVecIndex + Muls + Adds 一次构造（与 count 无关），
 *      运行期没有任何标量逐元素循环，也没有按 tile 的 PIPE_ALL 全流水栅栏。
 *
 *      ★ 坑 1：vgather 的 srcOffset 单位是**字节**，不是元素。
 *        传「元素下标 4i+1」时抽出来的值会逐位等于 src[i]（因为 (4i+1)/4 == i），
 *        表现为「大约一半结果错」的假象。下标必须乘 sizeof(float)。已真机验证。
 *
 *      ★ 坑 2：「DataCopyPad 每块 32B 槽」做列抽取在功能上正确（真机 14/14 PASS），
 *        但性能灾难：blockLen=4/blockCount=count 时每块有约 34~60ns 固定开销，
 *        B=4096 每核 86 行 × 3 列 = 258 块 → aiv_mte2 15.8µs、总时长 21.4µs。
 *        **DMA 描述符数量是致命变量，必须保持 O(1)**，列抽取只能在 UB 内用向量做。
 *
 *  (2) 两个实例化共用一个 TPipe，UB 缓冲按「一块 workBuf_ + 两个队列」分配：
 *      y1/y2/cy/yref 四个 float 槽、三条 int32 下标向量、fp16 输出兜底全部按偏移切开，
 *      InitBuffer 次数 8→3（fp32）/10→5（fp16）。每次 InitBuffer 约 30~40ns 标量开销，
 *      与数据量无关，但会累加在每次 launch 上。
 *      （历史上还有一条 KernelSbvScalar 标量回退路径，已在 fp16 也改为向量后删除。）
 *
 *  ---------------------------------------------------------------------------
 *  设计要点 1：tiling 参数用一次 DMA 取回，不做逐字段 GM 标量读
 *  ---------------------------------------------------------------------------
 *  实测：单次 GM 标量读约 0.48µs（对比「有/无 threshold」两个同 B 用例得出）。
 *  `GET_TILING_DATA_WITH_STRUCT` 是编译器内建，其 prologue 阶段实测有约 3.0µs
 *  落在 aiv_scalar 上，而当时 kernel 还没发起任何自己写的 GM 读 —— 即 TilingData
 *  的取回是这条固定开销的主因。这里改成：一次 DataCopyPad 把 32B TilingData
 *  搬进 UB，再按结构体读字段（UB 标量读约 8 cycle）。
 *
 *  ---------------------------------------------------------------------------
 *  设计要点 2：行划分在 kernel 内按 GetBlockNum() 自洽计算
 *  ---------------------------------------------------------------------------
 *  host 的 SetBlockDim 只是并行度请求，部分平台会裁剪。若 kernel 沿用 host 预算的
 *  rowsPerCore，被裁掉的 block 负责的行就永远不会被写（实测 B=4096 时 2032/4096 行未写）。
 *
 *  ---------------------------------------------------------------------------
 *  设计要点 3：尾块安全
 *  ---------------------------------------------------------------------------
 *  count 是运行期值（<= tileB <= TILE_ROWS），搬运长度、Gather 长度、向量长度
 *  全部用同一个 count，不按 32/64 对齐假设多读；每个缓冲都按 TILE_ROWS 满配 + 64B 余量。
 *
 *  ---------------------------------------------------------------------------
 *  设计要点 4：输出宽度跟着调用方的张量走（不是常数）
 *  ---------------------------------------------------------------------------
 *  题目正文要求输出 float32，官方模板 fp16 配置列的输出却是 fp16 —— 两者冲突，
 *  而 aclnn 的 dtype 校验是「整组列匹配」，猜错一列就是 161002、kernel 完全不下发。
 *  因此 host 侧读 violation_score 的真实 desc dtype 写进 TilingData.outIsF32。
 */

namespace sbv {

// 注意：TILE_ROWS 定义在共享头文件 safety_belt_violation_geometry_tiling.h 的 sbv 命名空间中
// （host 与 kernel 单一真源），这里只派生出与 UB 相关的容量常量。
constexpr uint32_t TILE_BYTES  = TILE_ROWS * sizeof(float);   // 单列紧凑容量
constexpr uint32_t PB_BYTES    = 4U * TILE_BYTES + 64U;       // [4*TILE_ROWS] 个 float + 余量
constexpr uint32_t AP_BYTES    = 2U * TILE_BYTES + 64U;       // [2*TILE_ROWS] 个 float + 余量
constexpr int32_t  QUEUE_DEPTH = 2;                           // 双缓冲流水
constexpr int32_t  VEC_QDEPTH  = 1;
// ---------------------------------------------------------------------------
//  工作缓冲合并（实测：InitBuffer 每次约 30~40ns 的标量开销，且与数据量无关）
//  y1/y2/cy/yref 四个 float 槽 + 3 条 int32 下标向量 + fp16 输出兜底，
//  全部落在一块 TBuf 里按偏移切，InitBuffer 次数从 8 降到 3（fp32）/10 降到 5（fp16）。
//  每个槽 TILE_BYTES+64：2112 字节，32B 整数倍，满足向量指令的对齐要求。
// ---------------------------------------------------------------------------
constexpr uint32_t WORK_SLOT_BYTES = TILE_BYTES + 64U;              // 2112
constexpr uint32_t WORK_SLOT_F     = WORK_SLOT_BYTES / 4U;          // 528（float 元素）
constexpr uint32_t WORK_IDX_OFF_F  = 4U * WORK_SLOT_F;              // 2112（int32 元素偏移）
constexpr uint32_t WORK_IDX_BYTES  = 3U * TILE_BYTES + 64U;         // 6208
constexpr uint32_t WORK_OUTH_OFF_H = (4U*WORK_SLOT_BYTES + WORK_IDX_BYTES) / 2U;
constexpr uint32_t WORK_BYTES      = 4U*WORK_SLOT_BYTES + WORK_IDX_BYTES
                                     + (TILE_BYTES / 2U + 64U);                           // 全向量路径每核 1~2 个 tile，单缓冲足够

// vgather 的 srcOffset 单位（实测结论，勿按文档直觉改成「元素」）：
// 传「元素下标 4i+1」时抽出来的值逐位等于 src[i]，即偏移被按 4 字节折算过 ——
// 也就是 srcOffset 是**字节偏移**。因此下标构造要乘元素宽度。
constexpr uint32_t IDX_UNIT = sizeof(float);   // 4

// >>> 临时消融开关（编译期常量，正常路径零成本）
//   SBV_ENTRY_ONLY: 0=完整  1=构造任何对象前就返回（纯 launch 地板）
//                   2=只跑 Init（tiling/行划分/InitBuffer/下标），跳过 Process
//   SBV_SKIP_IDX:   1=把下标向量的长度置 0（等价于不构造下标），仅配合 ENTRY_ONLY=2 使用
constexpr int SBV_ENTRY_ONLY = 0;
constexpr int SBV_SKIP_IDX   = 0;

// blockCount 走 uint16_t，且 DataCopyPad 上限 4095：编译期锁死容量依据
static_assert(TILE_ROWS <= MAX_BLOCK_COUNT, "TILE_ROWS 超限，count 转 uint16_t 会截断");

/* -------------------------------------------------------------------------- *
 *  编译期 dtype 判定（不用 <type_traits>，避免 STL 依赖）
 * -------------------------------------------------------------------------- */
template <typename T>
struct IsF32 { static constexpr bool value = false; };

template <>
struct IsF32<float> { static constexpr bool value = true; };

/* -------------------------------------------------------------------------- *
 *  标量类型转换：fp16 -> fp32（仅 fp16 老路径使用）
 * -------------------------------------------------------------------------- */
template <typename T>
struct ToF32 {
    __aicore__ static inline float Cvt(T v) { return static_cast<float>(v); }
};

template <>
struct ToF32<float> {
    __aicore__ static inline float Cvt(float v) { return v; }
};

}  // namespace sbv

/* ========================================================================== *
 *  单模板类实现：fp32 / fp16 共用一份源码
 *  --------------------------------------------------------------------------
 *  历史上 fp32 与 fp16 各写一个类，两类的 Init / ProcessTile / 成员声明几乎逐行
 *  相同，静态检查会报大量重复代码，单个函数也超过 50 行。这里合并为一个类模板：
 *    - 输入宽度只影响「整块载入 + 是否 Cast」，用 if constexpr 在编译期分派；
 *    - 其余（tiling、行划分、下标、gather、判定、写回）只有一份实现；
 *    - 每个成员函数都控制在 50 行以内。
 *  fp32 实例不会分配 pbF_/apF_，InitBuffer 次数与合并前完全一致（3 / 5 次），
 *  指令序列和参数逐条保持不变，因此数值结果与性能特征不变。
 * ========================================================================== */
template <typename InT>
class KernelSbvVec {
public:
    __aicore__ inline KernelSbvVec() {}

    __aicore__ inline void Init(GM_ADDR person_boxes,
                                GM_ADDR anchor_points,
                                GM_ADDR threshold,
                                GM_ADDR violation_score,
                                GM_ADDR tilingAddr)
    {
        ReadTiling(tilingAddr, threshold);
        if (sbv::SBV_ENTRY_ONLY == 4) { return; }
        SplitRows();
        if (rowsThisCore_ == 0U) { return; }
        SetupGm(person_boxes, anchor_points, violation_score);
        SetupBuffers();
        BuildIndex();
    }

    __aicore__ inline void Process()
    {
        uint32_t done = 0U;
        while (done < rowsThisCore_) {
            const uint32_t remain = rowsThisCore_ - done;
            const uint32_t count  = (remain < tileB_) ? remain : tileB_;
            ProcessTile(blockStart_ + done, count);
            done += count;
        }
    }

private:
    static constexpr bool kInF16 = !sbv::IsF32<InT>::value;

    /* ---- 1. tiling：一次 DMA 搬入 UB，事件只等 MTE2_S ---- */
    __aicore__ inline void ReadTiling(GM_ADDR tilingAddr, GM_ADDR threshold)
    {
        pipe_.InitBuffer(tilBuf_, 64U);
        LocalTensor<uint32_t> til = tilBuf_.Get<uint32_t>();
        GlobalTensor<uint32_t> tilGm;
        tilGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tilingAddr), 8U);
        DataCopyExtParams cpTil{};
        cpTil.blockCount = 1U;
        cpTil.blockLen   = 32U;
        cpTil.srcStride  = 0U;
        cpTil.dstStride  = 0U;
        DataCopyPadExtParams<uint32_t> padTil{};
        DataCopyPad(til, tilGm, cpTil, padTil);
        {
            TEventID evtTil = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
            SetFlag<HardEvent::MTE2_S>(evtTil);
            WaitFlag<HardEvent::MTE2_S>(evtTil);
        }
        __ubuf__ SafetyBeltViolationGeometryTilingData* pt =
            (__ubuf__ SafetyBeltViolationGeometryTilingData*)til.GetPhyAddr();
        batch_ = pt->batch;
        waistRatio_ = pt->waistRatio;
        if (!(waistRatio_ > 0.0f && waistRatio_ <= 1.0f)) { waistRatio_ = 1.0f; }
        hasThreshold_ = (pt->hasThreshold != 0U);
        thrIsF32_     = (pt->thrIsF32 != 0U);
        outIsF32_     = (pt->outIsF32 != 0U);
        thrValue_     = pt->thresholdValue;
        if (!(thrValue_ >= 0.0f && thrValue_ <= 1.0e30f)) { thrValue_ = 0.0f; }
        thrGm_ = threshold;
    }

    /* ---- 2. 行划分：按运行期实际启动的 block 数自洽计算 ---- */
    __aicore__ inline void SplitRows()
    {
        uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
        if (blockNum == 0U) { blockNum = 1U; }
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        rowsPerCore_ = (batch_ + blockNum - 1U) / blockNum;
        if (rowsPerCore_ == 0U) { rowsPerCore_ = 1U; }
        // 对齐到 8 行使 GM 起址与长度满足 32B 对齐，走无 padding 的 DataCopy 快路径
        rowsPerCore_ = (rowsPerCore_ + 7U) & ~7U;
        tileB_ = (rowsPerCore_ < sbv::TILE_ROWS) ? rowsPerCore_ : sbv::TILE_ROWS;
        const uint64_t startRow64 =
            static_cast<uint64_t>(coreIdx) * static_cast<uint64_t>(rowsPerCore_);
        blockStart_ = (startRow64 < static_cast<uint64_t>(batch_))
                          ? static_cast<uint32_t>(startRow64) : batch_;
        rowsThisCore_ = 0U;
        if (blockStart_ < batch_) {
            const uint32_t remain = batch_ - blockStart_;
            rowsThisCore_ = (remain < rowsPerCore_) ? remain : rowsPerCore_;
        }
    }

    /* ---- 3. GM 视图：输入宽度由 InT 决定，输出两种宽度都建 ---- */
    __aicore__ inline void SetupGm(GM_ADDR pb, GM_ADDR ap, GM_ADDR out)
    {
        pbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(pb),
                              static_cast<int64_t>(batch_) * 4);
        apGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(ap),
                              static_cast<int64_t>(batch_) * 2);
        outGmF32_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out),
                                  static_cast<int64_t>(batch_));
        outGmF16_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out),
                                  static_cast<int64_t>(batch_));
    }

    /* ---- 4. UB 分配：fp16 多两块 Cast 目标，fp32 仍是 3 次 InitBuffer ---- */
    __aicore__ inline void SetupBuffers()
    {
        pipe_.InitBuffer(inQue_, sbv::VEC_QDEPTH, sbv::PB_BYTES + sbv::AP_BYTES);
        if constexpr (kInF16) {
            pipe_.InitBuffer(pbF_, sbv::PB_BYTES);
            pipe_.InitBuffer(apF_, sbv::AP_BYTES);
        }
        pipe_.InitBuffer(workBuf_, sbv::WORK_BYTES);
        pipe_.InitBuffer(outQue_, sbv::VEC_QDEPTH, sbv::TILE_BYTES + 64U);
    }

    /* ---- 5. 字节下标向量：只造一条 ramp，y1/y2 由它向量派生 ---- */
    __aicore__ inline void BuildIndex()
    {
        LocalTensor<int32_t> idxAll = workBuf_.Get<int32_t>()[sbv::WORK_IDX_OFF_F];
        LocalTensor<int32_t> idxY1 = idxAll;
        LocalTensor<int32_t> idxY2 = idxAll[sbv::TILE_ROWS];
        LocalTensor<int32_t> idxCy = idxAll[2U * sbv::TILE_ROWS];
        const uint32_t nIdx = (sbv::SBV_SKIP_IDX != 0) ? 0U : tileB_;
        CreateVecIndex(idxCy, static_cast<int32_t>(0), nIdx);
        Muls(idxCy, idxCy, static_cast<int32_t>(2U * sbv::IDX_UNIT), nIdx);
        Adds(idxCy, idxCy, static_cast<int32_t>(1U * sbv::IDX_UNIT), nIdx);
        Muls(idxY1, idxCy, static_cast<int32_t>(2), nIdx);
        Adds(idxY1, idxY1, -static_cast<int32_t>(1U * sbv::IDX_UNIT), nIdx);
        Adds(idxY2, idxY1, static_cast<int32_t>(2U * sbv::IDX_UNIT), nIdx);
        PipeBarrier<PIPE_V>();
    }

    /* ---- 6. 整块载入：InT 在类模板实例化时确定，无需成员函数模板 ---- */
    __aicore__ inline void CopyIn(uint32_t rowStart, uint32_t count, uint32_t apElemOff)
    {
        DataCopyPadExtParams<InT> padParams{};
        LocalTensor<InT> inAll = inQue_.AllocTensor<InT>();
        LocalTensor<InT> pbRaw = inAll;
        LocalTensor<InT> apRaw = inAll[apElemOff];
        const bool fastPath = (((rowStart & 7U) == 0U) && ((count & 7U) == 0U));
        if (fastPath) {
            DataCopy(pbRaw, pbGm_[static_cast<int64_t>(rowStart) * 4],
                     static_cast<uint32_t>(4U * count));
            DataCopy(apRaw, apGm_[static_cast<int64_t>(rowStart) * 2],
                     static_cast<uint32_t>(2U * count));
        } else {
            DataCopyExtParams cpPb{}, cpAp{};
            cpPb.blockCount = 1U;
            cpPb.blockLen   = static_cast<uint32_t>(4U * count * sizeof(InT));
            cpPb.srcStride  = 0U;
            cpPb.dstStride  = 0U;
            cpAp.blockCount = 1U;
            cpAp.blockLen   = static_cast<uint32_t>(2U * count * sizeof(InT));
            cpAp.srcStride  = 0U;
            cpAp.dstStride  = 0U;
            DataCopyPad(pbRaw, pbGm_[static_cast<int64_t>(rowStart) * 4], cpPb, padParams);
            DataCopyPad(apRaw, apGm_[static_cast<int64_t>(rowStart) * 2], cpAp, padParams);
        }
        inQue_.EnQue(inAll);
    }

    /* ---- 7. threshold 的 GM 标量读：与 MTE2 重叠，全程只读一次 ---- */
    __aicore__ inline void LoadThreshold()
    {
        if (thrLoaded_) { return; }
        float raw = thrValue_;
        if (hasThreshold_ && thrGm_ != nullptr) {
            raw = thrIsF32_ ? *(reinterpret_cast<__gm__ float*>(thrGm_))
                            : static_cast<float>(*(reinterpret_cast<__gm__ half*>(thrGm_)));
        }
        thrValue_  = (raw >= 0.0f && raw <= 1.0e30f) ? raw : 0.0f;
        thrLoaded_ = true;
    }

    /* ---- 8. 载入 + 列抽取；fp16 先整块 Cast，再复用同一套字节下标 ---- */
    __aicore__ inline void LoadAndExtract(uint32_t rowStart, uint32_t count)
    {
        // fp32 的 ap 视图按字节容量切（PB_BYTES/4 个 float）；fp16 按元素数切
        constexpr uint32_t kApOff = kInF16
            ? (4U * sbv::TILE_ROWS)
            : (sbv::PB_BYTES / static_cast<uint32_t>(sizeof(float)));
        CopyIn(rowStart, count, kApOff);
        LoadThreshold();
        LocalTensor<InT> inDeq = inQue_.DeQue<InT>();
        if constexpr (kInF16) {
            LocalTensor<float> pbF = pbF_.Get<float>();
            LocalTensor<float> apF = apF_.Get<float>();
            Cast(pbF, inDeq, RoundMode::CAST_NONE, static_cast<int32_t>(4U * count));
            Cast(apF, inDeq[kApOff], RoundMode::CAST_NONE, static_cast<int32_t>(2U * count));
            inQue_.FreeTensor(inDeq);
            GatherColumns(pbF, apF, count);
        } else {
            GatherColumns(inDeq, inDeq[kApOff], count);
            inQue_.FreeTensor(inDeq);
        }
    }

    /* ---- 9. vgather 抽三列：y1/y2/cy 落到 workBuf 的三个槽 ---- */
    __aicore__ inline void GatherColumns(LocalTensor<float> pbF,
                                         LocalTensor<float> apF, uint32_t count)
    {
        LocalTensor<float> work = workBuf_.Get<float>();
        LocalTensor<float> y1 = work;
        LocalTensor<float> y2 = work[sbv::WORK_SLOT_F];
        LocalTensor<float> cy = work[2U * sbv::WORK_SLOT_F];
        LocalTensor<uint32_t> idxAllU = workBuf_.Get<uint32_t>()[sbv::WORK_IDX_OFF_F];
        LocalTensor<uint32_t> idxY1U = idxAllU;
        LocalTensor<uint32_t> idxY2U = idxAllU[sbv::TILE_ROWS];
        LocalTensor<uint32_t> idxCyU = idxAllU[2U * sbv::TILE_ROWS];
        Gather(y1, pbF, idxY1U, static_cast<uint32_t>(0), count);
        Gather(y2, pbF, idxY2U, static_cast<uint32_t>(0), count);
        Gather(cy, apF, idxCyU, static_cast<uint32_t>(0), count);
    }

    /* ---- 10. 判定：严格大于映射成精确 0.0/1.0（不改用 Compare+Select） ---- */
    __aicore__ inline void ComputeAndEnque(uint32_t count)
    {
        LocalTensor<float> work = workBuf_.Get<float>();
        LocalTensor<float> y1 = work;
        LocalTensor<float> y2 = work[sbv::WORK_SLOT_F];
        LocalTensor<float> cy = work[2U * sbv::WORK_SLOT_F];
        LocalTensor<float> yref = work[3U * sbv::WORK_SLOT_F];
        const int32_t n = static_cast<int32_t>(count);
        Sub(yref, y2, y1, n);
        Muls(yref, yref, waistRatio_, n);
        Add(yref, yref, y1, n);
        Adds(yref, yref, thrValue_, n);
        LocalTensor<float> outLocal = outQue_.AllocTensor<float>();
        Sub(outLocal, cy, yref, n);
        Maxs(outLocal, outLocal, 0.0f, n);
        Muls(outLocal, outLocal, 1.0e30f, n);
        Muls(outLocal, outLocal, 1.0e30f, n);
        Mins(outLocal, outLocal, 1.0f, n);
        Adds(outLocal, outLocal, 0.0f, n);
        outQue_.EnQue(outLocal);
    }

    /* ---- 11. 写回：输出宽度由 TilingData.outIsF32 决定 ---- */
    __aicore__ inline void StoreOutput(uint32_t rowStart, uint32_t count)
    {
        LocalTensor<float> res = outQue_.DeQue<float>();
        if (outIsF32_) {
            if (((rowStart & 7U) == 0U) && ((count & 7U) == 0U)) {
                DataCopy(outGmF32_[static_cast<int64_t>(rowStart)], res,
                         static_cast<uint32_t>(count));
            } else {
                DataCopyExtParams cpOut{};
                cpOut.blockCount = 1U;
                cpOut.blockLen   = static_cast<uint32_t>(count * sizeof(float));
                cpOut.srcStride  = 0U;
                cpOut.dstStride  = 0U;
                DataCopyPad(outGmF32_[static_cast<int64_t>(rowStart)], res, cpOut);
            }
            outQue_.FreeTensor(res);
            return;
        }
        StoreHalf(rowStart, count, res);
    }

    /* ---- 12. fp16 输出：整块 Cast + 多 tile 时的 MTE3 收尾栅栏 ---- */
    __aicore__ inline void StoreHalf(uint32_t rowStart, uint32_t count,
                                     LocalTensor<float> res)
    {
        LocalTensor<half> outHalf = workBuf_.Get<half>()[sbv::WORK_OUTH_OFF_H];
        Cast(outHalf, res, RoundMode::CAST_RINT, static_cast<int32_t>(count));
        PipeBarrier<PIPE_ALL>();
        outQue_.FreeTensor(res);
        DataCopyExtParams cpOutH{};
        cpOutH.blockCount = 1U;
        cpOutH.blockLen   = static_cast<uint32_t>(count * sizeof(half));
        cpOutH.srcStride  = 0U;
        cpOutH.dstStride  = 0U;
        DataCopyPad(outGmF16_[static_cast<int64_t>(rowStart)], outHalf, cpOutH);
        // outHalf 是手动 TBuf，框架不建立 MTE3 -> V 反向同步。仅当本核还有
        // 下一个 tile 时补栅栏，避免下一轮 Cast 覆盖上一轮未搬完的写回。
        if (rowStart + count < blockStart_ + rowsThisCore_) {
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessTile(uint32_t rowStart, uint32_t count)
    {
        LoadAndExtract(rowStart, count);
        ComputeAndEnque(count);
        StoreOutput(rowStart, count);
    }

    /* ---------------- UB 缓冲 ---------------- */
    TPipe pipe_;
    TQue<TPosition::VECIN,  sbv::VEC_QDEPTH> inQue_;
    TQue<TPosition::VECOUT, sbv::VEC_QDEPTH> outQue_;
    TBuf<TPosition::VECCALC> tilBuf_;
    TBuf<TPosition::VECCALC> pbF_;
    TBuf<TPosition::VECCALC> apF_;
    TBuf<TPosition::VECCALC> workBuf_;

    /* ---------------- GM 视图 ---------------- */
    GlobalTensor<InT>   pbGm_, apGm_;
    GlobalTensor<float> outGmF32_;
    GlobalTensor<half>  outGmF16_;
    __gm__ uint8_t*     thrGm_ = nullptr;

    /* ---------------- 标量参数 ---------------- */
    uint32_t batch_        = 0U;
    uint32_t rowsPerCore_  = 0U;
    uint32_t tileB_        = 0U;
    uint32_t blockStart_   = 0U;
    uint32_t rowsThisCore_ = 0U;
    float    waistRatio_   = 1.0f;
    float    thrValue_     = 0.0f;
    bool     hasThreshold_ = false;
    bool     thrIsF32_     = true;
    bool     outIsF32_     = true;
    bool     thrLoaded_    = false;
};

template <typename DT_PERSON_BOXES>
__global__ __aicore__ void safety_belt_violation_geometry(GM_ADDR person_boxes,
                                                          GM_ADDR anchor_points,
                                                          GM_ADDR threshold,
                                                          GM_ADDR violation_score,
                                                          GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    (void)workspace;  // 本算子不需要 workspace

    // tiling 结构体注册：编译期指令，必须出现在 kernel 入口这一层。
    // 它原先挂在 KernelSbvScalar::Init 里 —— 那个类运行期从未被实例化，
    // 但这条注册指令是"活的"：删掉那个类，14 例会全部 ERROR。
    REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);

    if (sbv::SBV_ENTRY_ONLY == 1) { return; }

    (void)sizeof(DT_PERSON_BOXES);
    if (sbv::IsF32<DT_PERSON_BOXES>::value) {
        KernelSbvVec<float> op;
        op.Init(person_boxes, anchor_points, threshold, violation_score, tiling);
        if (sbv::SBV_ENTRY_ONLY == 0) { op.Process(); }
    } else {
        KernelSbvVec<half> op;
        op.Init(person_boxes, anchor_points, threshold, violation_score, tiling);
        if (sbv::SBV_ENTRY_ONLY == 0) { op.Process(); }
    }
}
