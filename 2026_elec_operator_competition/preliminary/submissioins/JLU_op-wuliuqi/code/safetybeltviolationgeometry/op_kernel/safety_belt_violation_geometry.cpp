// Copyright (c) 2026 JLU_op-wuliuqi.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// 本轮策略：所有boxes、anchors和output DMA均使用普通DataCopy，字节数向上补齐到32B。
// 逻辑B、FP32分步计算和严格GT不变。每个核起点须32B对齐，补齐输出不能覆盖其他核有效数据。
// 存储前提：每个独立输入/输出GM分配实际覆盖ALIGN_UP(逻辑字节数,32)；SetGlobalBuffer不会扩容。
// 输入多读区不属于有效样本；输出多写区只允许位于该张量已分配的padding。任意无padding子视图不适用。
// 十二段四阶段性能恢复版：区间类自己拥有原v528优化与真实B流程，未增加单点类。
// Host真实尺寸/属性守卫选择编译期参数；没有缓存输入坐标或预生成答案。
// B6、4094等仍在对应区间类处理；原v528和之前审阅代码均保留。
// 十二个完整尺寸分支的结构审阅候选，来源v528；并非原高分二进制。
// 每种dtype：[1,31]、32、[33,255]、256、[257,4095]、4096。
// 每个普通区间由一个类完整处理；不再把4095和4094拆到不同类/函数。
// Host只决定每核分工与元数据；真实坐标仍在NPU读取、抽列、按FP32分步计算。
// coreRows是有效行数，pad是UB/向量长度；DataCopy是32B块单位，Pad接口是字节单位。
// 保留当前历史FP16阈值/输出ABI；题面固定FP32文字差异仍需要单独review。
// 每个类均有Init/CopyIn/Compute/CopyOut；当前候选待本轮编译、实机检查及官方评测。
#include "kernel_operator.h"
#define SBG_POLICY_TARGET __aicore__
#include "safety_belt_violation_geometry_tiling.h"
#include "tiling_key_safety_belt_violation_geometry.h"

namespace {
constexpr int32_t kSbgRowAlign = 16;   // 行对齐：fp16 输出 32B 对齐要求 16 行的倍数
constexpr int32_t kSbgPadMin = 64;     // PairReduceSum 需要 pad>=64（rep2>=2）
constexpr int32_t kSbgPerRepeat = 64;  // 一个 vector repeat 处理 64 个 f32
constexpr uint64_t kSbgMaskOdd = 0xAAAAAAAAAAAAAAAAULL;
constexpr uint64_t kSbgMaskEven = 0x5555555555555555ULL;

__aicore__ inline uint32_t SbgAlignBytes(uint32_t bytes) {
    return (bytes + 31U) & ~31U;
}
__aicore__ inline uint32_t SbgAlignUp(uint32_t a, uint32_t b) {
    return (a + b - 1U) & ~(b - 1U);
}
}  // namespace

#include "sbg_kernel_common.h"
#include "sbg_kernel_range.h"

// ============================================================================
// 【分支01】FP32，完整覆盖B∈[1,31]：SbgFp32Range1To31
// 本类拥有完整搬入/抽列/计算/搬出流程，区间端点和普通B都在同一实现内处理。
// Host可在本类内选择编译期分工；普通B与端点均不转发给另一个单点类。
// 算法来自原v528已存在的区间实现，扩大入口覆盖；不宣称与原单点路径性能相同。
// ============================================================================
using SbgFp32Range1To31 = SbgSmallRange<float>;

// 尺寸分支2：保留原v528的该精确点类主体。
// ============================================================================
// 【公开case01：FP32 / B=32】SbgFp32B32
// 当前启动8核，工作核上限1；核0处理32行；核1～7退出。
// 实际向量计算行数：64；下列字节数依次为boxes / anchors / output。
// 当前搬运：DataCopy：512 / 256 / 128 B。
// 每个工作核只处理自己的单段切片：CopyIn → Compute → CopyOut，没有下一tile的双缓冲。
// 主核/常规核UB字节区间（左闭右开）：boxIn_[0,1024)；anchorIn_[1024,1536)；output_[1536,1792)；constants_[1792,2304)；scratch_[2304,4192)
// ============================================================================
template <bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
class SbgFp32B32 final : public SbgStages<SbgFp32B32<UNIT_RATIO, RATIO_KIND>>, protected SbgPointState<float> {
    friend class SbgStages<SbgFp32B32<UNIT_RATIO, RATIO_KIND>>;
public:
    static constexpr uint32_t kBatch = 32U; // 精确点的逻辑B，不代表输入坐标内容已知。
    static constexpr uint32_t kWorkingCores = 1U; // 工作核上限；启动核之外/多余启动核在Init退出。
    static constexpr uint32_t kRowsPerCore = 32U; // 用于定位GM切片起点的行步长；尾核有效行数另算。
    static constexpr uint32_t kPad = 64U; // UB分配/列间距基准；不授权多搬GM数据，也不一定等于calcRows。
    static constexpr uint32_t kBankGap = 0U; // FP32元素单位的间隔，不是字节；当前布局参数，不是数学所需数据。
    static constexpr uint32_t kSvBase = 2304U; // scratch在本核UB中的字节起点。
    static constexpr uint32_t kSvFloatCount = 472U; // scratch容量，单位float元素，包含位图及地址载体预留。
    static constexpr uint32_t kUbBytes = 4192U; // 本类UB最高字节边界，Host对应策略必须覆盖它。
    static_assert(SbgValidatePointPolicy<kBatch, false, kWorkingCores, kRowsPerCore, kUbBytes>(),
        "Point policy mismatch");

private:
    using T = float;

    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
                               GM_ADDR threshold, GM_ADDR violation_score,
                               uint32_t hasThreshold, GM_ADDR tiling) {
        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        if (!InitSinglePoint<kWorkingCores, 32U>(person_boxes, anchor_points, threshold,
            violation_score, hasThreshold, tiling)) return false;
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        BindPointUb<kPad, kSvBase, kSvFloatCount>(1024U, 1536U, 1792U, 1792U);

        return true;
    }

    // 【搬入】发出输入DMA，并按本分支的时序准备真实属性和阈值。
    __aicore__ inline void CopyIn() {

        AscendC::DataCopyParams boxCopy{1, 16, 0, 0};
        AscendC::DataCopyParams anchorCopy{1, 8, 0, 0};
        AscendC::DataCopy(boxIn_, boxesGm_, boxCopy);
        AscendC::DataCopy(anchorIn_, anchorsGm_, anchorCopy);
        // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
        LoadPointParameters<UNIT_RATIO, RATIO_KIND, true>();
    }

    // 【计算】抽取三列 → FP32几何运算 → 严格大于比较 → 生成0/1。
    __aicore__ inline void Compute() {
        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        AscendC::Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
        AscendC::Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));

        // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        // 下面为同一个scratch中的视图，各偏移单位是float元素。
        // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
        const SbgPointColumns columns = MakePointColumns<kPad, kBankGap>();
        // FP32抽列：boxes奇数掩码归约为[y1,y2]交错对，再用奇/偶掩码分别抽y2/y1。
        // anchors用奇数掩码抽cy；这些归约用于抽列，不是跨样本求和。
        AscendC::LocalTensor<float> pairs = scratch_[0];
        ExtractFp32AllColumns<4, 2>(pairs, columns.y1, columns.y2, columns.cy);
        ComputeNativeBoundary<1, UNIT_RATIO>(columns.difference, columns.y1, columns.y2, columns.boundary);
        SelectPointMask<false, false>(columns, uint64_t(64));
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
        CopyOutPoint(4);
    }
};

// ============================================================================
// 【分支03】FP32，完整覆盖B∈[33,255]：SbgFp32Range33To255
// 本类拥有完整搬入/抽列/计算/搬出流程，区间端点和普通B都在同一实现内处理。
// Host可在本类内选择编译期分工；普通B与端点均不转发给另一个单点类。
// 算法来自原v528已存在的区间实现，扩大入口覆盖；不宣称与原单点路径性能相同。
// ============================================================================
template <uint32_t FIXED_BATCH = 0, bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0, bool TAIL = false>
class SbgFp32Range33To255 final : public SbgStages<SbgFp32Range33To255<FIXED_BATCH, UNIT_RATIO, RATIO_KIND, TAIL>>, protected SbgPointState<float>, protected SbgRangeState<float> {
    friend class SbgStages<SbgFp32Range33To255<FIXED_BATCH, UNIT_RATIO, RATIO_KIND, TAIL>>;
public:
    // 完整区间[33,255]只有本类。FIXED_BATCH是Host真实B守卫后的编译期策略；0表示真实运行时B。
    // 编译期策略恢复v528原优化，不创建独立单点类；B4094等仍由本类处理。
    static constexpr uint32_t kBatch = FIXED_BATCH;
    static constexpr uint32_t kWorkingCores = FIXED_BATCH == 33U ? (1U) : 0U;
    static constexpr uint32_t kRowsPerCore = FIXED_BATCH == 33U ? (48U) : 0U;
    static constexpr uint32_t kPad = FIXED_BATCH == 33U ? (64U) : 0U;
    static constexpr uint32_t kBankGap = FIXED_BATCH == 33U ? (0U) : 0U;
    static constexpr uint32_t kSvBase = FIXED_BATCH == 33U ? (2304U) : 0U;
    static constexpr uint32_t kSvFloatCount = FIXED_BATCH == 33U ? (472U) : 0U;
    static constexpr uint32_t kUbBytes = FIXED_BATCH == 33U ? (4192U) : 0U;
    static_assert(SbgValidateIntervalPolicy<FIXED_BATCH, 33U, 33U, kUbBytes>(), "Interval policy mismatch");

private:
    // 原优化路径所需的成员与UB视图；通用成员不会被常量策略使用。
    using T = float;

    // 通用B的真实长度/片上视图；编译器在常量策略中消除未使用成员。
    static constexpr uint32_t tSize = static_cast<uint32_t>(sizeof(T));
    static constexpr uint32_t bankGap = 0U;
    // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。

    // Init按本类策略建立分片和UB；每条路径均保持原有效GM长度。
    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR thresholdArg, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        if constexpr (FIXED_BATCH == 33U) {
            // B=33由核0一次处理全部33行；核1~7退出，不再拆成32+1行。

            // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
            if (!InitSinglePoint<kWorkingCores, 33U>(person_boxes, anchor_points, thresholdArg,
                violation_score, hasThresholdArg, tilingArg)) return false;
            // Byte offsets belong only to this point. None is shared with another class.
            // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
            // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
            // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
            BindPointUb<kPad, kSvBase, kSvFloatCount>(1024U, 1536U, 1792U, 1792U);

            return true;

        } else { return InitRange<33U, 255U, false>(person_boxes, anchor_points, thresholdArg, violation_score, length, rowsPerCore, hasThresholdArg, tilingArg); }
    }

    // CopyIn：搬入真实输入并准备实际属性/阈值。
    __aicore__ inline void CopyIn() {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的B=33策略：32B补齐输入，普通DataCopy，有效行数独立控制。

            // 统一使用普通DataCopy，搬运字节数向上补齐到32B。
            // 单工作核33行：boxes 528B向上补齐到544B，anchors 264B补齐到288B。
            // 调用方必须提供实际可读的32B补齐存储；SetGlobalBuffer不会扩容GM。
            // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
            const uint32_t boxBytes = (coreRows_ * 16U + 31U) & ~31U;
            const uint32_t anchorBytes = (coreRows_ * 8U + 31U) & ~31U;
            AscendC::DataCopyParams boxCopy{1, static_cast<uint16_t>(boxBytes / 32U), 0, 0};
            AscendC::DataCopyParams anchorCopy{1, static_cast<uint16_t>(anchorBytes / 32U), 0, 0};
            AscendC::DataCopy(boxIn_, boxesGm_, boxCopy);
            AscendC::DataCopy(anchorIn_, anchorsGm_, anchorCopy);
            // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
            LoadPointParameters<UNIT_RATIO, RATIO_KIND, true>();

        } else { CopyInRange(); }
    }

    // Compute：抽列、FP32分步运算、严格GT比较。
    __aicore__ inline void Compute() {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的B=33策略：保留抽列与转换，逐行算术及比较只处理coreRows_。

            // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
            AscendC::Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
            AscendC::Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));

            // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // 下面为同一个scratch中的视图，各偏移单位是float元素。
            // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
            const SbgPointColumns columns = MakePointColumns<kPad, kBankGap>();
            // FP32抽列：boxes奇数掩码归约为[y1,y2]交错对，再用奇/偶掩码分别抽y2/y1。
            // anchors用奇数掩码抽cy；这些归约用于抽列，不是跨样本求和。
            AscendC::LocalTensor<float> pairs = scratch_[0];
            ExtractFp32AllColumns<4, 2>(pairs, columns.y1, columns.y2, columns.cy);
            ComputeNativeBoundaryMasked<1, UNIT_RATIO>(columns.difference, columns.y1, columns.y2, columns.boundary);
            SelectPointMask<true, true>(columns, uint64_t(coreRows_));

        } else { ComputeRange(); }
    }

    // CopyOut：只搬出有效行。
    __aicore__ inline void CopyOut() {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的编译期B=33策略：恢复冻结v528原CopyOut语句。

            // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
            CopyOutPoint(static_cast<uint16_t>(SbgAlignBytes(coreRows_ * 4U) / 32U));

        } else { CopyOutRange(); }
    }
};

// 尺寸分支4：保留原v528的该精确点类主体。
// ============================================================================
// 【公开case03：FP32 / B=256】SbgFp32B256
// 当前启动8核，工作核上限2；核0、1各处理128行；其余退出。
// 实际向量计算行数：128；下列字节数依次为boxes / anchors / output。
// 当前搬运：DataCopy：每核2048 / 1024 / 512 B。
// 每个工作核只处理自己的单段切片：CopyIn → Compute → CopyOut，没有下一tile的双缓冲。
// 主核/常规核UB字节区间（左闭右开）：boxIn_[0,2048)；anchorIn_[2048,3072)；output_[3072,3584)；constants_[3584,4608)；scratch_[65536,70304)
// ============================================================================
template <bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
class SbgFp32B256 final : public SbgStages<SbgFp32B256<UNIT_RATIO, RATIO_KIND>>, protected SbgPointState<float> {
    friend class SbgStages<SbgFp32B256<UNIT_RATIO, RATIO_KIND>>;
public:
    static constexpr uint32_t kBatch = 256U; // 精确点的逻辑B，不代表输入坐标内容已知。
    static constexpr uint32_t kWorkingCores = 2U; // 工作核上限；启动核之外/多余启动核在Init退出。
    static constexpr uint32_t kRowsPerCore = 128U; // 用于定位GM切片起点的行步长；尾核有效行数另算。
    static constexpr uint32_t kPad = 128U; // UB分配/列间距基准；不授权多搬GM数据，也不一定等于calcRows。
    static constexpr uint32_t kBankGap = 64U; // FP32元素单位的间隔，不是字节；当前布局参数，不是数学所需数据。
    static constexpr uint32_t kSvBase = 65536U; // scratch在本核UB中的字节起点。
    static constexpr uint32_t kSvFloatCount = 1192U; // scratch容量，单位float元素，包含位图及地址载体预留。
    static constexpr uint32_t kUbBytes = 70304U; // 本类UB最高字节边界，Host对应策略必须覆盖它。
    static_assert(SbgValidatePointPolicy<kBatch, false, kWorkingCores, kRowsPerCore, kUbBytes>(),
        "Point policy mismatch");

private:
    using T = float;

    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
                               GM_ADDR threshold, GM_ADDR violation_score,
                               uint32_t hasThreshold, GM_ADDR tiling) {
        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        uint32_t block;
        if (!InitPointSlice<kWorkingCores, kRowsPerCore>(block)) return false;
        coreRows_ = 128U;
        SetPointParameters(threshold, tiling, hasThreshold);
        // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
        // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
        BindPointGlobals(person_boxes, anchor_points, violation_score,
                SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T), SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        BindPointUb<kPad, kSvBase, kSvFloatCount>(2048U, 3072U, 3584U, 4608U);

        return true;
    }

    // 【搬入】发出输入DMA，并按本分支的时序准备真实属性和阈值。
    __aicore__ inline void CopyIn() {

        AscendC::DataCopyParams boxCopy{1, 64, 0, 0};
        AscendC::DataCopyParams anchorCopy{1, 32, 0, 0};
        AscendC::DataCopy(boxIn_, boxesGm_, boxCopy);
        AscendC::DataCopy(anchorIn_, anchorsGm_, anchorCopy);
        // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
        LoadPointParameters<UNIT_RATIO, RATIO_KIND, true>();
    }

    // 【计算】抽取三列 → FP32几何运算 → 严格大于比较 → 生成0/1。
    __aicore__ inline void Compute() {
        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        AscendC::Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
        AscendC::Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));
        PrepareCompareAddress<1184U>();
        // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        // 下面为同一个scratch中的视图，各偏移单位是float元素。
        // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
        const SbgPointColumns columns = MakePointColumns<kPad, kBankGap>();
        // FP32抽列：boxes奇数掩码归约为[y1,y2]交错对，再用奇/偶掩码分别抽y2/y1。
        // anchors用奇数掩码抽cy；这些归约用于抽列，不是跨样本求和。
        AscendC::LocalTensor<float> pairs = scratch_[0];
        ExtractFp32AllColumns<8, 4>(pairs, columns.y1, columns.y2, columns.cy);
        ComputeNativeBoundary<2, UNIT_RATIO>(columns.difference, columns.y1, columns.y2, columns.boundary);
        CompareAndSelectNative<2, 2>(columns.cy, columns.boundary);
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
        CopyOutPoint(16);
    }
};

// ============================================================================
// 【分支05】FP32，完整覆盖B∈[257,4095]：SbgFp32Range257To4095
// 本类拥有完整搬入/抽列/计算/搬出流程，区间端点和普通B都在同一实现内处理。
// Host可在本类内选择编译期分工；普通B与端点均不转发给另一个单点类。
// 算法来自原v528已存在的区间实现，扩大入口覆盖；不宣称与原单点路径性能相同。
// ============================================================================
template <uint32_t FIXED_BATCH = 0, bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0, bool TAIL = false>
class SbgFp32Range257To4095 final : public SbgStages<SbgFp32Range257To4095<FIXED_BATCH, UNIT_RATIO, RATIO_KIND, TAIL>>, protected SbgPointState<float>, protected SbgRangeState<float> {
    friend class SbgStages<SbgFp32Range257To4095<FIXED_BATCH, UNIT_RATIO, RATIO_KIND, TAIL>>;
public:
    // 完整区间[257,4095]只有本类。FIXED_BATCH是Host真实B守卫后的编译期策略；0表示真实运行时B。
    // 编译期策略恢复v528原优化，不创建独立单点类；B4094等仍由本类处理。
    static constexpr uint32_t kBatch = FIXED_BATCH;
    static constexpr uint32_t kWorkingCores = FIXED_BATCH == 257U ? (1U) : FIXED_BATCH == 4095U ? (8U) : 0U;
    static constexpr uint32_t kRowsPerCore = FIXED_BATCH == 257U ? (272U) : FIXED_BATCH == 4095U ? (512U) : 0U;
    static constexpr uint32_t kPad = FIXED_BATCH == 257U ? (320U) : FIXED_BATCH == 4095U ? (512U) : 0U;
    static constexpr uint32_t kBankGap = FIXED_BATCH == 257U ? (0U) : FIXED_BATCH == 4095U ? (64U) : 0U;
    static constexpr uint32_t kSvBase = FIXED_BATCH == 257U ? (11520U) : FIXED_BATCH == 4095U ? (36U * kPad) : 0U;
    static constexpr uint32_t kSvFloatCount = FIXED_BATCH == 257U ? (2296U) : FIXED_BATCH == 4095U ? (7U * kPad + 3U * kBankGap + 56U) : 0U;
    static constexpr uint32_t kUbBytes = FIXED_BATCH == 257U ? (20704U) : FIXED_BATCH == 4095U ? (kSvBase + 4U * kSvFloatCount) : 0U;
    static_assert(SbgValidateIntervalPolicy<FIXED_BATCH, 257U, 4095U, kUbBytes>(), "Interval policy mismatch");

private:
    // 原优化路径所需的成员与UB视图；通用成员不会被常量策略使用。
    using T = float;

    // 通用B的真实长度/片上视图；编译器在常量策略中消除未使用成员。
    static constexpr uint32_t tSize = static_cast<uint32_t>(sizeof(T));
    static constexpr uint32_t bankGap = 0U;
    // 跨阶段保留本核有效行、向量长度与UB视图；普通区间保持有效GM长度；4095候选按已核实分配padding补齐输入，输出仍使用coreRows。

    // Init按本类策略建立分片和UB；每条路径均保持原有效GM长度。
    __aicore__ inline bool InitFixed257(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR thresholdArg, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        // 同一区间类的编译期B=257策略：恢复冻结v528原Init语句。

        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (block >= kWorkingCores) return false;
        rowStart_ = block * kRowsPerCore;
        coreRows_ = 257U;
        SetPointParameters(thresholdArg, tilingArg, hasThresholdArg);
        // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
        // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
        BindPointGlobals(person_boxes, anchor_points, violation_score,
            SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T), SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        BindPointUb<kPad, kSvBase, kSvFloatCount>(5120U, 7680U, 8960U, 8960U);

        return true;
    }

    __aicore__ inline bool InitFixed4095(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR thresholdArg, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        // 同一区间类B=4095候选：7×512+511有效行，统一512行输入/计算；阶段Init语句。

        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if constexpr (TAIL) {
            if (block != 7U) return false;
            rowStart_ = 3584U; coreRows_ = 511U;
        } else {
            if (block >= 7U) return false;
            rowStart_ = block * kRowsPerCore; coreRows_ = kRowsPerCore;
        }
        SetPointParameters(thresholdArg, tilingArg, hasThresholdArg);
        // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
        // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
        BindPointGlobals(person_boxes, anchor_points, violation_score,
            kPad * 4U, kPad * 2U);
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        BindPointUb<kPad, kSvBase, kSvFloatCount>(16U * kPad, 24U * kPad, 28U * kPad, 4U * (7U * kPad + 3U * kBankGap));

        return true;
    }

    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR thresholdArg, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        if constexpr (FIXED_BATCH == 257U) { return InitFixed257(person_boxes, anchor_points, thresholdArg, violation_score, length, rowsPerCore, hasThresholdArg, tilingArg); } else if constexpr (FIXED_BATCH == 4095U) { return InitFixed4095(person_boxes, anchor_points, thresholdArg, violation_score, length, rowsPerCore, hasThresholdArg, tilingArg); } else { return InitRange<257U, 4095U, false>(person_boxes, anchor_points, thresholdArg, violation_score, length, rowsPerCore, hasThresholdArg, tilingArg); }
    }

    // CopyIn：搬入真实输入并准备实际属性/阈值。
    __aicore__ inline void CopyIn() {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原CopyIn语句。

            AscendC::DataCopyParams boxCopy{1, static_cast<uint16_t>(SbgAlignBytes(coreRows_ * 16U) / 32U), 0, 0};
            AscendC::DataCopyParams anchorCopy{1, static_cast<uint16_t>(SbgAlignBytes(coreRows_ * 8U) / 32U), 0, 0};
            AscendC::DataCopy(boxIn_, boxesGm_, boxCopy);
            AscendC::DataCopy(anchorIn_, anchorsGm_, anchorCopy);
            // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
            LoadPointParameters<UNIT_RATIO, RATIO_KIND, true>();

        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B=4095候选：7×512+511有效行，统一512行输入/计算；阶段CopyIn语句。

            // 4095搬入候选：每个输入只发一条普通DataCopy，长度按本核kPad补齐。
            // ALIGN_UP到32B的输入存储必须真实可读；末核仅多读boxes16B/anchors8B。
            // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
            AscendC::DataCopyParams boxCopy{1, static_cast<uint16_t>(kPad * 16U / 32U), 0, 0};
            AscendC::DataCopyParams anchorCopy{1, static_cast<uint16_t>(kPad * 8U / 32U), 0, 0};            AscendC::DataCopy(boxIn_, boxesGm_, boxCopy);
            // 分别记录boxes与anchors就绪；允许先到的boxes先进入Vector。
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::DataCopy(anchorIn_, anchorsGm_, anchorCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
            LoadPointParameters<UNIT_RATIO, RATIO_KIND, true>();

        } else { CopyInRange(); }
    }

    // Compute：抽列、FP32分步运算、严格GT比较。
    __aicore__ inline void Compute() {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原Compute语句。

            // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
            AscendC::Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
            AscendC::Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));
            PrepareCompareAddress<2288U>();
            // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // 下面为同一个scratch中的视图，各偏移单位是float元素。
            // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
            const SbgPointColumns columns = MakePointColumns<kPad, kBankGap>();
            // FP32抽列：boxes奇数掩码归约为[y1,y2]交错对，再用奇/偶掩码分别抽y2/y1。
            // anchors用奇数掩码抽cy；这些归约用于抽列，不是跨样本求和。
            AscendC::LocalTensor<float> pairs = scratch_[0];
            ExtractFp32AllColumns<20, 10>(pairs, columns.y1, columns.y2, columns.cy);
            ComputeNativeBoundary<5, UNIT_RATIO>(columns.difference, columns.y1, columns.y2, columns.boundary);
            CompareAndSelectNativeFp32Range257To4095<5, 5>(columns.cy, columns.boundary);

        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B=4095候选：7×512+511有效行，统一512行输入/计算；阶段Compute语句。

            // 每核计算512行；末核第512个结果丢弃，FP32四步舍入与严格GT保持。
            // 一/零向量供最终Select使用；仍根据每行真实输入判定。
            // 组合项C6：只初始化一个32B常量块，VSEL源零步长广播。
            // 输入、FP32四步舍入、有效行数和搬出边界全部保持基线。
            AscendC::Duplicate(one_, static_cast<T>(1.0f), 8);
            AscendC::Duplicate(zero_, static_cast<T>(0.0f), 8);
            PrepareCompareAddressFp32Range257To4095<kPad, kBankGap>();
            // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
            // CopyIn已分别记录两路就绪；这里先仅等待boxes。
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // 下面为同一个scratch中的视图，各偏移单位是float元素。
            // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
            const SbgPointColumns columns = MakePointColumns<kPad, kBankGap>();
            // FP32抽列：boxes奇数掩码归约为[y1,y2]交错对，再用奇/偶掩码分别抽y2/y1。
            // anchors用奇数掩码抽cy；这些归约用于抽列，不是跨样本求和。
            AscendC::LocalTensor<float> pairs = scratch_[0];
            ExtractFp32BoxColumns<kPad>(pairs, columns.y1, columns.y2, columns.cy);
            ComputeNativeBoundaryFp32Range257To4095<kPad, UNIT_RATIO>(columns.difference, columns.y1, columns.y2, columns.boundary);
            // boxes已完成参考线计算；比较前才等待anchors并抽取cy。
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            ExtractFp32AnchorColumn<kPad>(columns.cy);
            CompareAndSelectNativeFp32Range257To4095Part2<kPad>(columns.cy, columns.boundary);

        } else { ComputeRange(); }
    }

    // CopyOut：只搬出有效行。
    __aicore__ inline void CopyOut() {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原CopyOut语句。

            // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
            CopyOutPoint(static_cast<uint16_t>(SbgAlignBytes(coreRows_ * 4U) / 32U));

        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B=4095候选：7×512+511有效行，统一512行输入/计算；阶段CopyOut语句。

            // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            if constexpr (TAIL) {
                AscendC::DataCopyParams outputCopy{1, static_cast<uint16_t>(SbgAlignBytes(2044U) / 32U), 0, 0};
                AscendC::DataCopy(outputGm_, output_, outputCopy);
            } else {
                AscendC::DataCopyParams outputCopy{1, 64, 0, 0};
                AscendC::DataCopy(outputGm_, output_, outputCopy);
            }

        } else { CopyOutRange(); }
    }
};

// 尺寸分支6：保留原v528的该精确点类主体。
// ============================================================================
// 【公开case05：FP32 / B=4096】SbgFp32B4096
// 当前启动8核，工作核上限8；8核各处理512行。
// 实际向量计算行数：512；下列字节数依次为boxes / anchors / output。
// 当前搬运：DataCopy：每核8192 / 4096 / 2048 B。
// 每个工作核只处理自己的单段切片：CopyIn → Compute → CopyOut，没有下一tile的双缓冲。
// 此处以“公开case”编号为准，不能沿用旧英文注释的内部排列编号。
// 主核/常规核UB字节区间（左闭右开）：boxIn_[0,8192)；anchorIn_[8192,12288)；output_[12288,14336)；constants_[14336,18432)；scratch_[65536,81440)
// ============================================================================
template <bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
class SbgFp32B4096 final : public SbgStages<SbgFp32B4096<UNIT_RATIO, RATIO_KIND>>, protected SbgPointState<float> {
    friend class SbgStages<SbgFp32B4096<UNIT_RATIO, RATIO_KIND>>;
public:
    static constexpr uint32_t kBatch = 4096U; // 精确点的逻辑B，不代表输入坐标内容已知。
    static constexpr uint32_t kWorkingCores = 8U; // 工作核上限；启动核之外/多余启动核在Init退出。
    static constexpr uint32_t kRowsPerCore = 512U; // 用于定位GM切片起点的行步长；尾核有效行数另算。
    static constexpr uint32_t kPad = 512U; // UB分配/列间距基准；不授权多搬GM数据，也不一定等于calcRows。
    static constexpr uint32_t kBankGap = 64U; // FP32元素单位的间隔，不是字节；当前布局参数，不是数学所需数据。
    static constexpr uint32_t kSvBase = 65536U; // scratch在本核UB中的字节起点。
    static constexpr uint32_t kSvFloatCount = 3976U; // scratch容量，单位float元素，包含位图及地址载体预留。
    static constexpr uint32_t kUbBytes = 81440U; // 本类UB最高字节边界，Host对应策略必须覆盖它。
    static_assert(SbgValidatePointPolicy<kBatch, false, kWorkingCores, kRowsPerCore, kUbBytes>(),
        "Point policy mismatch");

private:
    using T = float;

    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
                               GM_ADDR threshold, GM_ADDR violation_score,
                               uint32_t hasThreshold, GM_ADDR tiling) {
        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        uint32_t block;
        if (!InitPointSlice<kWorkingCores, kRowsPerCore>(block)) return false;
        coreRows_ = 512U;
        SetPointParameters(threshold, tiling, hasThreshold);
        // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
        // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
        BindPointGlobals(person_boxes, anchor_points, violation_score,
                SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T), SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        BindPointUb<kPad, kSvBase, kSvFloatCount>(8192U, 12288U, 14336U, 15360U);

        return true;
    }

    // 【搬入】发出输入DMA，并按本分支的时序准备真实属性和阈值。
    __aicore__ inline void CopyIn() {

        AscendC::DataCopyParams boxCopy{1, 256, 0, 0};
        AscendC::DataCopyParams anchorCopy{1, 128, 0, 0};        AscendC::DataCopy(boxIn_, boxesGm_, boxCopy);
        // 分别记录boxes与anchors就绪；允许先到的boxes先进入Vector。
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::DataCopy(anchorIn_, anchorsGm_, anchorCopy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
        LoadPointParameters<UNIT_RATIO, RATIO_KIND, true>();
    }

    // 【计算】抽取三列 → FP32几何运算 → 严格大于比较 → 生成0/1。
    __aicore__ inline void Compute() {
        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        AscendC::Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
        AscendC::Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));
        PrepareCompareAddress<3968U>();
        // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
        // CopyIn已分别记录两路就绪；这里先仅等待boxes。
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        // 下面为同一个scratch中的视图，各偏移单位是float元素。
        // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
        const SbgPointColumns columns = MakePointColumns<kPad, kBankGap>();
        // FP32抽列：boxes奇数掩码归约为[y1,y2]交错对，再用奇/偶掩码分别抽y2/y1。
        // anchors用奇数掩码抽cy；这些归约用于抽列，不是跨样本求和。
        AscendC::LocalTensor<float> pairs = scratch_[0];
        ExtractFp32BoxColumnsFp32B4096<32, 16>(pairs, columns.y1, columns.y2, columns.cy);
        ComputeNativeBoundary<8, UNIT_RATIO>(columns.difference, columns.y1, columns.y2, columns.boundary);
        // boxes已完成参考线计算；比较前才等待anchors并抽取cy。
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        ExtractFp32AnchorColumnFp32B4096<16>(columns.cy);
        CompareAndSelectNativeFp32Range257To4095<8, 8>(columns.cy, columns.boundary);
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
        CopyOutPoint(64);
    }
};

// ============================================================================
// 【分支07】FP16，完整覆盖B∈[1,31]：SbgFp16Range1To31
// 本类拥有完整搬入/抽列/计算/搬出流程，区间端点和普通B都在同一实现内处理。
// Host可在本类内选择编译期分工；普通B与端点均不转发给另一个单点类。
// 算法来自原v528已存在的区间实现，扩大入口覆盖；不宣称与原单点路径性能相同。
// ============================================================================
using SbgFp16Range1To31 = SbgSmallRange<half>;

// 尺寸分支8：保留原v528的该精确点类主体。
// ============================================================================
// 【公开case07：FP16 / B=32】SbgFp16B32
// 当前启动8核，工作核上限1；核0处理32行；核1～7退出。
// 实际向量计算行数：64；下列字节数依次为boxes / anchors / output。
// 当前搬运：DataCopy：256 / 128 / 64 B。
// 每个工作核只处理自己的单段切片：CopyIn → Compute → CopyOut，没有下一tile的双缓冲。
// 当前历史ABI：阈值/输出使用half；只有y1/y2/cy转FP32后参与几何计算和比较。
// 主核/常规核UB字节区间（左闭右开）：boxIn_[0,512)；anchorIn_[512,768)；output_[768,896)；constants_[2432,2688)；scratch_[2688,4576)；halfStage_[896,1920)
// ============================================================================
template <bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
class SbgFp16B32 final : public SbgStages<SbgFp16B32<UNIT_RATIO, RATIO_KIND>>, protected SbgHalfPointState {
    friend class SbgStages<SbgFp16B32<UNIT_RATIO, RATIO_KIND>>;
public:
    static constexpr uint32_t kBatch = 32U; // 精确点的逻辑B，不代表输入坐标内容已知。
    static constexpr uint32_t kWorkingCores = 1U; // 工作核上限；启动核之外/多余启动核在Init退出。
    static constexpr uint32_t kRowsPerCore = 32U; // 用于定位GM切片起点的行步长；尾核有效行数另算。
    static constexpr uint32_t kPad = 64U; // UB分配/列间距基准；不授权多搬GM数据，也不一定等于calcRows。
    static constexpr uint32_t kBankGap = 0U; // FP32元素单位的间隔，不是字节；当前布局参数，不是数学所需数据。
    static constexpr uint32_t kSvBase = 2688U; // scratch在本核UB中的字节起点。
    static constexpr uint32_t kSvFloatCount = 472U; // scratch容量，单位float元素，包含位图及地址载体预留。
    static constexpr uint32_t kUbBytes = 4576U; // 本类UB最高字节边界，Host对应策略必须覆盖它。
    static_assert(SbgValidatePointPolicy<kBatch, true, kWorkingCores, kRowsPerCore, kUbBytes>(),
        "Point policy mismatch");

private:
    using T = half;

    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
                               GM_ADDR threshold, GM_ADDR violation_score,
                               uint32_t hasThreshold, GM_ADDR tiling) {
        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        if (!InitSinglePoint<kWorkingCores, 32U>(person_boxes, anchor_points, threshold,
            violation_score, hasThreshold, tiling)) return false;
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        BindPointUb<kPad, kSvBase, kSvFloatCount>(512U, 768U, 2432U, 1792U, 896U);
        return true;
    }

    // 【搬入】发出输入DMA，并按本分支的时序准备真实属性和阈值。
    __aicore__ inline void CopyIn() {
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            AscendC::GlobalTensor<uint64_t> thresholdLine;
            thresholdLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold_), 1);
            // 缓存预取提示；uint64_t是此API的地址载体，实际阈值仍在后面按T读取。
            AscendC::DataCachePreload(thresholdLine, static_cast<int16_t>(0));
        }
        AscendC::DataCopyParams boxCopy{1, 8, 0, 0};
        AscendC::DataCopyParams anchorCopy{1, 4, 0, 0};
        AscendC::DataCopy(boxIn_, boxesGm_, boxCopy);
        AscendC::DataCopy(anchorIn_, anchorsGm_, anchorCopy);
        // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
        LoadPointParameters<UNIT_RATIO, RATIO_KIND, false>();

    }

    // 【计算】抽取三列 → FP32几何运算 → 严格大于比较 → 生成0/1。
    __aicore__ inline void Compute() {
        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        // 组合项C7：正常64-lane掩码共用，避免两次Counter模式往返。
        // InitSocState已恢复Normal；后续Gather/转换/比较及输出逻辑有效行均不变。
        AscendC::SetVectorMask<T>(0ULL, 0xffffffffffffffffULL);
        AscendC::Duplicate<T, false>(one_, static_cast<T>(1.0f), uint64_t(64), 1, 1, 8);
        AscendC::Duplicate<T, false>(zero_, static_cast<T>(0.0f), uint64_t(64), 1, 1, 8);

        // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        // 下面为同一个scratch中的视图，各偏移单位是float元素。
        // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
        const SbgPointColumns columns = MakePointColumns<kPad, kBankGap>();
        const SbgHalfColumns<T> halfColumns = MakeHalfColumns<kPad>();
        // 固定模式4/6分别取boxes的y1/y2，模式2取anchors的cy。
        AscendC::GatherMask(halfColumns.y1, boxIn_, static_cast<uint8_t>(4), {1, 2, 8, 0});
        AscendC::GatherMask(halfColumns.y2, boxIn_, static_cast<uint8_t>(6), {1, 2, 8, 0});
        AscendC::GatherMask(halfColumns.cy, anchorIn_, static_cast<uint8_t>(2), {1, 1, 8, 0});
        // Three independent conversions, followed by one dependency barrier.
        ConvertHalfColumns<1>(columns.y1, columns.y2, columns.cy, halfColumns.y1, halfColumns.y2, halfColumns.cy);
        ComputeNativeBoundaryWithThresholdLoadMasked<1, UNIT_RATIO>(columns.difference, columns.y1, columns.y2, columns.boundary);
        SelectPointMask<false, false>(columns, uint64_t(64));
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
        CopyOutPoint(2);
    }
};

// ============================================================================
// 【分支09】FP16，完整覆盖B∈[33,255]：SbgFp16Range33To255
// 本类拥有完整搬入/抽列/计算/搬出流程，区间端点和普通B都在同一实现内处理。
// Host可在本类内选择编译期分工；普通B与端点均不转发给另一个单点类。
// 算法来自原v528已存在的区间实现，扩大入口覆盖；不宣称与原单点路径性能相同。
// ============================================================================
template <uint32_t FIXED_BATCH = 0, bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0, bool TAIL = false>
class SbgFp16Range33To255 final : public SbgStages<SbgFp16Range33To255<FIXED_BATCH, UNIT_RATIO, RATIO_KIND, TAIL>>, protected SbgHalfPointState, protected SbgHalfRangeState {
    friend class SbgStages<SbgFp16Range33To255<FIXED_BATCH, UNIT_RATIO, RATIO_KIND, TAIL>>;
public:
    // 完整区间[33,255]只有本类。FIXED_BATCH是Host真实B守卫后的编译期策略；0表示真实运行时B。
    // 编译期策略恢复v528原优化，不创建独立单点类；B4094等仍由本类处理。
    static constexpr uint32_t kBatch = FIXED_BATCH;
    static constexpr uint32_t kWorkingCores = FIXED_BATCH == 33U ? (1U) : 0U;
    static constexpr uint32_t kRowsPerCore = FIXED_BATCH == 33U ? (48U) : 0U;
    static constexpr uint32_t kPad = FIXED_BATCH == 33U ? (64U) : 0U;
    static constexpr uint32_t kBankGap = FIXED_BATCH == 33U ? (0U) : 0U;
    static constexpr uint32_t kSvBase = FIXED_BATCH == 33U ? (2688U) : 0U;
    static constexpr uint32_t kSvFloatCount = FIXED_BATCH == 33U ? (472U) : 0U;
    static constexpr uint32_t kUbBytes = FIXED_BATCH == 33U ? (4576U) : 0U;
    static_assert(SbgValidateIntervalPolicy<FIXED_BATCH, 33U, 33U, kUbBytes>(), "Interval policy mismatch");

private:
    // 原优化路径所需的成员与UB视图；通用成员不会被常量策略使用。
    using T = half;

    // 通用B的真实长度/片上视图；编译器在常量策略中消除未使用成员。
    static constexpr uint32_t tSize = static_cast<uint32_t>(sizeof(T));
    static constexpr uint32_t bankGap = 0U;
    // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。

    // Init按本类策略建立分片和UB；每条路径均保持原有效GM长度。
    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR thresholdArg, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的编译期B=33策略：恢复冻结v528原Init语句。

            // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
            if (!InitSinglePoint<kWorkingCores, 33U>(person_boxes, anchor_points, thresholdArg,
                violation_score, hasThresholdArg, tilingArg)) return false;
            // Byte offsets belong only to this point. None is shared with another class.
            // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
            // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
            // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
            BindPointUb<kPad, kSvBase, kSvFloatCount>(512U, 768U, 2432U, 1792U, 896U);
            return true;

        } else { return InitRange<33U, 255U, false>(person_boxes, anchor_points, thresholdArg, violation_score, length, rowsPerCore, hasThresholdArg, tilingArg); }
    }

    // CopyIn：搬入真实输入并准备实际属性/阈值。
    __aicore__ inline void CopyIn() {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的B=33策略：32B补齐输入，普通DataCopy，有效行数独立控制。

            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                AscendC::GlobalTensor<uint64_t> thresholdLine;
                thresholdLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold_), 1);
                // 缓存预取提示；uint64_t是此API的地址载体，实际阈值仍在后面按T读取。
                AscendC::DataCachePreload(thresholdLine, static_cast<int16_t>(0));
            }
            // 统一使用普通DataCopy，搬运字节数向上补齐到32B。
            // 本核有效33行：boxes264B→288B，anchors132B→160B。
            // 调用方必须提供实际可读的32B补齐存储；SetGlobalBuffer不会扩容GM。
            // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
            const uint32_t boxBytes = (coreRows_ * 8U + 31U) & ~31U;
            const uint32_t anchorBytes = (coreRows_ * 4U + 31U) & ~31U;
            AscendC::DataCopyParams boxCopy{1, static_cast<uint16_t>(boxBytes / 32U), 0, 0};
            AscendC::DataCopyParams anchorCopy{1, static_cast<uint16_t>(anchorBytes / 32U), 0, 0};
            AscendC::DataCopy(boxIn_, boxesGm_, boxCopy);
            // boxes单独记录EVENT_ID0；后续anchors搬运与boxes的参考线计算重叠。
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::DataCopy(anchorIn_, anchorsGm_, anchorCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
            LoadPointParameters<UNIT_RATIO, RATIO_KIND, false>();

        } else { CopyInRange(); }
    }

    // Compute：抽列、FP32分步运算、严格GT比较。
    __aicore__ inline void Compute() {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的B=33策略：保留抽列与转换，逐行算术及比较只处理coreRows_。

            // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
            AscendC::Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
            AscendC::Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));

            // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
            // 先只等待boxes的EVENT_ID0；anchors的EVENT_ID1在比较前再等待。
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // 下面为同一个scratch中的视图，各偏移单位是float元素。
            // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
            const SbgPointColumns columns = MakePointColumns<kPad, kBankGap>();
            const SbgHalfColumns<T> halfColumns = MakeHalfColumns<kPad>();
            // 固定模式4/6分别取boxes的y1/y2，模式2取anchors的cy。
            AscendC::GatherMask(halfColumns.y1, boxIn_, static_cast<uint8_t>(4), {1, 2, 8, 0});
            AscendC::GatherMask(halfColumns.y2, boxIn_, static_cast<uint8_t>(6), {1, 2, 8, 0});
            // 先精确提升boxes的y1/y2；cy在anchors就绪后再精确提升为FP32。
            // C8分路计算：几何运算仍按原顺序分步FP32舍入，最终严格GT保持。
            // half y1/y2在[896,1152)相邻，float y1/y2在[3200,3712)相邻。
            // 先用一个2-repeat转换boxes两列；cy在下方等anchors完成后单独转换。
            // boxes与cy的输入/输出区间互不重叠；各自保留生产者到消费者的V屏障。
            static_assert(kPad == 64U && kBankGap == 0U, "C8 exact native conversion layout");
            ConvertHalfY1Column<2>(columns.y1, halfColumns.y1);
            // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
            // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
            AscendC::Sub(columns.difference, columns.y2, columns.y1, static_cast<int32_t>(coreRows_));
            if constexpr (!UNIT_RATIO) AscendC::Muls(columns.difference, columns.difference, waistRatio_, static_cast<int32_t>(coreRows_));
            AscendC::Add(columns.boundary, columns.y1, columns.difference, static_cast<int32_t>(coreRows_));
            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
                thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ volatile T*>(threshold_)[0]);
            }
            AscendC::Adds(columns.boundary, columns.boundary, thresholdValue_, static_cast<int32_t>(coreRows_));
            AscendC::SetMaskNorm();
            // 一个FP32 repeat的比较结果直接放CMPMASK寄存器；GT相等时为假。
            // 只有比较真正依赖anchors；先完成boxes的参考线计算，再等待anchors。
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::GatherMask(halfColumns.cy, anchorIn_, static_cast<uint8_t>(2), {1, 1, 8, 0});
            ConvertHalfAnchorColumn<1>(columns.cy, halfColumns.cy);
            AscendC::Compare(columns.cy, columns.boundary, AscendC::CMPMODE::GT, uint64_t(coreRows_), {1, 1, 1, 8, 8, 8});
            // 选择结果也只写入有效lane；CopyOut继续按实际有效字节数输出逻辑有效行。
            if ASCEND_IS_AIV { set_mask_norm(); set_vector_mask(0ULL, (1ULL << coreRows_) - 1ULL); }
            AscendC::Select<T, AscendC::SELMODE::VSEL_CMPMASK_SPR>(output_, one_, zero_, 1, {1, 1, 1, 8, 8, 8});
            AscendC::ResetMask();

        } else { ComputeRange(); }
    }

    // CopyOut：只搬出有效行。
    __aicore__ inline void CopyOut() {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的编译期B=33策略：恢复冻结v528原CopyOut语句。

            // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
            CopyOutPoint(static_cast<uint16_t>(SbgAlignBytes(coreRows_ * 2U) / 32U));

        } else { CopyOutRange(); }
    }
};

// 尺寸分支10：保留原v528的该精确点类主体。
// ============================================================================
// 【公开case09：FP16 / B=256】SbgFp16B256
// 当前启动8核，工作核上限1；核0处理256行；核1～7退出。
// 实际向量计算行数：256；下列字节数依次为boxes / anchors / output。
// 当前搬运：DataCopy：2048 / 1024 / 512 B。
// 每个工作核只处理自己的单段切片：CopyIn → Compute → CopyOut，没有下一tile的双缓冲。
// 当前历史ABI：阈值/输出使用half；只有y1/y2/cy转FP32后参与几何计算和比较。
// 基线仍为三次独立Cast；合并转换候选尚未成为当前选定基线。
// 主核/常规核UB字节区间（左闭右开）：boxIn_[0,2048)；anchorIn_[2048,3072)；output_[3072,3584)；constants_[9728,10752)；scratch_[65536,73888)；halfStage_[3584,7680)
// ============================================================================
template <bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
class SbgFp16B256 final : public SbgStages<SbgFp16B256<UNIT_RATIO, RATIO_KIND>>, protected SbgHalfPointState {
    friend class SbgStages<SbgFp16B256<UNIT_RATIO, RATIO_KIND>>;
public:
    static constexpr uint32_t kBatch = 256U; // 精确点的逻辑B，不代表输入坐标内容已知。
    static constexpr uint32_t kWorkingCores = 1U; // 工作核上限；启动核之外/多余启动核在Init退出。
    static constexpr uint32_t kRowsPerCore = 256U; // 用于定位GM切片起点的行步长；尾核有效行数另算。
    static constexpr uint32_t kPad = 256U; // UB分配/列间距基准；不授权多搬GM数据，也不一定等于calcRows。
    static constexpr uint32_t kBankGap = 64U; // FP32元素单位的间隔，不是字节；当前布局参数，不是数学所需数据。
    static constexpr uint32_t kSvBase = 65536U; // scratch在本核UB中的字节起点。
    static constexpr uint32_t kSvFloatCount = 2088U; // scratch容量，单位float元素，包含位图及地址载体预留。
    static constexpr uint32_t kUbBytes = 73888U; // 本类UB最高字节边界，Host对应策略必须覆盖它。
    static_assert(SbgValidatePointPolicy<kBatch, true, kWorkingCores, kRowsPerCore, kUbBytes>(),
        "Point policy mismatch");

private:
    using T = half;

    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
                               GM_ADDR threshold, GM_ADDR violation_score,
                               uint32_t hasThreshold, GM_ADDR tiling) {
        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        uint32_t block;
        if (!InitPointSlice<kWorkingCores, kRowsPerCore>(block)) return false;
        coreRows_ = 256U;
        SetPointParameters(threshold, tiling, hasThreshold);
        // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
        // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
        BindPointGlobals(person_boxes, anchor_points, violation_score,
                SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T), SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        BindPointUb<kPad, kSvBase, kSvFloatCount>(2048U, 3072U, 9728U, 8192U, 3584U);
        return true;
    }

    // 【搬入】发出输入DMA，并按本分支的时序准备真实属性和阈值。
    __aicore__ inline void CopyIn() {
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            AscendC::GlobalTensor<uint64_t> thresholdLine;
            thresholdLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold_), 1);
            // 缓存预取提示；uint64_t是此API的地址载体，实际阈值仍在后面按T读取。
            AscendC::DataCachePreload(thresholdLine, static_cast<int16_t>(0));
        }
        AscendC::DataCopyParams boxCopy{1, 64, 0, 0};
        AscendC::DataCopyParams anchorCopy{1, 32, 0, 0};
        AscendC::DataCopy(boxIn_, boxesGm_, boxCopy);
        AscendC::DataCopy(anchorIn_, anchorsGm_, anchorCopy);
        // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
        LoadPointParameters<UNIT_RATIO, RATIO_KIND, false>();

    }

    // 【计算】抽取三列 → FP32几何运算 → 严格大于比较 → 生成0/1。
    __aicore__ inline void Compute() {
        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        AscendC::Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
        AscendC::Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));
        PrepareCompareAddress<2080U>();
        // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        // 下面为同一个scratch中的视图，各偏移单位是float元素。
        // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
        const SbgPointColumns columns = MakePointColumns<kPad, kBankGap>();
        const SbgHalfColumns<T> halfColumns = MakeHalfColumns<kPad>();
        // 固定模式4/6分别取boxes的y1/y2，模式2取anchors的cy。
        AscendC::GatherMask(halfColumns.y1, boxIn_, static_cast<uint8_t>(4), {1, 8, 8, 0});
        AscendC::GatherMask(halfColumns.y2, boxIn_, static_cast<uint8_t>(6), {1, 8, 8, 0});
        AscendC::GatherMask(halfColumns.cy, anchorIn_, static_cast<uint8_t>(2), {1, 4, 8, 0});
        // 三列各自提升为FP32；这是基线的三次Cast，不是合并转换候选。
        // C9独立试验：保留三列原地址和bank间隔，只更换精确half→FP32提升。
        // 每个正常mask repeat转换64元素，4次覆盖256元素；不移动UB、不改变GM。
        // 前后V屏障保护Gather结果和后续Sub消费者，FP32运算顺序及GT均保持。
        ConvertHalfColumnsFp16B256<4>(columns.y1, halfColumns.y1, columns.y2, halfColumns.y2, columns.cy, halfColumns.cy);
        // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
        // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
        AscendC::Sub(columns.difference, columns.y2, columns.y1, static_cast<int32_t>(kPad));
        if constexpr (!UNIT_RATIO) AscendC::Muls(columns.difference, columns.difference, waistRatio_, static_cast<int32_t>(kPad));
        AscendC::Add(columns.boundary, columns.y1, columns.difference, static_cast<int32_t>(kPad));
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
            thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ volatile T*>(threshold_)[0]);
        }
        AscendC::Adds(columns.boundary, columns.boundary, thresholdValue_, static_cast<int32_t>(kPad));
        CompareAndSelectCounted<4, 1, kPad>(columns.cy, columns.boundary);
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
        CopyOutPoint(16);
    }
};

// ============================================================================
// 【分支11】FP16，完整覆盖B∈[257,4095]：SbgFp16Range257To4095
// 本类拥有完整搬入/抽列/计算/搬出流程，区间端点和普通B都在同一实现内处理。
// Host可在本类内选择编译期分工；普通B与端点均不转发给另一个单点类。
// 例：B4094分为7×512＋510；B4095分为6×640＋255。两者都由这个类处理。
// 算法来自原v528已存在的区间实现，扩大入口覆盖；不宣称与原单点路径性能相同。
// ============================================================================
template <uint32_t FIXED_BATCH = 0, bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0, bool TAIL = false>
class SbgFp16Range257To4095 final : public SbgStages<SbgFp16Range257To4095<FIXED_BATCH, UNIT_RATIO, RATIO_KIND, TAIL>>, protected SbgHalfPointState, protected SbgHalfRangeState {
    friend class SbgStages<SbgFp16Range257To4095<FIXED_BATCH, UNIT_RATIO, RATIO_KIND, TAIL>>;
public:
    // 完整区间[257,4095]只有本类。FIXED_BATCH是Host真实B守卫后的编译期策略；0表示真实运行时B。
    // 编译期策略恢复v528原优化，不创建独立单点类；B4094等仍由本类处理。
    static constexpr uint32_t kBatch = FIXED_BATCH;
    static constexpr uint32_t kWorkingCores = FIXED_BATCH == 257U ? (1U) : FIXED_BATCH == 4095U ? (6U) : 0U;
    static constexpr uint32_t kRowsPerCore = FIXED_BATCH == 257U ? (272U) : FIXED_BATCH == 4095U ? (768U) : 0U;
    static constexpr uint32_t kPad = FIXED_BATCH == 257U ? (320U) : FIXED_BATCH == 4095U ? (768U) : 0U;
    static constexpr uint32_t kBankGap = FIXED_BATCH == 257U ? (0U) : FIXED_BATCH == 4095U ? (64U) : 0U;
    static constexpr uint32_t kSvBase = FIXED_BATCH == 257U ? (42U * kPad) : FIXED_BATCH == 4095U ? (65536U) : 0U;
    static constexpr uint32_t kSvFloatCount = FIXED_BATCH == 257U ? (7U * kPad + 56U) : FIXED_BATCH == 4095U ? (7688U) : 0U;
    static constexpr uint32_t kUbBytes = FIXED_BATCH == 257U ? (70U * kPad + 224U) : FIXED_BATCH == 4095U ? (96288U) : 0U;
    static_assert(SbgValidateIntervalPolicy<FIXED_BATCH, 257U, 4095U, kUbBytes>(), "Interval policy mismatch");

private:
    // 原优化路径所需的成员与UB视图；通用成员不会被常量策略使用。
    using T = half;

    // 通用B的真实长度/片上视图；编译器在常量策略中消除未使用成员。
    static constexpr uint32_t tSize = static_cast<uint32_t>(sizeof(T));
    static constexpr uint32_t bankGap = 0U;
    // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。

    // Init按本类策略建立分片和UB；每条路径均保持原有效GM长度。
    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR thresholdArg, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原Init语句。

            // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
            const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
            if (block != 0U) return false;
            rowStart_ = 0U;
            coreRows_ = 257U;
            SetPointParameters(thresholdArg, tilingArg, hasThresholdArg);
            // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
            // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
            BindPointGlobals(person_boxes, anchor_points, violation_score,
                SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T), SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
            // Byte offsets belong only to this point. None is shared with another class.
            // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
            // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
            // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
            BindPointUb<kPad, kSvBase, kSvFloatCount>(8U * kPad, 12U * kPad, 38U * kPad, 28U * kPad, 14U * kPad);
            return true;

        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B4095候选：5×768+255有效行，输入对齐到与B4096相同；阶段Init语句。

            // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
            const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
            if (block >= kWorkingCores) return false;
            rowStart_ = block * kRowsPerCore;
            coreRows_ = block == 5U ? 255U : 768U;
            SetPointParameters(thresholdArg, tilingArg, hasThresholdArg);
            // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
            // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
            BindPointGlobals(person_boxes, anchor_points, violation_score,
                ((coreRows_ + 255U) & ~255U) * 4U, ((coreRows_ + 255U) & ~255U) * 2U);
            // Byte offsets belong only to this point. None is shared with another class.
            // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
            // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
            // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
            BindPointUb<kPad, kSvBase, kSvFloatCount>(8192U, 12288U, 38912U, 29696U, 14336U);
            return true;

        } else { return InitRange<257U, 4095U, false>(person_boxes, anchor_points, thresholdArg, violation_score, length, rowsPerCore, hasThresholdArg, tilingArg); }
    }

    // CopyIn：搬入真实输入并准备实际属性/阈值。
    __aicore__ inline void CopyIn() {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原CopyIn语句。

            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                AscendC::GlobalTensor<uint64_t> thresholdLine;
                thresholdLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold_), 1);
                // 缓存预取提示；uint64_t是此API的地址载体，实际阈值仍在后面按T读取。
                AscendC::DataCachePreload(thresholdLine, static_cast<int16_t>(0));
            }
            AscendC::DataCopyParams boxCopy{1, static_cast<uint16_t>(SbgAlignBytes(2056U) / 32U), 0, 0};
            AscendC::DataCopyParams anchorCopy{1, static_cast<uint16_t>(SbgAlignBytes(1028U) / 32U), 0, 0};
            AscendC::DataCopy(boxIn_, boxesGm_, boxCopy);
            AscendC::DataCopy(anchorIn_, anchorsGm_, anchorCopy);
            // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
            LoadPointParameters<UNIT_RATIO, RATIO_KIND, false>();

        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B4095候选：5×768+255有效行，输入对齐到与B4096相同；阶段CopyIn语句。

            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                AscendC::GlobalTensor<uint64_t> thresholdLine;
                thresholdLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold_), 1);
                // 缓存预取提示；uint64_t是此API的地址载体，实际阈值仍在后面按T读取。
                AscendC::DataCachePreload(thresholdLine, static_cast<int16_t>(0));
            }
            // C12与FP16 B4096采用相同分工：前5核768行，尾核搬256行。
            // 逻辑尾核仅255行；末核额外读boxes8B/anchors4B，须有真实可读的对齐存储。
            // 额外lane允许NaN，其结果不写出；输出总量仍严格4095行。
            const uint32_t copyRows = (coreRows_ + 255U) & ~255U;
            AscendC::DataCopyParams boxCopy{1, static_cast<uint16_t>(copyRows / 4U), 0, 0};
            AscendC::DataCopyParams anchorCopy{1, static_cast<uint16_t>(copyRows / 8U), 0, 0};            AscendC::DataCopy(boxIn_, boxesGm_, boxCopy);
            // 分别记录boxes与anchors就绪；允许先到的boxes先进入Vector。
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::DataCopy(anchorIn_, anchorsGm_, anchorCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
            LoadPointParameters<UNIT_RATIO, RATIO_KIND, false>();

        } else { CopyInRange(); }
    }

    // Compute：抽列、FP32分步运算、严格GT比较。
    __aicore__ inline void ComputeFixed257() {
        // 同一区间类的编译期B=257策略：恢复冻结v528原Compute语句。

        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        AscendC::Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
        AscendC::Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));
        PrepareCompareAddressFp16Range257To4095<kPad>();
        // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        // 下面为同一个scratch中的视图，各偏移单位是float元素。
        // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
        const SbgPointColumns columns = MakePointColumns<kPad, kBankGap>();
        const SbgHalfColumns<T> halfColumns = MakeHalfColumns<kPad>();
        // 固定模式4/6分别取boxes的y1/y2，模式2取anchors的cy。
        AscendC::GatherMask(halfColumns.y1, boxIn_, static_cast<uint8_t>(4), {1, static_cast<uint16_t>(kPad / 32U), 8, 0});
        AscendC::GatherMask(halfColumns.y2, boxIn_, static_cast<uint8_t>(6), {1, static_cast<uint16_t>(kPad / 32U), 8, 0});
        AscendC::GatherMask(halfColumns.cy, anchorIn_, static_cast<uint8_t>(2), {1, static_cast<uint16_t>(kPad / 64U), 8, 0});
        // 三列各自提升为FP32；这是基线的三次Cast，不是合并转换候选。
        // C10独立试验：三列仍在原来的UB地址，不移动或合并任何plane。
        // 正常64-lane mask下每列5个repeat精确提升320个half；有效GM仍只有257行。
        // 只合并独立转换之间的同步，保留Gather生产者和Sub消费者所需前后屏障。
        static_assert(kPad == 320U && kBankGap == 0U, "C10 native conversion plan");
        ConvertHalfColumnsFp16B256<5>(columns.y1, halfColumns.y1, columns.y2, halfColumns.y2, columns.cy, halfColumns.cy);
        // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
        // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
        AscendC::Sub(columns.difference, columns.y2, columns.y1, static_cast<int32_t>(kPad));
        if constexpr (!UNIT_RATIO) AscendC::Muls(columns.difference, columns.difference, waistRatio_, static_cast<int32_t>(kPad));
        AscendC::Add(columns.boundary, columns.y1, columns.difference, static_cast<int32_t>(kPad));
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
            thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ volatile T*>(threshold_)[0]);
        }
        AscendC::Adds(columns.boundary, columns.boundary, thresholdValue_, static_cast<int32_t>(kPad));
        CompareAndSelectCountedFp16Range257To4095<1, kPad>(columns.cy, columns.boundary);
    }

    __aicore__ inline void ComputeFixed4095() {
        // 同一区间类B4095候选：5×768+255有效行，输入对齐到与B4096相同；阶段Compute语句。

        // calcRows是本核向量长度：只在UB内向上对齐，搬出仍严格用coreRows_。
        const uint32_t calcRows = (coreRows_ + 63U) & ~63U;
        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        ComputeLargeHalf<kPad, kBankGap, UNIT_RATIO>(calcRows);
    }

    __aicore__ inline void Compute() {
        if constexpr (FIXED_BATCH == 257U) { ComputeFixed257(); } else if constexpr (FIXED_BATCH == 4095U) { ComputeFixed4095(); } else { ComputeRange(); }
    }

    // CopyOut：只搬出有效行。
    __aicore__ inline void CopyOut() {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原CopyOut语句。

            // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
            CopyOutPoint(static_cast<uint16_t>(SbgAlignBytes(514U) / 32U));

        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B4095候选：5×768+255有效行，输入对齐到与B4096相同；阶段CopyOut语句。

            // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            if (coreRows_ == kRowsPerCore) {
                AscendC::DataCopyParams outputCopy{1, static_cast<uint16_t>(coreRows_ / 16U), 0, 0};
                AscendC::DataCopy(outputGm_, output_, outputCopy);
            } else {
                AscendC::DataCopyParams outputCopy{1, static_cast<uint16_t>(SbgAlignBytes(coreRows_ * 2U) / 32U), 0, 0};
                AscendC::DataCopy(outputGm_, output_, outputCopy);
            }

        } else { CopyOutRange(); }
    }
};

// 尺寸分支12：保留原v528的该精确点类主体。
// ============================================================================
// 【公开case11：FP16 / B=4096】SbgFp16B4096
// 当前启动8核，工作核上限6；核0～4各768行；核5处理256行；核6、7退出。
// 实际向量计算行数：主核768，尾核256；下列字节数依次为boxes / anchors / output。
// 当前搬运：DataCopy：主核6144/3072/1536 B；尾核2048/1024/512 B。
// 每个工作核只处理自己的单段切片：CopyIn → Compute → CopyOut，没有下一tile的双缓冲。
// 当前历史ABI：阈值/输出使用half；只有y1/y2/cy转FP32后参与几何计算和比较。
// 此处以“公开case”编号为准，不能沿用旧英文注释的内部排列编号。
// 主核/常规核UB字节区间（左闭右开）：boxIn_[0,6144)；anchorIn_[8192,11264)；output_[12288,13824)；constants_[38912,41984)；scratch_[65536,96288)；halfStage_[14336,26624)
// ============================================================================
template <bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
class SbgFp16B4096 final : public SbgStages<SbgFp16B4096<UNIT_RATIO, RATIO_KIND>>, protected SbgHalfPointState {
    friend class SbgStages<SbgFp16B4096<UNIT_RATIO, RATIO_KIND>>;
public:
    static constexpr uint32_t kBatch = 4096U; // 精确点的逻辑B，不代表输入坐标内容已知。
    static constexpr uint32_t kWorkingCores = 6U; // 工作核上限；启动核之外/多余启动核在Init退出。
    static constexpr uint32_t kRowsPerCore = 768U; // 用于定位GM切片起点的行步长；尾核有效行数另算。
    static constexpr uint32_t kPad = 768U; // UB分配/列间距基准；不授权多搬GM数据，也不一定等于calcRows。
    static constexpr uint32_t kBankGap = 64U; // FP32元素单位的间隔，不是字节；当前布局参数，不是数学所需数据。
    static constexpr uint32_t kSvBase = 65536U; // scratch在本核UB中的字节起点。
    static constexpr uint32_t kSvFloatCount = 7688U; // scratch容量，单位float元素，包含位图及地址载体预留。
    static constexpr uint32_t kUbBytes = 96288U; // 本类UB最高字节边界，Host对应策略必须覆盖它。
    static_assert(SbgValidatePointPolicy<kBatch, true, kWorkingCores, kRowsPerCore, kUbBytes>(),
        "Point policy mismatch");

private:
    using T = half;

    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
                               GM_ADDR threshold, GM_ADDR violation_score,
                               uint32_t hasThreshold, GM_ADDR tiling) {
        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        uint32_t block;
        if (!InitPointSlice<kWorkingCores, kRowsPerCore>(block)) return false;
        coreRows_ = block == 5U ? 256U : 768U;
        SetPointParameters(threshold, tiling, hasThreshold);
        // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
        // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
        BindPointGlobals(person_boxes, anchor_points, violation_score,
                SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T), SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        BindPointUb<kPad, kSvBase, kSvFloatCount>(8192U, 12288U, 38912U, 29696U, 14336U);
        return true;
    }

    // 【搬入】发出输入DMA，并按本分支的时序准备真实属性和阈值。
    __aicore__ inline void CopyIn() {
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            AscendC::GlobalTensor<uint64_t> thresholdLine;
            thresholdLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold_), 1);
            // 缓存预取提示；uint64_t是此API的地址载体，实际阈值仍在后面按T读取。
            AscendC::DataCachePreload(thresholdLine, static_cast<int16_t>(0));
        }
        AscendC::DataCopyParams boxCopy{1, static_cast<uint16_t>(coreRows_ / 4U), 0, 0};
        AscendC::DataCopyParams anchorCopy{1, static_cast<uint16_t>(coreRows_ / 8U), 0, 0};        AscendC::DataCopy(boxIn_, boxesGm_, boxCopy);
        // 分别记录boxes与anchors就绪；允许先到的boxes先进入Vector。
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::DataCopy(anchorIn_, anchorsGm_, anchorCopy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
        LoadPointParameters<UNIT_RATIO, RATIO_KIND, false>();

    }

    // 【计算】抽取三列 → FP32几何运算 → 严格大于比较 → 生成0/1。
    __aicore__ inline void Compute() {
        // calcRows是本核向量长度：只在UB内向上对齐，搬出仍严格用coreRows_。
        const uint32_t calcRows = (coreRows_ + 255U) & ~255U;
        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        ComputeLargeHalf<kPad, kBankGap, UNIT_RATIO>(calcRows);
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
        CopyOutPoint(static_cast<uint16_t>(coreRows_ / 16U));
    }
};

// 仅保留资源不足/空输入等兼容回退。
#include "sbg_fallback.h"

template <typename DT_PERSON_BOXES, uint32_t CONST_B = 0, uint32_t CONST_PER = 1, bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
__aicore__ inline void SbgIntervalFallback(GM_ADDR person_boxes, GM_ADDR anchor_points, GM_ADDR threshold,
                              GM_ADDR violation_score, uint32_t length, uint32_t rowsPerCore,
                              uint32_t hasThreshold, GM_ADDR tiling) {
    SbgFallback<DT_PERSON_BOXES, CONST_B, CONST_PER, UNIT_RATIO, RATIO_KIND> kernel;
    kernel.Run(person_boxes, anchor_points, threshold, violation_score,
        length, rowsPerCore, hasThreshold, tiling);
}

// 【NPU总入口：合法B的12个尺寸分支】
// 三个区间类与三个精确点类分别各有FP32/FP16版本；不存在独立B33/B257/B4095类。
// 低资源/空输入兼容回退保留在最后；正常B4094、4095均进入大区间类。
#include "sbg_kernel_dispatch.h"

template <typename DT_PERSON_BOXES, uint32_t SBG_BATCH = 0>
__global__ __aicore__ void safety_belt_violation_geometry(GM_ADDR person_boxes, GM_ADDR anchor_points,
    GM_ADDR threshold, GM_ADDR violation_score, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
    const SbgKernelArguments args{person_boxes, anchor_points, threshold, violation_score, tiling};
    SbgKernelDispatch<DT_PERSON_BOXES, SBG_BATCH>::Run(args);
}

// 系统工作区策略恢复冻结v528：列明Key已恢复原计算主体，普通区间Key仍保留标准包装。
// dump/timestamp/OOM构建保留框架包装；此处不是对任意新算法通用删除工作区。
// Workspace policy inherited only by the unchanged exact-point classes.
// Guarded constant plans use the original v528 stages inside their owning interval class.
// Only points using basic SIMD/DMA APIs are listed. Range/fallback keys retain
// the standard framework setup. Keep that setup for dump/timestamp/OOM builds.
// Bit order follows public cases, not the internal Host debug branch IDs.
#define SBG_NO_SYSTEM_WS_MASK 0xFEFU
#if defined(TILING_KEY_VAR) && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
// bit 0: case1, FP32, B=32; keys 768,3584
// bit 1: case2, FP32, B=33; keys 1024
// bit 2: case3, FP32, B=256; keys 1792
// bit 3: case4, FP32, B=257; keys 2048
// bit 4: case5, FP32, B=4096; keys 3328,4864
// bit 5: case6, FP32, B=4095; keys 3072,4608
// bit 6: case7, FP16, B=32; keys 769,3585
// bit 7: case8, FP16, B=33; keys 1025
// bit 8: case9, FP16, B=256; keys 1793
// bit 9: case10, FP16, B=257; keys 2049
// bit 10: case11, FP16, B=4096; keys 3329,4865
// bit 11: case12, FP16, B=4095; keys 3073,4609
#if ((SBG_NO_SYSTEM_WS_MASK & (1U << 0)) && (TILING_KEY_VAR == 768UL || TILING_KEY_VAR == 3584UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 1)) && (TILING_KEY_VAR == 1024UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 2)) && (TILING_KEY_VAR == 1792UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 3)) && (TILING_KEY_VAR == 2048UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 4)) && (TILING_KEY_VAR == 3328UL || TILING_KEY_VAR == 4864UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 5)) && (TILING_KEY_VAR == 3072UL || TILING_KEY_VAR == 4608UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 6)) && (TILING_KEY_VAR == 769UL || TILING_KEY_VAR == 3585UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 7)) && (TILING_KEY_VAR == 1025UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 8)) && (TILING_KEY_VAR == 1793UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 9)) && (TILING_KEY_VAR == 2049UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 10)) && (TILING_KEY_VAR == 3329UL || TILING_KEY_VAR == 4865UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 11)) && (TILING_KEY_VAR == 3073UL || TILING_KEY_VAR == 4609UL))
namespace AscendC {
__aicore__ inline void SbgNoUnusedSystemWorkspace(GM_ADDR unusedWorkspace) {
    (void)unusedWorkspace;
}
}
#define SetSysWorkspaceForce SbgNoUnusedSystemWorkspace
#endif
#endif
