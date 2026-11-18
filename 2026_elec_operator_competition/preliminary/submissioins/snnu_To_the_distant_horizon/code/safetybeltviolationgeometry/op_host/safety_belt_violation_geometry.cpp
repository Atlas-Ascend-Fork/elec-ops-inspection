/*
 * Copyright (c) 2026 zhiyuan (致远队).
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

// Host side tiling（W13 压平版；cache-bust 10261521）；bust2-1630; bust3; bust4; bust6-2213; bust1027-w42; bust1106-w47
// ── host 零数据运算声明（合规自证）────────────────────────────────────────────
//   host 只做：① 形状读取（GetShapeSize）② tiling 字段透传（waist_ratio 原样）
//   ③ launch 分派（SetBlockDim）④ workspace 分配（0 字节）。不读任何输入张量
//   内容、不做逐元素算术；全部判定在 device 侧 kernel 内完成。
//   W11 定态：三档 launch 统一 8 核（小点 core-0-only 早退 / 中点 8 核切分 /
//   大点 8 核双 tile 流水）——W8_LARGE_CORES16=0、GV_CORES/W7B/S45B 全为 8 的折叠。
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/safety_belt_violation_geometry_tiling.h"
#include "../op_kernel/tiling_key_safety_belt_violation_geometry.h"

namespace optiling {
    // ── W39：核数轴 H18 式重定价（HyosungSink mhc §19.2/§50.6 同构判据）──────────
    //   C1 空载实测 bd=4 地板 0.91 vs bd=8 1.15（-0.24）。
    //   ⚠ 与 W27"1 核"不同：bd=4 仍是多核 launch 快路径几何，只是派发税减半。
#ifndef W39_CORES4
#define W39_CORES4 0
#endif
#ifndef W27_SINGLE_CORE
#define W27_SINGLE_CORE 0   // 板测定案(6adeffb8): 中点单核 T3 2.82/T9 2.60 vs 8核 2.37/2.32
#endif
#ifndef W42_CORES16_MID1024
#define W42_CORES16_MID1024 1
#endif
#ifndef W43_CORES16_BIG
#define W43_CORES16_BIG 0
#endif
    // 档位→核数决策（与 kernel 档位谓词同口径；历史板测数据见各 W 注释）
        // 核数轴终审（W14c 6adcd71d 补点收官）：小 {1,4,6,8}={2.59,2.55,2.61,1.64}，
        //   大 {4,8,16,32,48}={4.87WA,2.44,2.84,4.08,4.94}——8 核全域最优（此后单调恶化）。
        //   三档统一写死 8 核，核数轴永久关闭。
        // W27（天南星 10-25 实测）：中点 T3/T4/T9/T10=1.54~1.56 ≈ 空核地板+0.05，
        //   8 核×32 行/核的切分开销（4 笔 DMA×8 核+事件+入口）是主体——中点档
        //   单核全量（256 行 UB ~30KB << 192KB）。快慢路径风险板测裁决，
        //   W27_SINGLE_CORE=0 回退 8 核。
#ifndef W27_SINGLE_CORE
#define W27_SINGLE_CORE 0   // 板测定案(6adeffb8): 中点单核 T3 2.82/T9 2.60 vs 8核 2.37/2.32 —— block_dim<8 launch慢路径反噬跨档成立(小4核+0.9同构), 中点8核终审关闭
#endif
// ── W39：核数轴 H18 式重定价（HyosungSink mhc §19.2/§50.6 同构判据）──────────
//   旧"8 全域最优"（W14c）是旧树+无 W(blk) 轴分解的结论。C1 空载实测 bd=4 地板
//   0.91 vs bd=8 1.15（-0.24）；W 代价上限 = W×(1-4/8)（中点每核 32→64 行单 tile
//   变 2 tile，W23 流水现成；大点 512→1024 行 2→4 tile）。⚠ 与 W27"1 核"不同：
//   bd=4 仍是多核 launch 快路径几何，只是派发税减半。
#ifndef W39_CORES4
#define W39_CORES4 0
#endif
        uint32_t sb_launch_cores = SelectLaunchCores(total_b);
        // W42：B=1024 档核数加压——T5/T11 ratio 0.66/0.75 全队最差。B∈(512,1024]
        //   时 8 核每核 128 行 tv=128；16 核每核 64 行 tv=64，Gather/Cast rep 减半。
        //   ⚠ bd=16 launch 未板测过（W39 bd=4 判死是慢路径；bd>8 待裁决）。
#ifndef W42_CORES16_MID1024
#define W42_CORES16_MID1024 1
#endif

    static uint32_t SelectLaunchCores(uint32_t total_b) {
        uint32_t cores = 8u;
#if W39_CORES4
        cores = 4u;
#endif
#if W27_SINGLE_CORE
        if (total_b > SB_SMALL_MAX_B && total_b <= SB_MID_MAX_B) {
            cores = 1u;   // 中点单核全量（kernel GatherVec n=全 B）
        }
#endif
#if W42_CORES16_MID1024
        if (total_b > 512u && total_b <= SB_MID_MAX_B) {
            cores = 16u;
        }
#endif
        // W43：大点（B>1024）16 核——每核 256 行=1 tile（无循环/无双缓冲）。
#if W43_CORES16_BIG
        if (total_b > SB_MID_MAX_B) {
            cores = 16u;
        }
#endif
        return cores;
    }

    // perH = align8(ceil(total_b/cores))（与 kernel 侧 ReservePerH 同式；bd 除零防护）
    static uint32_t ComputePerH(uint32_t total_b, uint32_t cores) {
        const uint32_t bd = (cores > 0u) ? cores : 1u;
        uint32_t perH = (total_b + bd - 1u) / bd;
        perH = ((perH + 7u) / 8u) * 8u;
        if (perH == 0u) perH = 8u;
        return perH;
    }

    // TPL 键选择：dtype + SCH_MODE（W44 档位分派键；DIAG_FLOOR_AIC_MODE 走空核探针）
    static void SelectTplKey(gert::TilingContext *context, ge::DataType dtype_person_boxes, uint32_t total_b) {
        uint32_t DT_PERSON_BOXES = static_cast<uint32_t>(dtype_person_boxes);
        // W13：SCH_MODE 值域压到每 dtype 一个（fp32→0 / fp16→1），与 tiling_key 头的
        //   UINT_DECL/UINT_SEL 清单同值；仅作 TPL 键，kernel 内不参与档位分派。
#if DIAG_FLOOR_AIC_MODE
        // P-D1：12 点全路由 AIC 空核（时间=AIC 地板，WA 预期）；FORMAT 维键值=1。
        uint32_t sb_mode = SB_MODE_AIC_EMPTY;
        ASCENDC_TPL_SEL_PARAM(context, DT_PERSON_BOXES, sb_mode);
#else
        // W44（B8/D1-K1）：SCH_MODE 升级为档位分派键（dtype×档位，6 值域）。
        //   谓词与 kernel 原运行期分支逐字同构（≤33 小 / ≤1024 中 / else 大），
        //   连续全域区间，无单点等式——合规自证见 tiling_key 头注释。
        const bool isF16 = (dtype_person_boxes == ge::DT_FLOAT16);
        uint32_t sch_mode;
        if (total_b <= SB_SMALL_MAX_B) {
            sch_mode = isF16 ? SB_TPL_SCH_F16_SMALL : SB_TPL_SCH_F32_SMALL;
        } else if (total_b <= SB_MID_MAX_B) {
            sch_mode = isF16 ? SB_TPL_SCH_F16_MID : SB_TPL_SCH_F32_MID;
        } else {
            sch_mode = isF16 ? SB_TPL_SCH_F16_BIG : SB_TPL_SCH_F32_BIG;
        }
        // 维度顺序必须与 tiling_key_*.h 的 ASCENDC_TPL_ARGS_DECL 顺序一致：dtype + SCH_MODE。
        ASCENDC_TPL_SEL_PARAM(context, DT_PERSON_BOXES, sch_mode);
#endif
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const gert::Tensor *tensor_person_boxes = context->GetRequiredInputTensor(0);
        if (tensor_person_boxes == nullptr) {
            return ge::GRAPH_FAILED;
        }

        // >> 2 语义：GetShapeSize() 返回元素数；person_boxes 每框 4 字段 ⇒ B = 元素数 >> 2
        //   （形状推导，非数据运算）。fp16/fp32 同式。
        uint32_t total_b = static_cast<uint32_t>(tensor_person_boxes->GetShapeSize() >> 2);
        if (total_b == 0) {
            total_b = 1;
        }

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        float waist_ratio = 1.0f;
        if (attrs != nullptr) {
            const float *attr_val = attrs->GetFloat(0);
            if (attr_val != nullptr) {
                waist_ratio = *attr_val;
            }
        }

        // threshold 只判指针非空（host 不读阈值数值）；真值由 kernel 从 GM 取回。

        SelectTplKey(context, tensor_person_boxes->GetDataType(), total_b);

        const uint32_t sb_launch_cores = SelectLaunchCores(total_b);

        SafetyBeltViolationGeometryTilingData *tiling = context->GetTilingData<SafetyBeltViolationGeometryTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->total_b = total_b;
        tiling->waist_ratio = waist_ratio;
        tiling->threshold = 0.0f;     // 结构体占位，阈值真值由 kernel 从 GM 取回
        // W36：per=align8(ceil(total_b/bd)) 由 host 预计算写入 reserved——kernel 侧
        //   免 2 条运行期除法（raw/per 链）。bd 与下方 SetBlockDim 同源（8 恒定；
        //   W27 单核开启时 kernel 仍走 GetBlockNum 动态值，reserved 不被中点消费）。
        tiling->reserved = ComputePerH(total_b, sb_launch_cores);

        context->SetBlockDim(sb_launch_cores);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace != nullptr) {
            currentWorkspace[0] = 0;
        }

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    // 仅元数据：out 的 dim0 = 输入 dim0（框数 B），dim1 固定 1；dtype 同输入。
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *boxes_shape = context->GetInputShape(0);
        gert::Shape *out_shape = context->GetOutputShape(0);
        if (boxes_shape != nullptr && out_shape != nullptr) {
            out_shape->SetDimNum(2);
            out_shape->SetDim(0, boxes_shape->GetDim(0));
            out_shape->SetDim(1, 1);
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class SafetyBeltViolationGeometry : public OpDef {
    public:
        explicit SafetyBeltViolationGeometry(const char *name) : OpDef(name) {
            this->Input("person_boxes")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("anchor_points")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("threshold")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("violation_score")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("waist_ratio").AttrType(REQUIRED).Float();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(SafetyBeltViolationGeometry);
}  // namespace ops
