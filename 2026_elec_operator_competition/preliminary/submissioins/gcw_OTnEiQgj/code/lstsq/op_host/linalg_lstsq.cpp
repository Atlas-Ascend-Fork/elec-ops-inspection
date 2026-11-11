/*
 * Copyright (c) 2026 gcw_OTnEiQgj.
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
// Template API and registration are preserved; common code validates shapes and attrs.
#include "lstsq_plan.h"
#include "../op_kernel/linalg_lstsq_tiling.h"
#include "../op_kernel/tiling_key_linalg_lstsq.h"

namespace {
uint32_t select_small_fp32_lu(const elec::Config& c) {
    const uint64_t pad_a = (c.n + 7) & ~uint64_t(7);
    const bool pack_tail = (c.n + c.k <= pad_a);
    const bool zero_r = (c.flags & elec::LSTSQ_SQUARE_RESIDUAL) != 0;
    if (pad_a == 16 && c.k == 1) {
        return pack_tail ? (zero_r ? 11 : 5) : (zero_r ? 10 : 4);
    }
    if (c.k == 2 && ((pack_tail && pad_a == 56) || (!pack_tail && pad_a == 64 && c.n == pad_a))) {
        return pack_tail ? (zero_r ? 12 : 6) : (zero_r ? 13 : 7);
    }
    return 0;
}

uint32_t select_fast_fp32_lu(const elec::Config& c, uint64_t blocks) {
    if ((c.flags & ~elec::LSTSQ_SQUARE_RESIDUAL) != 0 || c.rhs_batch_stride != c.n * c.k) {
        return 0;
    }
    if (blocks == c.batch) {
        return select_small_fp32_lu(c);
    }
    if (blocks == c.batch * 8 && (c.n & 7) == 0 && c.k <= 4) {
        if (c.n == 32 + 7 * 24 && c.k == 4) return 8;
        if (c.n == 32 + 7 * 32 && c.k == 3) return 9;
    }
    return 0;
}

ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto status = elec::lstsq_host::tile(context, [](uint32_t) {});
    if (status != ge::GRAPH_SUCCESS) return status;
    const auto* data = context->GetTilingData<LinalgLstsqTilingData>();
    const auto& c = data->config;
    const uint32_t raw_type = static_cast<uint32_t>(context->GetRequiredInputDesc(0)->GetDataType());
    uint32_t use_lu = 0;
    if (c.m == c.n && c.k <= 8 && c.n <= 256) {
        if (raw_type == static_cast<uint32_t>(ge::DT_FLOAT)) {
            use_lu = select_fast_fp32_lu(c, context->GetBlockDim());
        }
        if (use_lu == 0) {
            use_lu = (c.n <= 64) ? 1 : 2;
        }
    }
    ASCENDC_TPL_SEL_PARAM(context, raw_type, use_lu);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShape(gert::InferShapeContext* context) {
    return elec::lstsq_host::infer_shape(context);
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    return elec::lstsq_host::infer_dtype(context);
}
}  // namespace

namespace ops {
    class LinalgLstsq : public OpDef {
    public:
        explicit LinalgLstsq(const char *name) : OpDef(name) {
            this->Input("A")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("B")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("solution")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("residuals")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("driver").AttrType(OPTIONAL).String("gels");
            this->Attr("rcond").AttrType(OPTIONAL).Float(-1.0);
            this->SetInferShape(InferShape).SetInferDataType(InferDataType);
            this->AICore()
                .SetTiling(TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(LinalgLstsq);
}  // namespace ops

#ifdef OP_TILING_LIB
// RUN packages carry host inference in the same private registry as tiling.
IMPL_OP_INFERSHAPE(LinalgLstsq).InferShape(InferShape).InferDataType(InferDataType);
#endif
