/*
 * Copyright (c) 2026 aodebiao.
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
// Host侧Tiling实现
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "../op_kernel/swi_glu_tiling.h"
#include "../op_kernel/tiling_key_swi_glu.h"

namespace optiling {
static inline uint32_t ToU32Positive(int64_t v)
{
    return v > 0 ? static_cast<uint32_t>(v) : 0U;
}

// SwiGLU 的五个必选输入。
struct SwiGluInputs {
    const gert::StorageShape *x = nullptr;
    const gert::StorageShape *w1 = nullptr;
    const gert::StorageShape *b1 = nullptr;
    const gert::StorageShape *w2 = nullptr;
    const gert::StorageShape *b2 = nullptr;
    ge::DataType dtype = ge::DT_UNDEFINED;
};

static bool FetchInputs(gert::TilingContext *context, SwiGluInputs &inputs)
{
    inputs.x = context->GetInputShape(0);
    inputs.w1 = context->GetInputShape(1);
    inputs.b1 = context->GetInputShape(2);
    inputs.w2 = context->GetInputShape(3);
    inputs.b2 = context->GetInputShape(4);
    const auto *desc = context->GetInputDesc(0);
    if (desc == nullptr) {
        return false;
    }
    inputs.dtype = desc->GetDataType();
    return inputs.x != nullptr && inputs.w1 != nullptr && inputs.b1 != nullptr &&
           inputs.w2 != nullptr && inputs.b2 != nullptr;
}

static bool ShapeElements(const gert::Shape &shape, uint64_t &elements)
{
    elements = 1U;
    for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
        const int64_t dimension = shape.GetDim(i);
        if (dimension < 0 || dimension > UINT32_MAX ||
            (dimension != 0 && elements > INT64_MAX / static_cast<uint64_t>(dimension))) {
            return false;
        }
        elements *= static_cast<uint64_t>(dimension);
    }
    return true;
}

static bool CheckWeights(const gert::Shape &w1, const gert::Shape &w2, uint64_t k, uint64_t n)
{
    for (const auto *weight : {&w1, &w2}) {
        uint64_t elements;
        if (!ShapeElements(*weight, elements) || elements != k * n) {
            return false;
        }
        const uint64_t d0 = static_cast<uint64_t>(weight->GetDim(weight->GetDimNum() - 2U));
        const uint64_t d1 = static_cast<uint64_t>(weight->GetDim(weight->GetDimNum() - 1U));
        if (!((d0 == k && d1 == n) || (d0 == n && d1 == k))) {
            return false;
        }
    }
    return true;
}

static bool CheckShapes(const gert::Shape *x, const gert::Shape *w1, const gert::Shape *b1,
                        const gert::Shape *w2, const gert::Shape *b2)
{
    if (x == nullptr || w1 == nullptr || b1 == nullptr || w2 == nullptr || b2 == nullptr ||
        x->GetDimNum() == 0U || w1->GetDimNum() < 2U || w2->GetDimNum() < 2U) {
        return false;
    }
    uint64_t xElements, b1Elements, b2Elements;
    if (!ShapeElements(*x, xElements) || !ShapeElements(*b1, b1Elements) ||
        !ShapeElements(*b2, b2Elements) || b1Elements > UINT32_MAX || b1Elements != b2Elements) {
        return false;
    }
    const uint64_t k = static_cast<uint64_t>(x->GetDim(x->GetDimNum() - 1U));
    uint64_t m = 1U;
    for (size_t i = 0U; i + 1U < x->GetDimNum(); ++i) {
        m *= static_cast<uint64_t>(x->GetDim(i));
        if (m > UINT32_MAX) {
            return false;
        }
    }
    const uint64_t n = b1Elements;
    if (k * n > INT64_MAX || m * n > INT64_MAX) {
        return false;
    }
    return CheckWeights(*w1, *w2, k, n);
}

// 权重的最后两维。
struct WeightDims {
    uint32_t d0 = 0U;
    uint32_t d1 = 0U;
};

// 取 shape 的最后两维，维度不足时置 0。
static inline WeightDims GetLastTwoDims(const gert::Shape &shape)
{
    WeightDims dims;
    if (shape.GetDimNum() < 2U) {
        return dims;
    }
    dims.d0 = ToU32Positive(shape.GetDim(shape.GetDimNum() - 2U));
    dims.d1 = ToU32Positive(shape.GetDim(shape.GetDimNum() - 1U));
    return dims;
}

// K 取 input_x 的最后一维。
static inline uint32_t ResolveK(const gert::Shape &xShape)
{
    uint32_t k = 0U;
    if (xShape.GetDimNum() > 0U) {
        k = ToU32Positive(xShape.GetDim(xShape.GetDimNum() - 1U));
    }
    return k;
}

// For this five-input SwiGLU, both linear branches produce the same N.
// Bias length is the most reliable source of N and also works for [1, N].
static inline uint32_t ResolveN(uint32_t b1Elems, uint32_t b2Elems, uint32_t k,
                                uint32_t w1D0, uint32_t w1D1)
{
    uint32_t n = b1Elems;
    if (n == 0U) {
        n = b2Elems;
    }
    if (n == 0U) {
        if (w1D0 == k && w1D1 != 0U) {
            n = w1D1;
        } else if (w1D1 == k && w1D0 != 0U) {
            n = w1D0;
        }
    }
    return n;
}

// 0: [K, N], index k*N+n. 1: [N, K], index n*K+k.
// The reference side uses Linear-style weights: [out_features, in_features]
// i.e. [N, K]. For non-square matrices the shape determines the layout.
// For square K == N matrices the shape is ambiguous, so prefer [N, K]
// instead of [K, N]. This is essential because transposing a square weight
// changes the result even though its shape is unchanged.
static inline uint32_t ResolveTransposed(uint32_t d0, uint32_t d1, uint32_t k, uint32_t n)
{
    if ((d0 == k && d1 == n) && !(d0 == n && d1 == k)) {
        return 0U;
    }
    if (d0 == n && d1 == k) {
        return 1U;
    }
    return 1U;
}

static inline uint32_t ComputeM(const gert::Shape &shape)
{
    uint32_t m = 1U;
    for (size_t i = 0U; i + 1U < shape.GetDimNum(); ++i) {
        m *= static_cast<uint32_t>(shape.GetDim(i));
    }
    return m;
}

static inline uint32_t ComputeBlockDim(int32_t numCoresAiv, uint32_t m, uint32_t n)
{
    const uint32_t tilesPerRow = (static_cast<uint64_t>(n) + SWIGLU_OUT_TILE - 1U) / SWIGLU_OUT_TILE;
    const uint64_t taskCount64 = static_cast<uint64_t>(m) * static_cast<uint64_t>(tilesPerRow);
    const uint32_t taskCount = taskCount64 > 0xFFFFFFFFULL
                                   ? 0xFFFFFFFFU
                                   : static_cast<uint32_t>(taskCount64);
    uint32_t blockDim = static_cast<uint32_t>(numCoresAiv);
    if (blockDim > taskCount) {
        blockDim = taskCount;
    }
    if (blockDim == 0U) {
        blockDim = 1U;
    }
    return blockDim;
}

static inline void FillTilingData(SwiGluTilingData *tiling, uint32_t m, uint32_t k, uint32_t n,
                                  uint32_t w1Transposed, uint32_t w2Transposed)
{
    tiling->m = m;
    tiling->k = k;
    tiling->n = n;
    tiling->weight1Transposed = w1Transposed;
    tiling->weight2Transposed = w2Transposed;
    tiling->useCube = 0;
    tiling->tileM = 32;
    tiling->tileN = 128;
}

// 取 tiling 数据并申请、清零 workspace。
static inline ge::graphStatus PrepareBuffers(gert::TilingContext *context,
                                             SwiGluTilingData *&tiling,
                                             size_t *&workspace)
{
    tiling = context->GetTilingData<SwiGluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

// Cost-based dispatch: matrix work dominates activation for K >= 16.
static inline void TryEnableCube(platform_ascendc::PlatformAscendC &platform,
                                 uint32_t m, uint32_t n, uint32_t k,
                                 SwiGluTilingData *tiling, uint32_t &blockDim,
                                 size_t *workspace)
{
    const bool cubeWorthIt = (k >= 16U) && (k > 512U || m > 4U) &&
                             (static_cast<uint64_t>(m) * n >= (4096ULL + k - 1U) / k);
    if (!cubeWorthIt || tiling->weight1Transposed == 0U || tiling->weight2Transposed == 0U ||
        m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) {
        return;
    }

    matmul_tiling::MatmulApiTiling mm(platform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT16);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT16, true);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT16);
    mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                   matmul_tiling::DataType::DT_FLOAT16);
    mm.SetOrgShape(m, n, k);
    mm.SetShape(m < 32U ? m : 32U, n < 128U ? n : 128U, k);
    mm.SetFixSplit(32, 128, -1);
    mm.EnableBias(false);
    mm.SetBufferSpace(-1, -1, -1);

    if (mm.GetTiling(tiling->cube) == -1) {
        return;
    }
    tiling->useCube = 1;
    uint32_t aic = platform.GetCoreNumAic();
    if (aic == 0U) {
        aic = 1U;
    }
    const uint64_t tasks = ((static_cast<uint64_t>(m) + 31U) / 32U) *
                           ((static_cast<uint64_t>(n) + 127U) / 128U);
    blockDim = tasks < aic ? tasks : aic;
    workspace[0] = static_cast<size_t>(m) * n * 4 + platform.GetLibApiWorkSpaceSize();
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t numCoresAiv = platform.GetCoreNumAiv();
    if (numCoresAiv <= 0) {
        numCoresAiv = 1;
    }

    SwiGluInputs inputs;
    if (!FetchInputs(context, inputs)) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckShapes(&inputs.x->GetStorageShape(), &inputs.w1->GetStorageShape(),
                     &inputs.b1->GetStorageShape(), &inputs.w2->GetStorageShape(),
                     &inputs.b2->GetStorageShape())) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dtypeInputX = inputs.dtype;
    const uint32_t b1Elems = ToU32Positive(inputs.b1->GetStorageShape().GetShapeSize());
    const uint32_t b2Elems = ToU32Positive(inputs.b2->GetStorageShape().GetShapeSize());
    const uint32_t k = ResolveK(inputs.x->GetStorageShape());
    const WeightDims w1Dims = GetLastTwoDims(inputs.w1->GetStorageShape());
    const WeightDims w2Dims = GetLastTwoDims(inputs.w2->GetStorageShape());
    const uint32_t n = ResolveN(b1Elems, b2Elems, k, w1Dims.d0, w1Dims.d1);
    const uint32_t w1Transposed = ResolveTransposed(w1Dims.d0, w1Dims.d1, k, n);
    const uint32_t w2Transposed = ResolveTransposed(w2Dims.d0, w2Dims.d1, k, n);
    const uint32_t m = ComputeM(inputs.x->GetStorageShape());
    uint32_t blockDim = ComputeBlockDim(numCoresAiv, m, n);

    SwiGluTilingData *tiling = nullptr;
    size_t *currentWorkspace = nullptr;
    if (PrepareBuffers(context, tiling, currentWorkspace) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    FillTilingData(tiling, m, k, n, w1Transposed, w2Transposed);
    TryEnableCube(platform, m, n, k, tiling, blockDim, currentWorkspace);

    tiling->blockDim = blockDim;
    context->SetBlockDim(blockDim);
    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtypeInputX);
    uint32_t KERNEL_MODE = tiling->useCube ? ASCENDC_TPL_MIX_AIC_1_1 : ASCENDC_TPL_AIV_ONLY;
    bool USE_CUBE = tiling->useCube != 0;
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, KERNEL_MODE, USE_CUBE);

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    const gert::Shape *w1Shape = context->GetInputShape(1);
    const gert::Shape *b1Shape = context->GetInputShape(2);
    const gert::Shape *w2Shape = context->GetInputShape(3);
    const gert::Shape *b2Shape = context->GetInputShape(4);
    gert::Shape *outShape = context->GetOutputShape(0);
    if (outShape == nullptr || !optiling::CheckShapes(xShape, w1Shape, b1Shape, w2Shape, b2Shape)) {
        return GRAPH_FAILED;
    }

    *outShape = *xShape;
    if (outShape->GetDimNum() == 0U) {
        return GRAPH_SUCCESS;
    }

    int64_t n = 0;
    n = b1Shape->GetShapeSize();

    // Fallback when bias shape metadata is unavailable: infer N from W1 and K.
    if (n <= 0 && w1Shape != nullptr && w1Shape->GetDimNum() >= 2U) {
        const int64_t k = xShape->GetDim(xShape->GetDimNum() - 1U);
        const int64_t d0 = w1Shape->GetDim(w1Shape->GetDimNum() - 2U);
        const int64_t d1 = w1Shape->GetDim(w1Shape->GetDimNum() - 1U);
        if (d0 == k) {
            n = d1;
        } else if (d1 == k) {
            n = d0;
        }
    }
    outShape->SetDim(outShape->GetDimNum() - 1U, n);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SwiGlu : public OpDef {
public:
    explicit SwiGlu(const char *name) : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("weight_1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("bias_1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("weight_2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("bias_2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(SwiGlu);
}  // namespace ops
