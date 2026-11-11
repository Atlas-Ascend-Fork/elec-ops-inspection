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
#include <cstdint>
#include <cstddef>
#include <cstring>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/spline_conv_tiling.h"
#include "../op_kernel/tiling_key_spline_conv.h"
namespace optiling {
    // Attr defaults, overridden by the optional attributes of the operator.
    struct SplineConvAttrValues {
        uint32_t dim = 2U;
        uint32_t degree = 1U;
        uint32_t aggrMean = 1U;
        uint32_t kernel[3] = {
            2U, 2U, 1U };
        uint32_t open[3] = {
            1U, 1U, 1U };
    };
    struct SplineConvShapeValues {
        uint32_t n = 0U;
        uint32_t e = 0U;
        uint32_t cIn = 0U;
        uint32_t cOut = 0U;
    };
    static bool ReadKernelSizeAttr( const gert::RuntimeAttrs *attrs, SplineConvAttrValues &values) {
        const gert::TypedContinuousVector<int64_t> *kernelSize = attrs->GetListInt(1);
        if (kernelSize == nullptr) {
            return true;
        }
        const int64_t *data = kernelSize->GetData();
        if (data == nullptr || kernelSize->GetSize() < values.dim) {
            return false;
        }
        for (uint32_t d = 0U; d < 3U; ++d) {
            if (d >= values.dim) {
                values.kernel[d] = 1U;
                continue;
            }
            if (data[d] < 1 || data[d] > 64) {
                return false;
            }
            values.kernel[d] = static_cast<uint32_t>(data[d]);
        }
        return true;
    }
    static bool ReadOpenAttr( const gert::RuntimeAttrs *attrs, SplineConvAttrValues &values) {
        const gert::TypedContinuousVector<int64_t> *isOpen = attrs->GetListInt(2);
        if (isOpen == nullptr) {
            return true;
        }
        const int64_t *data = isOpen->GetData();
        if (data == nullptr || isOpen->GetSize() < values.dim) {
            return false;
        }
        for (uint32_t d = 0U; d < values.dim; ++d) {
            if (data[d] != 0 && data[d] != 1) {
                return false;
            }
            values.open[d] = static_cast<uint32_t>(data[d]);
        }
        return true;
    }
    static bool ParseAttrs( const gert::RuntimeAttrs *attrs, SplineConvAttrValues &values) {
        if (attrs == nullptr) {
            return true;
        }
        const int64_t *dimPtr = attrs->GetInt(0);
        if (dimPtr != nullptr) {
            if (*dimPtr < 1 || *dimPtr > 3) {
                return false;
            }
            values.dim = static_cast<uint32_t>( *dimPtr);
        }
        // kernel_size / is_open_spline depend on the effective dim, keep the original order.
        if (!ReadKernelSizeAttr(attrs, values) || !ReadOpenAttr(attrs, values)) {
            return false;
        }
        const int64_t *degreePtr = attrs->GetInt(3);
        if (degreePtr != nullptr) {
            if (*degreePtr < 0 || *degreePtr > 3) {
                return false;
            }
            values.degree = static_cast<uint32_t>( *degreePtr);
        }
        uint64_t kernelCount = 1U;
        for (uint32_t d = 0U; d < values.dim; ++d) {
            if (values.kernel[d] < values.degree * values.open[d]) {
                return false;
            }
            kernelCount *= values.kernel[d];
        }
        if (kernelCount > 64U) {
            return false;
        }
        const char *aggr = attrs->GetStr(4);
        if (aggr == nullptr) {
            return true;
        }
        if (std::strcmp( aggr, "add") == 0) {
            values.aggrMean = 0U;
        }
        else {
            values.aggrMean = 1U;
        }
        return true;
    }
    static ge::graphStatus ReadShapeInfo( gert::TilingContext *context, SplineConvShapeValues &values) {
        const auto *xShapeDesc = context->GetInputShape(0);
        const auto *edgeIndexShapeDesc = context->GetInputShape(1);
        const auto *edgeAttrShapeDesc = context->GetInputShape(2);
        const auto *weightShapeDesc = context->GetInputShape(3);
        if (xShapeDesc == nullptr || edgeIndexShapeDesc == nullptr ||
            edgeAttrShapeDesc == nullptr || weightShapeDesc == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape &xShape = xShapeDesc->GetOriginShape();
        const gert::Shape &edgeIndexShape = edgeIndexShapeDesc->GetOriginShape();
        const gert::Shape &weightShape = weightShapeDesc->GetOriginShape();
        const gert::Shape &edgeAttrShape = edgeAttrShapeDesc->GetOriginShape();
        if (xShape.GetDimNum() != 2U || edgeIndexShape.GetDimNum() != 2U ||
            edgeIndexShape.GetDim(0) != 2 || edgeAttrShape.GetDimNum() != 2U ||
            weightShape.GetDimNum() != 3U || xShape.GetDim(0) < 0 ||
            xShape.GetDim(0) > UINT32_MAX || edgeIndexShape.GetDim(1) < 0 ||
            edgeIndexShape.GetDim(1) > UINT32_MAX || edgeAttrShape.GetDim(0) != edgeIndexShape.GetDim(1) ||
            xShape.GetDim(1) < 1 || xShape.GetDim(1) > 128 ||
            weightShape.GetDim(1) != xShape.GetDim(1) ||
            weightShape.GetDim(2) < 1 || weightShape.GetDim(2) > 128) {
            return ge::GRAPH_FAILED;
        }
        values.n = static_cast<uint32_t>( xShape.GetDim(0));
        values.cIn = static_cast<uint32_t>( xShape.GetDim(1));
        values.e = static_cast<uint32_t>( edgeIndexShape.GetDim(1));
        values.cOut = static_cast<uint32_t>( weightShape.GetDim(2));
        return ge::GRAPH_SUCCESS;
    }
    static uint32_t ComputeBasisSlots( uint32_t dim, uint32_t degree) {
        const uint32_t basisPerDim = degree + 1U;
        uint32_t basisSlots = 1U;
        for (uint32_t d = 0U;
        d < dim;
        ++d) {
            basisSlots *= basisPerDim;
        }
        return basisSlots;
    }
    static uint64_t ComputeAccStrideFloats( uint32_t n, uint32_t cOut) {
        const uint64_t accElements64 = static_cast<uint64_t>( n) * static_cast<uint64_t>( cOut);
        return (accElements64 + 15U) & (~15ULL);
    }
    static uint32_t ComputeBlockDim( uint64_t work, uint32_t maxCores, uint32_t edgeCount) {
        uint32_t blockDim = 1U;
        if (work > 500000ULL) {
            blockDim = 8U;
        }
        if (work > 4000000ULL) {
            blockDim = maxCores;
        }
        if (blockDim > maxCores) {
            blockDim = maxCores;
        }
        if (blockDim > edgeCount) {
            blockDim = edgeCount;
        }
        if (blockDim == 0U) {
            blockDim = 1U;
        }
        return blockDim;
    }
    static void FillTilingData( SplineConvTilingData *tiling, const SplineConvShapeValues &shape,
        const SplineConvAttrValues &attrValues, uint32_t basisSlots, uint64_t accStrideFloats) {
        tiling->n = shape.n;
        tiling->e = shape.e;
        tiling->cIn = shape.cIn;
        tiling->cOut = shape.cOut;
        tiling->dim = attrValues.dim;
        tiling->degree = attrValues.degree;
        tiling->basisSlots = basisSlots;
        tiling->aggrMean = attrValues.aggrMean;
        tiling->kernel0 = attrValues.kernel[0];
        tiling->kernel1 = attrValues.kernel[1];
        tiling->kernel2 = attrValues.kernel[2];
        tiling->open0 = attrValues.open[0];
        tiling->open1 = attrValues.open[1];
        tiling->open2 = attrValues.open[2];
        tiling->accStrideFloats = accStrideFloats;
    }
    static ge::graphStatus TilingFunc( gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC( context->GetPlatformInfo());
        SplineConvShapeValues shape;
        if (ReadShapeInfo(context, shape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        SplineConvAttrValues attrValues;
        if (!ParseAttrs(context->GetAttrs(), attrValues)) {
            return ge::GRAPH_FAILED;
        }
        const auto &pseudoShape = context->GetInputShape(2)->GetOriginShape();
        const auto &weightShape = context->GetInputShape(3)->GetOriginShape();
        const uint64_t kernelCount = static_cast<uint64_t>(attrValues.kernel[0]) *
                                     attrValues.kernel[1] * attrValues.kernel[2];
        if (pseudoShape.GetDim(1) != attrValues.dim || weightShape.GetDim(0) != kernelCount) {
            return ge::GRAPH_FAILED;
        }
        const uint32_t basisSlots = ComputeBasisSlots( attrValues.dim, attrValues.degree);
        const uint64_t accStrideFloats = ComputeAccStrideFloats( shape.n, shape.cOut);
        SplineConvTilingData *tiling = context->GetTilingData< SplineConvTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        FillTilingData( tiling, shape, attrValues, basisSlots, accStrideFloats);
        const auto *xDesc = context->GetInputDesc(0);
        ge::DataType dtypeX = ge::DT_FLOAT;
        if (xDesc != nullptr) {
            dtypeX = xDesc->GetDataType();
        }
        uint32_t DT_X = static_cast<uint32_t>( dtypeX);
        ASCENDC_TPL_SEL_PARAM( context, DT_X);
        int32_t maxCoresSigned = platform.GetCoreNumAiv();
        if (maxCoresSigned <= 0) {
            maxCoresSigned = 1;
        }
        const uint32_t maxCores = static_cast<uint32_t>( maxCoresSigned);
        const uint64_t work = static_cast<uint64_t>( shape.e) * static_cast<uint64_t>( basisSlots) *
                              static_cast<uint64_t>( shape.cIn) * static_cast<uint64_t>( shape.cOut);
        const uint32_t blockDim = ComputeBlockDim( work, maxCores, shape.e);
        context->SetBlockDim( blockDim);
        const size_t userWorkspaceBytes = ( static_cast<size_t>( accStrideFloats) + static_cast<size_t>( shape.n) ) * sizeof(float);
        const uint32_t systemWorkspaceBytes = platform.GetLibApiWorkSpaceSize();
        size_t *workspaceSizes = context->GetWorkspaceSizes(1);
        if (workspaceSizes == nullptr) {
            return ge::GRAPH_FAILED;
        }
        workspaceSizes[0] = static_cast<size_t>( systemWorkspaceBytes) + userWorkspaceBytes;
        return ge::GRAPH_SUCCESS;
    }
}
namespace ge {
    static graphStatus InferShape( gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        const gert::Shape *weightShape = context->GetInputShape(3);
        gert::Shape *outShape = context->GetOutputShape(0);
        if (xShape == nullptr || weightShape == nullptr || outShape == nullptr ||
            xShape->GetDimNum() != 2U || weightShape->GetDimNum() != 3U) {
            return GRAPH_FAILED;
        }
        *outShape = *xShape;
        outShape->SetDim( 1, weightShape->GetDim(2));
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType( gert::InferDataTypeContext *context) {
        context->SetOutputDataType( 0, context->GetInputDataType(0));
        return GRAPH_SUCCESS;
    }
}
namespace ops {
    class SplineConv : public OpDef {
        public: explicit SplineConv( const char *name) : OpDef(name) {
            this->Input("x") .ParamType(REQUIRED) .DataType({
                ge::DT_FLOAT, ge::DT_BF16 }) .Format({
                ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("edge_index") .ParamType(REQUIRED) .DataType({
                ge::DT_INT64, ge::DT_INT64 }) .Format({
                ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("edge_attr") .ParamType(REQUIRED) .DataType({
                ge::DT_FLOAT, ge::DT_BF16 }) .Format({
                ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("weight") .ParamType(REQUIRED) .DataType({
                ge::DT_FLOAT, ge::DT_BF16 }) .Format({
                ge::FORMAT_ND, ge::FORMAT_ND });
            this->Output("out") .ParamType(REQUIRED) .DataType({
                ge::DT_FLOAT, ge::DT_BF16 }) .Format({
                ge::FORMAT_ND, ge::FORMAT_ND });
            this->Attr("dim") .AttrType(OPTIONAL) .Int(2);
            this->Attr("kernel_size") .AttrType(OPTIONAL) .ListInt({
                2, 2 });
            this->Attr("is_open_spline") .AttrType(OPTIONAL) .ListInt({
                1, 1 });
            this->Attr("degree") .AttrType(OPTIONAL) .Int(1);
            this->Attr("aggr") .AttrType(OPTIONAL) .String("mean");
            this->SetInferShape( ge::InferShape) .SetInferDataType( ge::InferDataType);
            this->AICore() .SetTiling( optiling::TilingFunc) .AddConfig( "ascend910b");
        }
    };
    OP_ADD( SplineConv);
}
