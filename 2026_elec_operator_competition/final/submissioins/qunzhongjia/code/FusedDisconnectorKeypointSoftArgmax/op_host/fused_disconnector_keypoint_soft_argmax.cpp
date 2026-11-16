#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include "../op_kernel/fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "../op_kernel/tiling_key_fused_disconnector_keypoint_soft_argmax.h"
namespace {
constexpr int64_t kFixedCin = 32;
constexpr int64_t kFixedK = 4;
constexpr int64_t kFixedKernel = 3;
bool ValidateTensorShapes(const gert::Shape* featShape,
const gert::Shape* weightShape,
const gert::Shape* biasShape,
uint32_t& B,
uint32_t& C,
uint32_t& H,
uint32_t& W)
{
if (featShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
return false;
}
if (featShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4 || biasShape->GetDimNum() != 1) {
return false;
}
const int64_t bDim = featShape->GetDim(0);
const int64_t cDim = featShape->GetDim(1);
const int64_t hDim = featShape->GetDim(2);
const int64_t wDim = featShape->GetDim(3);
if (bDim < 1 || cDim != kFixedCin || hDim < 1 || wDim < 1) {
return false;
}
if (bDim > 65535 || hDim > 16384 || wDim > 16384) {
return false;
}
if (weightShape->GetDim(0) != kFixedK || weightShape->GetDim(1) != kFixedCin ||
weightShape->GetDim(2) != kFixedKernel || weightShape->GetDim(3) != kFixedKernel) {
return false;
}
if (biasShape->GetDim(0) != kFixedK) {
return false;
}
B = static_cast<uint32_t>(bDim);
C = static_cast<uint32_t>(cDim);
H = static_cast<uint32_t>(hDim);
W = static_cast<uint32_t>(wDim);
return true;
}
uint32_t CeilDiv(uint32_t a, uint32_t b)
{
return (a + b - 1U) / b;
}
uint32_t SelectKernelPathAndStripes(uint32_t B,
uint32_t C,
uint32_t H,
uint32_t W,
uint32_t cores,
uint32_t& stripesPerBatch,
uint32_t& usedCores,
size_t& userWsFloats)
{
static const bool forceBaseline = []() {
const char* env = std::getenv("KEYPOINT_FORCE_PATH0");
return env != nullptr && env[0] == '1';
}();
if (forceBaseline) {
const uint32_t totalHeatmaps = B * 4U;
stripesPerBatch = 1U;
usedCores = std::min(cores, totalHeatmaps);
userWsFloats = static_cast<size_t>(totalHeatmaps) * 16U;
return 0U;
}
const uint32_t totalHm = B * 4U;
if (C == 32U && cores >= 36U && B >= 1U && B <= 4U) {
if (H == 48U && W == 64U) {
constexpr uint32_t kStripes48[4] = {8U, 4U, 3U, 2U};
stripesPerBatch = kStripes48[B - 1U];
usedCores = totalHm * stripesPerBatch;
userWsFloats = 0U;
return B;
}
if (H == 64U && W == 64U) {
constexpr uint32_t kStripes64[4] = {8U, 4U, 2U, 2U};
stripesPerBatch = kStripes64[B - 1U];
usedCores = totalHm * stripesPerBatch;
userWsFloats = 0U;
return 4U + B;
}
if (H == 72U && W == 96U) {
constexpr uint32_t kStripes72[4] = {6U, 4U, 3U, 2U};
stripesPerBatch = kStripes72[B - 1U];
usedCores = totalHm * stripesPerBatch;
userWsFloats = 0U;
return 8U + B;
}
}
if (C == 32U && W >= 1U && W <= 128U && H >= 1U && H <= 128U && B >= 1U && B <= 16U) {
const uint32_t maxLimitRows = (W <= 64U) ? 48U : ((W <= 96U) ? 32U : 20U);
const uint32_t neededS = CeilDiv(H, maxLimitRows);
if (neededS >= 1U && neededS <= 9U) {
stripesPerBatch = neededS;
usedCores = std::min(cores, totalHm);
userWsFloats = 0U;
if ((W % 8U) == 0U) {
if (W == 64U) {
return 13U;
}
if (W == 96U) {
return 14U;
}
return 15U;
}
if (W <= 96U) {
return 16U;
}
}
}
if (C == 32U && W >= 1U && W <= 256U) {
stripesPerBatch = 1U;
usedCores = std::min(cores, totalHm);
userWsFloats = 0U;
return 17U;
}
stripesPerBatch = 1U;
usedCores = std::min(cores, B);
userWsFloats = 0U;
return 18U;
}
}
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
if (context == nullptr || context->GetPlatformInfo() == nullptr) {
return ge::GRAPH_FAILED;
}
const auto* featShape = context->GetInputShape(0);
const auto* weightShape = context->GetInputShape(1);
const auto* biasShape = context->GetInputShape(2);
if (featShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
return ge::GRAPH_FAILED;
}
uint32_t B = 0;
uint32_t C = 0;
uint32_t H = 0;
uint32_t W = 0;
if (!ValidateTensorShapes(&featShape->GetStorageShape(),
&weightShape->GetStorageShape(),
&biasShape->GetStorageShape(),
B, C, H, W)) {
return ge::GRAPH_FAILED;
}
const auto* d0 = context->GetInputDesc(0);
const auto* d1 = context->GetInputDesc(1);
const auto* d2 = context->GetInputDesc(2);
if (d0 == nullptr || d1 == nullptr || d2 == nullptr ||
d0->GetDataType() != ge::DT_FLOAT ||
d1->GetDataType() != ge::DT_FLOAT ||
d2->GetDataType() != ge::DT_FLOAT) {
return ge::GRAPH_FAILED;
}
const gert::RuntimeAttrs* attrs = context->GetAttrs();
if (attrs == nullptr) {
return ge::GRAPH_FAILED;
}
const float* tauPtr = attrs->GetFloat(0);
const int64_t* stridePtr = attrs->GetInt(1);
if (tauPtr == nullptr || stridePtr == nullptr) {
return ge::GRAPH_FAILED;
}
const float tau = *tauPtr;
const int64_t stride = *stridePtr;
if (!std::isfinite(tau) || tau <= 0.0f || stride <= 0 || stride > 65536) {
return ge::GRAPH_FAILED;
}
auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
const uint32_t cores = std::max(1U, static_cast<uint32_t>(platform.GetCoreNumAiv()));
uint32_t stripesPerBatch = 1U;
uint32_t usedCores = 1U;
size_t userWsFloats = 0U;
uint32_t pathId = SelectKernelPathAndStripes(
B, C, H, W, cores, stripesPerBatch, usedCores, userWsFloats);
if (pathId == 4U && tau == 1.0f && stride == 4) {
pathId = 19U;
stripesPerBatch = 1U;
usedCores = 1U;
userWsFloats = 0U;
}
auto* td = context->GetTilingData<FusedDisconnectorKeypointSoftArgmaxTilingData>();
size_t* workspaces = context->GetWorkspaceSizes(1);
if (td == nullptr || workspaces == nullptr) {
return ge::GRAPH_FAILED;
}
const float invTau = static_cast<float>(1.0 / static_cast<double>(tau));
const uint32_t exactInvTau = ((invTau * tau) == 1.0f) ? 1U : 0U;
td->B = B;
td->C = C;
td->H = H;
td->W = W;
td->stripesPerBatch = stripesPerBatch;
td->usedCores = usedCores;
td->exactInvTau = exactInvTau;
td->tau = tau;
td->invTau = invTau;
td->strideFloat = static_cast<float>(stride);
for (uint32_t i = 0U; i < 6U; ++i) {
td->pad[i] = 0U;
}
for (uint32_t i = 0U; i < 28U * 16U; ++i) {
td->syncSlots[i] = 0.0f;
}
context->GetRawTilingData()->SetDataSize(sizeof(FusedDisconnectorKeypointSoftArgmaxTilingData));
ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(ge::DT_FLOAT), pathId);
context->SetBlockDim(usedCores);
workspaces[0] = 0U;
return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
if (context == nullptr) {
return GRAPH_FAILED;
}
const gert::Shape* featShape = context->GetInputShape(0);
const gert::Shape* weightShape = context->GetInputShape(1);
const gert::Shape* biasShape = context->GetInputShape(2);
gert::Shape* kpShape = context->GetOutputShape(0);
gert::Shape* scShape = context->GetOutputShape(1);
if (featShape == nullptr || weightShape == nullptr || biasShape == nullptr ||
kpShape == nullptr || scShape == nullptr) {
return GRAPH_FAILED;
}
if (featShape->GetDimNum() != 4) {
return GRAPH_FAILED;
}
const int64_t bDim = featShape->GetDim(0);
kpShape->SetDimNum(3);
kpShape->SetDim(0, bDim);
kpShape->SetDim(1, 4);
kpShape->SetDim(2, 2);
scShape->SetDimNum(2);
scShape->SetDim(0, bDim);
scShape->SetDim(1, 4);
return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
if (context == nullptr) {
return GRAPH_FAILED;
}
const auto dtype = context->GetInputDataType(0);
context->SetOutputDataType(0, dtype);
context->SetOutputDataType(1, dtype);
return GRAPH_SUCCESS;
}
}
namespace ops {
class FusedDisconnectorKeypointSoftArgmax : public OpDef {
public:
explicit FusedDisconnectorKeypointSoftArgmax(const char* name) : OpDef(name)
{
this->Input("feat")
.AutoContiguous()
.ParamType(REQUIRED)
.DataType({ge::DT_FLOAT})
.Format({ge::FORMAT_ND});
this->Input("weight")
.AutoContiguous()
.ParamType(REQUIRED)
.DataType({ge::DT_FLOAT})
.Format({ge::FORMAT_ND});
this->Input("bias")
.AutoContiguous()
.ParamType(REQUIRED)
.DataType({ge::DT_FLOAT})
.Format({ge::FORMAT_ND});
this->Output("keypoints")
.ParamType(REQUIRED)
.DataType({ge::DT_FLOAT})
.Format({ge::FORMAT_ND});
this->Output("scores")
.ParamType(REQUIRED)
.DataType({ge::DT_FLOAT})
.Format({ge::FORMAT_ND});
this->Attr("tau").AttrType(REQUIRED).Float();
this->Attr("stride").AttrType(REQUIRED).Int();
this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
}
};
OP_ADD(FusedDisconnectorKeypointSoftArgmax);
}
