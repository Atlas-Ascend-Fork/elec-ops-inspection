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
#pragma once
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_lstsq_tiling.h"
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace elec::lstsq_host {
using Shape = std::vector<int64_t>;
constexpr uint64_t memory_limit = uint64_t(1) << 30;

inline void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

inline uint64_t multiply(uint64_t a, uint64_t b) {
    if (a == 0 || b == 0) return 0;
    require(b <= memory_limit / a, "lstsq allocation exceeds 1 GiB");
    return a * b;
}

inline uint64_t add(uint64_t a, uint64_t b) {
    require(a <= memory_limit && b <= memory_limit - a, "lstsq allocation exceeds 1 GiB");
    return a + b;
}

inline uint64_t elements(const Shape& shape) {
    uint64_t result = 1;
    for (const int64_t dimension : shape) {
        require(dimension > 0, "lstsq input dimensions must be positive");
        result = multiply(result, static_cast<uint64_t>(dimension));
    }
    return result;
}

inline bool empty_shape(const Shape& shape) {
    bool has_zero = false;
    for (const int64_t d : shape) {
        if (d < 0) return false;
        has_zero = has_zero || (d == 0);
    }
    return has_zero;
}

inline uint64_t output_elements(const Shape& shape) {
    if (empty_shape(shape)) return 0;
    return elements(shape);
}

inline uint64_t align64(uint64_t bytes) {
    return add(bytes, (64 - bytes % 64) % 64);
}

inline Shape append_shape(Shape s, std::initializer_list<int64_t> tail) {
    s.insert(s.end(), tail);
    return s;
}

inline Shape get_shape(const gert::Shape* s) {
    require(s != nullptr, "missing shape");
    Shape result;
    for (size_t i = 0; i < s->GetDimNum(); ++i) {
        result.push_back(s->GetDim(i));
    }
    return result;
}

inline uint32_t map_dtype(ge::DataType t) {
    if (t == ge::DT_FLOAT) return F32;
    if (t == ge::DT_BF16) return BF16;
    throw std::invalid_argument("unsupported lstsq dtype");
}

inline Config parse_attrs(const gert::RuntimeAttrs* attrs) {
    Config c;
    const char* driver = attrs ? attrs->GetStr(0) : nullptr;
    require(!driver || std::strcmp(driver, "gels") == 0, "only driver=gels is supported");
    return c;
}

struct Plan {
    TemplateTilingData tiling;
    std::vector<Shape> outputs;
    Shape batch_shape;
    bool vector_rhs = false;
    uint64_t tensor_bytes = 0;
    uint64_t per_core_bytes = 0;
};

inline void configure_plan_workspace(
    Plan& plan, uint32_t type, uint64_t input_a, uint64_t input_b, bool residual_requested) {
    const Config& c = plan.tiling.config;
    const uint64_t bytes = (type == F32) ? 4 : 2;
    const uint64_t output_x = multiply(multiply(c.batch, c.n), c.k);
    const uint64_t output_r = residual_requested ? multiply(c.batch, c.k) : 0;
    plan.tensor_bytes = multiply(add(add(input_a, input_b), add(output_x, output_r)), bytes);
    if (c.m == c.n && c.n <= 64 && c.k <= 8) {
        plan.tiling.scratch_bytes = 0;
        plan.per_core_bytes = 0;
        return;
    }
    const uint64_t pad_n = (c.n + 7) & ~uint64_t(7);
    if (c.m == c.n && c.n <= 256 && c.k <= 8) {
        plan.tiling.scratch_bytes = multiply(add(multiply(c.n, pad_n), 4096), 4);
        plan.per_core_bytes = align64(plan.tiling.scratch_bytes);
        add(plan.tensor_bytes, plan.per_core_bytes);
        return;
    }
    const uint64_t scratch_elems = add(add(multiply(c.m, pad_n), multiply(multiply(2, c.m), c.k)), c.m);
    plan.tiling.scratch_bytes = multiply(scratch_elems, 4);
    const uint64_t staged_x = align64(multiply(multiply(c.n, c.k), bytes));
    const uint64_t staged_r = residual_requested ? align64(multiply(c.k, bytes)) : 0;
    plan.per_core_bytes = add(add(align64(plan.tiling.scratch_bytes), staged_x), staged_r);
    add(plan.tensor_bytes, plan.per_core_bytes);
}

inline Plan make(const std::vector<Shape>& inputs, uint32_t type, Config c,
                 bool matrix_rhs = false) {
    require(inputs.size() == 2 && (type == F32 || type == BF16), "invalid lstsq inputs/type");
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    require(a.size() >= 2, "lstsq A must have at least two dimensions");
    const uint64_t input_a = elements(a);
    const uint64_t input_b = elements(b);
    Plan plan;
    plan.batch_shape.assign(a.begin(), a.end() - 2);
    c.batch = elements(plan.batch_shape);
    c.m = a[a.size() - 2];
    c.n = a.back();
    require(c.m >= c.n, "lstsq requires m >= n");
    const Shape vector_batch = append_shape(plan.batch_shape, {int64_t(c.m)});
    plan.vector_rhs = !matrix_rhs && (b == Shape{int64_t(c.m)} || b == vector_batch);
    if (plan.vector_rhs) {
        c.k = 1;
    } else {
        require(b.size() >= 2, "lstsq B must be a vector or matrix");
        c.k = b.back();
    }
    const Shape tail = plan.vector_rhs ? Shape{int64_t(c.m)} : Shape{int64_t(c.m), int64_t(c.k)};
    if (b == tail) {
        c.rhs_batch_stride = 0;
    } else {
        Shape batched = plan.batch_shape;
        batched.insert(batched.end(), tail.begin(), tail.end());
        require(b == batched, "input shape does not match the operator template");
        c.rhs_batch_stride = multiply(c.m, c.k);
    }
    c.dtype = F32;
    plan.tiling.config = c;
    for (auto& input_type : plan.tiling.input_types) input_type = type;
    for (auto& output_type : plan.tiling.output_types) output_type = type;
    plan.outputs.push_back(plan.vector_rhs
        ? append_shape(plan.batch_shape, {int64_t(c.n)})
        : append_shape(plan.batch_shape, {int64_t(c.n), int64_t(c.k)}));
    const bool residual_requested = c.m > c.n || (c.flags & LSTSQ_SQUARE_RESIDUAL);
    plan.outputs.push_back(residual_requested
        ? append_shape(plan.batch_shape, {int64_t(c.k)}) : Shape{0});
    configure_plan_workspace(plan, type, input_a, input_b, residual_requested);
    return plan;
}

inline bool concrete_hint(const gert::Shape* shape) {
    if (!shape || shape->GetDimNum() == 0) return false;
    for (size_t dimension = 0; dimension < shape->GetDimNum(); ++dimension) {
        if (shape->GetDim(dimension) < 0) return false;
    }
    return true;
}

inline std::string shape_text(const Shape& shape) {
    std::string result = "[";
    for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
        if (dimension) result += ",";
        result += std::to_string(shape[dimension]);
    }
    return result + "]";
}

inline void append_tensor_shapes(std::string& message, const char* name,
                                 const gert::StorageShape* shape) {
    message += std::string(" ") + name;
    if (!shape) {
        message += "=missing";
        return;
    }
    message += "_origin=" + shape_text(get_shape(&shape->GetOriginShape()));
    message += std::string(" ") + name + "_storage=" + shape_text(get_shape(&shape->GetStorageShape()));
}

inline std::string output_mismatch(gert::TilingContext* context, const Plan& plan,
                                   size_t index, const Shape& actual,
                                   uint64_t actual_count, uint64_t expected_count) {
    std::string message = "lstsq output view element count mismatch output_index=" +
        std::to_string(index) + " actual_elements=" + std::to_string(actual_count) +
        " expected_elements=" + std::to_string(expected_count) +
        " actual_shape=" + shape_text(actual) +
        " expected_shape=" + shape_text(plan.outputs[index]);
    append_tensor_shapes(message, "A", context->GetRequiredInputShape(0));
    append_tensor_shapes(message, "B", context->GetRequiredInputShape(1));
    append_tensor_shapes(message, "output0", context->GetOutputShape(0));
    append_tensor_shapes(message, "output1", context->GetOutputShape(1));
    const auto& config = plan.tiling.config;
    return message + " m=" + std::to_string(config.m) + " n=" + std::to_string(config.n) +
        " k=" + std::to_string(config.k) + " batch=" + std::to_string(config.batch) +
        " rhs=" + (plan.vector_rhs ? "vector" : "matrix");
}

inline bool positive_dims(const Shape& s) {
    for (const int64_t dimension : s) {
        if (dimension <= 0) return false;
    }
    return true;
}

inline void apply_inferred_outputs(gert::InferShapeContext* context, const std::vector<Shape>& outputs) {
    for (size_t index = 0; index < outputs.size(); ++index) {
        auto* output = context->GetOutputShape(index);
        if (!output || concrete_hint(output)) continue;
        output->SetDimNum(outputs[index].size());
        for (size_t dimension = 0; dimension < outputs[index].size(); ++dimension) {
            output->SetDim(dimension, outputs[index][dimension]);
        }
    }
}

inline ge::graphStatus infer_shape(gert::InferShapeContext* context) {
    try {
        auto* solution = context->GetOutputShape(0);
        require(solution != nullptr, "missing lstsq solution shape");
        if (concrete_hint(solution)) return ge::GRAPH_SUCCESS;
        const auto* input_a = context->GetInputShape(0);
        const auto* input_b = context->GetInputShape(1);
        if (!input_a || !input_b) return ge::GRAPH_SUCCESS;
        const Shape a = get_shape(input_a);
        const Shape b = get_shape(input_b);
        if (a.size() < 2 || b.empty() || !positive_dims(a) || !positive_dims(b)) {
            return ge::GRAPH_SUCCESS;
        }
        const Shape batch(a.begin(), a.end() - 2);
        const int64_t m = a[a.size() - 2];
        const int64_t n = a.back();
        const Shape matrix_tail{m, b.back()};
        Shape matrix_batch = batch;
        matrix_batch.insert(matrix_batch.end(), matrix_tail.begin(), matrix_tail.end());
        const bool matrix_rhs = b.size() >= 2 && (b == matrix_tail || b == matrix_batch);
        const bool vector_rhs = b == Shape{m} || b == append_shape(batch, {m});
        if (matrix_rhs == vector_rhs) return ge::GRAPH_SUCCESS;
        const int64_t k = matrix_rhs ? b.back() : 1;
        const std::vector<Shape> outputs{
            matrix_rhs ? append_shape(batch, {n, k}) : append_shape(batch, {n}),
            m > n ? append_shape(batch, {k}) : Shape{0}};
        apply_inferred_outputs(context, outputs);
        return ge::GRAPH_SUCCESS;
    } catch (const std::exception&) {
        return ge::GRAPH_FAILED;
    }
}

inline ge::graphStatus infer_dtype(gert::InferDataTypeContext* context) {
    const auto t = context->GetInputDataType(0);
    if (context->SetOutputDataType(0, t) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (context->GetComputeNodeInfo()->GetOutputsNum() > 1) {
        return context->SetOutputDataType(1, t);
    }
    return ge::GRAPH_SUCCESS;
}

inline void validate_tile_outputs(gert::TilingContext* context, const Plan& plan, uint32_t type) {
    for (size_t index = 0; index < plan.outputs.size(); ++index) {
        const auto* output = context->GetOutputShape(index);
        if (!output) {
            require(index == 1, "missing lstsq solution");
            continue;
        }
        const auto actual = get_shape(&output->GetStorageShape());
        const uint64_t actual_count = output_elements(actual);
        const uint64_t expected_count = output_elements(plan.outputs[index]);
        if (actual_count != expected_count) {
            throw std::invalid_argument(
                output_mismatch(context, plan, index, actual, actual_count, expected_count));
        }
        const auto* description = context->GetOutputDesc(index);
        require(description && map_dtype(description->GetDataType()) == type,
                "lstsq output dtype mismatch");
    }
}

inline std::vector<Shape> read_tile_inputs(gert::TilingContext* context, uint32_t type) {
    std::vector<Shape> inputs;
    for (uint32_t index = 0; index < 2; ++index) {
        const auto* shape = context->GetRequiredInputShape(index);
        const auto* desc = context->GetRequiredInputDesc(index);
        require(shape && desc && map_dtype(desc->GetDataType()) == type, "lstsq input mismatch");
        inputs.push_back(get_shape(&shape->GetStorageShape()));
    }
    return inputs;
}

template <typename Select>
ge::graphStatus tile(gert::TilingContext* context, Select select) {
    try {
        const auto* first = context->GetRequiredInputDesc(0);
        require(first != nullptr, "missing lstsq input descriptor");
        const uint32_t type = map_dtype(first->GetDataType());
        const auto inputs = read_tile_inputs(context, type);
        const auto* primary = context->GetOutputShape(0);
        require(primary != nullptr, "missing lstsq solution");
        auto attributes = parse_attrs(context->GetAttrs());
        const auto* residual = context->GetOutputShape(1);
        if (inputs[0].size() >= 2 && inputs[0][inputs[0].size() - 2] == inputs[0].back() &&
            residual && output_elements(get_shape(&residual->GetStorageShape())) != 0) {
            attributes.flags |= LSTSQ_SQUARE_RESIDUAL;
        }
        auto plan = make(inputs, type, attributes);
        const auto primary_shape = get_shape(&primary->GetStorageShape());
        if (output_elements(primary_shape) != output_elements(plan.outputs[0])) {
            plan = make(inputs, type, attributes, true);
        }
        validate_tile_outputs(context, plan, type);
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const uint64_t available = platform.GetCoreNumAiv();
        require(available > 0, "lstsq requires vector cores");
        const auto& cfg = plan.tiling.config;
        const bool p8_panel = (cfg.m == cfg.n && cfg.k <= 8 && (cfg.n & 7) == 0 &&
                               ((cfg.n >= 184 && cfg.n <= 208) || (cfg.n >= 232 && cfg.n <= 256)));
        const uint64_t desired_blocks = (p8_panel && multiply(cfg.batch, 8) <= available)
            ? multiply(cfg.batch, 8) : cfg.batch;
        const uint64_t blocks = desired_blocks < available ? desired_blocks : available;
        const uint64_t user_ws = plan.per_core_bytes == 0 ? 0ULL : add(16384ULL, multiply(blocks, plan.per_core_bytes));
        const uint64_t workspace_bytes = add(platform.GetLibApiWorkSpaceSize(), user_ws);
        add(plan.tensor_bytes, workspace_bytes);
        auto* tiling = context->GetTilingData<TemplateTilingData>();
        auto* workspace = context->GetWorkspaceSizes(1);
        require(tiling && workspace, "missing lstsq tiling/workspace storage");
        *tiling = plan.tiling;
        workspace[0] = workspace_bytes;
        select(static_cast<uint32_t>(first->GetDataType()));
        context->SetBlockDim(static_cast<uint32_t>(blocks));
        return ge::GRAPH_SUCCESS;
    } catch (const std::exception&) {
        return ge::GRAPH_FAILED;
    }
}
}  // namespace elec::lstsq_host
