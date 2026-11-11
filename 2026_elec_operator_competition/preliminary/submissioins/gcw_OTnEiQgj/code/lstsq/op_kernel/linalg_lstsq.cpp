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
#include "linalg_lstsq_tiling.h"
#include "tiling_key_linalg_lstsq.h"
#include "lstsq_common.h"
#include "lstsq_small.h"
#include "lstsq_large_p8.h"
#include "lstsq_large_lu.h"
#include "lstsq_qr.h"

namespace {
ELEC_FN void PublishOutput(GM_ADDR source, GM_ADDR target, uint64_t bytes,
                           AscendC::LocalTensor<uint16_t> buffer) {
    AscendC::GlobalTensor<uint16_t> staged, output;
    staged.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(source));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(target));
    for (uint64_t offset = 0; offset < bytes; offset += 4096) {
        const uint32_t count = static_cast<uint32_t>(bytes - offset < 4096 ? bytes - offset : 4096);
        const AscendC::DataCopyExtParams copy{1, count, 0, 0, 0};
        const AscendC::DataCopyPadExtParams<uint16_t> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(buffer, staged[offset >> 1], copy, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(output[offset >> 1], buffer, copy);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

ELEC_FN void solve_qr_batches(
    const LinalgLstsqTilingData& tiling_data, GM_ADDR A, GM_ADDR B,
    GM_ADDR solution, GM_ADDR residuals, GM_ADDR workspace) {
    GM_ADDR inputs[2] = {A, B};
    GM_ADDR outputs[2] = {solution, residuals};
    elec::Tensor input_tensors[2], output_tensors[2];
    elec::bind_template_tensors(tiling_data, inputs, outputs, input_tensors, output_tensors);
    elec::Config config = tiling_data.config;
    const uint64_t total_batch = config.batch, element_bytes = tiling_data.output_types[0] == elec::F32 ? 4 : 2;
    const bool residual_requested = config.m > config.n || (config.flags & elec::LSTSQ_SQUARE_RESIDUAL);
    const uint64_t scratch_bytes = (tiling_data.scratch_bytes + 63u) & ~uint64_t(63u);
    const uint64_t solution_bytes = config.n * config.k * element_bytes;
    const uint64_t staged_solution_bytes = (solution_bytes + 63u) & ~uint64_t(63u);
    const uint64_t residual_bytes = residual_requested ? config.k * element_bytes : 0;
    const uint64_t per_core_bytes = scratch_bytes + staged_solution_bytes + ((residual_bytes + 63u) & ~uint64_t(63u));
    GM_ADDR user_workspace = AscendC::GetUserWorkspace(workspace) + AscendC::GetBlockIdx() * per_core_bytes;
    GM_ADDR staged_solution = user_workspace + scratch_bytes;
    GM_ADDR staged_residual = staged_solution + staged_solution_bytes;
    elec::Tensor scratch(user_workspace);
    config.batch = 1;
    for (uint64_t first_batch = AscendC::GetBlockIdx(); first_batch < total_batch; first_batch += AscendC::GetBlockNum()) {
        elec::Tensor local_inputs[2] = {
            input_tensors[0].slice(first_batch * config.m * config.n),
            input_tensors[1].slice(first_batch * config.rhs_batch_stride)};
        elec::Tensor local_outputs[2] = {
            elec::Tensor(staged_solution, tiling_data.output_types[0]),
            elec::Tensor(residual_requested ? staged_residual : nullptr, tiling_data.output_types[1])};
        { elec::Math math; elec::check_template_status(elec::compute_lstsq(config, local_inputs, local_outputs, scratch, math)); }
        local_outputs[0].flush();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> copy_buffer;
        pipe.InitBuffer(copy_buffer, 4096);
        auto buffer = copy_buffer.Get<uint16_t>();
        PublishOutput(staged_solution, solution + first_batch * solution_bytes, solution_bytes, buffer);
        if (residual_requested && residuals != nullptr) {
            PublishOutput(staged_residual, residuals + first_batch * residual_bytes, residual_bytes, buffer);
        }
    }
}
template <typename DT_A>
ELEC_FN void dispatch_large_square(
    const LinalgLstsqTilingData& tiling_data, GM_ADDR A, GM_ADDR B,
    GM_ADDR solution, GM_ADDR residuals, GM_ADDR workspace, GM_ADDR tiling) {
    const uint64_t per_core = (tiling_data.scratch_bytes + 63u) & ~uint64_t(63u);
    GM_ADDR ws_base = AscendC::GetUserWorkspace(workspace);
    const uint64_t n = tiling_data.config.n;
    const bool p8_panel = ((n & 7u) == 0u && ((n >= 184u && n <= 208u) || (n >= 232u && n <= 256u)));
    if (p8_panel && AscendC::GetBlockNum() == tiling_data.config.batch * 8u) {
        const __gm__ LinalgLstsqTilingData* td = reinterpret_cast<const __gm__ LinalgLstsqTilingData*>(tiling);
        elec::check_template_status(
            elec::lstsq_fast::solve_square_large_p8<DT_A>(&td->config, A, B, solution, residuals, ws_base, per_core));
    } else {
        elec::check_template_status(
            elec::lstsq_fast::solve_square_large<DT_A>(
                tiling_data.config, A, B, solution, residuals, ws_base + 16384u + AscendC::GetBlockIdx() * per_core));
    }
}
}  // namespace

template <typename DT_A, uint32_t USE_LU>
__global__ __aicore__ void linalg_lstsq(GM_ADDR A, GM_ADDR B, GM_ADDR solution, GM_ADDR residuals, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(512, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(539, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(1792, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(2048, KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(LinalgLstsqTilingData);
    const __gm__ LinalgLstsqTilingData* td = reinterpret_cast<const __gm__ LinalgLstsqTilingData*>(tiling);
    if constexpr ((USE_LU >= 4 && USE_LU <= 7) || (USE_LU >= 10 && USE_LU <= 13)) {
        constexpr bool zero_r = (USE_LU >= 10);
        constexpr uint32_t regime = zero_r ? (USE_LU - 9u) : (USE_LU - 3u);
        elec::lstsq_fast::solve_square_small<DT_A, regime, zero_r>(
            &td->config, A, B, solution, zero_r ? residuals : nullptr);
        return;
    }
    if constexpr (USE_LU == 8 || USE_LU == 9) {
        GM_ADDR ws_base = AscendC::GetUserWorkspace(workspace);
        const uint64_t per_core = (td->scratch_bytes + 63u) & ~uint64_t(63u);
        elec::lstsq_fast::solve_square_large_p8<DT_A, USE_LU == 9 ? 2u : 1u>(
            &td->config, A, B, solution, residuals, ws_base, per_core);
        return;
    }
    if constexpr (USE_LU == 1) {
        elec::check_template_status(elec::lstsq_fast::solve_square_small<DT_A>(&td->config, A, B, solution, residuals));
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(LinalgLstsqTilingData, tiling_data, tiling);
    if constexpr (USE_LU == 2) {
        dispatch_large_square<DT_A>(tiling_data, A, B, solution, residuals, workspace, tiling);
        return;
    }
    solve_qr_batches(tiling_data, A, B, solution, residuals, workspace);
}
