/*
 * Copyright (c) 2026 dihidao.
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
// Private device implementation; included by ld_device.h.
#if (defined(__DAV_C220_CUBE__))
__aicore__ inline void LuN256BatchCube(GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
    uint32_t groups = *reinterpret_cast<const __gm__ uint32_t *>(tiling + 24U);
    uint32_t cores = *reinterpret_cast<const __gm__ uint32_t *>(tiling + 28U);
    uint32_t group = GetBlockIdx() / cores, rank = GetBlockIdx() % cores;
    if (group >= groups)
        return;
    GM_ADDR ws = GetUserWorkspace(workspace) + static_cast<uint64_t>(group) * ::kN256GroupBytes;
    GM_ADDR neg = ws + ::kN256WorkspaceBytes;
    GM_ADDR u = neg + 32768U;
    GM_ADDR flags = u + 65536U;
    if (td.batchCount == 1U && cores == 4U) {
        LuCube op;
        op.RunTail208(ws, neg, u, flags, tiling + 64U, td.n, td.stride, 1U);
        LuPublish(flags + (13U + GetBlockIdx()) * 64U, 0x6ADBD431U);
        return;
    }
    for (uint32_t mat = group; mat < td.batchCount; mat += groups) {
        uint32_t epoch = mat / groups + 1U;
        {
            LuCube op;
            op.RunN256Group(ws, neg, u, flags, tiling + 64U, td.n, epoch, rank, cores);
        }
        LuPublish(flags + (13U + rank) * 64U, 0x6ADBD400U + epoch);
    }
}
#endif

#if (defined(__DAV_C220_CUBE__))
__aicore__ inline void LuWideBatchCube(GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
    uint32_t groups = *reinterpret_cast<const __gm__ uint32_t *>(tiling + 24U);
    uint32_t cores = *reinterpret_cast<const __gm__ uint32_t *>(tiling + 28U);
    uint32_t group = GetBlockIdx() / cores, rank = GetBlockIdx() % cores;
    if (group >= groups)
        return;
    GM_ADDR ws = GetUserWorkspace(workspace) + static_cast<uint64_t>(group) * LuWideGroupBytes(td.n);
    GM_ADDR neg = ws + LuWideWorkspaceBytes(td.n);
    GM_ADDR u = neg + 65536U;
    GM_ADDR flags = u + 65536U;
    for (uint32_t mat = group; mat < td.batchCount; mat += groups) {
        uint32_t epoch = mat / groups + 1U;
        {
            LuCube op;
            op.RunWideGroup(ws, neg, u, flags, tiling + 64U, td.n, epoch, rank, cores);
        }
        LuPublish(flags + (13U + rank) * 64U, 0x6ADBD400U + epoch);
    }
}
#endif

#if (defined(__DAV_C220_VEC__))
template <bool WIDE, class Tiling>
__aicore__ inline void LuBatchVectorLoop46(GM_ADDR A, uint32_t cores, GM_ADDR det, GM_ADDR flags, uint32_t group,
                                           uint32_t groups, GM_ADDR neg, uint32_t rank, GM_ADDR u, uint32_t vectors,
                                           GM_ADDR wave, GM_ADDR ws, Tiling &td) {

    if (rank == 0U)
        LuClear(flags, 27U);
    LinalgDetTilingData one = td;
    one.batchCount = 1U;
    one.n = td.n;
    one.stride = WIDE ? 512U : 256U;
    one.matPerCore = 1U;
    one.isLarge = 2U;
    one.groupSize = vectors;
    for (uint32_t mat = group; mat < td.batchCount; mat += groups) {
        uint32_t epoch = mat / groups + 1U;
        KernelLinalgDet<float, 2, true, false, false, false, false, 0, true, true, true, true, true, WIDE> op;
        op.SetCube(neg, u, flags, epoch);
        op.InitUserWorkspace(A + static_cast<uint64_t>(mat) * td.n * td.n * sizeof(float),
                             det + static_cast<uint64_t>(mat) * sizeof(float), ws, one, rank);
        op.Process();
        PipeBarrier<PIPE_ALL>();
        LuPublish(wave + rank * 64U, epoch);
        if (rank == 0U) {
            for (uint32_t peer = 0; peer < vectors; ++peer)
                LuPoll(wave + peer * 64U, epoch);
            for (uint32_t peer = 0; peer < cores; ++peer)
                LuPoll(flags + (13U + peer) * 64U, 0x6ADBD400U + epoch);
            LuClear(flags);
            LuPublish(wave + 8U * 64U, epoch);
        } else
            LuPoll(wave + 8U * 64U, epoch);
    }
}
#endif

#if (defined(__DAV_C220_VEC__))
__aicore__ inline void LuN256BatchVector(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
    uint32_t groups = *reinterpret_cast<const __gm__ uint32_t *>(tiling + 24U);
    uint32_t cores = *reinterpret_cast<const __gm__ uint32_t *>(tiling + 28U);
    uint32_t vectors = 2U * cores, group = GetBlockIdx() / vectors, rank = GetBlockIdx() % vectors;
    if (group >= groups)
        return;
    GM_ADDR ws = GetUserWorkspace(workspace) + static_cast<uint64_t>(group) * ::kN256GroupBytes;
    GM_ADDR neg = ws + ::kN256WorkspaceBytes;
    GM_ADDR u = neg + 32768U;
    GM_ADDR flags = u + 65536U;
    GM_ADDR wave = flags + 18U * 64U;
    if (td.batchCount == 1U && cores == 4U) {
        KernelLinalgDet<float, 2, true, false, false, false, false, 0, true, true, true, true, true> op;
        op.SetCube(neg, u, flags, 1U);
        op.InitUserWorkspace(A, det, ws, td);
        op.Process();
        if (GetBlockIdx() == 0U) {
            for (uint32_t peer = 0; peer < 4U; ++peer)
                LuPoll(flags + (13U + peer) * 64U, 0x6ADBD431U);
            LuClear(flags, 27U);
        }
        return;
    }
    LuBatchVectorLoop46<false>(A, cores, det, flags, group, groups, neg, rank, u, vectors, wave, ws, td);
}
#endif

#if (defined(__DAV_C220_VEC__))
__aicore__ inline void LuWideBatchVector(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
    uint32_t groups = *reinterpret_cast<const __gm__ uint32_t *>(tiling + 24U);
    uint32_t cores = *reinterpret_cast<const __gm__ uint32_t *>(tiling + 28U);
    uint32_t vectors = 2U * cores, group = GetBlockIdx() / vectors, rank = GetBlockIdx() % vectors;
    if (group >= groups)
        return;
    GM_ADDR ws = GetUserWorkspace(workspace) + static_cast<uint64_t>(group) * LuWideGroupBytes(td.n);
    GM_ADDR neg = ws + LuWideWorkspaceBytes(td.n);
    GM_ADDR u = neg + 65536U;
    GM_ADDR flags = u + 65536U;
    GM_ADDR wave = flags + 18U * 64U;
    LuBatchVectorLoop46<true>(A, cores, det, flags, group, groups, neg, rank, u, vectors, wave, ws, td);
}
#endif
