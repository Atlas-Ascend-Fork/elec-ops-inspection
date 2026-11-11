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
#include "kernel_operator.h"
#include "spline_conv_tiling.h"
#include "tiling_key_spline_conv.h"
static constexpr uint32_t MAX_CIN = 128U;
static constexpr uint32_t MAX_COUT = 128U;
static constexpr uint32_t MAX_KERNEL_COUNT = 64U;
static constexpr uint32_t BASIS_TABLE_SIZE = 12U;
static constexpr uint32_t ZERO_CHUNK = 256U;
static constexpr uint32_t WEIGHT_TILE_ELEMS = 2048U;
__aicore__ inline float SplineToFloat( float value) {
    return value;
}
__aicore__ inline float SplineToFloat( bfloat16_t value) {
    return AscendC::ToFloat( value);
}
template <typename T> __aicore__ inline T SplineFromFloat( float value);
template <> __aicore__ inline float SplineFromFloat<float>( float value) {
    return value;
}
template <> __aicore__ inline bfloat16_t SplineFromFloat<bfloat16_t>( float value) {
    return AscendC::ToBfloat16( value);
}
template <class DT_X> class KernelSplineConv {
    public: __aicore__ inline KernelSplineConv() {
    }
    __aicore__ inline void Init( GM_ADDR x, GM_ADDR edge_index, GM_ADDR edge_attr, GM_ADDR weight, GM_ADDR out, GM_ADDR workspace, const SplineConvTilingData &tiling) {
        InitTilingParams(tiling);
        InitDerivedParams();
        InitGlobalBuffers(x, edge_index, edge_attr, weight, out, workspace);
        InitPipeBuffers();
    }
    __aicore__ inline void Process() {
        const uint32_t coreIdx = static_cast<uint32_t>( AscendC::GetBlockIdx());
        uint32_t coreNum = static_cast<uint32_t>( AscendC::GetBlockNum());
        if (coreNum == 0U) {
            coreNum = 1U;
        }
        LoadWeightCache();
        if (localAggregate_) {
            ProcessLocalAggregate();
            return;
        }
        ZeroWorkspace( coreIdx, coreNum);
        if (coreNum > 1U) {
            AscendC::SyncAll<true>();
        }
        AccumulateEdgesAtomic(coreIdx, coreNum);
        // Finish output/degree DMA before scalar degree reads and finalization.
        auto aggregateDone = pipe_.FetchEventID(AscendC::HardEvent::MTE3_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(aggregateDone);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(aggregateDone);
        if (coreNum > 1U) {
            AscendC::SyncAll<true>();
        }
        AscendC::DataCacheCleanAndInvalid< float, AscendC::CacheLine::ENTIRE_DATA_CACHE>( accGm_);
        AscendC::DataCacheCleanAndInvalid< float, AscendC::CacheLine::ENTIRE_DATA_CACHE>( degGm_);
        FinalizeNodes( coreIdx, coreNum);
    }
    private:
    // Copy the tiling record into the kernel state.
    __aicore__ inline void InitTilingParams( const SplineConvTilingData &tiling) {
        n_ = tiling.n;
        e_ = tiling.e;
        cIn_ = tiling.cIn;
        cOut_ = tiling.cOut;
        dim_ = tiling.dim;
        degree_ = tiling.degree;
        basisSlots_ = tiling.basisSlots;
        aggrMean_ = tiling.aggrMean;
        kernel_[0] = tiling.kernel0;
        kernel_[1] = tiling.kernel1;
        kernel_[2] = tiling.kernel2;
        open_[0] = tiling.open0;
        open_[1] = tiling.open1;
        open_[2] = tiling.open2;
        accStrideFloats_ = tiling.accStrideFloats;
    }
    // Derive every stride / tile size / buffering decision from the tiling record.
    __aicore__ inline void InitDerivedParams() {
        basisPerDim_ = degree_ + 1U;
        kernelOffset_[0] = 1U;
        kernelOffset_[1] = kernel_[0];
        kernelOffset_[2] = kernel_[0] * kernel_[1];
        kernelCount_ = kernel_[0] * kernel_[1] * kernel_[2];
        weightMatrixStride_ = cIn_ * cOut_;
        const uint32_t alignment = 32U / sizeof(DT_X);
        weightRowStride_ = (cOut_ + alignment - 1U) / alignment * alignment;
        ciTile_ = WEIGHT_TILE_ELEMS / weightRowStride_;
        if (ciTile_ == 0U) {
            ciTile_ = 1U;
        }
        if (ciTile_ > cIn_) {
            ciTile_ = cIn_;
        }
        weightTileCount_ = ( cIn_ + ciTile_ - 1U ) / ciTile_;
        mergeWeightIndex_ = 0U;
        for (uint32_t d = 0U;
        d < dim_;
        ++d) {
            if (open_[d] == 0U && kernel_[d] < basisPerDim_) {
                mergeWeightIndex_ = 1U;
            }
        }
        localAccStride_ = (cOut_ + 7U) & ~7U;
        localAggregate_ = AscendC::GetBlockNum() == 1U &&
                          static_cast<uint64_t>(n_) * localAccStride_ <= 4096U;
        cachedWeightElements_ = kernelCount_ * cIn_ * weightRowStride_;
        const uint32_t cacheLimit = localAggregate_ ? 32768U : 36864U;
        cacheWeights_ = cachedWeightElements_ <= cacheLimit;
    }
    __aicore__ inline void InitGlobalBuffers( GM_ADDR x, GM_ADDR edge_index, GM_ADDR edge_attr, GM_ADDR weight, GM_ADDR out, GM_ADDR workspace) {
        xGm_.SetGlobalBuffer( reinterpret_cast< __gm__ DT_X *>( x), static_cast<uint64_t>(n_) * cIn_);
        edgeIndexGm_.SetGlobalBuffer( reinterpret_cast< __gm__ int64_t *>( edge_index), 2ULL * e_);
        edgeAttrGm_.SetGlobalBuffer( reinterpret_cast< __gm__ DT_X *>( edge_attr), static_cast<uint64_t>(e_) * dim_);
        weightGm_.SetGlobalBuffer( reinterpret_cast< __gm__ DT_X *>( weight), kernelCount_ * weightMatrixStride_);
        outGm_.SetGlobalBuffer( reinterpret_cast< __gm__ DT_X *>( out), static_cast<uint64_t>(n_) * cOut_);
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace( workspace);
        __gm__ float *workspaceFloat = reinterpret_cast< __gm__ float *>( userWorkspace);
        accGm_.SetGlobalBuffer( workspaceFloat, static_cast<uint64_t>(n_) * cOut_);
        degGm_.SetGlobalBuffer( workspaceFloat + accStrideFloats_, n_);
    }
    // The InitBuffer order defines the UB layout, keep it unchanged.
    __aicore__ inline void InitPipeBuffers() {
        pipe_.InitBuffer( messageQueue_, 1, MAX_COUT * sizeof(float));
        pipe_.InitBuffer( weightTileQueue_, 2, WEIGHT_TILE_ELEMS * sizeof(float));
        pipe_.InitBuffer( finalInputQueue_, 1, MAX_COUT * sizeof(float));
        pipe_.InitBuffer( xCacheBuf_, MAX_CIN * sizeof(float));
        pipe_.InitBuffer(effectiveWeightBuf_, WEIGHT_TILE_ELEMS * sizeof(float));
        pipe_.InitBuffer( outputBuf_, MAX_COUT * sizeof(float));
        pipe_.InitBuffer( zeroBuf_, ZERO_CHUNK * sizeof(float));
        pipe_.InitBuffer( oneBuf_, 32U);
        if (localAggregate_) {
            InitLocalAggregateBuffers();
        }
        InitWeightCacheBuffer();
    }
    __aicore__ inline void InitLocalAggregateBuffers() {
        pipe_.InitBuffer(localAccBuf_, n_ * localAccStride_ * sizeof(float));
        pipe_.InitBuffer(localDegreeBuf_, ((n_ + 7U) & ~7U) * sizeof(uint32_t));
        pipe_.InitBuffer(localMessageBuf_, MAX_COUT * sizeof(float));
    }
    __aicore__ inline void InitWeightCacheBuffer() {
        if (cacheWeights_) {
            pipe_.InitBuffer(weightCacheBuf_, cachedWeightElements_ * sizeof(float));
        } else {
            pipe_.InitBuffer(weightFloatBuf_, WEIGHT_TILE_ELEMS * sizeof(float));
        }
    }
    __aicore__ inline float ScalePseudo( float pseudo, uint32_t scale) const {
        if (scale == 1U) {
            return pseudo;
        }
        if (scale == 2U) {
            return pseudo * 2.0f;
        }
        if (scale == 3U) {
            return pseudo * 3.0f;
        }
        return pseudo * static_cast<float>(static_cast<int32_t>(scale));
    }
    __aicore__ inline float SmallIntToFloat( int32_t value) const {
        if (value <= 0) {
            return 0.0f;
        }
        if (value == 1) {
            return 1.0f;
        }
        if (value == 2) {
            return 2.0f;
        }
        if (value == 3) {
            return 3.0f;
        }
        return static_cast<float>(value);
    }
    __aicore__ inline float Basis1D( float v, uint32_t kMod) const {
        if (degree_ == 0U) {
            return 1.0f;
        }
        if (degree_ == 1U) {
            if (kMod == 0U) {
                return 1.0f - v;
            }
            return v;
        }
        const float v2 = v * v;
        if (degree_ == 2U) {
            if (kMod == 0U) {
                return 0.5f * v2 - v + 0.5f;
            }
            if (kMod == 1U) {
                return -v2 + v + 0.5f;
            }
            return 0.5f * v2;
        }
        const float v3 = v2 * v;
        if (kMod == 0U) {
            const float t = 1.0f - v;
            return t * t * t * (1.0f / 6.0f);
        }
        if (kMod == 1U) {
            return ( 3.0f * v3 - 6.0f * v2 + 4.0f ) * (1.0f / 6.0f);
        }
        if (kMod == 2U) {
            return ( -3.0f * v3 + 3.0f * v2 + 3.0f * v + 1.0f ) * (1.0f / 6.0f);
        }
        return v3 * (1.0f / 6.0f);
    }
    __aicore__ inline void PrepareEdgeTables( uint64_t edge, float basisTable[BASIS_TABLE_SIZE], uint32_t indexTable[BASIS_TABLE_SIZE]) {
        for (uint32_t d = 0U;
        d < dim_;
        ++d) {
            const float pseudo = SplineToFloat( edgeAttrGm_.GetValue( edge * dim_ + d));
            const uint32_t scale = kernel_[d] - degree_ * open_[d];
            const float vd = ScalePseudo( pseudo, scale);
            const int32_t floorSigned = AscendC::ScalarCast< float, int32_t, AscendC::RoundMode::CAST_FLOOR>( vd);
            const uint32_t floorValue = static_cast<uint32_t>( floorSigned);
            const float frac = vd - SmallIntToFloat( floorSigned);
            const uint32_t tableBase = d * 4U;
            for (uint32_t k = 0U;
            k < basisPerDim_;
            ++k) {
                basisTable[ tableBase + k ] = Basis1D( frac, k);
                indexTable[ tableBase + k ] = ( ( floorValue + k ) % kernel_[d] ) * kernelOffset_[d];
            }
        }
    }
    __aicore__ inline void LoadXCache( uint32_t src, AscendC::LocalTensor<float> &xCache) {
        const uint64_t xBase = static_cast<uint64_t>(src) * cIn_;
        for (uint32_t ci = 0U;
        ci < cIn_;
        ++ci) {
            xCache.SetValue( ci, SplineToFloat( xGm_.GetValue( xBase + ci)));
        }
    }
    __aicore__ inline void ZeroWorkspace( uint32_t coreIdx, uint32_t coreNum) {
        AscendC::LocalTensor<float> zero = zeroBuf_.Get<float>();
        AscendC::Duplicate( zero, 0.0f, ZERO_CHUNK);
        auto zeroEvent = pipe_.FetchEventID(AscendC::HardEvent::V_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(zeroEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(zeroEvent);
        ZeroRange( accGm_, static_cast<uint64_t>(n_) * cOut_, coreIdx, coreNum, zero);
        ZeroRange( degGm_, n_, coreIdx, coreNum, zero);
    }
    // Map [0, total) onto this core's slice. G.EXP.22: the shared divisor is
    // normalized once here so that every range split in the kernel reuses it.
    __aicore__ inline void SplitCoreRange( uint64_t total, uint32_t coreIdx, uint32_t coreNum,
        uint64_t &start, uint64_t &end) const {
        if (coreNum == 0U) {
            coreNum = 1U;
        }
        start = (total * coreIdx) / coreNum;
        end = (total * (coreIdx + 1U)) / coreNum;
    }
    __aicore__ inline void ZeroRange( AscendC::GlobalTensor<float> &dst, uint64_t total, uint32_t coreIdx, uint32_t coreNum, const AscendC::LocalTensor<float> &zero) {
        uint64_t start = 0U;
        uint64_t end = 0U;
        SplitCoreRange( total, coreIdx, coreNum, start, end);
        AscendC::DataCopyExtParams cp;
        cp.blockCount = 1;
        cp.srcStride = 0;
        cp.dstStride = 0;
        cp.rsv = 0;
        while (start < end) {
            uint32_t count = end - start;
            if (count > ZERO_CHUNK) {
                count = ZERO_CHUNK;
            }
            cp.blockLen = count * sizeof(float);
            AscendC::DataCopyPad( dst[start], zero, cp);
            start += count;
        }
    }
    __aicore__ inline void EnqueueWeightTile( uint32_t weightBase, uint32_t elementCount, float *) {
        AscendC::LocalTensor<float> tile = weightTileQueue_.AllocTensor< float>();
        AscendC::DataCopyExtParams cp;
        cp.blockCount = elementCount / cOut_;
        cp.blockLen = cOut_ * sizeof(float);
        cp.srcStride = 0;
        const uint32_t rowBytes = cOut_ * sizeof(float);
        const uint32_t roundedRowBytes = (rowBytes + 31U) & ~31U;
        cp.dstStride = (weightRowStride_ * sizeof(float) - roundedRowBytes) / 32U;
        cp.rsv = 0;
        AscendC::DataCopyPadExtParams<float> pad = {
            true, 0, static_cast<uint8_t>((8U - cOut_ % 8U) % 8U), 0.0f };
        AscendC::DataCopyPad( tile, weightGm_[ weightBase ], cp, pad);
        weightTileQueue_.EnQue< float>( tile);
    }
    __aicore__ inline void EnqueueWeightTile( uint32_t weightBase, uint32_t elementCount, bfloat16_t *) {
        AscendC::LocalTensor<bfloat16_t> tile = weightTileQueue_.AllocTensor< bfloat16_t>();
        AscendC::DataCopyExtParams cp;
        cp.blockCount = elementCount / cOut_;
        cp.blockLen = cOut_ * sizeof(bfloat16_t);
        cp.srcStride = 0;
        const uint32_t rowBytes = cOut_ * sizeof(bfloat16_t);
        const uint32_t roundedRowBytes = (rowBytes + 31U) & ~31U;
        cp.dstStride = (weightRowStride_ * sizeof(bfloat16_t) - roundedRowBytes) / 32U;
        cp.rsv = 0;
        AscendC::DataCopyPadExtParams< bfloat16_t> pad = {
            true, 0, static_cast<uint8_t>(weightRowStride_ - cOut_), AscendC::ToBfloat16( 0.0f) };
        AscendC::DataCopyPad( tile, weightGm_[ weightBase ], cp, pad);
        weightTileQueue_.EnQue< bfloat16_t>( tile);
    }
    __aicore__ inline void ComputeWeightTile(
    AscendC::LocalTensor<float> &tile, uint32_t ciStart, uint32_t ciCount,
    float basis, AscendC::LocalTensor<float> &xCache,
    AscendC::LocalTensor<float> &message, float *) {
        for (uint32_t ci = 0;
        ci < ciCount;
        ++ci) {
            float scale = xCache.GetValue(ciStart + ci) * basis;
            if (scale != 0.0f) {
                AscendC::Axpy(message, tile[ci * weightRowStride_], scale, cOut_);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void ComputeWeightTile(
    AscendC::LocalTensor<bfloat16_t> &tile, uint32_t ciStart, uint32_t ciCount,
    float basis, AscendC::LocalTensor<float> &xCache,
    AscendC::LocalTensor<float> &message, bfloat16_t *) {
        auto converted = weightFloatBuf_.Get<float>();
        AscendC::Cast(converted, tile, AscendC::RoundMode::CAST_NONE,
        ciCount * weightRowStride_);
        AscendC::PipeBarrier<PIPE_V>();
        ComputeWeightTile(converted, ciStart, ciCount, basis, xCache, message,
        static_cast<float *>(nullptr));
    }
    __aicore__ inline void BuildMerged2D( const float basisTable[BASIS_TABLE_SIZE], const uint32_t indexTable[BASIS_TABLE_SIZE], float mergedBasis[MAX_KERNEL_COUNT]) {
        for (uint32_t k1 = 0U;
        k1 < basisPerDim_;
        ++k1) {
            const float b1 = basisTable[ 4U + k1 ];
            const uint32_t i1 = indexTable[ 4U + k1 ];
            for (uint32_t k0 = 0U;
            k0 < basisPerDim_;
            ++k0) {
                const uint32_t wi = indexTable[k0] + i1;
                mergedBasis[wi] += basisTable[k0] * b1;
            }
        }
    }
    __aicore__ inline void BuildMerged3D( const float basisTable[BASIS_TABLE_SIZE], const uint32_t indexTable[BASIS_TABLE_SIZE], float mergedBasis[MAX_KERNEL_COUNT]) {
        for (uint32_t k2 = 0U;
        k2 < basisPerDim_;
        ++k2) {
            const float b2 = basisTable[ 8U + k2 ];
            const uint32_t i2 = indexTable[ 8U + k2 ];
            for (uint32_t k1 = 0U;
            k1 < basisPerDim_;
            ++k1) {
                const float b12 = basisTable[ 4U + k1 ] * b2;
                const uint32_t i12 = indexTable[ 4U + k1 ] + i2;
                for (uint32_t k0 = 0U;
                k0 < basisPerDim_;
                ++k0) {
                    const uint32_t wi = indexTable[k0] + i12;
                    mergedBasis[wi] += basisTable[k0] * b12;
                }
            }
        }
    }
    __aicore__ inline void ConvertWeightCache(
        AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<float> &src,
        uint32_t count, float *) {
        AscendC::Muls(dst, src, 1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ConvertWeightCache(
        AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<bfloat16_t> &src,
        uint32_t count, bfloat16_t *) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadWeightCache() {
        if (!cacheWeights_) return;
        // Stream through the existing queue: never allocate a full raw copy.
        const uint32_t totalRows = kernelCount_ * cIn_;
        const uint32_t rowsPerTile = WEIGHT_TILE_ELEMS / weightRowStride_;
        auto cached = weightCacheBuf_.Get<float>();
        uint32_t firstRows = totalRows < rowsPerTile ? totalRows : rowsPerTile;
        EnqueueWeightTile(0U, firstRows * cOut_, static_cast<DT_X *>(nullptr));
        for (uint32_t rowStart = 0; rowStart < totalRows; rowStart += rowsPerTile) {
            uint32_t rows = totalRows - rowStart;
            if (rows > rowsPerTile) rows = rowsPerTile;
            uint32_t nextStart = rowStart + rowsPerTile;
            if (nextStart < totalRows) {
                uint32_t nextRows = totalRows - nextStart;
                if (nextRows > rowsPerTile) nextRows = rowsPerTile;
                EnqueueWeightTile(nextStart * cOut_, nextRows * cOut_,
                                  static_cast<DT_X *>(nullptr));
            }
            auto raw = weightTileQueue_.DeQue<DT_X>();
            auto dest = cached[rowStart * weightRowStride_];
            ConvertWeightCache(dest, raw, rows * weightRowStride_,
                               static_cast<DT_X *>(nullptr));
            weightTileQueue_.FreeTensor(raw);
        }
    }
    // Interpolate an entire weight tile, then multiply by x once.
    // All polynomial values and products are evaluated in float32.
    __aicore__ inline void BlendTile(
        AscendC::LocalTensor<float> &tile, float basis,
        AscendC::LocalTensor<float> &effective, uint32_t count, float *) {
        AscendC::Axpy(effective, tile, basis, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void BlendTile(
        AscendC::LocalTensor<bfloat16_t> &tile, float basis,
        AscendC::LocalTensor<float> &effective, uint32_t count, bfloat16_t *) {
        auto converted = weightFloatBuf_.Get<float>();
        AscendC::Cast(converted, tile, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        BlendTile(converted, basis, effective, count, static_cast<float *>(nullptr));
    }
    __aicore__ inline void ComputeMessage(
        uint64_t edge, uint32_t src, AscendC::LocalTensor<float> &message) {
        auto xCache = xCacheBuf_.Get<float>();
        LoadXCache(src, xCache);
        float basisTable[BASIS_TABLE_SIZE];
        uint32_t indexTable[BASIS_TABLE_SIZE];
        PrepareEdgeTables(edge, basisTable, indexTable);
        float merged[MAX_KERNEL_COUNT];
        BuildMergedBasis(basisTable, indexTable, merged);
        uint32_t active[MAX_KERNEL_COUNT];
        const uint32_t activeCount = CollectActiveBasis(merged, active);
        AscendC::Duplicate(message, 0.0f, cOut_);
        AscendC::PipeBarrier<PIPE_V>();
        AccumulateTileMessages(xCache, active, activeCount, merged, message);
    }
    // Zero the merged basis table, then fill it for the current dimensionality.
    __aicore__ inline void BuildMergedBasis( const float basisTable[BASIS_TABLE_SIZE],
        const uint32_t indexTable[BASIS_TABLE_SIZE], float merged[MAX_KERNEL_COUNT]) {
        for (uint32_t wi = 0; wi < kernelCount_; ++wi) {
            merged[wi] = 0.0f;
        }
        if (dim_ == 1U) {
            for (uint32_t k = 0U; k < basisPerDim_; ++k) {
                merged[indexTable[k]] += basisTable[k];
            }
        }
        else if (dim_ == 2U) {
            BuildMerged2D(basisTable, indexTable, merged);
        }
        else {
            BuildMerged3D(basisTable, indexTable, merged);
        }
    }
    // Kernel slots with a non-zero merged basis, in increasing order.
    __aicore__ inline uint32_t CollectActiveBasis( const float merged[MAX_KERNEL_COUNT],
        uint32_t active[MAX_KERNEL_COUNT]) const {
        uint32_t activeCount = 0;
        for (uint32_t wi = 0; wi < kernelCount_; ++wi) {
            if (merged[wi] != 0.0f) {
                active[activeCount++] = wi;
            }
        }
        return activeCount;
    }
    // Blend one ci tile of every active kernel from the UB weight cache.
    __aicore__ inline void BlendCachedTiles( const uint32_t active[MAX_KERNEL_COUNT], uint32_t activeCount,
        const float merged[MAX_KERNEL_COUNT], uint32_t ciStart,
        AscendC::LocalTensor<float> &effective, uint32_t paddedCount) {
        auto cached = weightCacheBuf_.Get<float>();
        for (uint32_t slot = 0; slot < activeCount; ++slot) {
            const uint32_t offset = (active[slot] * cIn_ + ciStart) * weightRowStride_;
            auto tile = cached[offset];
            BlendTile(tile, merged[active[slot]], effective, paddedCount,
                      static_cast<float *>(nullptr));
        }
    }
    // Blend one ci tile of every active kernel, streaming them through the queue.
    __aicore__ inline void BlendStreamedTiles( const uint32_t active[MAX_KERNEL_COUNT], uint32_t activeCount,
        const float merged[MAX_KERNEL_COUNT], uint32_t messageBase, uint32_t denseCount,
        AscendC::LocalTensor<float> &effective, uint32_t paddedCount) {
        if (activeCount != 0U) {
            EnqueueWeightTile(active[0] * weightMatrixStride_ + messageBase,
                              denseCount, static_cast<DT_X *>(nullptr));
        }
        for (uint32_t slot = 0; slot < activeCount; ++slot) {
            if (slot + 1U < activeCount) {
                EnqueueWeightTile(active[slot + 1U] * weightMatrixStride_ + messageBase,
                                  denseCount, static_cast<DT_X *>(nullptr));
            }
            auto tile = weightTileQueue_.DeQue<DT_X>();
            BlendTile(tile, merged[active[slot]], effective, paddedCount,
                      static_cast<DT_X *>(nullptr));
            weightTileQueue_.FreeTensor(tile);
        }
    }
    __aicore__ inline void BlendTileRange( const uint32_t active[MAX_KERNEL_COUNT], uint32_t activeCount,
        const float merged[MAX_KERNEL_COUNT], uint32_t ciStart, uint32_t denseCount,
        AscendC::LocalTensor<float> &effective, uint32_t paddedCount) {
        if (cacheWeights_) {
            BlendCachedTiles(active, activeCount, merged, ciStart, effective, paddedCount);
        }
        else {
            BlendStreamedTiles(active, activeCount, merged, ciStart * cOut_, denseCount,
                               effective, paddedCount);
        }
    }
    // message = sum_k basis_k * (W_k * x), accumulated ci tile by ci tile.
    __aicore__ inline void AccumulateTileMessages( AscendC::LocalTensor<float> &xCache,
        const uint32_t active[MAX_KERNEL_COUNT], uint32_t activeCount,
        const float merged[MAX_KERNEL_COUNT], AscendC::LocalTensor<float> &message) {
        auto effective = effectiveWeightBuf_.Get<float>();
        for (uint32_t ciStart = 0; ciStart < cIn_; ciStart += ciTile_) {
            uint32_t ciCount = cIn_ - ciStart;
            if (ciCount > ciTile_) {
                ciCount = ciTile_;
            }
            const uint32_t paddedCount = ciCount * weightRowStride_;
            const uint32_t denseCount = ciCount * cOut_;
            AscendC::Duplicate(effective, 0.0f, paddedCount);
            AscendC::PipeBarrier<PIPE_V>();
            BlendTileRange(active, activeCount, merged, ciStart, denseCount, effective, paddedCount);
            ComputeWeightTile(effective, ciStart, ciCount, 1.0f, xCache, message,
                              static_cast<float *>(nullptr));
        }
    }
    // Single-core, small output: retain fp32 aggregation entirely in UB.
    __aicore__ inline void ProcessLocalAggregate() {
        auto acc = localAccBuf_.Get<float>();
        auto degree = localDegreeBuf_.Get<float>();
        AscendC::Duplicate(acc, 0.0f, n_ * localAccStride_);
        AscendC::Duplicate(degree, 0.0f, (n_ + 7U) & ~7U);
        AscendC::PipeBarrier<PIPE_V>();
        auto ready = pipe_.FetchEventID(AscendC::HardEvent::V_S);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(ready);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(ready);
        auto message = localMessageBuf_.Get<float>();
        for (uint32_t edge = 0; edge < e_; ++edge) {
            uint32_t src = static_cast<uint32_t>(edgeIndexGm_.GetValue(edge));
            uint32_t dst = static_cast<uint32_t>(edgeIndexGm_.GetValue(static_cast<uint64_t>(e_) + edge));
            if (src >= n_ || dst >= n_) continue;
            if (aggrMean_ != 0U) degree.SetValue(dst, degree.GetValue(dst) + 1.0f);
            ComputeMessage(edge, src, message);
            auto row = acc[dst * localAccStride_];
            AscendC::Add(row, row, message, cOut_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t node = 0; node < n_; ++node) {
            float scale = 1.0f;
            if (aggrMean_ != 0U) {
                float count = degree.GetValue(node);
                if (count > 0.0f) scale = 1.0f / count;
            }
            auto output = messageQueue_.AllocTensor<float>();
            AscendC::Muls(output, acc[node * localAccStride_], scale, cOut_);
            messageQueue_.EnQue<float>(output);
            output = messageQueue_.DeQue<float>();
            StoreFinalRow(static_cast<uint64_t>(node) * cOut_, output, static_cast<DT_X *>(nullptr));
            messageQueue_.FreeTensor(output);
        }
    }
    __aicore__ inline void AccumulateEdgesAtomic( uint32_t coreIdx, uint32_t coreNum) {
        AscendC::LocalTensor<float> one = oneBuf_.Get<float>();
        AscendC::Duplicate(one, 1.0f, 8U);
        auto oneEvent = pipe_.FetchEventID(AscendC::HardEvent::V_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(oneEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(oneEvent);
        AscendC::DataCopyExtParams messageCopy;
        messageCopy.blockCount = 1;
        messageCopy.blockLen = cOut_ * sizeof(float);
        messageCopy.srcStride = 0;
        messageCopy.dstStride = 0;
        messageCopy.rsv = 0;
        AscendC::DataCopyExtParams degreeCopy;
        degreeCopy.blockCount = 1;
        degreeCopy.blockLen = sizeof(float);
        degreeCopy.srcStride = 0;
        degreeCopy.dstStride = 0;
        degreeCopy.rsv = 0;
        uint64_t startEdge = 0U;
        uint64_t endEdge = 0U;
        SplitCoreRange( e_, coreIdx, coreNum, startEdge, endEdge);
        AscendC::SetAtomicAdd< float>();
        for (uint64_t edge = startEdge;
        edge < endEdge;
        ++edge) {
            const uint32_t src = static_cast<uint32_t>( edgeIndexGm_.GetValue( edge));
            const uint32_t dst = static_cast<uint32_t>( edgeIndexGm_.GetValue( e_ + edge));
            if (src >= n_ || dst >= n_) {
                continue;
            }
            if (aggrMean_ != 0U) {
                AscendC::DataCopyPad( degGm_[ dst ], one, degreeCopy);
            }
            AscendC::LocalTensor<float> message = messageQueue_.AllocTensor< float>();
            ComputeMessage( edge, src, message);
            messageQueue_.EnQue< float>( message);
            message = messageQueue_.DeQue< float>();
            AscendC::DataCopyPad( accGm_[ static_cast<uint64_t>(dst) * cOut_ ], message, messageCopy);
            messageQueue_.FreeTensor( message);
        }
        AscendC::DisableDmaAtomic();
    }
    __aicore__ inline void StoreFinalRow( uint64_t outputBase, AscendC::LocalTensor<float> &value, float *) {
        AscendC::DataCopyExtParams cp;
        cp.blockCount = 1;
        cp.blockLen = cOut_ * sizeof(float);
        cp.srcStride = 0;
        cp.dstStride = 0;
        cp.rsv = 0;
        AscendC::DataCopyPad( outGm_[ outputBase ], value, cp);
    }
    __aicore__ inline void StoreFinalRow( uint64_t outputBase, AscendC::LocalTensor<float> &value, bfloat16_t *) {
        AscendC::LocalTensor<bfloat16_t> output = outputBuf_.Get< bfloat16_t>();
        AscendC::Cast(output, value, AscendC::RoundMode::CAST_RINT, cOut_);
        AscendC::PipeBarrier<PIPE_V>();
        auto event = pipe_.FetchEventID(AscendC::HardEvent::V_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::DataCopyExtParams cp;
        cp.blockCount = 1;
        cp.blockLen = cOut_ * sizeof(bfloat16_t);
        cp.srcStride = 0;
        cp.dstStride = 0;
        cp.rsv = 0;
        AscendC::DataCopyPad( outGm_[ outputBase ], output, cp);
        auto done = pipe_.FetchEventID(AscendC::HardEvent::MTE3_V);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(done);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(done);
    }
    __aicore__ inline void FinalizeNodes( uint32_t coreIdx, uint32_t coreNum) {
        uint64_t startNode = 0U;
        uint64_t endNode = 0U;
        SplitCoreRange( n_, coreIdx, coreNum, startNode, endNode);
        AscendC::DataCopyExtParams inputCopy;
        inputCopy.blockCount = 1;
        inputCopy.blockLen = cOut_ * sizeof(float);
        inputCopy.srcStride = 0;
        inputCopy.dstStride = 0;
        inputCopy.rsv = 0;
        AscendC::DataCopyPadExtParams<float> inputPad = {
            true, 0, static_cast<uint8_t>((8U - cOut_ % 8U) % 8U), 0.0f };
        for (uint64_t node = startNode;
        node < endNode;
        ++node) {
            const uint64_t outputBase = node * cOut_;
            float scale = 1.0f;
            if (aggrMean_ != 0U) {
                const float degreeValue = degGm_.GetValue( node);
                if (degreeValue > 0.0f) {
                    scale = 1.0f / degreeValue;
                }
            }
            AscendC::LocalTensor<float> input = finalInputQueue_.AllocTensor< float>();
            AscendC::DataCopyPad( input, accGm_[ outputBase ], inputCopy, inputPad);
            finalInputQueue_.EnQue< float>( input);
            input = finalInputQueue_.DeQue< float>();
            AscendC::LocalTensor<float> output = messageQueue_.AllocTensor< float>();
            AscendC::Muls( output, input, scale, cOut_);
            messageQueue_.EnQue< float>( output);
            finalInputQueue_.FreeTensor( input);
            output = messageQueue_.DeQue< float>();
            StoreFinalRow( outputBase, output, static_cast< DT_X *>( nullptr));
            messageQueue_.FreeTensor( output);
        }
    }
    private: AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<int64_t> edgeIndexGm_;
    AscendC::GlobalTensor<DT_X> edgeAttrGm_;
    AscendC::GlobalTensor<DT_X> weightGm_;
    AscendC::GlobalTensor<DT_X> outGm_;
    AscendC::GlobalTensor<float> accGm_;
    AscendC::GlobalTensor<float> degGm_;
    AscendC::TPipe pipe_;
    AscendC::TQue< AscendC::QuePosition::VECOUT, 1> messageQueue_;
    AscendC::TQue< AscendC::QuePosition::VECIN, 2> weightTileQueue_;
    AscendC::TQue< AscendC::QuePosition::VECIN, 1> finalInputQueue_;
    AscendC::TBuf< AscendC::TPosition::VECCALC> xCacheBuf_;
    AscendC::TBuf< AscendC::TPosition::VECCALC> weightFloatBuf_;
    AscendC::TBuf< AscendC::TPosition::VECCALC> outputBuf_;
    AscendC::TBuf< AscendC::TPosition::VECCALC> zeroBuf_;
    AscendC::TBuf< AscendC::TPosition::VECCALC> oneBuf_;
    uint32_t n_ = 0U;
    uint32_t e_ = 0U;
    uint32_t cIn_ = 0U;
    uint32_t cOut_ = 0U;
    uint32_t dim_ = 0U;
    uint32_t degree_ = 0U;
    uint32_t basisSlots_ = 0U;
    uint32_t basisPerDim_ = 0U;
    uint32_t aggrMean_ = 0U;
    uint32_t kernel_[3] = {
        1U, 1U, 1U };
    uint32_t open_[3] = {
        1U, 1U, 1U };
    uint32_t kernelOffset_[3] = {
        1U, 1U, 1U };
    uint32_t kernelCount_ = 1U;
    uint32_t weightMatrixStride_ = 0U;
    AscendC::TBuf<AscendC::TPosition::VECCALC> effectiveWeightBuf_;
    uint32_t weightRowStride_ = 16U;
    uint32_t ciTile_ = 1U;
    uint32_t weightTileCount_ = 1U;
    uint32_t mergeWeightIndex_ = 0U;
    uint64_t accStrideFloats_ = 0U;
    bool localAggregate_ = false;
    uint32_t localAccStride_ = 0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> localAccBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> localDegreeBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> localMessageBuf_;
    bool cacheWeights_ = false;
    uint32_t cachedWeightElements_ = 0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weightCacheBuf_;

};
template <typename DT_X> __global__ __aicore__ void spline_conv( GM_ADDR x, GM_ADDR edge_index, GM_ADDR edge_attr, GM_ADDR weight, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT( SplineConvTilingData);
    GET_TILING_DATA_WITH_STRUCT( SplineConvTilingData, tilingData, tiling);
    KernelSplineConv< DT_X> op;
    op.Init( x, edge_index, edge_attr, weight, out, workspace, tilingData);
    op.Process();
}
