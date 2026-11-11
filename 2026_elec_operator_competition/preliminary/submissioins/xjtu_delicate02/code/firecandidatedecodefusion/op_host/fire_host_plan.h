/*
 * Copyright (c) 2026 Delicate02. All rights reserved.
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
#include <algorithm>
#include <array>
#include <vector>
#include <cstdint>
#include "../op_kernel/fire_candidate_decode_fusion_tiling.h"

struct FireHostPlan {
    FireCandidateDecodeFusionPackedTilingData packet{};
    uint32_t launchBlocks=0, tile=0;
};
struct FirePlanPartition {
    FireCandidateDecodeFusionTilingData tiling{};
    bool compactSingleTile=false,batchOne=false;
    uint32_t batchAlign=0,batchUnits=0,batchWorkers=0,maxBatchRows=0;
};
inline uint32_t fire_mid_tile(uint32_t rows,bool halfInput,bool batchOne) {
    return rows<=512?512:(rows<=528?528:(rows<=640?640:
        ((halfInput && batchOne && rows<=896)?896:1024)));
}
inline FirePlanPartition fire_partition_sizes(uint32_t batch,uint32_t n,bool halfInput,uint32_t available) {
    FirePlanPartition partition;
    if(batch==0 || n==0 || available==0)return partition;
    auto* tiling=&partition.tiling;
        tiling->batch = batch;
        tiling->n = n;
        const uint32_t totalRows=tiling->batch*tiling->n;
        const uint32_t cores=available;
        const bool compactSingleTile=totalRows<=8192;
        const uint32_t batchAlign=halfInput?16:8;
        const uint32_t batchUnits=(n+batchAlign-1)/batchAlign;
        // Amortize preparation with one bounded 1024-row resource per owner.
        // The original aligned ownership still handles each partial tile.
        const uint32_t desiredOwners=n/1024+uint32_t(n%1024!=0);
        const bool balanced1024Owners=!compactSingleTile && totalRows<cores*1664*3 &&
            uint64_t(batch)*desiredOwners<=cores;
        const uint32_t batchWorkers=balanced1024Owners?desiredOwners:
            std::min({cores/batch,batchUnits,std::max(1u,n/512)});
        const uint32_t maxBatchRows=batchWorkers?
            (batchUnits+batchWorkers-1)/batchWorkers*batchAlign:0;
        const bool batchOne=!compactSingleTile && totalRows<cores*1664*3 &&
            batchWorkers && maxBatchRows<=1024;

    partition.compactSingleTile=compactSingleTile;partition.batchOne=batchOne;
    partition.batchAlign=batchAlign;partition.batchUnits=batchUnits;
    partition.batchWorkers=batchWorkers;partition.maxBatchRows=maxBatchRows;
    return partition;
}
inline bool fire_select_partition(FirePlanPartition& partition,bool halfInput,uint32_t available) {
    auto* tiling=&partition.tiling;
    const uint32_t batch=tiling->batch,totalRows=batch*tiling->n,cores=available;
    const bool compactSingleTile=partition.compactSingleTile,batchOne=partition.batchOne;
    const uint32_t batchWorkers=partition.batchWorkers,maxBatchRows=partition.maxBatchRows;
        if(compactSingleTile) {
            tiling->tile=tiling->n<=32?32:(tiling->n<=64?64:128);
            uint32_t tiles=tiling->batch*((tiling->n+tiling->tile-1)/tiling->tile);
            tiling->blocks=std::min(cores,tiles);
        } else if(totalRows>=cores*1664*3) {
            const uint32_t originalTiles=(tiling->n+1663)/1664;
            const uint32_t balancedRows=(tiling->n+originalTiles-1)/originalTiles;
            const uint32_t rowAlign=halfInput?16:8;
            const uint32_t alignedRows=(balancedRows+rowAlign-1)/rowAlign*rowAlign;
            tiling->tile=!halfInput && alignedRows<=1640?1640:1664;
            tiling->blocks=std::min(cores,tiling->batch*((tiling->n+tiling->tile-1)/tiling->tile));
        } else if(batchOne) {
            tiling->tile=fire_mid_tile(maxBatchRows,halfInput,true);
            tiling->blocks=batchWorkers*batch;
        } else {
            uint32_t rowAlign=halfInput?16:8;
            uint32_t workers=std::min(cores,std::max(1u,totalRows/512));
            if(workers>=8)workers=workers/8*8;
            if (workers == 0) return false;
            uint32_t units=(totalRows+rowAlign-1)/rowAlign;
            uint32_t maxWorkerRows=(units+workers-1)/workers*rowAlign;
            tiling->blocks=workers;
            tiling->tile=fire_mid_tile(maxWorkerRows,false,false);
        }
    return tiling->blocks!=0;
}
inline uint32_t fire_complete_partition(FirePlanPartition& partition,bool halfInput,uint32_t available) {
    auto* tiling=&partition.tiling;
    const uint32_t batch=tiling->batch,totalRows=batch*tiling->n,cores=available;
    const bool compactSingleTile=partition.compactSingleTile,batchOne=partition.batchOne;
    const uint32_t batchWorkers=partition.batchWorkers,maxBatchRows=partition.maxBatchRows;
    const uint32_t batchUnits=partition.batchUnits;

        uint32_t launchBlocks=tiling->blocks;
        if(compactSingleTile)launchBlocks=std::min(cores,std::max(launchBlocks,8u));
        else launchBlocks=std::min(cores,((launchBlocks+7)/8)*8);
        const uint32_t rowAlign=halfInput?16:8;
        const uint32_t units=(totalRows+rowAlign-1)/rowAlign;
        tiling->unitsPerWorker=batchOne?batchUnits/batchWorkers:units/tiling->blocks;
        tiling->extraUnits=batchOne?batchUnits%batchWorkers:units%tiling->blocks;
        tiling->totalRows=totalRows;
        tiling->denseOne=(tiling->tile<=128 && tiling->n<=tiling->tile &&
            tiling->blocks==tiling->batch) ||
            (tiling->tile>128 && tiling->tile<=1024 && tiling->n>=tiling->tile &&
             (tiling->unitsPerWorker+(tiling->extraUnits!=0))*rowAlign<=tiling->tile);
    return launchBlocks;
}
inline uint32_t fire_uniform_units(uint32_t n,uint32_t TILE,bool halfInput,const FireCandidateDecodeFusionTilingData* tiling) {
    const uint32_t batch=tiling->batch;
        uint32_t packedUnits=tiling->tile>1024?0u:tiling->unitsPerWorker;
        if(!halfInput && TILE==1640) {
            const uint64_t rounded=uint64_t(n)+TILE-1;
            const uint64_t tiles=(rounded/TILE)*batch;
            if(rounded<=UINT32_MAX && tiles<=UINT32_MAX/tiling->blocks &&
                tiles%tiling->blocks==0 && tiles/tiling->blocks<=65535)
                packedUnits=static_cast<uint32_t>(tiles/tiling->blocks);
        }
    return packedUnits;
}
inline FireHostPlan fire_make_plan(uint32_t batch,uint32_t n,bool halfInput,uint32_t available) {
    FireHostPlan result;
    if(batch==0 || n==0 || available==0)return result;
    auto partition=fire_partition_sizes(batch,n,halfInput,available);
    if(!fire_select_partition(partition,halfInput,available))return result;
    result.launchBlocks=fire_complete_partition(partition,halfInput,available);
    auto* tiling=&partition.tiling;auto* packet=&result.packet;
    const uint32_t TILE=tiling->tile,batchAlign=partition.batchAlign;
    const uint32_t batchUnits=partition.batchUnits,batchWorkers=partition.batchWorkers,maxBatchRows=partition.maxBatchRows;
    const bool batchOne=partition.batchOne;
    const uint32_t packedUnits=fire_uniform_units(n,TILE,halfInput,tiling);
        packet->n=tiling->n;packet->totalRows=tiling->totalRows;
        packet->batch=tiling->batch;
        // A full-capacity partition has the same constant count on every
        // owner. All other aligned/tail partitions retain the generic path.
        // The new bounded resource retains the original partial-entry flag.
        const bool uniformFull=batchOne && TILE!=896 && n%batchAlign==0 &&
            batchUnits%batchWorkers==0 && maxBatchRows==(TILE+127)/128*128;
        const uint32_t packedDense=batchOne?uint32_t(uniformFull):tiling->denseOne;
        packet->control=packedUnits|(tiling->extraUnits<<16)|
            (tiling->blocks<<22)|(packedDense<<28)|(uint32_t(batchOne)<<29);
        if(batchOne && !(batch&(batch-1))) {
            const uint32_t shift=(batch>1)+(batch>2)+(batch>4);
            packet->control|=shift<<30;
        }

    result.tile=TILE;
    return result;
}
