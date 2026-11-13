/*
 * Copyright (c) 2026 ruby0451 contributors.
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

#ifndef RUBY0451_SOLVE_SOLVE_COLUMN_H
#define RUBY0451_SOLVE_SOLVE_COLUMN_H
#include "solve_common.h"
template<uint32_t FAMILY=0,uint32_t STATIC_N=0,uint32_t STATIC_WORKERS=0,uint32_t STATIC_K=0,bool PARTIAL_CAPACITY=false> class KernelColumnSolve {
public:
template<bool SPLIT>
__aicore__ inline void ResidualTileCompensated(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows);


template<bool SPLIT>
__aicore__ inline void ComputeCompensatedProducts(LocalTensor<float> sourceA,LocalTensor<float> sourceHigh,LocalTensor<float> sourceLow,LocalTensor<float> product,LocalTensor<float> low,LocalTensor<float> temp,LocalTensor<float> ah,LocalTensor<float> xh,LocalTensor<float> xl,uint32_t q,uint32_t rows);


template<bool persistent,bool retiredMetadata>
__aicore__ inline void GatherWindowMetadata(uint32_t lane,LocalTensor<float> packetView,uint32_t panel,uint32_t steps);


__aicore__ inline void PublishUnpivotedOwner(uint32_t panel,uint32_t panelEnd,uint32_t end,uint32_t begin,uint32_t localCols);


template<bool retiredMetadata>
__aicore__ inline void JoinCollapsedOwner();



__aicore__ inline void TransposeColumnGroup(uint32_t group,uint32_t dstColumn,uint32_t col);


template<uint32_t PREFIX>
 __aicore__ inline void ChooseCollapsedPivot(LocalTensor<float> absolute,LocalTensor<float> reduced,LocalTensor<float> f,uint64_t* active,uint32_t rowBase,uint32_t firstRow,uint32_t& row,float& diagonal);


__aicore__ inline void CorrectRhsBlock(LocalTensor<float> rhs,uint32_t block,uint32_t stop,uint32_t slot);


__aicore__ inline void ComputeJointProducts(LocalTensor<float> xh,LocalTensor<float> xl,LocalTensor<float> product,LocalTensor<float> sourceHigh,LocalTensor<float> sourceLow,LocalTensor<float> sourceA,LocalTensor<float> low,LocalTensor<float> temp,uint32_t rows,uint32_t size);


__aicore__ inline void ComputeOriginalProducts(LocalTensor<float> product,LocalTensor<float> original,LocalTensor<float> ah,LocalTensor<float> al,LocalTensor<float> xh,LocalTensor<float> xl,LocalTensor<float> low,LocalTensor<float> temp,uint32_t q,uint32_t rows);


__aicore__ inline void ReduceCompensatedRows(LocalTensor<float> product,LocalTensor<float> low,LocalTensor<float> total,LocalTensor<float> z,LocalTensor<float> temp,LocalTensor<float> other,uint32_t rows);

__aicore__ inline void StoreResidualRows(LocalTensor<float> other,LocalTensor<float> savedB,LocalTensor<float> original,LocalTensor<float> ah,LocalTensor<float> al,LocalTensor<float> z,LocalTensor<float> total,LocalTensor<float> temp,LocalTensor<uint32_t>& leftIndex,uint64_t boff,uint32_t rows,uint32_t q,uint32_t base);

__aicore__ inline void ReduceResidualRows(LocalTensor<float> product,LocalTensor<float> low,LocalTensor<float> original,LocalTensor<float> ah,uint32_t rows);

__aicore__ inline void LoadResidualRhs(LocalTensor<float> savedB,uint64_t boff,uint32_t rows);

template<bool localJoins>
 __aicore__ inline void StoreBlockedCorrection(uint32_t batch,uint32_t lane);


__aicore__ inline void ComputeBlockedResidual(uint32_t batch,uint32_t lane);

template<uint32_t PREFIX,uint32_t rows,uint32_t pitch>
 __aicore__ inline void StoreCollapsedFactors(LocalTensor<float> matrix);

template<uint32_t rows,uint32_t pitch>
 __aicore__ inline void UpdateCollapsedColumns(uint32_t p,uint32_t cols,uint32_t firstRow,float inv,LocalTensor<float> f,LocalTensor<float> trailing);


__aicore__ inline void LoadWindowColumns(uint32_t batch,uint32_t lane,uint32_t rhsLane,uint32_t begin,uint32_t localCols);

template<bool persistent,bool retiredMetadata>
 __aicore__ inline void JoinWindowTail(uint32_t lane);

template<bool retiredMetadata>
 __aicore__ inline void StoreWindowFactors(uint32_t begin,uint32_t localCols,uint32_t lane,uint32_t rhsLane);

template<bool persistent>
 __aicore__ inline void ComposeWindowPairs(uint32_t begin,uint32_t end);


__aicore__ inline void ComposeWindowQuartets(uint32_t begin,uint32_t end);

template<bool persistent,bool retiredMetadata,bool boundaryReady>
__aicore__ inline bool ProcessWindowPanel(uint32_t panel,uint32_t panelWidth,uint32_t window,uint32_t begin,uint32_t end,uint32_t lane,uint32_t rhsLane,uint32_t localCols,uint64_t* activeRows,bool& pendingBoundary,LocalTensor<float> absolute,LocalTensor<float> reduced);

template<bool boundaryReady>
 __aicore__ inline void UpdateWindowColumns(uint32_t panel,uint32_t panelEnd,uint32_t begin,uint32_t end,uint32_t lane,uint32_t rhsLane,uint32_t localCols,uint32_t rowBase,bool owner,uint64_t* activeRows,bool& pendingBoundary);

template<bool persistent,bool boundaryReady>
 __aicore__ inline void PublishWindowOwner(uint32_t panel,uint32_t panelEnd,uint32_t begin,uint32_t rowBase,uint32_t window,uint32_t packet,uint32_t steps,uint64_t* activeRows,bool& pendingBoundary,LocalTensor<float> absolute,LocalTensor<float> reduced);

template<bool persistent,bool boundaryReady>
 __aicore__ inline void NormalizeWindowPivot(uint32_t p,uint32_t panel,uint32_t panelEnd,uint32_t begin,uint32_t rowBase,uint32_t window,uint64_t* activeRows,bool& pendingBoundary,LocalTensor<float> absolute,LocalTensor<float> reduced);

template<bool persistent,bool boundaryReady>
 __aicore__ inline void JoinWindowCoefficients(uint32_t p,uint32_t panel,bool& pendingBoundary);

template<bool persistent,bool boundaryReady>
 __aicore__ inline void WaitWindowPivot(uint32_t p,uint32_t panel,bool pendingBoundary);

template<bool persistent>
 __aicore__ inline void ChooseWindowPivot(uint32_t rowBase,uint32_t rowCount,uint64_t* activeRows,LocalTensor<float> absolute,LocalTensor<float> reduced,uint32_t& row,float& diagonal);

template<bool persistent>
 __aicore__ inline void StoreUnpivotedFactors(uint32_t begin,uint32_t end,uint32_t localCols,uint32_t lane,uint32_t rhsLane);

template<bool persistent>
 __aicore__ inline void ReceiveUnpivotedPanel(uint32_t panel,uint32_t panelEnd,uint32_t begin,uint32_t end,uint32_t lane,uint32_t rhsLane,uint32_t localCols,uint32_t packet,uint32_t steps,bool owner);

template<bool persistent>
 __aicore__ inline void NormalizeUnpivotedOwner(uint32_t panel,uint32_t panelEnd,uint32_t begin,uint32_t packet,uint32_t steps);

__aicore__ inline ~KernelColumnSolve();
template<HardEvent event> __aicore__ inline void Fence();
__aicore__ inline void CreateVecIndex(LocalTensor<int32_t> target,int32_t first,uint32_t count);
__aicore__ inline void Init(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR work,const LinalgSolveTilingData& t);
__aicore__ inline void LoadMatrix(GlobalTensor<float> source,uint64_t offset,uint32_t cols,uint32_t firstColumn);
__aicore__ inline LocalTensor<float> ReadColumn(uint32_t j);
__aicore__ inline void WriteColumn(uint32_t j,LocalTensor<float> value);
__aicore__ inline void Process();

__aicore__ inline void LoadRange(GlobalTensor<float> source,uint64_t offset,uint32_t stride,uint32_t count,uint32_t dstColumn);
__aicore__ inline void LoadAlignedWideRange(GlobalTensor<float> source,uint64_t offset,
                                               uint32_t stride,uint32_t count,uint32_t dstColumn);
__aicore__ inline void FactorPaired();
    // Build the fixed private layout directly from bounded GM reads. Missing
    // rows/columns form diag(A,4I); extra RHS columns are zero. The added block
    // is independent, so its solution is zero and the original AX=B is retained.
__aicore__ inline void LoadCapacityRange(GlobalTensor<float> source,uint64_t offset,
                                            uint32_t stride,uint32_t count,uint32_t dstColumn,
                                            uint32_t validCols,uint32_t firstDiagonal);
__aicore__ inline void FactorUnpivoted();
__aicore__ inline void InitializePublication();
__aicore__ inline void PublishPersistentPanel(uint32_t panel);
__aicore__ inline void WaitPersistentPanel(uint32_t panel);
__aicore__ inline void FactorWindow();
template<uint32_t PREFIX>
    __aicore__ inline void FactorCollapsedTail();
__aicore__ inline void ProcessCooperative();
__aicore__ inline void StoreCooperativeOutput(uint32_t batch);
__aicore__ inline void PrepareCompactThreeIndices();
__aicore__ inline void StoreCompactThreeOutput(uint32_t batch);
__aicore__ inline void ProcessBlocked();
__aicore__ inline void ForwardBlocked();
__aicore__ inline void BacksolveBlocked(bool coefficientsReady=false);
__aicore__ inline void NormalizeOwnedPanelEarly(LocalTensor<float> tile,uint32_t cols,uint32_t row,float inv);
__aicore__ inline void NormalizeOwnedPanel(LocalTensor<float> tile,uint32_t cols,uint32_t row,float inv);
__aicore__ inline void UpdateColumnsWindowReady(LocalTensor<float> tile,uint32_t cols,
                                                     uint32_t row,uint32_t readyBase);
__aicore__ inline void UpdateColumns(LocalTensor<float> tile,uint32_t cols,uint32_t row,float inv);

__aicore__ inline void RowMul(LocalTensor<float> dst,LocalTensor<float> a0,LocalTensor<float> b0,uint32_t rows);
__aicore__ inline void PrepareResidualIndices();
__aicore__ inline void ResidualTileOriginal(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows);
__aicore__ inline void ResidualTile(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows);
__aicore__ inline void ResidualPlainThreeRhs32(uint32_t batch,uint32_t base);
__aicore__ inline void ResidualTileRounded(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows);
__aicore__ inline void ResidualTileAccurate(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows);
__aicore__ inline void ResidualJointFour(uint32_t batch,uint32_t base,uint32_t rows);
__aicore__ inline void ResidualTileSplitProduct(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows);
__aicore__ inline void AddWide(float term,float& sum,float& low);
__aicore__ inline void Assemble(LocalTensor<float> rhs,LocalTensor<float> target,bool add,uint32_t rhsPitch=0,bool scale=true,bool useOrder=true);
__aicore__ inline void CorrectSingleRhs();
__aicore__ inline void ApplyCorrection();
__aicore__ inline void Refine(uint32_t batch);
private:
    GlobalTensor<uint32_t> publication;
    LocalTensor<float> mat,input,transposed,factor,column,output,inverse,baseFactor;LocalTensor<uint32_t> indices,order;
    GlobalTensor<float> a,b,x,w;
    uint32_t n,k,na,cp,logicalN,logicalK,batches,cores,firstResident,initialResident,spillGroup,workers;bool broadcast,parallel;
};
// FP32 transpose: 16 source rows x 8 columns per repeat.


#include "solve_column_memory.h"
#include "solve_column_publication.h"
#include "solve_column_factor.h"
#include "solve_column_window_1.h"
#include "solve_column_window_2.h"
#include "solve_column_residual.h"
#include "solve_column_correction.h"
#include "solve_column_dispatch.h"

#endif
