// Copyright 2026 XDU_XDU-BTD Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lstsq_transport_alias.h"

#include "kernel_operator.h"
#include "lstsq_scalar_guard.h"
#include "lib/matmul_intf.h"
#include "linalg_lstsq_tiling.h"
#ifndef LSTSQ_STANDALONE
#include "tiling_key_linalg_lstsq.h"
#endif

namespace lstsq_detail{
using namespace AscendC;
template<Zq5 event>
__aicore__ inline void ZqB(int32_t id=0){SetFlag<event>(id);WaitFlag<event>(id);}

struct Pair { float hi, lo; };

struct VolatileRounding {
__aicore__ static inline float RoundAdd(float a,float b){volatile float v=a+b;return v;}
__aicore__ static inline float RoundSub(float a,float b){volatile float v=a-b;return v;}
__aicore__ static inline float RoundMul(float a,float b){volatile float v=a*b;return v;}
};

struct NoContractRounding {
__aicore__ static inline float RoundAdd(float a,float b){
#pragma clang fp contract(off) reassociate(off)
return a+b;
}
__aicore__ static inline float RoundSub(float a,float b){
#pragma clang fp contract(off) reassociate(off)
return a-b;
}
__aicore__ static inline float RoundMul(float a,float b){
#pragma clang fp contract(off) reassociate(off)
return a*b;
}
};

template<class Rounding>
struct CompensatedArithmetic : Rounding {
__aicore__ static inline Pair Sum(float a,float b){
float s=Rounding::RoundAdd(a,b),v=Rounding::RoundSub(s,a);
return{s,Rounding::RoundAdd(Rounding::RoundSub(a,Rounding::RoundSub(s,v)),Rounding::RoundSub(b,v))};
}
__aicore__ static inline float High(float a){
union{float f;Zq2 u;}v;v.f=a;v.u&=0xfffff000u;return v.f;
}
__aicore__ static inline Pair Product(float a,float b){
float h=Rounding::RoundMul(a,b),ah=High(a),bh=High(b);
float al=Rounding::RoundSub(a,ah),bl=Rounding::RoundSub(b,bh);
float e=Rounding::RoundAdd(Rounding::RoundAdd(Rounding::RoundSub(Rounding::RoundMul(ah,bh),h),Rounding::RoundMul(ah,bl)),Rounding::RoundMul(al,bh));
return{h,Rounding::RoundAdd(e,Rounding::RoundMul(al,bl))};
}
__aicore__ static inline float Divide(float a,float b){
if(b==0.0f)return scalar_guard::QuietNaN();
volatile float seed=a/b;
float q=seed;
for(Zq2 it=0;it<2u;++it){
Pair p=Product(q,b),r=Sum(a,-p.hi);
float err=Rounding::RoundAdd(r.hi,Rounding::RoundSub(r.lo,p.lo));
volatile float correction=err/b;
q=Rounding::RoundAdd(q,correction);
}
return q;
}
};

using scalar_math = CompensatedArithmetic<VolatileRounding>;
using square_math = CompensatedArithmetic<NoContractRounding>;

__aicore__ inline float RoundAdd(float a,float b) { return scalar_math::RoundAdd(a,b); }
__aicore__ inline float RoundSub(float a,float b) { return scalar_math::RoundSub(a,b); }
__aicore__ inline float RoundMul(float a,float b) { return scalar_math::RoundMul(a,b); }
__aicore__ inline Pair Sum(float a,float b) { return scalar_math::Sum(a,b); }
__aicore__ inline Pair Product(float a,float b) { return scalar_math::Product(a,b); }
__aicore__ inline float Divide(float a,float b) { return scalar_math::Divide(a,b); }

#include "lstsq_compensated_tile_product.h"
#include "lstsq_residual_reduction.h"
#include "lstsq_rectangular_solve.h"
#include "tiny/lstsq_tiny_scalar_triangular.h"
#include "lstsq_shared_triangular_solve.h"
#include "lstsq_shared_column_triangular_solve.h"
#include "lstsq_shared_lu_factorization.h"
#include "lstsq_blocked_lu256.h"
#include "lstsq_panel_lu200.h"
#include "lstsq_shared_producer_residual.h"
#include "lstsq_shared_refinement.h"
#include "lstsq_shared_residual.h"
#include "lstsq_shared_workspace.h"
#include "lstsq_shared_tiny_fallback.h"
#include "lstsq_fixed_workspace.h"
#include "legacy/lstsq_legacy.h"

#include "square/lstsq_generic_square.h"
#include "square/lstsq_large_pipe200.h"
#include "qr/lstsq_qr_factor_diagnostic.h"
#include "square/lstsq_large_pipe256.h"
#include "square/lstsq_small_pipe64.h"
#include "packed/lstsq_fast_context64.h"
#include "packed/lstsq_fast_context50.h"

#include "tiny/lstsq_tiny_fallback.h"
#include "tiny/lstsq_tiny_fast_context15.h"
#include "tiny/lstsq_tiny_fast_context16.h"


template<Zq2 ColdKey>
__aicore__ __attribute__((noinline))void LstsqCold15OutOfLine(
GM_ADDR A,GM_ADDR B,GM_ADDR solution,GM_ADDR residuals,GM_ADDR workspace,
LinalgLstsqTilingData data,Zq2 batch){
lstsq_detail::KernelTinyAReuse15<15u>cold;
cold.Init(A,B,solution,residuals,workspace,data);cold.ProcessOneMatrix(batch,true);
}

template<Zq2 Cold16Key>
__aicore__ __attribute__((noinline))void LstsqCold16OutOfLine(
GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR R,GM_ADDR W,LinalgLstsqTilingData data,Zq2 batch){
KernelTiny<16u>cold;cold.Init(A,B,X,R,W,data);cold.ProcessOneCold16(batch);
}

template<Zq2 Cold64Key>
__aicore__ __attribute__((noinline))void LstsqCold64OutOfLine(
GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR R,GM_ADDR W,LinalgLstsqTilingData data,Zq2 task){
KernelLarge64Next<64u,2u>cold;
cold.Init(A,B,X,R,W,data);cold.ProcessOneColdTask64(task);
}

template<Zq2 Fast50Key>
__aicore__ __attribute__((noinline))bool LstsqFast50TaskOutOfLine(
GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR R,GM_ADDR W,LinalgLstsqTilingData data,Zq2 task){
KernelFastContext50<50u,2u>fast;
fast.Init(A,B,X,R,W,data);
return fast.TryTask50(task);
}

template<Zq2 Cold50Key>
__aicore__ __attribute__((noinline))void LstsqCold50OutOfLine(
GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR R,GM_ADDR W,LinalgLstsqTilingData data,Zq2 task){
KernelLarge<50u,2u>cold;
cold.Init(A,B,X,R,W,data);cold.ProcessOneColdTask50(task);
}

#ifndef LSTSQ_STANDALONE
template<Zq2 N, Zq2 Rhs>
__aicore__ inline void RunMixedQr(GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR residuals,
                                  GM_ADDR workspace, LinalgLstsqTilingData data)
{
    constexpr bool compact200 = N == 200u;
    qr_factor_probe::Run<N, Rhs, !compact200>(a, b, x, workspace, data);
    if ASCEND_IS_AIV {
        Zq1<PIPE_ALL>();
        SyncAll<true>();
        const Zq2 task = GetBlockIdx() / 2u;
        const Zq2 batch = task / 4u;
        const Zq2 rhs = (task % 4u) % Rhs;
        const Zq3 shared = Zq3(batch) * 397824u;
        bool accepted;
        {
            TPipe consumerPipe;
            KernelLargePipe256<256u, 3u, !compact200, compact200, !compact200> consumer;
            consumer.Init(a, b, x, residuals, workspace, data, &consumerPipe, true, true);
            accepted = consumer.QrUniformDomain(data.batches, batch, rhs, shared, N, Rhs);
            consumer.QrCubeDecision(accepted, shared);
            if (accepted) {
                consumer.ProcessQrCubeRhs(batch, rhs, shared);
            }
        }
        if (!accepted && GetSubBlockIdx() == 0u) {
            if constexpr (compact200) {
                KernelLarge200<200u, 4u> fallback;
                fallback.Init(a, b, x, residuals, workspace, data);
                fallback.Process(true);
            } else {
                TPipe fallbackPipe;
                KernelLargePipe256<256u, 3u> fallback;
                fallback.Init(a, b, x, residuals, workspace, data, &fallbackPipe, false, true);
                fallback.Process();
            }
        }
    }
}

template<Zq2 Shape>
__aicore__ inline void RunSmallSquareTasks(GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR residuals,
                                           GM_ADDR workspace, LinalgLstsqTilingData data)
{
    for (Zq2 task = GetBlockIdx(); task < data.batches * 2u; task += GetBlockNum()) {
        if constexpr (Shape == 9u) {
            bool accepted;
            {
                KernelFastContext64<64u, 2u> fast;
                fast.Init(a, b, x, residuals, workspace, data);
                accepted = fast.TryTask64(task);
            }
            if (!accepted) {
                LstsqCold64OutOfLine<4864u>(a, b, x, residuals, workspace, data, task);
            }
        } else {
            const bool accepted = LstsqFast50TaskOutOfLine<1792u>(a, b, x, residuals, workspace, data, task);
            if (!accepted) {
                LstsqCold50OutOfLine<1792u>(a, b, x, residuals, workspace, data, task);
            }
        }
    }
}

template<Zq2 Shape>
__aicore__ inline void RunLarge(GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR residuals, GM_ADDR workspace,
                                LinalgLstsqTilingData data)
{
    if constexpr (Shape == 7u) {
        RunMixedQr<256u, 3u>(a, b, x, residuals, workspace, data);
    } else if constexpr (Shape == 8u) {
        RunMixedQr<200u, 4u>(a, b, x, residuals, workspace, data);
    } else if constexpr (Shape == 2u) {
        TPipe pipe;
        KernelLargePipe256<256u, 3u> kernel;
        kernel.Init(a, b, x, residuals, workspace, data, &pipe);
        kernel.Process();
    } else if constexpr (Shape == 1u) {
        KernelLarge200<200u, 4u> kernel;
        kernel.Init(a, b, x, residuals, workspace, data);
        kernel.Process();
    } else if constexpr (Shape == 3u || Shape == 9u) {
        RunSmallSquareTasks<Shape>(a, b, x, residuals, workspace, data);
    } else if constexpr (Shape == 4u) {
        KernelLarge64<64u, 2u> kernel;
        kernel.Init(a, b, x, residuals, workspace, data);
        kernel.Process();
    } else {
        constexpr Zq2 fixedK = Shape >= 3u ? 2u : 0u;
        KernelLarge<0u, fixedK> kernel;
        kernel.Init(a, b, x, residuals, workspace, data);
        kernel.Process();
    }
}

template<Zq2 Shape>
__aicore__ inline void RunTiny(GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR residuals, GM_ADDR workspace,
                               LinalgLstsqTilingData data)
{
    Zq2 owner = GetBlockIdx();
    Zq2 stride = GetBlockNum();
    if (Zq3(stride) == Zq3(data.batches) * 2u) {
        if ((owner & 1u) != 0u) {
            return;
        }
        owner >>= 1u;
        stride >>= 1u;
    }
    for (Zq2 batch = owner; batch < data.batches; batch += stride) {
        bool done;
        if constexpr (Shape == 5u) {
            {
                KernelTinyFastContext15<15u> fast;
                fast.Init(a, b, x, residuals, workspace, data);
                done = fast.ProcessFastSingle(batch);
            }
            if (!done) {
                LstsqCold15OutOfLine<2560u>(a, b, x, residuals, workspace, data, batch);
            }
        } else {
            {
                KernelTinyFastContext16<16u> fast;
                fast.Init(a, b, x, residuals, workspace, data);
                done = fast.ProcessFastSingle(batch);
            }
            if (!done) {
                LstsqCold16OutOfLine<3072u>(a, b, x, residuals, workspace, data, batch);
            }
        }
    }
}
#endif

}

#ifndef LSTSQ_STANDALONE
template<typename DT_A, bool BIG, Zq2 SHAPE>
__global__ __aicore__ void linalg_lstsq(GM_ADDR A, GM_ADDR B, GM_ADDR solution,
                                      GM_ADDR residuals, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(1280, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(3840, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(4352, KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(LinalgLstsqTilingData);

    // The host selects these keys only for doubled small-batch Tiny16 dispatch.
    if constexpr (!BIG && (SHAPE == 10u || SHAPE == 11u)) {
        const Zq2 block = AscendC::GetBlockIdx();
        if ((block & 1u) != 0u) {
            return;
        }
        const Zq2 batch = block >> 1u;
        const Zq3 residualCount = SHAPE == 11u ? Zq3(AscendC::GetBlockNum() >> 1u) : Zq3(0u);
        bool done;
        {
            lstsq_detail::KernelTinyFastContext16<16u> fast;
            fast.InitImplicitGeometry(A, B, solution, residuals, residualCount);
            done = fast.ProcessFastSingle(batch);
        }
        if (!done) {
            GET_TILING_DATA_WITH_STRUCT(LinalgLstsqTilingData, coldData, tiling);
            lstsq_detail::LstsqCold16OutOfLine<3072u>(A, B, solution, residuals, workspace, coldData, batch);
        }
    } else {
        GET_TILING_DATA_WITH_STRUCT(LinalgLstsqTilingData, data, tiling);
        if constexpr (BIG) {
            lstsq_detail::RunLarge<SHAPE>(A, B, solution, residuals, workspace, data);
        } else if constexpr (SHAPE == 5u || SHAPE == 6u) {
            lstsq_detail::RunTiny<SHAPE>(A, B, solution, residuals, workspace, data);
        } else {
            lstsq_detail::KernelLegacy kernel;
            kernel.Init(A, B, solution, residuals, workspace, data);
            kernel.Process();
        }
    }
}
#endif
