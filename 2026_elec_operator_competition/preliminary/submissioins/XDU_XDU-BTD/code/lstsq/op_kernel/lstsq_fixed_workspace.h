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

#pragma once

class PackedWorkspaceState {
protected:
__gm__ float*a_=nullptr,*b_=nullptr,*x_=nullptr,*r_=nullptr,*w_=nullptr;
TPipe pipe_;
Zq7<float>factors_,rows_,block_,panel_;
Zq7<Zq2>residualTile_;
Zq2 factorPitch_=0u;
bool localFactors_=false,columnLocal_=false;
Zq7<float>solveY_;
bool vectorSolve_=false;
Zq7<float>output_,vr_;
bool vectorResidual_=false;
Zq2 vectorPitch_=8u,currentBatch_=0u;
bool dmaOutputs_=false;
bool splitRhs_=false;
Zq2 currentRhs_=0u,rhsWidth_=0u;
Zq7<float>negative64_;
bool negative64Ready_=false;
bool diagonalKnown_=false,diagonalSafe_=false;
LinalgLstsqTilingData tiling_;
};

class SquareWorkspaceState {
protected:
    TBuf<TPosition::VECCALC>factorBuf_,rowBuf_,blockBuf_,panelBuf_,residualTileBuf_;
    Zq7<float>factors_,rows_,block_,panel_;
    Zq7<Zq2>residualTile_;
    Zq2 factorPitch_=0u;
    bool localFactors_=false,columnLocal_=false;
    TBuf<TPosition::VECCALC>solveBuf_;
    Zq7<float>solveY_;
    bool vectorSolve_=false;
    TBuf<TPosition::VECCALC>outputBuf_;
    TBuf<TPosition::VECCALC>residualBuf_;
    Zq7<float>output_,vr_;
    bool vectorResidual_=false;
    Zq2 vectorPitch_=8u,currentBatch_=0u;
    bool dmaOutputs_=false;
    bool splitRhs_=false;
    Zq2 currentRhs_=0u,rhsWidth_=0u;
};

// Shared ownership for private fixed-square input, output and factor workspace.
class PrivateSquareStorage {
protected:
__gm__ float*a_=nullptr,*b_=nullptr,*x_=nullptr,*r_=nullptr,*w_=nullptr;
TPipe pipe_;
TBuf<TPosition::VECCALC>negative64Buf_;
Zq7<float>negative64_;
bool negative64Ready_=false;
bool diagonalKnown_=false,diagonalSafe_=false;
LinalgLstsqTilingData tiling_;
};

class KernelScalarOperations {
protected:
__aicore__ static inline float SeedInverse(float diagonal){return scalar_guard::ReciprocalSeed(diagonal);}
__aicore__ static inline float AbsF(float value){return value<0.0f?-value:value;}
};

// All fixed square kernels process one right-hand side per private workspace.
template<Zq2 Rows, Zq2 Rhs, Zq2 VectorWidth>
struct FixedSquareLayout {
    __aicore__ static inline Zq2 M() { return Rows; }
    __aicore__ static inline Zq2 N() { return Rows; }
    __aicore__ static inline Zq2 K() { return 1u; }
    __aicore__ static inline Zq2 RhsWidth() { return Rhs; }
    __aicore__ static inline Zq2 FactorPitch() { return (Rows + 7u) / 8u * 8u; }
    __aicore__ static inline Zq2 VectorPitch() { return VectorWidth; }
    __aicore__ static inline Zq2 ResidualMaskPlane() { return Rows == 256u ? 8u : 11u; }
    __aicore__ static inline Zq2 ResidualZeroPlane() { return Rows == 256u ? 9u : 12u; }
    __aicore__ static inline bool UseSplitRhs() { return Rhs > 1u; }
    __aicore__ static inline bool UseDmaOutputs() { return true; }


    __aicore__ static inline Zq3 OffA(Zq3 base) { return base; }
    __aicore__ static inline Zq3 OffLU(Zq3 base) { return base + Zq3(Rows) * Rows; }
    __aicore__ static inline Zq3 OffB(Zq3 base) { return OffLU(base) + Zq3(Rows) * Rows; }
    __aicore__ static inline Zq3 OffX(Zq3 base) { return OffB(base) + Rows; }
    __aicore__ static inline Zq3 OffR(Zq3 base) { return OffX(base) + Rows; }
    __aicore__ static inline Zq3 OffD(Zq3 base) { return OffR(base) + Rows; }
    __aicore__ static inline Zq3 OffP(Zq3 base) { return OffD(base) + Rows; }
    __aicore__ static inline Zq3 OffInv(Zq3 base) { return OffP(base) + Rows; }
    __aicore__ static inline Zq3 PerBatch() {
        return (OffInv(0u) + Rows + 15u) / 16u * 16u;
    }
};

struct FixedWorkspace {
    template<class T, TPosition Position>
    __aicore__ static inline Zq7<T> Allocate(TPipe& pipe, TBuf<Position>& buffer, Zq2 elements)
    {
        pipe.InitBuffer(buffer, elements * sizeof(T));
        return buffer.template Get<T>();
    }

    template<class Kernel>
    __aicore__ static inline void BindAddresses(Kernel& kernel, GM_ADDR a, GM_ADDR b, GM_ADDR x,
                                                GM_ADDR residuals, GM_ADDR workspace)
    {
        kernel.a_ = reinterpret_cast<__gm__ float*>(a);
        kernel.b_ = reinterpret_cast<__gm__ float*>(b);
        kernel.x_ = reinterpret_cast<__gm__ float*>(x);
        kernel.r_ = reinterpret_cast<__gm__ float*>(residuals);
        const Zq3 user = reinterpret_cast<Zq3>(AscendC::GetUserWorkspace(workspace));
        kernel.w_ = reinterpret_cast<__gm__ float*>((user + 63u) & ~Zq3(63u));
    }

    __aicore__ static inline void InitResidualPlanes(Zq7<float> vr, Zq2 pitch, Zq2 maskPlane,
                                                    Zq2 zeroPlane)
    {
        Duplicate(vr[maskPlane * pitch].template Zq6<Zq2>(), Zq2(0xfffff000u), pitch);
        Duplicate(vr, 0.0f, pitch);
        Duplicate(vr[zeroPlane * pitch], 0.0f, pitch);
        Duplicate(vr[pitch], 0.0f, pitch);
    }

    template<Zq2 N, Zq2 Rhs, bool Compact200 = false, class Kernel>
    __aicore__ static inline void InitSquare(Kernel& kernel, TPipe& pipe,
                                            LinalgLstsqTilingData data, bool compactQr = false)
    {
        constexpr Zq2 pitch = N == 64u ? 64u : 256u;
        data.m = N;
        data.n = N;
        data.k = 1u;
        kernel.rhsWidth_ = Rhs;
        kernel.splitRhs_ = true;
        kernel.tiling_ = data;
        kernel.dmaOutputs_ = true;
        kernel.output_ = Allocate<float>(pipe, kernel.outputBuf_, N * 8u + 8u);
        kernel.localFactors_ = N != 256u;
        kernel.columnLocal_ = N != 256u;
        kernel.factorPitch_ = N;
        constexpr Zq2 factors = N == 64u ? 64u * 72u : N == 200u ? 200u * 200u
                                                                 : Compact200 ? 8320u : 32u * 256u;
        kernel.factors_ = Allocate<float>(pipe, kernel.factorBuf_, factors);
        kernel.rows_ = Allocate<float>(pipe, kernel.rowBuf_, 208u);
        if constexpr (N == 256u) {
            const Zq2 block = compactQr ? (Compact200 ? 25152u : 24768u)
                                         : 16u * 256u + 64u + 2u * 48u * 264u;
            kernel.block_ = Allocate<float>(pipe, kernel.blockBuf_, block);
            kernel.panel_ = Allocate<float>(pipe, kernel.panelBuf_, 256u * 8u);
            const Zq2 tile = compactQr ? (Compact200 ? 8320u : 8192u) : 2u * 8u * 256u;
            kernel.residualTile_ = Allocate<Zq2>(pipe, kernel.residualTileBuf_, tile);
        } else {
            constexpr Zq2 block = N == 200u ? 9u * 200u : 8u * 64u;
            constexpr Zq2 panel = N == 200u ? 8u : 16u * 2048u + 512u;
            kernel.block_ = Allocate<float>(pipe, kernel.blockBuf_, block);
            kernel.panel_ = Allocate<float>(pipe, kernel.panelBuf_, panel);
            kernel.residualTile_ = Allocate<Zq2>(pipe, kernel.residualTileBuf_, 8u * N);
        }
        kernel.vectorResidual_ = true;
        kernel.vectorPitch_ = pitch;
        kernel.vr_ = Allocate<float>(pipe, kernel.residualBuf_, (N == 256u ? 10u : 13u) * pitch);
        kernel.vectorSolve_ = true;
        kernel.solveY_ = Allocate<float>(pipe, kernel.solveBuf_, pitch);
    }

    template<Zq2 N, class Kernel>
    __aicore__ static inline void InitTinyFallback(Kernel& kernel, LinalgLstsqTilingData data)
    {
        constexpr Zq2 pitch = 16u;
        data.m = N;
        data.n = N;
        data.k = 1u;
        auto arena = Allocate<float>(kernel.pipe_, kernel.arenaBuf_, 4936u + pitch * 24u);
        kernel.tinyNegative_ = Allocate<float>(kernel.pipe_, kernel.tinyNegativeBuf_, 304u);
        auto negativeTail = kernel.tinyNegative_[264u].template Zq6<Zq2>();
        for (Zq2 i = 0; i < 8u; ++i) {
            negativeTail.SetValue(i, 0u);
        }
        kernel.rhsWidth_ = 1u;
        kernel.splitRhs_ = false;
        kernel.tiling_ = data;
        kernel.dmaOutputs_ = true;
        kernel.output_ = arena[0u];
        kernel.localFactors_ = true;
        kernel.columnLocal_ = true;
        kernel.factorPitch_ = pitch;
        kernel.factors_ = Allocate<float>(kernel.pipe_, kernel.tinyAugmentedBuf_, 512u);
        kernel.rows_ = arena[24u + pitch * 16u];
        kernel.block_ = arena[232u + pitch * 16u];
        kernel.panel_ = arena[360u + pitch * 16u];
        kernel.residualTile_ = arena[4712u + pitch * 16u].template Zq6<Zq2>();
        kernel.vectorResidual_ = true;
        kernel.vectorPitch_ = pitch;
        kernel.vr_ = arena[4712u + pitch * 24u];
        kernel.vectorSolve_ = true;
        kernel.solveY_ = arena[4920u + pitch * 24u];
        InitTinyTranspose(kernel.panel_, kernel.tinyNegative_, kernel.vr_);
        InitResidualPlanes(kernel.vr_, pitch, 11u, 12u);
    }

    __aicore__ static inline void InitTinyTranspose(Zq7<float> panel, Zq7<float> negative, Zq7<float> vr)
    {
        auto transpose = panel[16u * 256u].template Zq6<int32_t>();
        auto features = negative[272u].template Zq6<int32_t>();
        auto rowBytes = negative[288u].template Zq6<int32_t>();
        for (Zq2 i = 0; i < 16u; ++i) {
            features.SetValue(i, int32_t(i * 4u));
            rowBytes.SetValue(i, int32_t(i * 64u));
        }
        ZqB<Zq5::S_V>();
        SetMaskNorm();
        ResetMask();
        Brcb(transpose, features, Zq4(2), Zq9{2, 16});
        Brcb(transpose[8u], features, Zq4(2), Zq9{2, 16});
        Zq1<PIPE_V>();
        Add<int32_t>(transpose, transpose, rowBytes, Zq3(16), Zq4(16), Zq0{1, 1, 1, 2, 2, 0});
        Zq1<PIPE_V>();
        auto columns = vr[10u * 16u].template Zq6<Zq2>();
        for (Zq2 col = 0; col < 16u; ++col) {
            columns.SetValue(col, col * 16u * 4u);
        }
    }
};
