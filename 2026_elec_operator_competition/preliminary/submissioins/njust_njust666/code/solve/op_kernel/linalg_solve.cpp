#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "linalg_solve_tiling.h"
#include "tiling_key_linalg_solve.h"
using namespace AscendC;

template <HardEvent E> __aicore__ inline void Fence() {
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}

template <uint32_t N> class TinyInverse {
    static constexpr uint32_t D = 16, RP = 40, ND = N * D;
    static constexpr uint32_t MA = 0, AA = N * RP, PP = AA + ND, EE = PP + ND;
    static constexpr uint32_t SS = EE + ND, ZZ = SS + ND, TT = ZZ + ND, LL = TT + ND;
    static constexpr uint32_t HH = LL + ND, HL = HH + ND, AH = HL + ND, AL = AH + ND;
    static constexpr uint32_t XT = AL + ND, XH = XT + ND, XL = XH + ND, IV = XL + ND;
    static constexpr uint32_t XX = IV + ND, RR = XX + D, BB = RR + D, DD = BB + D;
    static constexpr uint32_t CC = DD + D, AB = CC + D, MM = AB + D, RW = MM + 8;
    static constexpr uint32_t BC = RW + RP, TOTAL = BC + D * 8;
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR out, const LinalgSolveTilingData &data) {
        t = data;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        mem = LocalTensor<float>(TPosition::VECCALC, 0, TOTAL);
        mat = mem[MA]; aa = mem[AA]; p = mem[PP]; e = mem[EE]; s = mem[SS]; z = mem[ZZ];
        temp = mem[TT]; low = mem[LL]; h = mem[HH]; hl = mem[HL]; ah = mem[AH]; al = mem[AL];
        xt = mem[XT]; xh = mem[XH]; xl = mem[XL]; inverse = mem[IV];
        x = mem[XX]; r = mem[RR]; bvalues = mem[BB]; delta = mem[DD];
        coef = mem[CC]; ab = mem[AB]; maximum = mem[MM]; row = mem[RW]; broad = mem[BC];
        colIndex = LocalTensor<uint32_t>(TPosition::VECCALC, TOTAL * 4, D);
        resultIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + D) * 4, D);
        tileIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D) * 4, ND);
        inverseIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D + ND) * 4, ND);
        shiftIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D + 2 * ND) * 4, 4 * ND);
        auto positions = coef.ReinterpretCast<int32_t>();
        auto shifted = row.ReinterpretCast<int32_t>();
        auto bitMask = ab.ReinterpretCast<int16_t>();
        auto ci = colIndex.ReinterpretCast<int32_t>();
        auto ri = resultIndex.ReinterpretCast<int32_t>();
        auto ti = tileIndex.ReinterpretCast<int32_t>();
        auto ii = inverseIndex.ReinterpretCast<int32_t>();
        for (uint32_t i = 0; i < D; ++i) positions.SetValue(i, int32_t(i));
        Fence<HardEvent::S_V>();
        Muls(ci, positions, int32_t(RP * 4), D);
        Muls(ri, positions, int32_t(D * 4), D); PipeBarrier<PIPE_V>();
        UnaryRepeatParams repeat{1, 1, D / 8, 0};
        Muls(ti, positions, int32_t(4), uint64_t(D), uint8_t(N), repeat); PipeBarrier<PIPE_V>();
        Brcb(broad, colIndex.ReinterpretCast<float>(), D / 8, {1, 8}); PipeBarrier<PIPE_V>();
        Muls(ii, broad.ReinterpretCast<int32_t>(), int32_t(1), uint64_t(D), uint8_t(N), {1, 0, D / 8, 1}); PipeBarrier<PIPE_V>();
        Add(ii, ii, ti, ND); PipeBarrier<PIPE_V>();
        Adds(ii, ii, int32_t(D * 4), ND); PipeBarrier<PIPE_V>();
        Brcb(broad, resultIndex.ReinterpretCast<float>(), D / 8, {1, 8});
        Duplicate(bitMask, int16_t(15), 2 * D); PipeBarrier<PIPE_V>();
        for (uint32_t level = 0; level < 4; ++level) {
            const int32_t half = D >> (level + 1);
            Adds(shifted, positions, half, D); PipeBarrier<PIPE_V>();
            And(shifted.ReinterpretCast<int16_t>(), shifted.ReinterpretCast<int16_t>(), bitMask, 2 * D); PipeBarrier<PIPE_V>();
            auto si = shiftIndex[level * ND].ReinterpretCast<int32_t>();
            Muls(si, shifted, int32_t(4), uint64_t(D), uint8_t(N), repeat); PipeBarrier<PIPE_V>();
            Add(si, si, broad.ReinterpretCast<int32_t>(), uint64_t(D), uint8_t(N), {1, 1, 0, D / 8, D / 8, 1}); PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Process() {
        for (uint32_t batch = GetBlockIdx(); batch < t.batches; batch += t.blocks) {
            uint32_t coords = batch, ai = 0, bi = 0;
            for (int d = static_cast<int>(t.batchRank) - 1; d >= 0; --d) {
                uint32_t c = coords % t.batchDims[d]; coords /= t.batchDims[d];
                ai += c * t.aStrides[d]; bi += c * t.bStrides[d];
            }
            Duplicate(mem, 0.0f, TOTAL); PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            DataCopyPadExtParams<float> pad{true, 0, 0, 0};
            DataCopyExtParams pa{static_cast<uint16_t>(N), N * 4, 0, 0, 0};
            DataCopyExtParams pb{static_cast<uint16_t>(N), 4, 0, (RP - 8) / 8, 0};
            DataCopyExtParams pv{1, N * 4, 0, 0, 0};
            DataCopyPad(aa, ag[uint64_t(ai) * N * N], pa, pad);
            DataCopyPad(mat[2 * D], bg[uint64_t(bi) * N], pb, pad);
            DataCopyPad(bvalues, bg[uint64_t(bi) * N], pv, pad);
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
            Muls(mat, aa, 1.0f, uint64_t(D), uint8_t(N), {1, 1, RP / 8, D / 8});
            PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
            for (uint32_t i = 0; i < N; ++i) mat.SetValue(i * RP + D + i, 1.0f);
            Fence<HardEvent::S_V>();
            GaussJordan();
            AscendC::Gather(x, mat, colIndex, 2 * D * 4, N); PipeBarrier<PIPE_V>();
            AscendC::Gather(inverse, mat, inverseIndex, 0, ND); PipeBarrier<PIPE_V>();
            for (uint32_t iteration = 0; iteration < 1; ++iteration) {
                Residual();
                AscendC::Gather(xt, r, tileIndex, 0, ND); PipeBarrier<PIPE_V>();
                Mul(p, inverse, xt, ND); PipeBarrier<PIPE_V>();
                WholeReduceSum(delta, p, int32_t(D), int32_t(N), 1, 1, D / 8); PipeBarrier<PIPE_V>();
                Add(x, x, delta, D); PipeBarrier<PIPE_V>();
            }
            Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
            DataCopyExtParams out{1, N * 4, 0, 0, 0};
            DataCopyPad(xg[uint64_t(batch) * N], x, out);
            Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
        }
    }
private:
    __aicore__ inline void GaussJordan() {
        for (uint32_t j = 0; j < N; ++j) {
            AscendC::Gather(coef, mat, colIndex, j * 4, N); PipeBarrier<PIPE_V>();
            Abs(ab, coef, D); PipeBarrier<PIPE_V>();
            if (j) { Duplicate(ab, 0.0f, j); PipeBarrier<PIPE_V>(); }
            WholeReduceMax(maximum, ab, int32_t(N), 1, 1, 1, D / 8); PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();
            uint32_t pivot = maximum.ReinterpretCast<uint32_t>().GetValue(1);
            float diagonal = coef.GetValue(pivot);
            if (pivot != j) {
                coef.SetValue(pivot, coef.GetValue(j));
                DataCopy(row, mat[j * RP], RP); PipeBarrier<PIPE_V>();
                DataCopy(mat[j * RP], mat[pivot * RP], RP); PipeBarrier<PIPE_V>();
                DataCopy(mat[pivot * RP], row, RP); PipeBarrier<PIPE_V>();
            }
            Muls(mat[j * RP], mat[j * RP], 1.0f / diagonal, RP); PipeBarrier<PIPE_V>();
            DataCopy(row, mat[j * RP], RP); PipeBarrier<PIPE_V>();
            coef.SetValue(j, 0.0f); Fence<HardEvent::S_V>();
            Muls(coef, coef, -1.0f, D); PipeBarrier<PIPE_V>();
            Brcb(broad, coef, D / 8, {1, 8}); PipeBarrier<PIPE_V>();
            MulAddDst(mat, broad, row, uint64_t(RP), uint8_t(N), {1, 0, 1, RP / 8, 1, 0});
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Residual() {
        AscendC::Gather(xt, x, tileIndex, 0, ND); PipeBarrier<PIPE_V>();
        Muls(xh, xt, 4097.0f, ND); PipeBarrier<PIPE_V>();
        Sub(xl, xh, xt, ND); PipeBarrier<PIPE_V>();
        Sub(xh, xh, xl, ND); PipeBarrier<PIPE_V>();
        Sub(xl, xt, xh, ND); PipeBarrier<PIPE_V>();
        Mul(p, aa, xt, ND); PipeBarrier<PIPE_V>();
        Muls(temp, aa, 4097.0f, ND); PipeBarrier<PIPE_V>();
        Sub(al, temp, aa, ND); PipeBarrier<PIPE_V>();
        Sub(ah, temp, al, ND); PipeBarrier<PIPE_V>();
        Sub(al, aa, ah, ND); PipeBarrier<PIPE_V>();
        Mul(e, ah, xh, ND); PipeBarrier<PIPE_V>();
        Sub(e, p, e, ND); PipeBarrier<PIPE_V>();
        Mul(temp, al, xh, ND); PipeBarrier<PIPE_V>();
        Sub(e, e, temp, ND); PipeBarrier<PIPE_V>();
        Mul(temp, ah, xl, ND); PipeBarrier<PIPE_V>();
        Sub(e, e, temp, ND); PipeBarrier<PIPE_V>();
        Mul(temp, al, xl, ND); PipeBarrier<PIPE_V>();
        Sub(e, temp, e, ND); PipeBarrier<PIPE_V>();
        for (uint32_t level = 0; level < 4; ++level) {
            AscendC::Gather(h, p, shiftIndex[level * ND], 0, ND); PipeBarrier<PIPE_V>();
            AscendC::Gather(hl, e, shiftIndex[level * ND], 0, ND); PipeBarrier<PIPE_V>();
            Add(s, p, h, ND); PipeBarrier<PIPE_V>();
            Sub(z, s, p, ND); PipeBarrier<PIPE_V>();
            Sub(temp, s, z, ND); PipeBarrier<PIPE_V>();
            Sub(temp, p, temp, ND); PipeBarrier<PIPE_V>();
            Sub(z, h, z, ND); PipeBarrier<PIPE_V>();
            Add(temp, temp, z, ND); PipeBarrier<PIPE_V>();
            Add(low, e, hl, ND); PipeBarrier<PIPE_V>();
            Add(low, low, temp, ND); PipeBarrier<PIPE_V>();
            DataCopy(p, s, ND); PipeBarrier<PIPE_V>();
            DataCopy(e, low, ND); PipeBarrier<PIPE_V>();
        }
        AscendC::Gather(coef, p, resultIndex, 0, N); PipeBarrier<PIPE_V>();
        AscendC::Gather(ab, e, resultIndex, 0, N); PipeBarrier<PIPE_V>();
        Sub(r, bvalues, coef, D); PipeBarrier<PIPE_V>();
        Sub(z, bvalues, r, D); PipeBarrier<PIPE_V>();
        Add(temp, r, z, D); PipeBarrier<PIPE_V>();
        Sub(temp, bvalues, temp, D); PipeBarrier<PIPE_V>();
        Sub(z, z, coef, D); PipeBarrier<PIPE_V>();
        Add(temp, temp, z, D); PipeBarrier<PIPE_V>();
        Sub(temp, temp, ab, D); PipeBarrier<PIPE_V>();
        Add(r, r, temp, D); PipeBarrier<PIPE_V>();
    }
    LinalgSolveTilingData t;
    GlobalTensor<float> ag, bg, xg;
    LocalTensor<float> mem, mat, aa, p, e, s, z, temp, low, h, hl, ah, al, xt, xh, xl, inverse;
    LocalTensor<float> x, r, bvalues, delta, coef, ab, maximum, row, broad;
    LocalTensor<uint32_t> colIndex, resultIndex, tileIndex, inverseIndex, shiftIndex;
};

template <uint32_t N> class TinyExact {
    static constexpr uint32_t D = 16, RP = 40, ND = N * D;
    static constexpr uint32_t MA = 0, AA = N * RP, PP = AA + ND, EE = PP + ND;
    static constexpr uint32_t SS = EE + ND, ZZ = SS + ND, TT = ZZ + ND, LL = TT + ND;
    static constexpr uint32_t HH = LL + ND, HL = HH + ND, AH = HL + ND, AL = AH + ND;
    static constexpr uint32_t XT = AL + ND, XH = XT + ND, XL = XH + ND, IV = XL + ND;
    static constexpr uint32_t XX = IV + ND, RR = XX + D, BB = RR + D, DD = BB + D;
    static constexpr uint32_t CC = DD + D, AB = CC + D, MM = AB + D, RW = MM + 8;
    static constexpr uint32_t BC = RW + RP, TOTAL = BC + D * 8;
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR out) {
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        mem = LocalTensor<float>(TPosition::VECCALC, 0, TOTAL);
        mat = mem[MA]; aa = mem[AA]; p = mem[PP]; e = mem[EE]; s = mem[SS]; z = mem[ZZ];
        temp = mem[TT]; low = mem[LL]; h = mem[HH]; hl = mem[HL]; ah = mem[AH]; al = mem[AL];
        xt = mem[XT]; xh = mem[XH]; xl = mem[XL]; inverse = mem[IV];
        x = mem[XX]; r = mem[RR]; bvalues = mem[BB]; delta = mem[DD];
        coef = mem[CC]; ab = mem[AB]; maximum = mem[MM]; row = mem[RW]; broad = mem[BC];
        colIndex = LocalTensor<uint32_t>(TPosition::VECCALC, TOTAL * 4, D);
        resultIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + D) * 4, D);
        tileIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D) * 4, ND);
        inverseIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D + ND) * 4, ND);
        shiftIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D + 2 * ND) * 4, 4 * ND);
        auto positions = coef.ReinterpretCast<int32_t>();
        auto shifted = row.ReinterpretCast<int32_t>();
        auto bitMask = ab.ReinterpretCast<int16_t>();
        auto ci = colIndex.ReinterpretCast<int32_t>();
        auto ri = resultIndex.ReinterpretCast<int32_t>();
        auto ti = tileIndex.ReinterpretCast<int32_t>();
        auto ii = inverseIndex.ReinterpretCast<int32_t>();
        for (uint32_t i = 0; i < D; ++i) positions.SetValue(i, int32_t(i));
        Fence<HardEvent::S_V>();
        Muls(ci, positions, int32_t(RP * 4), D);
        Muls(ri, positions, int32_t(D * 4), D); PipeBarrier<PIPE_V>();
        UnaryRepeatParams repeat{1, 1, D / 8, 0};
        Muls(ti, positions, int32_t(4), uint64_t(D), uint8_t(N), repeat); PipeBarrier<PIPE_V>();
        Brcb(broad, colIndex.ReinterpretCast<float>(), D / 8, {1, 8}); PipeBarrier<PIPE_V>();
        Muls(ii, broad.ReinterpretCast<int32_t>(), int32_t(1), uint64_t(D), uint8_t(N), {1, 0, D / 8, 1}); PipeBarrier<PIPE_V>();
        Add(ii, ii, ti, ND); PipeBarrier<PIPE_V>();
        Adds(ii, ii, int32_t(D * 4), ND); PipeBarrier<PIPE_V>();
        Brcb(broad, resultIndex.ReinterpretCast<float>(), D / 8, {1, 8});
        Duplicate(bitMask, int16_t(15), 2 * D); PipeBarrier<PIPE_V>();
        for (uint32_t level = 0; level < 4; ++level) {
            const int32_t half = D >> (level + 1);
            Adds(shifted, positions, half, D); PipeBarrier<PIPE_V>();
            And(shifted.ReinterpretCast<int16_t>(), shifted.ReinterpretCast<int16_t>(), bitMask, 2 * D); PipeBarrier<PIPE_V>();
            auto si = shiftIndex[level * ND].ReinterpretCast<int32_t>();
            Muls(si, shifted, int32_t(4), uint64_t(D), uint8_t(N), repeat); PipeBarrier<PIPE_V>();
            Add(si, si, broad.ReinterpretCast<int32_t>(), uint64_t(D), uint8_t(N), {1, 1, 0, D / 8, D / 8, 1}); PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Process() {
        const uint32_t batch=GetBlockIdx();
            Duplicate(mem, 0.0f, TOTAL); PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            DataCopyPadExtParams<float> pad{true, 0, 0, 0};
            DataCopyExtParams pa{static_cast<uint16_t>(N), N * 4, 0, 0, 0};
            DataCopyExtParams pb{static_cast<uint16_t>(N), 4, 0, (RP - 8) / 8, 0};
            DataCopyExtParams pv{1, N * 4, 0, 0, 0};
            DataCopyPad(aa, ag[uint64_t(batch) * N * N], pa, pad);
            DataCopyPad(mat[2 * D], bg[uint64_t(batch) * N], pb, pad);
            DataCopyPad(bvalues, bg[uint64_t(batch) * N], pv, pad);
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
            Muls(mat, aa, 1.0f, uint64_t(D), uint8_t(N), {1, 1, RP / 8, D / 8});
            PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
            for (uint32_t i = 0; i < N; ++i) mat.SetValue(i * RP + D + i, 1.0f);
            Fence<HardEvent::S_V>();
            GaussJordan();
            AscendC::Gather(x, mat, colIndex, 2 * D * 4, N); PipeBarrier<PIPE_V>();
            AscendC::Gather(inverse, mat, inverseIndex, 0, ND); PipeBarrier<PIPE_V>();
            for (uint32_t iteration = 0; iteration < 1; ++iteration) {
                Residual();
                AscendC::Gather(xt, r, tileIndex, 0, ND); PipeBarrier<PIPE_V>();
                Mul(p, inverse, xt, ND); PipeBarrier<PIPE_V>();
                WholeReduceSum(delta, p, int32_t(D), int32_t(N), 1, 1, D / 8); PipeBarrier<PIPE_V>();
                Add(x, x, delta, D); PipeBarrier<PIPE_V>();
            }
            Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
            DataCopyExtParams out{1, N * 4, 0, 0, 0};
            DataCopyPad(xg[uint64_t(batch) * N], x, out);
            Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
    }
private:
    __aicore__ inline void GaussJordan() {
        for (uint32_t j = 0; j < N; ++j) {
            AscendC::Gather(coef, mat, colIndex, j * 4, N); PipeBarrier<PIPE_V>();
            Abs(ab, coef, D); PipeBarrier<PIPE_V>();
            if (j) { Duplicate(ab, 0.0f, j); PipeBarrier<PIPE_V>(); }
            WholeReduceMax(maximum, ab, int32_t(N), 1, 1, 1, D / 8); PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();
            uint32_t pivot = maximum.ReinterpretCast<uint32_t>().GetValue(1);
            float diagonal = coef.GetValue(pivot);
            if (pivot != j) {
                coef.SetValue(pivot, coef.GetValue(j));
                DataCopy(row, mat[j * RP], RP); PipeBarrier<PIPE_V>();
                DataCopy(mat[j * RP], mat[pivot * RP], RP); PipeBarrier<PIPE_V>();
                DataCopy(mat[pivot * RP], row, RP); PipeBarrier<PIPE_V>();
            }
            Muls(mat[j * RP], mat[j * RP], 1.0f / diagonal, RP); PipeBarrier<PIPE_V>();
            DataCopy(row, mat[j * RP], RP); PipeBarrier<PIPE_V>();
            coef.SetValue(j, 0.0f); Fence<HardEvent::S_V>();
            Muls(coef, coef, -1.0f, D); PipeBarrier<PIPE_V>();
            Brcb(broad, coef, D / 8, {1, 8}); PipeBarrier<PIPE_V>();
            MulAddDst(mat, broad, row, uint64_t(RP), uint8_t(N), {1, 0, 1, RP / 8, 1, 0});
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Residual() {
        AscendC::Gather(xt, x, tileIndex, 0, ND); PipeBarrier<PIPE_V>();
        Muls(xh, xt, 4097.0f, ND); PipeBarrier<PIPE_V>();
        Sub(xl, xh, xt, ND); PipeBarrier<PIPE_V>();
        Sub(xh, xh, xl, ND); PipeBarrier<PIPE_V>();
        Sub(xl, xt, xh, ND); PipeBarrier<PIPE_V>();
        Mul(p, aa, xt, ND); PipeBarrier<PIPE_V>();
        Muls(temp, aa, 4097.0f, ND); PipeBarrier<PIPE_V>();
        Sub(al, temp, aa, ND); PipeBarrier<PIPE_V>();
        Sub(ah, temp, al, ND); PipeBarrier<PIPE_V>();
        Sub(al, aa, ah, ND); PipeBarrier<PIPE_V>();
        Mul(e, ah, xh, ND); PipeBarrier<PIPE_V>();
        Sub(e, p, e, ND); PipeBarrier<PIPE_V>();
        Mul(temp, al, xh, ND); PipeBarrier<PIPE_V>();
        Sub(e, e, temp, ND); PipeBarrier<PIPE_V>();
        Mul(temp, ah, xl, ND); PipeBarrier<PIPE_V>();
        Sub(e, e, temp, ND); PipeBarrier<PIPE_V>();
        Mul(temp, al, xl, ND); PipeBarrier<PIPE_V>();
        Sub(e, temp, e, ND); PipeBarrier<PIPE_V>();
        for (uint32_t level = 0; level < 4; ++level) {
            AscendC::Gather(h, p, shiftIndex[level * ND], 0, ND); PipeBarrier<PIPE_V>();
            AscendC::Gather(hl, e, shiftIndex[level * ND], 0, ND); PipeBarrier<PIPE_V>();
            Add(s, p, h, ND); PipeBarrier<PIPE_V>();
            Sub(z, s, p, ND); PipeBarrier<PIPE_V>();
            Sub(temp, s, z, ND); PipeBarrier<PIPE_V>();
            Sub(temp, p, temp, ND); PipeBarrier<PIPE_V>();
            Sub(z, h, z, ND); PipeBarrier<PIPE_V>();
            Add(temp, temp, z, ND); PipeBarrier<PIPE_V>();
            Add(low, e, hl, ND); PipeBarrier<PIPE_V>();
            Add(low, low, temp, ND); PipeBarrier<PIPE_V>();
            DataCopy(p, s, ND); PipeBarrier<PIPE_V>();
            DataCopy(e, low, ND); PipeBarrier<PIPE_V>();
        }
        AscendC::Gather(coef, p, resultIndex, 0, N); PipeBarrier<PIPE_V>();
        AscendC::Gather(ab, e, resultIndex, 0, N); PipeBarrier<PIPE_V>();
        Sub(r, bvalues, coef, D); PipeBarrier<PIPE_V>();
        Sub(z, bvalues, r, D); PipeBarrier<PIPE_V>();
        Add(temp, r, z, D); PipeBarrier<PIPE_V>();
        Sub(temp, bvalues, temp, D); PipeBarrier<PIPE_V>();
        Sub(z, z, coef, D); PipeBarrier<PIPE_V>();
        Add(temp, temp, z, D); PipeBarrier<PIPE_V>();
        Sub(temp, temp, ab, D); PipeBarrier<PIPE_V>();
        Add(r, r, temp, D); PipeBarrier<PIPE_V>();
    }
    GlobalTensor<float> ag, bg, xg;
    LocalTensor<float> mem, mat, aa, p, e, s, z, temp, low, h, hl, ah, al, xt, xh, xl, inverse;
    LocalTensor<float> x, r, bvalues, delta, coef, ab, maximum, row, broad;
    LocalTensor<uint32_t> colIndex, resultIndex, tileIndex, inverseIndex, shiftIndex;
};

// Independent right-hand sides own a complete FP32 inverse and correction path.
template <uint32_t N> class TinyOriginalRows {
    static constexpr uint32_t D = 16, RP = 40, ND = N * D;
    static constexpr uint32_t MA = 0, AA = N * RP, PP = AA + ND, EE = PP + ND;
    static constexpr uint32_t SS = EE + ND, ZZ = SS + ND, TT = ZZ + ND, LL = TT + ND;
    static constexpr uint32_t HH = LL + ND, HL = HH + ND, AH = HL + ND, AL = AH + ND;
    static constexpr uint32_t XT = AL + ND, XH = XT + ND, XL = XH + ND, IV = XL + ND;
    static constexpr uint32_t XX = IV + ND, RR = XX + D, BB = RR + D, DD = BB + D;
    static constexpr uint32_t CC = DD + D, AB = CC + D, MM = AB + D, RW = MM + 8;
    static constexpr uint32_t BC = RW + RP, TOTAL = BC + D * 8;
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR out) {
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        mem = LocalTensor<float>(TPosition::VECCALC, 0, TOTAL);
        mat = mem[MA]; aa = mem[AA]; p = mem[PP]; e = mem[EE]; s = mem[SS]; z = mem[ZZ];
        temp = mem[TT]; low = mem[LL]; h = mem[HH]; hl = mem[HL]; ah = mem[AH]; al = mem[AL];
        xt = mem[XT]; xh = mem[XH]; xl = mem[XL]; inverse = mem[IV];
        x = mem[XX]; r = mem[RR]; bvalues = mem[BB]; delta = mem[DD];
        coef = mem[CC]; ab = mem[AB]; maximum = mem[MM]; row = mem[RW]; broad = mem[BC];
        colIndex = LocalTensor<uint32_t>(TPosition::VECCALC, TOTAL * 4, D);
        resultIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + D) * 4, D);
        tileIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D) * 4, ND);
        inverseIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D + ND) * 4, ND);
        shiftIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D + 2 * ND) * 4, 4 * ND);
        auto positions = coef.ReinterpretCast<int32_t>();
        auto shifted = row.ReinterpretCast<int32_t>();
        auto bitMask = ab.ReinterpretCast<int16_t>();
        auto ci = colIndex.ReinterpretCast<int32_t>();
        auto ri = resultIndex.ReinterpretCast<int32_t>();
        auto ti = tileIndex.ReinterpretCast<int32_t>();
        auto ii = inverseIndex.ReinterpretCast<int32_t>();
        for (uint32_t i = 0; i < D; ++i) positions.SetValue(i, int32_t(i));
        Fence<HardEvent::S_V>();
        Muls(ci, positions, int32_t(RP * 4), D);
        Muls(ri, positions, int32_t(D * 4), D); PipeBarrier<PIPE_V>();
        UnaryRepeatParams repeat{1, 1, D / 8, 0};
        Muls(ti, positions, int32_t(4), uint64_t(D), uint8_t(N), repeat); PipeBarrier<PIPE_V>();
        Brcb(broad, colIndex.ReinterpretCast<float>(), D / 8, {1, 8}); PipeBarrier<PIPE_V>();
        Muls(ii, broad.ReinterpretCast<int32_t>(), int32_t(1), uint64_t(D), uint8_t(N), {1, 0, D / 8, 1}); PipeBarrier<PIPE_V>();
        Add(ii, ii, ti, ND); PipeBarrier<PIPE_V>();
        Adds(ii, ii, int32_t(D * 4), ND); PipeBarrier<PIPE_V>();
        Brcb(broad, resultIndex.ReinterpretCast<float>(), D / 8, {1, 8});
        Duplicate(bitMask, int16_t(15), 2 * D); PipeBarrier<PIPE_V>();
        for (uint32_t level = 0; level < 4; ++level) {
            const int32_t half = D >> (level + 1);
            Adds(shifted, positions, half, D); PipeBarrier<PIPE_V>();
            And(shifted.ReinterpretCast<int16_t>(), shifted.ReinterpretCast<int16_t>(), bitMask, 2 * D); PipeBarrier<PIPE_V>();
            auto si = shiftIndex[level * ND].ReinterpretCast<int32_t>();
            Muls(si, shifted, int32_t(4), uint64_t(D), uint8_t(N), repeat); PipeBarrier<PIPE_V>();
            Add(si, si, broad.ReinterpretCast<int32_t>(), uint64_t(D), uint8_t(N), {1, 1, 0, D / 8, D / 8, 1}); PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Process() {
        const uint32_t batch=GetBlockIdx();
            Duplicate(mem, 0.0f, TOTAL); PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            DataCopyPadExtParams<float> pad{true, 0, 0, 0};
            DataCopyExtParams pa{static_cast<uint16_t>(N), N * 4, 0, 0, 0};
            DataCopyExtParams pb{static_cast<uint16_t>(N), 4, 0, (RP - 8) / 8, 0};
            DataCopyExtParams pv{1, N * 4, 0, 0, 0};
            DataCopyPad(aa, ag[uint64_t(batch) * N * N], pa, pad);
            DataCopyPad(mat[2 * D], bg[uint64_t(batch) * N], pb, pad);
            DataCopyPad(bvalues, bg[uint64_t(batch) * N], pv, pad);
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
            Muls(mat, aa, 1.0f, uint64_t(D), uint8_t(N), {1, 1, RP / 8, D / 8});
            PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
            for (uint32_t i = 0; i < N; ++i) mat.SetValue(i * RP + D + i, 1.0f);
            Fence<HardEvent::S_V>();
            Duplicate(xt,1.0f,D);PipeBarrier<PIPE_V>();GaussJordan();
            Sub(coef.ReinterpretCast<int32_t>(),delta.ReinterpretCast<int32_t>(),colIndex.ReinterpretCast<int32_t>(),D);PipeBarrier<PIPE_V>();
            Brcb(broad.ReinterpretCast<int32_t>(),coef.ReinterpretCast<int32_t>(),D/8,{1,8});PipeBarrier<PIPE_V>();
            Add(inverseIndex.ReinterpretCast<int32_t>(),inverseIndex.ReinterpretCast<int32_t>(),broad.ReinterpretCast<int32_t>(),uint64_t(D),uint8_t(N),{1,1,0,D/8,D/8,1});PipeBarrier<PIPE_V>();
            if constexpr(N<D){Duplicate(coef,0.0f,D);PipeBarrier<PIPE_V>();}
            AscendC::Gather(x, mat, delta.ReinterpretCast<uint32_t>(), 2 * D * 4, N); PipeBarrier<PIPE_V>();
            AscendC::Gather(inverse, mat, inverseIndex, 0, ND); PipeBarrier<PIPE_V>();
            for (uint32_t iteration = 0; iteration < 1; ++iteration) {
                Residual();
                AscendC::Gather(xt, r, tileIndex, 0, ND); PipeBarrier<PIPE_V>();
                Mul(p, inverse, xt, ND); PipeBarrier<PIPE_V>();
                WholeReduceSum(delta, p, int32_t(D), int32_t(N), 1, 1, D / 8); PipeBarrier<PIPE_V>();
                Add(x, x, delta, D); PipeBarrier<PIPE_V>();
            }
            Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
            DataCopyExtParams out{1, N * 4, 0, 0, 0};
            DataCopyPad(xg[uint64_t(batch) * N], x, out);
            Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
    }
private:
    __aicore__ inline void GaussJordan() {
        for (uint32_t j = 0; j < N; ++j) {
            AscendC::Gather(coef, mat, colIndex, j * 4, N); PipeBarrier<PIPE_V>();
            Abs(ab, coef, D); PipeBarrier<PIPE_V>();
            Mul(ab,ab,xt,D);PipeBarrier<PIPE_V>();
            WholeReduceMax(maximum, ab, int32_t(N), 1, 1, 1, D / 8); PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();
            uint32_t pivot = maximum.ReinterpretCast<uint32_t>().GetValue(1);
            float diagonal = coef.GetValue(pivot);
            Muls(mat[pivot * RP], mat[pivot * RP], 1.0f / diagonal, RP); PipeBarrier<PIPE_V>();
            DataCopy(row, mat[pivot * RP], RP); PipeBarrier<PIPE_V>();
            coef.SetValue(pivot, 0.0f);xt.SetValue(pivot,0.0f);delta.ReinterpretCast<uint32_t>().SetValue(j,pivot*RP*4);Fence<HardEvent::S_V>();
            Muls(coef, coef, -1.0f, D); PipeBarrier<PIPE_V>();
            Brcb(broad, coef, D / 8, {1, 8}); PipeBarrier<PIPE_V>();
            MulAddDst(mat, broad, row, uint64_t(RP), uint8_t(N), {1, 0, 1, RP / 8, 1, 0});
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Residual() {
        AscendC::Gather(xt, x, tileIndex, 0, ND); PipeBarrier<PIPE_V>();
        Muls(xh, xt, 4097.0f, ND); PipeBarrier<PIPE_V>();
        Sub(xl, xh, xt, ND); PipeBarrier<PIPE_V>();
        Sub(xh, xh, xl, ND); PipeBarrier<PIPE_V>();
        Sub(xl, xt, xh, ND); PipeBarrier<PIPE_V>();
        Mul(p, aa, xt, ND); PipeBarrier<PIPE_V>();
        Muls(temp, aa, 4097.0f, ND); PipeBarrier<PIPE_V>();
        Sub(al, temp, aa, ND); PipeBarrier<PIPE_V>();
        Sub(ah, temp, al, ND); PipeBarrier<PIPE_V>();
        Sub(al, aa, ah, ND); PipeBarrier<PIPE_V>();
        Mul(e, ah, xh, ND); PipeBarrier<PIPE_V>();
        Sub(e, p, e, ND); PipeBarrier<PIPE_V>();
        Mul(temp, al, xh, ND); PipeBarrier<PIPE_V>();
        Sub(e, e, temp, ND); PipeBarrier<PIPE_V>();
        Mul(temp, ah, xl, ND); PipeBarrier<PIPE_V>();
        Sub(e, e, temp, ND); PipeBarrier<PIPE_V>();
        Mul(temp, al, xl, ND); PipeBarrier<PIPE_V>();
        Sub(e, temp, e, ND); PipeBarrier<PIPE_V>();
        for (uint32_t level = 0; level < 4; ++level) {
            AscendC::Gather(h, p, shiftIndex[level * ND], 0, ND); PipeBarrier<PIPE_V>();
            AscendC::Gather(hl, e, shiftIndex[level * ND], 0, ND); PipeBarrier<PIPE_V>();
            Add(s, p, h, ND); PipeBarrier<PIPE_V>();
            Sub(z, s, p, ND); PipeBarrier<PIPE_V>();
            Sub(temp, s, z, ND); PipeBarrier<PIPE_V>();
            Sub(temp, p, temp, ND); PipeBarrier<PIPE_V>();
            Sub(z, h, z, ND); PipeBarrier<PIPE_V>();
            Add(temp, temp, z, ND); PipeBarrier<PIPE_V>();
            Add(low, e, hl, ND); PipeBarrier<PIPE_V>();
            Add(low, low, temp, ND); PipeBarrier<PIPE_V>();
            DataCopy(p, s, ND); PipeBarrier<PIPE_V>();
            DataCopy(e, low, ND); PipeBarrier<PIPE_V>();
        }
        AscendC::Gather(coef, p, resultIndex, 0, N); PipeBarrier<PIPE_V>();
        AscendC::Gather(ab, e, resultIndex, 0, N); PipeBarrier<PIPE_V>();
        Sub(r, bvalues, coef, D); PipeBarrier<PIPE_V>();
        Sub(z, bvalues, r, D); PipeBarrier<PIPE_V>();
        Add(temp, r, z, D); PipeBarrier<PIPE_V>();
        Sub(temp, bvalues, temp, D); PipeBarrier<PIPE_V>();
        Sub(z, z, coef, D); PipeBarrier<PIPE_V>();
        Add(temp, temp, z, D); PipeBarrier<PIPE_V>();
        Sub(temp, temp, ab, D); PipeBarrier<PIPE_V>();
        Add(r, r, temp, D); PipeBarrier<PIPE_V>();
    }
    GlobalTensor<float> ag, bg, xg;
    LocalTensor<float> mem, mat, aa, p, e, s, z, temp, low, h, hl, ah, al, xt, xh, xl, inverse;
    LocalTensor<float> x, r, bvalues, delta, coef, ab, maximum, row, broad;
    LocalTensor<uint32_t> colIndex, resultIndex, tileIndex, inverseIndex, shiftIndex;
};

// Independent right-hand sides own a complete FP32 inverse and correction path.
template <uint32_t N> class SmallCrout {
    static constexpr uint32_t D = 64, RP = D + 8, ND = N * D, NS = N * 8;
    static constexpr uint32_t AA = N * RP, PP = AA + ND, EE = PP + ND;
    static constexpr uint32_t SS = EE + ND, ZZ = SS + ND, TT = ZZ + ND, LL = TT + ND;
    static constexpr uint32_t HH = LL + ND, HL = HH + NS, SP = HL + NS, SE = SP + NS;
    static constexpr uint32_t XX = SE + NS, XH = XX + D, XL = XH + D, RR = XL + D;
    static constexpr uint32_t BB = RR + D, DD = BB + D, CC = DD + D, AB = CC + D;
    static constexpr uint32_t MM = AB + D, RW = MM + 8, BC = RW + RP, INV = BC + D * 8, TOTAL = INV + D;
    static_assert((TOTAL+4*D+3*NS)*4<=196608,"Static small LU must fit UB");
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR out, const LinalgSolveTilingData &data) {
        t = data;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a)); bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        mem = LocalTensor<float>(TPosition::VECCALC, 0, TOTAL);
        mat = mem; aa = mem[AA]; pp = mem[PP]; ee = mem[EE]; ss = mem[SS]; zz = mem[ZZ];
        tt = mem[TT]; ll = mem[LL]; hh = mem[HH]; hl = mem[HL]; smallP = mem[SP]; smallE = mem[SE];
        x = mem[XX]; xh = mem[XH]; xl = mem[XL]; r = mem[RR]; bvalues = mem[BB]; delta = mem[DD];
        invDiag=mem[INV];coef = mem[CC]; ab = mem[AB]; maximum = mem[MM]; row = mem[RW]; broad = mem[BC];
        colIndex = LocalTensor<uint32_t>(TPosition::VECCALC, TOTAL * 4, D);
        resultIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + D) * 4, D);
        shifts = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D) * 4, 3 * NS);
        permIndex=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+2*D+3*NS)*4,D);
        zeros=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+3*D+3*NS)*4,D);
        auto positions = coef.ReinterpretCast<int32_t>(); auto shifted = row.ReinterpretCast<int32_t>();
        for (uint32_t q = 0; q < D; ++q) positions.SetValue(q, int32_t(q));
        Fence<HardEvent::S_V>();
        Muls(colIndex.ReinterpretCast<int32_t>(), positions, int32_t(RP * 4), D);
        Muls(resultIndex.ReinterpretCast<int32_t>(), positions, int32_t(8 * 4), D); PipeBarrier<PIPE_V>();
        Brcb(broad, resultIndex.ReinterpretCast<float>(), D / 8, {1, 8});
        Duplicate(ab.ReinterpretCast<int16_t>(), int16_t(7), 16); PipeBarrier<PIPE_V>();
        for (uint32_t level = 0; level < 3; ++level) {
            Adds(shifted, positions, int32_t(4 >> level), 8); PipeBarrier<PIPE_V>();
            And(shifted.ReinterpretCast<int16_t>(), shifted.ReinterpretCast<int16_t>(), ab.ReinterpretCast<int16_t>(), 16); PipeBarrier<PIPE_V>();
            Muls(shifted, shifted, int32_t(4), 8); PipeBarrier<PIPE_V>();
            DataCopy(shifts[level * NS].ReinterpretCast<float>(), broad, NS); PipeBarrier<PIPE_V>();
            auto dest = shifts[level * NS].ReinterpretCast<int32_t>();
            Add(dest, dest, shifted, uint64_t(8), uint8_t(N), {1, 1, 1, 1, 1, 0}); PipeBarrier<PIPE_V>();
        }
        Muls(permIndex.ReinterpretCast<int32_t>(),positions,int32_t(4),D);Duplicate(zeros.ReinterpretCast<int32_t>(),int32_t(0),D);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() {
        const uint32_t batch = GetBlockIdx() / t.k, c = GetBlockIdx() % t.k;
        uint32_t coords = batch, ai = 0, bi = 0;
        for (int q = static_cast<int>(t.batchRank) - 1; q >= 0; --q) {
            uint32_t v = coords % t.batchDims[q]; coords /= t.batchDims[q];
            ai += v * t.aStrides[q]; bi += v * t.bStrides[q];
        }
        Duplicate(mem, 0.0f, TOTAL); PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
        DataCopyPadExtParams<float> pad{true, 0, 0, 0};
        DataCopyExtParams pa{N, N * 4, 0, (D - (N + 7) / 8 * 8) / 8, 0};
        DataCopyExtParams pb{N, 4, (t.k - 1) * 4, (RP - 8) / 8, 0};
        DataCopyPad(aa, ag[uint64_t(ai) * N * N], pa, pad);
        DataCopyPad(mat[D], bg[uint64_t(bi) * N * t.k + c], pb, pad);
        Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        Muls(mat, aa, 1.0f, uint64_t(D), uint8_t(N), {1, 1, RP / 8, D / 8}); PipeBarrier<PIPE_V>();
        AscendC::Gather(bvalues, mat, colIndex, D * 4, N); PipeBarrier<PIPE_V>();
        FactorNormalized();
        AscendC::Gather(x,mat,colIndex,D*4,N);PipeBarrier<PIPE_V>();
        Upper(x);Residual();
        Duplicate(delta,0.0f,D);PipeBarrier<PIPE_V>();AscendC::Gather(delta,r,permIndex,0,N);PipeBarrier<PIPE_V>();
        Mul(delta,delta,invDiag,D);PipeBarrier<PIPE_V>();
        Forward(delta);Upper(delta);Add(x,x,delta,D);PipeBarrier<PIPE_V>();
        Brcb(broad, x, D / 8, {1, 8}); PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
        DataCopyExtParams out{N, 4, 0, (t.k - 1) * 4, 0};
        DataCopyPad(xg[uint64_t(batch) * N * t.k + c], broad, out);
        Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
    }
private:
    __aicore__ inline void FactorNormalized(){
        for(uint32_t j=0;j<N;++j){
            AscendC::Gather(coef,mat,colIndex,j*4,N);PipeBarrier<PIPE_V>();Abs(ab,coef,D);PipeBarrier<PIPE_V>();
            if(j){Duplicate(ab,0.0f,j);PipeBarrier<PIPE_V>();}
            WholeReduceMax(maximum,ab,int32_t(N),1,1,1,D/8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
            uint32_t pivot=maximum.ReinterpretCast<uint32_t>().GetValue(1);float diagonal=coef.GetValue(pivot);
            const float reciprocal=1.0f/diagonal;invDiag.SetValue(j,reciprocal);
            if(pivot!=j){
                coef.SetValue(pivot,coef.GetValue(j));uint32_t original=permIndex.GetValue(j);permIndex.SetValue(j,permIndex.GetValue(pivot));permIndex.SetValue(pivot,original);
                DataCopy(row,mat[j*RP],RP);PipeBarrier<PIPE_V>();DataCopy(mat[j*RP],mat[pivot*RP],RP);PipeBarrier<PIPE_V>();DataCopy(mat[pivot*RP],row,RP);PipeBarrier<PIPE_V>();
            }
            // PA = D * Ls * U: the row's earlier lower entries are scaled by its own pivot.
            Fence<HardEvent::S_V>();Muls(mat[j*RP],mat[j*RP],reciprocal,RP);PipeBarrier<PIPE_V>();
            DataCopy(row,mat[j*RP],RP);PipeBarrier<PIPE_V>();
            Duplicate(row,0.0f,j+1);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,D);PipeBarrier<PIPE_V>();Brcb(broad,coef,D/8,{1,8});PipeBarrier<PIPE_V>();
            if(j+1<N)for(uint32_t off=0;off<RP;off+=64){
                MulAddDst(mat[(j+1)*RP+off],broad[(j+1)*8],row[off],uint64_t(RP-off<64?RP-off:64),uint8_t(N-j-1),{1,0,1,RP/8,1,0});PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void Upper(LocalTensor<float> rhs){
        for(int j=int(N)-1;j>0;--j){
            AscendC::Gather(coef,mat,colIndex,uint32_t(j)*4,N);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,D);PipeBarrier<PIPE_V>();
            AscendC::Gather(xh,rhs,zeros,uint32_t(j)*4,D);PipeBarrier<PIPE_V>();MulAddDst(rhs,coef,xh,uint64_t(j),uint8_t(1),{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Forward(LocalTensor<float> rhs){
        constexpr uint64_t all=(~0ULL)>>(64-N);
        for(uint32_t j=0;j+1<N;++j){
            AscendC::Gather(xh,rhs,zeros,j*4,D);PipeBarrier<PIPE_V>();
            AscendC::Gather(coef,mat,colIndex,j*4,N);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,D);PipeBarrier<PIPE_V>();
            uint64_t below[2]={all&(~0ULL<<(j+1)),0};MulAddDst(rhs,coef,xh,below,uint8_t(1),{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Product(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b) {
        Mul(dst, a, b, uint64_t(D), uint8_t(N), {1, 1, 1, D / 8, D / 8, 0}); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Residual() {
        Muls(xh, x, 4097.0f, D); PipeBarrier<PIPE_V>();
        Sub(xl, xh, x, D); PipeBarrier<PIPE_V>();
        Sub(xh, xh, xl, D); PipeBarrier<PIPE_V>();
        Sub(xl, x, xh, D); PipeBarrier<PIPE_V>();
        Product(pp, aa, x);
        Muls(tt, aa, 4097.0f, ND); PipeBarrier<PIPE_V>();
        Sub(zz, tt, aa, ND); PipeBarrier<PIPE_V>();
        Sub(ss, tt, zz, ND); PipeBarrier<PIPE_V>();
        Sub(zz, aa, ss, ND); PipeBarrier<PIPE_V>();
        Product(ee, ss, xh);
        Sub(ee, pp, ee, ND); PipeBarrier<PIPE_V>();
        Product(tt, zz, xh);
        Sub(ee, ee, tt, ND); PipeBarrier<PIPE_V>();
        Product(tt, ss, xl);
        Sub(ee, ee, tt, ND); PipeBarrier<PIPE_V>();
        Product(tt, zz, xl);
        Sub(ee, tt, ee, ND); PipeBarrier<PIPE_V>();
        BinaryRepeatParams reduce{1, 1, 1, D / 8, D / 8, D / 8};
        UnaryRepeatParams copy{1, 1, D / 8, D / 8};
        for (uint32_t span = D; span > 8; span >>= 1) {
            const uint32_t half = span >> 1;
            Add(ss, pp, pp[half], uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Sub(zz, ss, pp, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Sub(tt, ss, zz, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Sub(tt, pp, tt, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Sub(zz, pp[half], zz, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Add(tt, tt, zz, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Add(ll, ee, ee[half], uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Add(ll, ll, tt, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Muls(pp, ss, 1.0f, uint64_t(half), uint8_t(N), copy); PipeBarrier<PIPE_V>();
            Muls(ee, ll, 1.0f, uint64_t(half), uint8_t(N), copy); PipeBarrier<PIPE_V>();
        }
        DataCopyParams pack{N, 1, D / 8 - 1, 0};
        DataCopy(smallP, pp, pack); DataCopy(smallE, ee, pack); PipeBarrier<PIPE_V>();
        for (uint32_t level = 0; level < 3; ++level) {
            AscendC::Gather(hh, smallP, shifts[level * NS], 0, NS); PipeBarrier<PIPE_V>();
            AscendC::Gather(hl, smallE, shifts[level * NS], 0, NS); PipeBarrier<PIPE_V>();
            Add(ss, smallP, hh, NS); PipeBarrier<PIPE_V>();
            Sub(zz, ss, smallP, NS); PipeBarrier<PIPE_V>();
            Sub(tt, ss, zz, NS); PipeBarrier<PIPE_V>();
            Sub(tt, smallP, tt, NS); PipeBarrier<PIPE_V>();
            Sub(zz, hh, zz, NS); PipeBarrier<PIPE_V>();
            Add(tt, tt, zz, NS); PipeBarrier<PIPE_V>();
            Add(ll, smallE, hl, NS); PipeBarrier<PIPE_V>();
            Add(ll, ll, tt, NS); PipeBarrier<PIPE_V>();
            DataCopy(smallP, ss, NS); DataCopy(smallE, ll, NS); PipeBarrier<PIPE_V>();
        }
        AscendC::Gather(hh, smallP, resultIndex, 0, N); PipeBarrier<PIPE_V>();
        AscendC::Gather(hl, smallE, resultIndex, 0, N); PipeBarrier<PIPE_V>();
        Sub(r, bvalues, hh, D); PipeBarrier<PIPE_V>();
        Sub(zz, bvalues, r, D); PipeBarrier<PIPE_V>();
        Add(tt, r, zz, D); PipeBarrier<PIPE_V>();
        Sub(tt, bvalues, tt, D); PipeBarrier<PIPE_V>();
        Sub(zz, zz, hh, D); PipeBarrier<PIPE_V>();
        Add(tt, tt, zz, D); PipeBarrier<PIPE_V>();
        Sub(tt, tt, hl, D); PipeBarrier<PIPE_V>();
        Add(r, r, tt, D); PipeBarrier<PIPE_V>();
    }
    LinalgSolveTilingData t;
    GlobalTensor<float> ag, bg, xg;
    LocalTensor<float> mem, mat, aa, pp, ee, ss, zz, tt, ll, hh, hl, smallP, smallE;
    LocalTensor<float> x, xh, xl, r, bvalues, delta, coef, ab, maximum, row, broad,invDiag;
    LocalTensor<uint32_t> colIndex, resultIndex, shifts,permIndex,zeros;
};

template <uint32_t N> class SmallComplete {
    static constexpr uint32_t D = 64, RP = D + 8, ND = N * D, NS = N * 8;
    static constexpr uint32_t AA = N * RP, PP = AA + ND, EE = PP + ND;
    static constexpr uint32_t SS = EE + ND, ZZ = SS + ND, TT = ZZ + ND, LL = TT + ND;
    static constexpr uint32_t HH = LL + ND, HL = HH + NS, SP = HL + NS, SE = SP + NS;
    static constexpr uint32_t XX = SE + NS, XH = XX + D, XL = XH + D, RR = XL + D;
    static constexpr uint32_t BB = RR + D, DD = BB + D, CC = DD + D, AB = CC + D;
    static constexpr uint32_t MM = AB + D, RW = MM + 8, BC = RW + RP, INV = BC + D * 8, TOTAL = INV + D;
    static_assert((TOTAL+6*D+3*NS)*4<=196608,"Static small LU must fit UB");
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR out) {
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a)); bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        mem = LocalTensor<float>(TPosition::VECCALC, 0, TOTAL);
        mat = mem; aa = mem[AA]; pp = mem[PP]; ee = mem[EE]; ss = mem[SS]; zz = mem[ZZ];
        tt = mem[TT]; ll = mem[LL]; hh = mem[HH]; hl = mem[HL]; smallP = mem[SP]; smallE = mem[SE];
        x = mem[XX]; xh = mem[XH]; xl = mem[XL]; r = mem[RR]; bvalues = mem[BB]; delta = mem[DD];
        invDiag=mem[INV];coef = mem[CC]; ab = mem[AB]; maximum = mem[MM]; row = mem[RW]; broad = mem[BC];
        colIndex = LocalTensor<uint32_t>(TPosition::VECCALC, TOTAL * 4, D);
        resultIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + D) * 4, D);
        shifts = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D) * 4, 3 * NS);
        permIndex=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+2*D+3*NS)*4,D);
        zeros=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+3*D+3*NS)*4,D);
        columnPerm=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+4*D+3*NS)*4,D);inverseColumn=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+5*D+3*NS)*4,D);
        auto positions = coef.ReinterpretCast<int32_t>(); auto shifted = row.ReinterpretCast<int32_t>();
        for (uint32_t q = 0; q < D; ++q) positions.SetValue(q, int32_t(q));
        Fence<HardEvent::S_V>();
        Muls(colIndex.ReinterpretCast<int32_t>(), positions, int32_t(RP * 4), D);
        Muls(resultIndex.ReinterpretCast<int32_t>(), positions, int32_t(8 * 4), D); PipeBarrier<PIPE_V>();
        Brcb(broad, resultIndex.ReinterpretCast<float>(), D / 8, {1, 8});
        Duplicate(ab.ReinterpretCast<int16_t>(), int16_t(7), 16); PipeBarrier<PIPE_V>();
        for (uint32_t level = 0; level < 3; ++level) {
            Adds(shifted, positions, int32_t(4 >> level), 8); PipeBarrier<PIPE_V>();
            And(shifted.ReinterpretCast<int16_t>(), shifted.ReinterpretCast<int16_t>(), ab.ReinterpretCast<int16_t>(), 16); PipeBarrier<PIPE_V>();
            Muls(shifted, shifted, int32_t(4), 8); PipeBarrier<PIPE_V>();
            DataCopy(shifts[level * NS].ReinterpretCast<float>(), broad, NS); PipeBarrier<PIPE_V>();
            auto dest = shifts[level * NS].ReinterpretCast<int32_t>();
            Add(dest, dest, shifted, uint64_t(8), uint8_t(N), {1, 1, 1, 1, 1, 0}); PipeBarrier<PIPE_V>();
        }
        Muls(permIndex.ReinterpretCast<int32_t>(),positions,int32_t(4),D);Duplicate(zeros.ReinterpretCast<int32_t>(),int32_t(0),D);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() {
        const uint32_t batch=GetBlockIdx()/2,c=GetBlockIdx()%2;
        Duplicate(mem, 0.0f, TOTAL); PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
        DataCopyPadExtParams<float> pad{true, 0, 0, 0};
        DataCopyExtParams pa{N, N * 4, 0, (D - (N + 7) / 8 * 8) / 8, 0};
        DataCopyExtParams pb{N, 4, (2 - 1) * 4, (RP - 8) / 8, 0};
        DataCopyPad(aa, ag[uint64_t(batch) * N * N], pa, pad);
        DataCopyPad(mat[D], bg[uint64_t(batch) * N * 2 + c], pb, pad);
        Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        Muls(mat, aa, 1.0f, uint64_t(D), uint8_t(N), {1, 1, RP / 8, D / 8}); PipeBarrier<PIPE_V>();
        AscendC::Gather(bvalues, mat, colIndex, D * 4, N); PipeBarrier<PIPE_V>();
        FactorNormalized();
        for(uint32_t q=0;q<D;++q)inverseColumn.SetValue(columnPerm.GetValue(q)/4,q*4);Fence<HardEvent::S_V>();
        AscendC::Gather(x,mat,colIndex,D*4,N);PipeBarrier<PIPE_V>();
        Upper(x);Duplicate(delta,0.0f,D);PipeBarrier<PIPE_V>();AscendC::Gather(delta,x,inverseColumn,0,N);PipeBarrier<PIPE_V>();DataCopy(x,delta,D);PipeBarrier<PIPE_V>();Residual();
        Duplicate(delta,0.0f,D);PipeBarrier<PIPE_V>();AscendC::Gather(delta,r,permIndex,0,N);PipeBarrier<PIPE_V>();
        Mul(delta,delta,invDiag,D);PipeBarrier<PIPE_V>();
        Forward(delta);Upper(delta);Duplicate(coef,0.0f,D);PipeBarrier<PIPE_V>();AscendC::Gather(coef,delta,inverseColumn,0,N);PipeBarrier<PIPE_V>();Add(x,x,coef,D);PipeBarrier<PIPE_V>();
        Brcb(broad, x, D / 8, {1, 8}); PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
        DataCopyExtParams out{N, 4, 0, (2 - 1) * 4, 0};
        DataCopyPad(xg[uint64_t(batch) * N * 2 + c], broad, out);
        Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
    }
private:
    __aicore__ inline void FactorNormalized(){
        for(uint32_t q=0;q<D;++q)columnPerm.SetValue(q,q*4);Fence<HardEvent::S_V>();
        for(uint32_t j=0;j<N;++j){
            Muls(pp,mat,1.0f,uint64_t(D),uint8_t(N),{1,1,D/8,RP/8});PipeBarrier<PIPE_V>();Abs(pp,pp,ND);PipeBarrier<PIPE_V>();
            if(j){Duplicate(pp,0.0f,j*D);PipeBarrier<PIPE_V>();for(uint32_t rr=j;rr<N;++rr){Duplicate(pp[rr*D],0.0f,j);PipeBarrier<PIPE_V>();}}
            Duplicate(ss,0.0f,2*D);PipeBarrier<PIPE_V>();WholeReduceMax(ss,pp,int32_t(D),uint8_t(N),1,1,D/8);PipeBarrier<PIPE_V>();
            WholeReduceMax(maximum,ss,int32_t(D),uint8_t(2),1,1,D/8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
            uint32_t half=maximum.GetValue(2)>maximum.GetValue(0)?1:0;
            uint32_t pivot=(half*D+maximum.ReinterpretCast<uint32_t>().GetValue(half*2+1))/2;
            uint32_t pivotColumn=ss.ReinterpretCast<uint32_t>().GetValue(pivot*2+1);
            if(pivot!=j){
                uint32_t original=permIndex.GetValue(j);permIndex.SetValue(j,permIndex.GetValue(pivot));permIndex.SetValue(pivot,original);
                DataCopy(row,mat[j*RP],RP);PipeBarrier<PIPE_V>();DataCopy(mat[j*RP],mat[pivot*RP],RP);PipeBarrier<PIPE_V>();DataCopy(mat[pivot*RP],row,RP);PipeBarrier<PIPE_V>();
            }
            if(pivotColumn!=j){
                AscendC::Gather(coef,mat,colIndex,j*4,N);AscendC::Gather(ab,mat,colIndex,pivotColumn*4,N);PipeBarrier<PIPE_V>();
                Brcb(broad,coef,D/8,{1,8});PipeBarrier<PIPE_V>();uint64_t pm[2]={1ULL<<(pivotColumn&7),0};
                Muls(mat[pivotColumn/8*8],broad,1.0f,pm,uint8_t(N),{1,1,RP/8,1});PipeBarrier<PIPE_V>();
                Brcb(broad,ab,D/8,{1,8});PipeBarrier<PIPE_V>();uint64_t jm[2]={1ULL<<(j&7),0};
                Muls(mat[j/8*8],broad,1.0f,jm,uint8_t(N),{1,1,RP/8,1});PipeBarrier<PIPE_V>();
                uint32_t old=columnPerm.GetValue(j);columnPerm.SetValue(j,columnPerm.GetValue(pivotColumn));columnPerm.SetValue(pivotColumn,old);
            }
            AscendC::Gather(coef,mat,colIndex,j*4,N);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();float diagonal=coef.GetValue(j);const float reciprocal=1.0f/diagonal;invDiag.SetValue(j,reciprocal);
            Fence<HardEvent::S_V>();Muls(mat[j*RP],mat[j*RP],reciprocal,RP);PipeBarrier<PIPE_V>();DataCopy(row,mat[j*RP],RP);PipeBarrier<PIPE_V>();
            Duplicate(row,0.0f,j+1);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,D);PipeBarrier<PIPE_V>();Brcb(broad,coef,D/8,{1,8});PipeBarrier<PIPE_V>();
            if(j+1<N)for(uint32_t off=0;off<RP;off+=64){
                MulAddDst(mat[(j+1)*RP+off],broad[(j+1)*8],row[off],uint64_t(RP-off<64?RP-off:64),uint8_t(N-j-1),{1,0,1,RP/8,1,0});PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void Upper(LocalTensor<float> rhs){
        for(int j=int(N)-1;j>0;--j){
            AscendC::Gather(coef,mat,colIndex,uint32_t(j)*4,N);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,D);PipeBarrier<PIPE_V>();
            AscendC::Gather(xh,rhs,zeros,uint32_t(j)*4,D);PipeBarrier<PIPE_V>();MulAddDst(rhs,coef,xh,uint64_t(j),uint8_t(1),{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Forward(LocalTensor<float> rhs){
        constexpr uint64_t all=(~0ULL)>>(64-N);
        for(uint32_t j=0;j+1<N;++j){
            AscendC::Gather(xh,rhs,zeros,j*4,D);PipeBarrier<PIPE_V>();
            AscendC::Gather(coef,mat,colIndex,j*4,N);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,D);PipeBarrier<PIPE_V>();
            uint64_t below[2]={all&(~0ULL<<(j+1)),0};MulAddDst(rhs,coef,xh,below,uint8_t(1),{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Product(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b) {
        Mul(dst, a, b, uint64_t(D), uint8_t(N), {1, 1, 1, D / 8, D / 8, 0}); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Residual() {
        Muls(xh, x, 4097.0f, D); PipeBarrier<PIPE_V>();
        Sub(xl, xh, x, D); PipeBarrier<PIPE_V>();
        Sub(xh, xh, xl, D); PipeBarrier<PIPE_V>();
        Sub(xl, x, xh, D); PipeBarrier<PIPE_V>();
        Product(pp, aa, x);
        Muls(tt, aa, 4097.0f, ND); PipeBarrier<PIPE_V>();
        Sub(zz, tt, aa, ND); PipeBarrier<PIPE_V>();
        Sub(ss, tt, zz, ND); PipeBarrier<PIPE_V>();
        Sub(zz, aa, ss, ND); PipeBarrier<PIPE_V>();
        Product(ee, ss, xh);
        Sub(ee, pp, ee, ND); PipeBarrier<PIPE_V>();
        Product(tt, zz, xh);
        Sub(ee, ee, tt, ND); PipeBarrier<PIPE_V>();
        Product(tt, ss, xl);
        Sub(ee, ee, tt, ND); PipeBarrier<PIPE_V>();
        Product(tt, zz, xl);
        Sub(ee, tt, ee, ND); PipeBarrier<PIPE_V>();
        BinaryRepeatParams reduce{1, 1, 1, D / 8, D / 8, D / 8};
        UnaryRepeatParams copy{1, 1, D / 8, D / 8};
        for (uint32_t span = D; span > 8; span >>= 1) {
            const uint32_t half = span >> 1;
            Add(ss, pp, pp[half], uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Sub(zz, ss, pp, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Sub(tt, ss, zz, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Sub(tt, pp, tt, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Sub(zz, pp[half], zz, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Add(tt, tt, zz, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Add(ll, ee, ee[half], uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Add(ll, ll, tt, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Muls(pp, ss, 1.0f, uint64_t(half), uint8_t(N), copy); PipeBarrier<PIPE_V>();
            Muls(ee, ll, 1.0f, uint64_t(half), uint8_t(N), copy); PipeBarrier<PIPE_V>();
        }
        DataCopyParams pack{N, 1, D / 8 - 1, 0};
        DataCopy(smallP, pp, pack); DataCopy(smallE, ee, pack); PipeBarrier<PIPE_V>();
        for (uint32_t level = 0; level < 3; ++level) {
            AscendC::Gather(hh, smallP, shifts[level * NS], 0, NS); PipeBarrier<PIPE_V>();
            AscendC::Gather(hl, smallE, shifts[level * NS], 0, NS); PipeBarrier<PIPE_V>();
            Add(ss, smallP, hh, NS); PipeBarrier<PIPE_V>();
            Sub(zz, ss, smallP, NS); PipeBarrier<PIPE_V>();
            Sub(tt, ss, zz, NS); PipeBarrier<PIPE_V>();
            Sub(tt, smallP, tt, NS); PipeBarrier<PIPE_V>();
            Sub(zz, hh, zz, NS); PipeBarrier<PIPE_V>();
            Add(tt, tt, zz, NS); PipeBarrier<PIPE_V>();
            Add(ll, smallE, hl, NS); PipeBarrier<PIPE_V>();
            Add(ll, ll, tt, NS); PipeBarrier<PIPE_V>();
            DataCopy(smallP, ss, NS); DataCopy(smallE, ll, NS); PipeBarrier<PIPE_V>();
        }
        AscendC::Gather(hh, smallP, resultIndex, 0, N); PipeBarrier<PIPE_V>();
        AscendC::Gather(hl, smallE, resultIndex, 0, N); PipeBarrier<PIPE_V>();
        Sub(r, bvalues, hh, D); PipeBarrier<PIPE_V>();
        Sub(zz, bvalues, r, D); PipeBarrier<PIPE_V>();
        Add(tt, r, zz, D); PipeBarrier<PIPE_V>();
        Sub(tt, bvalues, tt, D); PipeBarrier<PIPE_V>();
        Sub(zz, zz, hh, D); PipeBarrier<PIPE_V>();
        Add(tt, tt, zz, D); PipeBarrier<PIPE_V>();
        Sub(tt, tt, hl, D); PipeBarrier<PIPE_V>();
        Add(r, r, tt, D); PipeBarrier<PIPE_V>();
    }
    GlobalTensor<float> ag, bg, xg;
    LocalTensor<float> mem, mat, aa, pp, ee, ss, zz, tt, ll, hh, hl, smallP, smallE;
    LocalTensor<float> x, xh, xl, r, bvalues, delta, coef, ab, maximum, row, broad,invDiag;
    LocalTensor<uint32_t> colIndex, resultIndex, shifts,permIndex,zeros,columnPerm,inverseColumn;
};


template <uint32_t N> class SmallExact {
    static constexpr uint32_t D = 64, RP = D + 8, ND = N * D, NS = N * 8;
    static constexpr uint32_t AA = N * RP, PP = AA + ND, EE = PP + ND;
    static constexpr uint32_t SS = EE + ND, ZZ = SS + ND, TT = ZZ + ND, LL = TT + ND;
    static constexpr uint32_t HH = LL + ND, HL = HH + NS, SP = HL + NS, SE = SP + NS;
    static constexpr uint32_t XX = SE + NS, XH = XX + D, XL = XH + D, RR = XL + D;
    static constexpr uint32_t BB = RR + D, DD = BB + D, CC = DD + D, AB = CC + D;
    static constexpr uint32_t MM = AB + D, RW = MM + 8, BC = RW + RP, INV = BC + D * 8, TOTAL = INV + D;
    static_assert((TOTAL+4*D+3*NS)*4<=196608,"Static small LU must fit UB");
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR out) {
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a)); bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        mem = LocalTensor<float>(TPosition::VECCALC, 0, TOTAL);
        mat = mem; aa = mem[AA]; pp = mem[PP]; ee = mem[EE]; ss = mem[SS]; zz = mem[ZZ];
        tt = mem[TT]; ll = mem[LL]; hh = mem[HH]; hl = mem[HL]; smallP = mem[SP]; smallE = mem[SE];
        x = mem[XX]; xh = mem[XH]; xl = mem[XL]; r = mem[RR]; bvalues = mem[BB]; delta = mem[DD];
        invDiag=mem[INV];coef = mem[CC]; ab = mem[AB]; maximum = mem[MM]; row = mem[RW]; broad = mem[BC];
        colIndex = LocalTensor<uint32_t>(TPosition::VECCALC, TOTAL * 4, D);
        resultIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + D) * 4, D);
        shifts = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D) * 4, 3 * NS);
        permIndex=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+2*D+3*NS)*4,D);
        zeros=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+3*D+3*NS)*4,D);
        auto positions = coef.ReinterpretCast<int32_t>(); auto shifted = row.ReinterpretCast<int32_t>();
        for (uint32_t q = 0; q < D; ++q) positions.SetValue(q, int32_t(q));
        Fence<HardEvent::S_V>();
        Muls(colIndex.ReinterpretCast<int32_t>(), positions, int32_t(RP * 4), D);
        Muls(resultIndex.ReinterpretCast<int32_t>(), positions, int32_t(8 * 4), D); PipeBarrier<PIPE_V>();
        Brcb(broad, resultIndex.ReinterpretCast<float>(), D / 8, {1, 8});
        Duplicate(ab.ReinterpretCast<int16_t>(), int16_t(7), 16); PipeBarrier<PIPE_V>();
        for (uint32_t level = 0; level < 3; ++level) {
            Adds(shifted, positions, int32_t(4 >> level), 8); PipeBarrier<PIPE_V>();
            And(shifted.ReinterpretCast<int16_t>(), shifted.ReinterpretCast<int16_t>(), ab.ReinterpretCast<int16_t>(), 16); PipeBarrier<PIPE_V>();
            Muls(shifted, shifted, int32_t(4), 8); PipeBarrier<PIPE_V>();
            DataCopy(shifts[level * NS].ReinterpretCast<float>(), broad, NS); PipeBarrier<PIPE_V>();
            auto dest = shifts[level * NS].ReinterpretCast<int32_t>();
            Add(dest, dest, shifted, uint64_t(8), uint8_t(N), {1, 1, 1, 1, 1, 0}); PipeBarrier<PIPE_V>();
        }
        Muls(permIndex.ReinterpretCast<int32_t>(),positions,int32_t(4),D);Duplicate(zeros.ReinterpretCast<int32_t>(),int32_t(0),D);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool Process() {
        const uint32_t batch=GetBlockIdx()/2,c=GetBlockIdx()%2;
        Duplicate(mem, 0.0f, TOTAL); PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
        DataCopyPadExtParams<float> pad{true, 0, 0, 0};
        DataCopyExtParams pa{N, N * 4, 0, (D - (N + 7) / 8 * 8) / 8, 0};
        DataCopyExtParams pb{N, 4, (2 - 1) * 4, (RP - 8) / 8, 0};
        DataCopyPad(aa, ag[uint64_t(batch) * N * N], pa, pad);
        DataCopyPad(mat[D], bg[uint64_t(batch) * N * 2 + c], pb, pad);
        Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        Muls(mat, aa, 1.0f, uint64_t(D), uint8_t(N), {1, 1, RP / 8, D / 8}); PipeBarrier<PIPE_V>();
        AscendC::Gather(bvalues, mat, colIndex, D * 4, N); PipeBarrier<PIPE_V>();
        FactorNormalized();
        AscendC::Gather(x,mat,colIndex,D*4,N);PipeBarrier<PIPE_V>();
        Upper(x);Residual();
        Duplicate(delta,0.0f,D);PipeBarrier<PIPE_V>();AscendC::Gather(delta,r,permIndex,0,N);PipeBarrier<PIPE_V>();
        Mul(delta,delta,invDiag,D);PipeBarrier<PIPE_V>();
        Forward(delta);Upper(delta);Add(x,x,delta,D);PipeBarrier<PIPE_V>();
        Abs(ab,delta,D);Abs(coef,x,D);PipeBarrier<PIPE_V>();Adds(coef,coef,1.0f,D);PipeBarrier<PIPE_V>();Div(ab,ab,coef,D);PipeBarrier<PIPE_V>();
        WholeReduceMax(maximum,ab,int32_t(N),1,1,1,D/8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();float change=maximum.GetValue(0);
        if(!(change>=0.0f && change<=.1f))return false;
        Brcb(broad, x, D / 8, {1, 8}); PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
        DataCopyExtParams out{N, 4, 0, (2 - 1) * 4, 0};
        DataCopyPad(xg[uint64_t(batch) * N * 2 + c], broad, out);
        Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
        return true;
    }
private:
    __aicore__ inline void FactorNormalized(){
        for(uint32_t j=0;j<N;++j){
            AscendC::Gather(coef,mat,colIndex,j*4,N);PipeBarrier<PIPE_V>();Abs(ab,coef,D);PipeBarrier<PIPE_V>();
            if(j){Duplicate(ab,0.0f,j);PipeBarrier<PIPE_V>();}
            WholeReduceMax(maximum,ab,int32_t(N),1,1,1,D/8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
            uint32_t pivot=maximum.ReinterpretCast<uint32_t>().GetValue(1);float diagonal=coef.GetValue(pivot);
            const float reciprocal=1.0f/diagonal;invDiag.SetValue(j,reciprocal);
            if(pivot!=j){
                coef.SetValue(pivot,coef.GetValue(j));uint32_t original=permIndex.GetValue(j);permIndex.SetValue(j,permIndex.GetValue(pivot));permIndex.SetValue(pivot,original);
                DataCopy(row,mat[j*RP],RP);PipeBarrier<PIPE_V>();DataCopy(mat[j*RP],mat[pivot*RP],RP);PipeBarrier<PIPE_V>();DataCopy(mat[pivot*RP],row,RP);PipeBarrier<PIPE_V>();
            }
            // PA = D * Ls * U: the row's earlier lower entries are scaled by its own pivot.
            Fence<HardEvent::S_V>();Muls(mat[j*RP],mat[j*RP],reciprocal,RP);PipeBarrier<PIPE_V>();
            DataCopy(row,mat[j*RP],RP);PipeBarrier<PIPE_V>();
            Duplicate(row,0.0f,j+1);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,D);PipeBarrier<PIPE_V>();Brcb(broad,coef,D/8,{1,8});PipeBarrier<PIPE_V>();
            if(j+1<N)for(uint32_t off=0;off<RP;off+=64){
                MulAddDst(mat[(j+1)*RP+off],broad[(j+1)*8],row[off],uint64_t(RP-off<64?RP-off:64),uint8_t(N-j-1),{1,0,1,RP/8,1,0});PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void Upper(LocalTensor<float> rhs){
        for(int j=int(N)-1;j>0;--j){
            AscendC::Gather(coef,mat,colIndex,uint32_t(j)*4,N);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,D);PipeBarrier<PIPE_V>();
            AscendC::Gather(xh,rhs,zeros,uint32_t(j)*4,D);PipeBarrier<PIPE_V>();MulAddDst(rhs,coef,xh,uint64_t(j),uint8_t(1),{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Forward(LocalTensor<float> rhs){
        constexpr uint64_t all=(~0ULL)>>(64-N);
        for(uint32_t j=0;j+1<N;++j){
            AscendC::Gather(xh,rhs,zeros,j*4,D);PipeBarrier<PIPE_V>();
            AscendC::Gather(coef,mat,colIndex,j*4,N);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,D);PipeBarrier<PIPE_V>();
            uint64_t below[2]={all&(~0ULL<<(j+1)),0};MulAddDst(rhs,coef,xh,below,uint8_t(1),{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Product(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b) {
        Mul(dst, a, b, uint64_t(D), uint8_t(N), {1, 1, 1, D / 8, D / 8, 0}); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Residual() {
        Muls(xh, x, 4097.0f, D); PipeBarrier<PIPE_V>();
        Sub(xl, xh, x, D); PipeBarrier<PIPE_V>();
        Sub(xh, xh, xl, D); PipeBarrier<PIPE_V>();
        Sub(xl, x, xh, D); PipeBarrier<PIPE_V>();
        Product(pp, aa, x);
        Muls(tt, aa, 4097.0f, ND); PipeBarrier<PIPE_V>();
        Sub(zz, tt, aa, ND); PipeBarrier<PIPE_V>();
        Sub(ss, tt, zz, ND); PipeBarrier<PIPE_V>();
        Sub(zz, aa, ss, ND); PipeBarrier<PIPE_V>();
        Product(ee, ss, xh);
        Sub(ee, pp, ee, ND); PipeBarrier<PIPE_V>();
        Product(tt, zz, xh);
        Sub(ee, ee, tt, ND); PipeBarrier<PIPE_V>();
        Product(tt, ss, xl);
        Sub(ee, ee, tt, ND); PipeBarrier<PIPE_V>();
        Product(tt, zz, xl);
        Sub(ee, tt, ee, ND); PipeBarrier<PIPE_V>();
        BinaryRepeatParams reduce{1, 1, 1, D / 8, D / 8, D / 8};
        UnaryRepeatParams copy{1, 1, D / 8, D / 8};
        for (uint32_t span = D; span > 8; span >>= 1) {
            const uint32_t half = span >> 1;
            Add(ss, pp, pp[half], uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Sub(zz, ss, pp, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Sub(tt, ss, zz, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Sub(tt, pp, tt, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Sub(zz, pp[half], zz, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Add(tt, tt, zz, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Add(ll, ee, ee[half], uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Add(ll, ll, tt, uint64_t(half), uint8_t(N), reduce); PipeBarrier<PIPE_V>();
            Muls(pp, ss, 1.0f, uint64_t(half), uint8_t(N), copy); PipeBarrier<PIPE_V>();
            Muls(ee, ll, 1.0f, uint64_t(half), uint8_t(N), copy); PipeBarrier<PIPE_V>();
        }
        DataCopyParams pack{N, 1, D / 8 - 1, 0};
        DataCopy(smallP, pp, pack); DataCopy(smallE, ee, pack); PipeBarrier<PIPE_V>();
        for (uint32_t level = 0; level < 3; ++level) {
            AscendC::Gather(hh, smallP, shifts[level * NS], 0, NS); PipeBarrier<PIPE_V>();
            AscendC::Gather(hl, smallE, shifts[level * NS], 0, NS); PipeBarrier<PIPE_V>();
            Add(ss, smallP, hh, NS); PipeBarrier<PIPE_V>();
            Sub(zz, ss, smallP, NS); PipeBarrier<PIPE_V>();
            Sub(tt, ss, zz, NS); PipeBarrier<PIPE_V>();
            Sub(tt, smallP, tt, NS); PipeBarrier<PIPE_V>();
            Sub(zz, hh, zz, NS); PipeBarrier<PIPE_V>();
            Add(tt, tt, zz, NS); PipeBarrier<PIPE_V>();
            Add(ll, smallE, hl, NS); PipeBarrier<PIPE_V>();
            Add(ll, ll, tt, NS); PipeBarrier<PIPE_V>();
            DataCopy(smallP, ss, NS); DataCopy(smallE, ll, NS); PipeBarrier<PIPE_V>();
        }
        AscendC::Gather(hh, smallP, resultIndex, 0, N); PipeBarrier<PIPE_V>();
        AscendC::Gather(hl, smallE, resultIndex, 0, N); PipeBarrier<PIPE_V>();
        Sub(r, bvalues, hh, D); PipeBarrier<PIPE_V>();
        Sub(zz, bvalues, r, D); PipeBarrier<PIPE_V>();
        Add(tt, r, zz, D); PipeBarrier<PIPE_V>();
        Sub(tt, bvalues, tt, D); PipeBarrier<PIPE_V>();
        Sub(zz, zz, hh, D); PipeBarrier<PIPE_V>();
        Add(tt, tt, zz, D); PipeBarrier<PIPE_V>();
        Sub(tt, tt, hl, D); PipeBarrier<PIPE_V>();
        Add(r, r, tt, D); PipeBarrier<PIPE_V>();
    }
    GlobalTensor<float> ag, bg, xg;
    LocalTensor<float> mem, mat, aa, pp, ee, ss, zz, tt, ll, hh, hl, smallP, smallE;
    LocalTensor<float> x, xh, xl, r, bvalues, delta, coef, ab, maximum, row, broad,invDiag;
    LocalTensor<uint32_t> colIndex, resultIndex, shifts,permIndex,zeros;
};

// Two column panels retain FP32 partial-pivot LU in UB. The owner publishes
// batched elementary row operations; the other panel applies the same pivots.
template <bool Right> class ColumnFactor {
    static constexpr uint32_t N = 256, C = 128, CP = Right ? 136 : 128;
    static constexpr uint32_t ROW = N * CP, OTHER = ROW + CP, COEF = OTHER + CP;
    static constexpr uint32_t ABS = COEF + N, MAX = ABS + N, COL = MAX + 8;
    static constexpr uint32_t INDEX = COL + 8 * N, LOG = INDEX + N;
    static constexpr uint32_t PIV = LOG + 8 * (N + 8), TOTAL = PIV + 8;
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR workspace, const LinalgSolveTilingData &data) {
        t = data;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        const uint64_t elements = uint64_t(N) * (t.nPad + 2 * t.kPad);
        base.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace))
                             + uint64_t(GetBlockIdx() / 2) * 2 * elements);
        logs.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace))
                             + uint64_t(GetBlockIdx() / 2) * 2 * elements + elements);
        pg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(GetUserWorkspace(workspace))
                          + uint64_t(GetBlockIdx() / 2) * 2 * elements + N * t.rowPitch);
        mem = LocalTensor<float>(TPosition::VECCALC, 0, TOTAL);
        mat = mem; row = mem[ROW]; other = mem[OTHER]; coef = mem[COEF];
        ab = mem[ABS]; maximum = mem[MAX]; column = mem[COL];
        index = mem[INDEX].ReinterpretCast<uint32_t>(); record = mem[LOG];
        pivotValues = mem[PIV].ReinterpretCast<int32_t>();
        for (uint32_t i = 0; i < N; ++i) index.SetValue(i, i * CP * 4);
        Fence<HardEvent::S_V>();
        uint32_t coords = GetBlockIdx() / 2, ai = 0, bi = 0;
        for (int d = static_cast<int>(t.batchRank) - 1; d >= 0; --d) {
            uint32_t q = coords % t.batchDims[d]; coords /= t.batchDims[d];
            ai += q * t.aStrides[d]; bi += q * t.bStrides[d];
        }
        Duplicate(mat, 0.0f, N * CP); PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
        DataCopyExtParams pa{N, C * 4, (N - C) * 4, (CP - C) / 8, 0};
        DataCopyPadExtParams<float> pad{true, 0, 0, 0};
        DataCopyPad(mat, ag[uint64_t(ai) * N * N + (Right ? C : 0)], pa, pad);
        if constexpr (Right) {
            DataCopyExtParams pb{N, t.k * 4, 0, C / 8, 0};
            DataCopyPad(mat[C], bg[uint64_t(bi) * N * t.k], pb, pad);
        }
        Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline void Process() {
        for (uint32_t chunk = 0; chunk < C / 8; ++chunk) {
            if constexpr (!Right) {
                Duplicate(record, 0.0f, 8 * (N + 8)); PipeBarrier<PIPE_V>();
                for (uint32_t q = 0; q < 8; ++q) Factor(chunk * 8 + q, q, true);
                Send(logs[(chunk & 1) * 8 * (N + 8)], record, 8 * (N + 8));
                Fence<HardEvent::S_MTE3>(); DataCopy(pg[chunk * 8], pivotValues, 8); Fence<HardEvent::MTE3_S>();
            }
            SyncAll<true>();
            if constexpr (Right) {
                Receive(record, logs[(chunk & 1) * 8 * (N + 8)], 8 * (N + 8));
                for (uint32_t q = 0; q < 8; ++q) {
                    uint32_t j = chunk * 8 + q;
                    uint32_t pivot = record.ReinterpretCast<uint32_t>().GetValue(q * (N + 8));
                    Swap(j, pivot);
                    DataCopy(coef, record[q * (N + 8) + 8], N); PipeBarrier<PIPE_V>();
                    DataCopy(row, mat[j * CP], CP); PipeBarrier<PIPE_V>();
                    Eliminate(j, 0, false);
                }
            }
        }
        for (uint32_t chunk = 0; chunk < C / 8; ++chunk) {
            if constexpr (Right) {
                for (uint32_t q = 0; q < 8; ++q) Factor(C + chunk * 8 + q, q, false);
                Fence<HardEvent::S_MTE3>(); DataCopy(pg[C + chunk * 8], pivotValues, 8); Fence<HardEvent::MTE3_S>();
            }
            SyncAll<true>();
            if constexpr (!Right) {
                Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
                DataCopy(pivotValues, pg[C + chunk * 8], 8); Fence<HardEvent::MTE2_S>();
                for (uint32_t q = 0; q < 8; ++q) Swap(C + chunk * 8 + q, pivotValues.GetValue(q));
            }
        }
        Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
        DataCopyExtParams out{N, CP * 4, 0, (t.rowPitch - CP) * 4, 0};
        DataCopyPad(base[Right ? C : 0], mat, out);
        Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
        SyncAll<true>();
    }
private:
    __aicore__ inline void Send(GlobalTensor<float> dst, LocalTensor<float> src, uint32_t count) {
        Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>(); DataCopy(dst, src, count);
        Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
    }
    __aicore__ inline void Receive(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t count) {
        Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>(); Fence<HardEvent::MTE3_MTE2>();
        DataCopy(dst, src, count); Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline void Swap(uint32_t j, uint32_t pivot) {
        if (j == pivot) return;
        Fence<HardEvent::S_V>();
        DataCopy(row, mat[j * CP], CP); PipeBarrier<PIPE_V>();
        DataCopy(other, mat[pivot * CP], CP); PipeBarrier<PIPE_V>();
        DataCopy(mat[pivot * CP], row, CP); DataCopy(mat[j * CP], other, CP); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Eliminate(uint32_t j, uint32_t local, bool retain) {
        Brcb(column, coef, N / 8, {1, 8}); PipeBarrier<PIPE_V>();
        if (j + 1 == N) return;
        for (uint32_t offset = 0; offset < CP; offset += 64) {
            const uint64_t mask = CP - offset < 64 ? CP - offset : 64;
            MulAddDst(mat[(j + 1) * CP + offset], column[(j + 1) * 8], row[offset], mask,
                      static_cast<uint8_t>(N - j - 1), {1, 0, 1, CP / 8, 1, 0});
            PipeBarrier<PIPE_V>();
        }
        if (retain) {
            uint64_t mask[2] = {1ULL << (local & 7), 0};
            Muls(mat[(j + 1) * CP + (local / 8) * 8], column[(j + 1) * 8], -1.0f, mask,
                 static_cast<uint8_t>(N - j - 1), {1, 1, CP / 8, 1}); PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Factor(uint32_t j, uint32_t slot, bool publish) {
        const uint32_t local = Right ? j - C : j;
        AscendC::Gather(coef, mat, index, local * 4, N); PipeBarrier<PIPE_V>();
        Abs(ab, coef, N); PipeBarrier<PIPE_V>();
        if (j) { Duplicate(ab, 0.0f, j); PipeBarrier<PIPE_V>(); }
        AscendC::ReduceMax(maximum, ab, column, int32_t(N), true); PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_S>();
        uint32_t pivot = maximum.ReinterpretCast<uint32_t>().GetValue(1);
        float diagonal = coef.GetValue(pivot);
        if (pivot != j) { coef.SetValue(pivot, coef.GetValue(j)); coef.SetValue(j, diagonal); }
        pivotValues.SetValue(slot, pivot);
        if (publish) record.ReinterpretCast<uint32_t>().SetValue(slot * (N + 8), pivot);
        Fence<HardEvent::S_V>(); Swap(j, pivot);
        DataCopy(row, mat[j * CP], CP); PipeBarrier<PIPE_V>();
        Duplicate(row, 0.0f, local + 1); PipeBarrier<PIPE_V>();
        Muls(coef, coef, -1.0f / diagonal, N); PipeBarrier<PIPE_V>();
        if (publish) { DataCopy(record[slot * (N + 8) + 8], coef, N); PipeBarrier<PIPE_V>(); }
        Eliminate(j, local, true);
    }
    TPipe pipe;
    LinalgSolveTilingData t;
    LocalTensor<float> mem, mat, row, other, coef, ab, maximum, column, record;
    LocalTensor<uint32_t> index;
    LocalTensor<int32_t> pivotValues;
    GlobalTensor<float> ag, bg, base, logs;
    GlobalTensor<int32_t> pg;
};

// FP32 right-looking blocked LU, one vector/cube pair per system.
class BlockedFactor {
    static constexpr uint32_t N=200, RP=208, ELEMENTS=N*216;
    static constexpr uint32_t ROW=N*RP, OTHER=ROW+RP, COEF=OTHER+RP;
    static constexpr uint32_t ABS=COEF+N, MAX=ABS+256, COL=MAX+8;
    static constexpr uint32_t IDX=COL+8*N, PIV=IDX+N, TEMP=PIV+N;
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR workspace,const LinalgSolveTilingData &data) {
        static_assert((TEMP+16*RP)*4<=192*1024,"N200 padded argmax fits UB withoutSelect");
        t=data;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a)); bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        auto base=reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace));
        wg.SetGlobalBuffer(base+uint64_t(GetBlockIdx())*ELEMENTS);
        product.SetGlobalBuffer(base+uint64_t(t.blocks)*ELEMENTS+uint64_t(GetBlockIdx())*N*RP);
        if ASCEND_IS_AIV {
            mem=LocalTensor<float>(TPosition::VECCALC,0,TEMP+16*RP); mat=mem; row=mem[ROW]; other=mem[OTHER];
            coef=mem[COEF]; ab=mem[ABS]; maximum=mem[MAX]; column=mem[COL]; temp=mem[TEMP];
            index=mem[IDX].ReinterpretCast<uint32_t>(); piv=mem[PIV].ReinterpretCast<int32_t>();
            for(uint32_t i=0;i<N;++i) index.SetValue(i,i*RP*4);
            Fence<HardEvent::S_V>(); Duplicate(mat,0.0f,N*RP);Duplicate(ab,0.0f,256); PipeBarrier<PIPE_V>();
            uint32_t coords=t.rhsSplit ? GetBlockIdx()/t.k : GetBlockIdx(),ai=0,bi=0;
            for(int d=int(t.batchRank)-1;d>=0;--d) { uint32_t q=coords%t.batchDims[d]; coords/=t.batchDims[d];ai+=q*t.aStrides[d];bi+=q*t.bStrides[d]; }
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            DataCopyPadExtParams<float> pad{true,0,0,0};
            DataCopyExtParams pa{N,N*4,0,1,0},pb{N,t.k*4,0,N/8,0};
            DataCopyPad(mat,ag[uint64_t(ai)*N*N],pa,pad); DataCopyPad(mat[N],bg[uint64_t(bi)*N*t.k],pb,pad);
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        }
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC {
            SetHF32Mode(HF32Mode::DISABLE);
            for(uint32_t p=0;p+16<N;p+=16) {
                uint32_t end=p+16; CrossCoreWaitFlag<2>(0);
                for(uint32_t r=end;r<N;r+=128) for(uint32_t c=end;c<RP;c+=128)
                    Multiply(p,r,c,N-r<128?N-r:128,RP-c<128?RP-c:128);
                CrossCoreSetFlag<2,PIPE_FIX>(1);
            }
        } else {
            for(uint32_t p=0;p<N;p+=16) {
                uint32_t end=p+16<N?p+16:N;
                for(uint32_t j=p;j<end;++j) PanelPivot(p,end,j);
                // Solve the panel's U12 block; RHS columns are part of the right tail.
                for(uint32_t j=p;j<end;++j) {
                    AscendC::Gather(coef,mat,index,j*4,N); PipeBarrier<PIPE_V>();
                    Muls(coef,coef,-1.0f,N); PipeBarrier<PIPE_V>(); Brcb(column,coef,N/8,{1,8}); PipeBarrier<PIPE_V>();
                    DataCopy(row,mat[j*RP],RP); PipeBarrier<PIPE_V>();
                    if(j+1<end) for(uint32_t c=end;c<RP;c+=64) {
                        MulAddDst(mat[(j+1)*RP+c],column[(j+1)*8],row[c],uint64_t(RP-c<64?RP-c:64),uint8_t(end-j-1),{1,0,1,RP/8,1,0});
                        PipeBarrier<PIPE_V>();
                    }
                }
                if(end<N) {
                    Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
                    DataCopyExtParams pl{uint16_t(N-end),16*4,(RP-16)/8,(RP-16)*4,0};
                    DataCopyExtParams pu{16,(RP-end)*4,end/8,end*4,0};
                    DataCopyPad(wg[end*RP+p],mat[end*RP+p],pl); DataCopyPad(wg[p*RP+end],mat[p*RP+end],pu);
                    CrossCoreSetFlag<2,PIPE_MTE3>(0); CrossCoreWaitFlag<2>(1);
                    for(uint32_t first=end;first<N;first+=16) {
                        uint32_t rows=N-first<16?N-first:16;
                        Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
                        DataCopyExtParams pc{uint16_t(rows),(RP-end)*4,end*4,end/8,0};
                        DataCopyPadExtParams<float> pad{false,0,0,0};
                        DataCopyPad(temp,product[first*RP+end],pc,pad); Fence<HardEvent::MTE2_V>();
                        for(uint32_t c=0;c<RP-end;c+=64) {
                            Sub(mat[first*RP+end+c],mat[first*RP+end+c],temp[c],uint64_t(RP-end-c<64?RP-end-c:64),uint8_t(rows),{1,1,1,RP/8,RP/8,RP/8});
                            PipeBarrier<PIPE_V>();
                        }
                    }
                }
            }
            Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
            DataCopy(wg,mat,N*RP); DataCopy(wg[N*RP],piv.ReinterpretCast<float>(),N);
            Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
        }
    }
private:
    __aicore__ inline void PanelPivot(uint32_t p,uint32_t end,uint32_t j) {
        AscendC::Gather(coef,mat,index,j*4,N); PipeBarrier<PIPE_V>(); Abs(ab,coef,N); PipeBarrier<PIPE_V>();
        if(j) { Duplicate(ab,0.0f,j); PipeBarrier<PIPE_V>(); }
        WholeReduceMax(maximum,ab,int32_t(64),uint8_t(4),1,1,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
        uint32_t maximumBits,pivot;GetReduceRepeatMaxMinSpr<float>(maximumBits,pivot);float diagonal=coef.GetValue(pivot);
        if(pivot!=j) { coef.SetValue(pivot,coef.GetValue(j));coef.SetValue(j,diagonal); }
        piv.SetValue(j,pivot); Fence<HardEvent::S_V>();
        DataCopy(row,mat[j*RP],RP); PipeBarrier<PIPE_V>();
        if(pivot!=j) {
            DataCopy(other,mat[pivot*RP],RP); PipeBarrier<PIPE_V>();
            DataCopy(mat[pivot*RP],row,RP); DataCopy(mat[j*RP],other,RP); PipeBarrier<PIPE_V>();
            DataCopy(row,other,RP); PipeBarrier<PIPE_V>();
        }
        Duplicate(row[p],0.0f,j-p+1); PipeBarrier<PIPE_V>();
        Muls(coef,coef,-1.0f/diagonal,N); PipeBarrier<PIPE_V>(); Brcb(column,coef,N/8,{1,8}); PipeBarrier<PIPE_V>();
        if(j+1<N) {
            MulAddDst(mat[(j+1)*RP+p],column[(j+1)*8],row[p],uint64_t(end-p),uint8_t(N-j-1),{1,0,1,RP/8,1,0}); PipeBarrier<PIPE_V>();
            uint64_t mask[2]={1ULL<<(j&7),0};
            Muls(mat[(j+1)*RP+(j/8)*8],column[(j+1)*8],-1.0f,mask,uint8_t(N-j-1),{1,1,RP/8,1}); PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Multiply(uint32_t p,uint32_t r,uint32_t c,uint32_t m,uint32_t n) {
        uint32_t mp=(m+15)/16*16,np=(n+15)/16*16;
        LocalTensor<float> a1(TPosition::A1,0,128*16), b1(TPosition::B1,128*16*4,128*16);
        LocalTensor<float> a2(TPosition::A2,0,128*16), b2(TPosition::B2,0,128*16), cc(TPosition::CO1,0,128*128);
        Nd2NzParams pa{1,uint16_t(m),16,0,RP,uint16_t(mp),1,0};
        Nd2NzParams pb{1,16,uint16_t(n),0,RP,16,1,0};
        DataCopy(a1,wg[r*RP+p],pa); DataCopy(b1,wg[p*RP+c],pb);
        Fence<HardEvent::MTE2_MTE1>();
        LoadData2DParams la;la.startIndex=0;la.repeatTimes=2;la.srcStride=mp/16;la.dstGap=0;la.ifTranspose=false;
        for(uint32_t i=0;i<mp/16;++i) LoadData(a2[i*256],a1[i*128],la);
        LoadData2dTransposeParams lb;lb.startIndex=0;lb.repeatTimes=np/16;lb.srcStride=1;lb.dstGap=0;lb.dstFracGap=np/16-1;
        LoadDataWithTranspose(b2,b1,lb); Fence<HardEvent::MTE1_M>();
        MmadParams mm;mm.m=m;mm.n=n;mm.k=16;mm.cmatrixInitVal=true;Mmad(cc,a2,b2,mm);
        Fence<HardEvent::M_FIX>();
        FixpipeParamsV220 fp;fp.ndNum=1;fp.srcNdStride=0;fp.dstNdStride=0;fp.mSize=m;fp.nSize=n;fp.srcStride=mp;fp.dstStride=RP;
        Fixpipe(product[r*RP+c],cc,fp); Fence<HardEvent::FIX_MTE2>();
    }
    TPipe pipe;
    LinalgSolveTilingData t;
    GlobalTensor<float> ag,bg,wg,product;
    LocalTensor<float> mem,mat,row,other,coef,ab,maximum,column,temp;
    LocalTensor<uint32_t> index; LocalTensor<int32_t> piv;
};

// Full partial pivoting with two resident column partitions and a shared Cube.
template<bool Right> class PairBlockedFactor {
    static constexpr uint32_t N=256,C=128,CP=Right?136:128,RP=264,ELEMENTS=N*272;
    static constexpr uint32_t ROW=N*CP,OTHER=ROW+CP,COEF=OTHER+CP,ABS=COEF+N,MAX=ABS+N;
    static constexpr uint32_t COL=MAX+8,IDX=COL+8*N,PIV=IDX+N,REC=PIV+N,TEMP=REC+288,OLDTOTAL=TEMP+16*CP,PANEL=OLDTOTAL,PANELIDX=PANEL+N*16,TOTAL=PANELIDX+128;
    static_assert(TOTAL*4<=196608,"Pair LU must fit 910B3 192KiB UB");
public:
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR B,GM_ADDR workspace,const LinalgSolveTilingData &data){
        t=data;uint32_t batch;
        if ASCEND_IS_AIC {batch=GetBlockIdx();}else{batch=GetBlockIdx()/2;}
        auto user=reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace));
        wg.SetGlobalBuffer(user+uint64_t(batch)*2*ELEMENTS);logs.SetGlobalBuffer(user+uint64_t(batch)*2*ELEMENTS+ELEMENTS);
        pg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(user)+uint64_t(batch)*2*ELEMENTS+N*RP);
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A));bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(B));
        if ASCEND_IS_AIV {
            mem=LocalTensor<float>(TPosition::VECCALC,0,TOTAL);mat=mem;row=mem[ROW];other=mem[OTHER];coef=mem[COEF];ab=mem[ABS];maximum=mem[MAX];
            column=mem[COL];index=mem[IDX].ReinterpretCast<uint32_t>();pivots=mem[PIV].ReinterpretCast<int32_t>();record=mem[REC];temp=mem[TEMP];
            panelIndex=record[272].ReinterpretCast<uint32_t>();panel=mem[PANEL];packedIndex=mem[PANELIDX].ReinterpretCast<uint32_t>();
            for(uint32_t r=0;r<8;++r)for(uint32_t q=0;q<16;++q)packedIndex.SetValue(r*16+q,(q*N+r)*4);
            for(uint32_t i=0;i<N;++i)index.SetValue(i,i*CP*4);
            for(uint32_t i=0;i<16;++i)panelIndex.SetValue(i,i*16*4);
            uint32_t coords=t.rhsSplit?batch/t.k:batch,ai=0,bi=0;
            for(int d=int(t.batchRank)-1;d>=0;--d){uint32_t q=coords%t.batchDims[d];coords/=t.batchDims[d];ai+=q*t.aStrides[d];bi+=q*t.bStrides[d];}
            Fence<HardEvent::S_V>();Duplicate(mat,0.0f,N*CP);PipeBarrier<PIPE_V>();Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
            DataCopyExtParams pa{N,C*4,(N-C)*4,(CP-C)/8,0};DataCopyPadExtParams<float>pad{true,0,0,0};
            DataCopyPad(mat,ag[uint64_t(ai)*N*N+(Right?C:0)],pa,pad);
            if constexpr(Right){DataCopyExtParams pb{N,t.k*4,0,C/8,0};DataCopyPad(mat[C],bg[uint64_t(bi)*N*t.k],pb,pad);}
            Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        }
    }
    __aicore__ inline void Process(){
        if ASCEND_IS_AIC {
            SetHF32Mode(HF32Mode::DISABLE);
            for(uint32_t p=0;p+16<N;p+=16){
                uint32_t end=p+16;CrossCoreWaitFlag<2>(0);
                for(uint32_t r=end;r<N;r+=128)Multiply(p,r,end,N-r<128?N-r:128,16);
                CrossCoreSetFlag<2,PIPE_FIX>(2);
                for(uint32_t r=end;r<N;r+=128)for(uint32_t c=end+16;c<RP;c+=128)
                    Multiply(p,r,c,N-r<128?N-r:128,RP-c<128?RP-c:128);
                CrossCoreSetFlag<2,PIPE_FIX>(1);
            }
        }else{
            constexpr uint32_t c0=Right?C:0;bool prepared=false;
            for(uint32_t p=0;p<N;p+=16){
                uint32_t end=p+16;bool owner=(p/C)==(Right?1:0);
                if(owner){
                    if(!prepared){for(uint32_t j=p;j<end;++j)PanelPivot(p,j);}else prepared=false;
                    DataCopyParams q{16,2,CP/8-2,0};DataCopy(record[16],mat[p*CP+p-c0],q);PipeBarrier<PIPE_V>();
                    Send(logs,record,272);
                }
                PairSync();
                if(!owner){
                    Receive(record,logs,272);
                    for(uint32_t j=p;j<end;++j){uint32_t pivot=record.ReinterpretCast<uint32_t>().GetValue(j-p);pivots.SetValue(j,pivot);Swap(j,pivot);}
                }
                uint32_t firstColumn=end>c0?end-c0:0;
                if(firstColumn<CP){
                    for(uint32_t j=p;j+1<end;++j){
                        AscendC::Gather(coef,record[16],panelIndex,(j-p)*4,16);PipeBarrier<PIPE_V>();
                        Muls(coef,coef,-1.0f,16);PipeBarrier<PIPE_V>();Brcb(column,coef,2,{1,8});PipeBarrier<PIPE_V>();
                        DataCopy(row,mat[j*CP],CP);PipeBarrier<PIPE_V>();
                        for(uint32_t c=firstColumn;c<CP;c+=64){
                            MulAddDst(mat[(j+1)*CP+c],column[(j+1-p)*8],row[c],uint64_t(CP-c<64?CP-c:64),uint8_t(end-j-1),{1,0,1,CP/8,1,0});PipeBarrier<PIPE_V>();
                        }
                    }
                }
                if(end<N){
                    Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();
                    if(owner){DataCopyExtParams pl{uint16_t(N-end),16*4,(CP-16)/8,(RP-16)*4,0};DataCopyPad(wg[end*RP+p],mat[end*CP+p-c0],pl);}
                    if(firstColumn<CP){uint32_t tail=CP-firstColumn;DataCopyExtParams pu{16,tail*4,firstColumn/8,(RP-tail)*4,0};DataCopyPad(wg[p*RP+c0+firstColumn],mat[p*CP+firstColumn],pu);}
                    CrossCoreSetFlag<2,PIPE_MTE3>(0);CrossCoreWaitFlag<2>(2);
                    bool nextOwner=(end/C)==(Right?1:0);
                    if(nextOwner){
                        Subtract(end,end-c0,16);
                        for(uint32_t q=0;q<16;++q){AscendC::Gather(panel[q*N],mat,index,(end-c0+q)*4,N);PipeBarrier<PIPE_V>();}
                        for(uint32_t j=end;j<end+16;++j)PackedPivot(end,j);
                    }
                    CrossCoreWaitFlag<2>(1);
                    uint32_t rest=end+16>c0?end+16-c0:0;
                    if(rest<CP)Subtract(end,rest,CP-rest);
                    if(nextOwner){
                        for(uint32_t j=end;j<end+16;++j)Swap(j,pivots.GetValue(j));
                        for(uint32_t first=0;first<N;first+=8){
                            AscendC::Gather(column,panel,packedIndex,first*4,128);PipeBarrier<PIPE_V>();
                            Muls(mat[first*CP+end-c0],column,1.0f,uint64_t(16),uint8_t(8),{1,1,CP/8,2});PipeBarrier<PIPE_V>();
                        }
                        prepared=true;
                    }
                }
            }
            Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();
            DataCopyExtParams out{N,CP*4,0,(RP-CP)*4,0};DataCopyPad(wg[Right?C:0],mat,out);
            if constexpr(!Right)DataCopy(pg,pivots,N);
            Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();PairSync();
        }
    }
private:
    __aicore__ inline void PairSync(){
        CrossCoreSetFlag<1,PIPE_MTE3>(3);CrossCoreWaitFlag<1>(3);
    }
    __aicore__ inline void Send(GlobalTensor<float>dst,LocalTensor<float>src,uint32_t count){
        Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();DataCopy(dst,src,count);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
    }
    __aicore__ inline void Receive(LocalTensor<float>dst,GlobalTensor<float>src,uint32_t count){
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopy(dst,src,count);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline void Swap(uint32_t j,uint32_t pivot){
        if(j==pivot)return;Fence<HardEvent::S_V>();DataCopy(row,mat[j*CP],CP);DataCopy(other,mat[pivot*CP],CP);PipeBarrier<PIPE_V>();
        DataCopy(mat[pivot*CP],row,CP);DataCopy(mat[j*CP],other,CP);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void PanelPivot(uint32_t p,uint32_t j){
        constexpr uint32_t c0=Right?C:0;uint32_t local=j-c0,start=p-c0;
        AscendC::Gather(coef,mat,index,local*4,N);PipeBarrier<PIPE_V>();Abs(ab,coef,N);PipeBarrier<PIPE_V>();
        if(j){Duplicate(ab,0.0f,j);PipeBarrier<PIPE_V>();}
        AscendC::ReduceMax(maximum,ab,column,int32_t(N),true);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
        uint32_t pivot=maximum.ReinterpretCast<uint32_t>().GetValue(1);float diagonal=coef.GetValue(pivot);
        if(pivot!=j){coef.SetValue(pivot,coef.GetValue(j));coef.SetValue(j,diagonal);}
        pivots.SetValue(j,pivot);record.ReinterpretCast<uint32_t>().SetValue(j-p,pivot);Fence<HardEvent::S_V>();Swap(j,pivot);
        DataCopy(row,mat[j*CP],CP);PipeBarrier<PIPE_V>();Duplicate(row[start],0.0f,j-p+1);PipeBarrier<PIPE_V>();
        Muls(coef,coef,-1.0f/diagonal,N);PipeBarrier<PIPE_V>();Brcb(column,coef,N/8,{1,8});PipeBarrier<PIPE_V>();
        if(j+1<N){
            MulAddDst(mat[(j+1)*CP+start],column[(j+1)*8],row[start],uint64_t(16),uint8_t(N-j-1),{1,0,1,CP/8,1,0});PipeBarrier<PIPE_V>();
            uint64_t mask[2]={1ULL<<(local&7),0};Muls(mat[(j+1)*CP+(local/8)*8],column[(j+1)*8],-1.0f,mask,uint8_t(N-j-1),{1,1,CP/8,1});PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Subtract(uint32_t end,uint32_t firstColumn,uint32_t tail){
        constexpr uint32_t c0=Right?C:0;
        for(uint32_t first=end;first<N;first+=16){
            uint32_t rows=N-first<16?N-first:16;Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
            DataCopyExtParams pc{uint16_t(rows),tail*4,(RP-tail)*4,(CP-tail)/8,0};DataCopyPadExtParams<float>pad{false,0,0,0};
            DataCopyPad(temp,logs[first*RP+c0+firstColumn],pc,pad);Fence<HardEvent::MTE2_V>();
            for(uint32_t q=0;q<tail;q+=64){Sub(mat[first*CP+firstColumn+q],mat[first*CP+firstColumn+q],temp[q],uint64_t(tail-q<64?tail-q:64),uint8_t(rows),{1,1,1,CP/8,CP/8,CP/8});PipeBarrier<PIPE_V>();}
        }
    }
    __aicore__ inline void PackedPivot(uint32_t p,uint32_t j){
        uint32_t local=j-p;DataCopy(coef,panel[local*N],N);PipeBarrier<PIPE_V>();Abs(ab,coef,N);PipeBarrier<PIPE_V>();
        if(j){Duplicate(ab,0.0f,j);PipeBarrier<PIPE_V>();}
        AscendC::ReduceMax(maximum,ab,column,int32_t(N),true);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
        uint32_t pivot=maximum.ReinterpretCast<uint32_t>().GetValue(1);float diagonal=coef.GetValue(pivot);
        if(pivot!=j){coef.SetValue(pivot,coef.GetValue(j));coef.SetValue(j,diagonal);}
        pivots.SetValue(j,pivot);record.ReinterpretCast<uint32_t>().SetValue(j-p,pivot);Fence<HardEvent::S_V>();
        if(pivot!=j){
            AscendC::Gather(row,panel,packedIndex,j*4,16);AscendC::Gather(other,panel,packedIndex,pivot*4,16);PipeBarrier<PIPE_V>();
            uint64_t pm[2]={1ULL<<(pivot&7),0},jm[2]={1ULL<<(j&7),0};Brcb(column,row,2,{1,8});PipeBarrier<PIPE_V>();
            Muls(panel[pivot/8*8],column,1.0f,pm,uint8_t(16),{1,1,N/8,1});PipeBarrier<PIPE_V>();Brcb(column,other,2,{1,8});PipeBarrier<PIPE_V>();
            Muls(panel[j/8*8],column,1.0f,jm,uint8_t(16),{1,1,N/8,1});PipeBarrier<PIPE_V>();
        }
        AscendC::Gather(row,panel,packedIndex,j*4,16);PipeBarrier<PIPE_V>();Duplicate(row,0.0f,local+1);PipeBarrier<PIPE_V>();
        Brcb(column,row,2,{1,8});PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f/diagonal,N);PipeBarrier<PIPE_V>();
        for(uint32_t first=(j+1)/64*64;first<N&&j+1<N;first+=64){
            uint32_t hi=N-first<64?N-first:64,lo=j+1>first?j+1-first:0;uint64_t bits=(hi==64?~0ULL:(1ULL<<hi)-1)&(~0ULL<<lo),mask[2]={bits,0};
            MulAddDst(panel[first],coef[first],column,mask,uint8_t(16),{1,1,0,N/8,0,1});PipeBarrier<PIPE_V>();
            Muls(panel[local*N+first],coef[first],-1.0f,mask,uint8_t(1),{1,1,0,0});PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Multiply(uint32_t p,uint32_t r,uint32_t c,uint32_t m,uint32_t n){
        uint32_t mp=(m+15)/16*16,np=(n+15)/16*16;
        LocalTensor<float>a1(TPosition::A1,0,128*16),b1(TPosition::B1,128*16*4,128*16);
        LocalTensor<float>a2(TPosition::A2,0,128*16),b2(TPosition::B2,0,128*16),cc(TPosition::CO1,0,128*128);
        Nd2NzParams pa{1,uint16_t(m),16,0,RP,uint16_t(mp),1,0},pb{1,16,uint16_t(n),0,RP,16,1,0};
        DataCopy(a1,wg[r*RP+p],pa);DataCopy(b1,wg[p*RP+c],pb);Fence<HardEvent::MTE2_MTE1>();
        LoadData2DParams la;la.startIndex=0;la.repeatTimes=2;la.srcStride=mp/16;la.dstGap=0;la.ifTranspose=false;
        for(uint32_t i=0;i<mp/16;++i)LoadData(a2[i*256],a1[i*128],la);
        LoadData2dTransposeParams lb;lb.startIndex=0;lb.repeatTimes=np/16;lb.srcStride=1;lb.dstGap=0;lb.dstFracGap=np/16-1;LoadDataWithTranspose(b2,b1,lb);Fence<HardEvent::MTE1_M>();
        MmadParams mm;mm.m=m;mm.n=n;mm.k=16;mm.cmatrixInitVal=true;Mmad(cc,a2,b2,mm);Fence<HardEvent::M_FIX>();
        FixpipeParamsV220 fp;fp.ndNum=1;fp.srcNdStride=0;fp.dstNdStride=0;fp.mSize=m;fp.nSize=n;fp.srcStride=mp;fp.dstStride=RP;Fixpipe(logs[r*RP+c],cc,fp);Fence<HardEvent::FIX_MTE2>();
    }
    TPipe pipe;LinalgSolveTilingData t;GlobalTensor<float>ag,bg,wg,logs;GlobalTensor<int32_t>pg;
    LocalTensor<float>mem,mat,row,other,coef,ab,maximum,column,record,temp,panel;LocalTensor<uint32_t>index,panelIndex,packedIndex;LocalTensor<int32_t>pivots;
};

// Keep Schur quadrants in original-row order; defer the logical row permutation.
class OriginalRowSchur {
    static constexpr uint32_t N=256,PC=264,RP=264,ELEMENTS=N*272;
public:
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR B,GM_ADDR workspace,const LinalgSolveTilingData &data){
        t=data;
        if ASCEND_IS_AIC{matrix=GetBlockIdx()/4;quad=GetBlockIdx()%4;}
        else{matrix=GetBlockIdx()/8;quad=(GetBlockIdx()/2)%4;}
        r0=quad/2*128;c0=quad%2*128;
        auto user=reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace));
        shared.SetGlobalBuffer(user+t.schurOffset/4+uint64_t(matrix)*kOS_Total);
        privateWg.SetGlobalBuffer(user+uint64_t(GetBlockIdx())*ELEMENTS);
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A));bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(B));
        uint32_t coords=matrix,ai=0,bi=0;
        for(int d=int(t.batchRank)-1;d>=0;--d){uint32_t q=coords%t.batchDims[d];coords/=t.batchDims[d];ai+=q*t.aStrides[d];bi+=q*t.bStrides[d];}
        ao=uint64_t(ai)*N*N;bo=uint64_t(bi)*N*t.k;
        if ASCEND_IS_AIV{mem=LocalTensor<float>(TPosition::VECCALC,0,32768);}
    }
    __aicore__ inline void Process(){
        if ASCEND_IS_AIC{CubeProcess();}
        else{
            if(GetBlockIdx()%8==0){
                Duplicate(mem,0.0f,128*128);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
                for(uint32_t i=0;i<128;++i)mem.SetValue(i*128+i,1.0f);
                Tx(shared[kOS_Identity],mem,128*128);
                Duplicate(mem[8984],1.0f,N);Duplicate(mem[9496].ReinterpretCast<int32_t>(),int32_t(0),N);PipeBarrier<PIPE_V>();
                auto pi=mem[10280].ReinterpretCast<int32_t>();CreateVecIndex(pi,int32_t(0),N);PipeBarrier<PIPE_V>();Muls(pi,pi,int32_t(16*4),N);PipeBarrier<PIPE_V>();
                auto ci=mem[10008].ReinterpretCast<uint32_t>();for(uint32_t q=0;q<16;++q)ci.SetValue(q,q*PC*4);Fence<HardEvent::S_V>();
            }
            if(!(GetBlockIdx()&1)){
                auto ix=mem[23000].ReinterpretCast<uint32_t>();for(uint32_t i=0;i<16;++i)ix.SetValue(i,i*16*4);Fence<HardEvent::S_V>();
            }else{
                auto ix=mem[6144].ReinterpretCast<uint32_t>();for(uint32_t r=0;r<8;++r)for(uint32_t q=0;q<16;++q)ix.SetValue(r*16+q,(q*136+r)*4);Fence<HardEvent::S_V>();
            }
            SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(0);
            for(uint32_t p=0;p<N;p+=16){
                CrossCoreWaitFlag<2>(1);SyncAll<true>();
                if(GetBlockIdx()%8==0)Lanen256_PanelTail(p);
                SyncAll<true>();
                if(!(GetBlockIdx()&1))U12(p);else PrepareL(p);
                SyncAll<true>();
                if(p+16<N){
                    if(GetBlockIdx()&1)PrepareNegativeU(p);
                    CrossCoreSetFlag<1,PIPE_MTE3>(3);CrossCoreWaitFlag<1>(3);CrossCoreSetFlag<2,PIPE_MTE3>(0);
                }
            }
            Assemble();SyncAll<true>();
            if(GetBlockIdx()&1)WideDiagonalInverse<true>();else WideDiagonalInverse<false>();
            SyncAll<true>();
            if(!(GetBlockIdx()&1) && quad<t.k)PrivateFactor();
        }
    }
private:
    __aicore__ inline void Tx(GlobalTensor<float>dst,LocalTensor<float>src,uint32_t count){
        Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();DataCopy(dst,src,count);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
    }
    __aicore__ inline void Rx(LocalTensor<float>dst,GlobalTensor<float>src,uint32_t count){
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopy(dst,src,count);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline void RxRows(LocalTensor<float>dst,GlobalTensor<float>src,DataCopyExtParams p){
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopyPadExtParams<float>pad{false,0,0,0};DataCopyPad(dst,src,p,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline void TxRows(GlobalTensor<float>dst,LocalTensor<float>src,DataCopyExtParams p){
        Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();DataCopyPad(dst,src,p);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
    }
    __aicore__ inline void CubeMultiply(uint32_t p,bool identity,bool initialize){
        LocalTensor<float>a1(TPosition::A1,0,128*16),b1(TPosition::B1,128*16*4,128*16);
        LocalTensor<float>a2(TPosition::A2,0,128*16),b2(TPosition::B2,0,128*16),acc(TPosition::CO1,0,128*128);
        Fence<HardEvent::M_MTE1>();Fence<HardEvent::MTE1_MTE2>();
        Nd2NzParams pa{1,128,16,0,uint16_t(identity?N:16),128,1,0},pb{1,16,128,0,128,16,1,0};
        if(identity){DataCopy(a1,ag[ao+r0*N+c0+p],pa);DataCopy(b1,shared[kOS_Identity+p*128],pb);}
        else{DataCopy(a1,shared[kOS_Private+quad*4096],pa);DataCopy(b1,shared[kOS_Private+quad*4096+2048],pb);}
        Fence<HardEvent::MTE2_MTE1>();
        LoadData2DParams la;la.startIndex=0;la.repeatTimes=2;la.srcStride=8;la.dstGap=0;la.ifTranspose=false;
        for(uint32_t i=0;i<8;++i)LoadData(a2[i*256],a1[i*128],la);
        LoadData2dTransposeParams lb;lb.startIndex=0;lb.repeatTimes=8;lb.srcStride=1;lb.dstGap=0;lb.dstFracGap=7;LoadDataWithTranspose(b2,b1,lb);Fence<HardEvent::MTE1_M>();
        MmadParams mm;mm.m=128;mm.n=128;mm.k=16;mm.cmatrixInitVal=initialize;Mmad(acc,a2,b2,mm);
    }
    __aicore__ inline void CubePublish(){
        LocalTensor<float>acc(TPosition::CO1,0,128*128);Fence<HardEvent::M_FIX>();
        FixpipeParamsV220 fp;fp.ndNum=1;fp.srcNdStride=0;fp.dstNdStride=0;fp.mSize=128;fp.nSize=128;fp.srcStride=128;fp.dstStride=N;
        Fixpipe(shared[kOS_S+r0*N+c0],acc,fp);CrossCoreSetFlag<2,PIPE_FIX>(1);Fence<HardEvent::FIX_MTE2>();
    }
    __aicore__ inline void CubeProcess(){
        SetHF32Mode(HF32Mode::DISABLE);CrossCoreWaitFlag<2>(0);
        for(uint32_t q=0;q<128;q+=16)CubeMultiply(q,true,q==0);CubePublish();
        for(uint32_t p=0;p+16<N;p+=16){CrossCoreWaitFlag<2>(0);CubeMultiply(p,false,false);CubePublish();}
    }
    // Lanen256_: consumed LU panel columns are not published orreadagain.
    __aicore__ inline void Lanen256_PanelTail(uint32_t p){
        auto panel=mem;auto lc=mem[4224];auto coef=mem[8320];auto ab=mem[8576];auto maximum=mem[8832];auto row=mem[8840];auto broad=mem[8856];auto active=mem[8984];auto reduce=mem[9240];
        auto perm=mem[9496].ReinterpretCast<int32_t>();auto pb=mem[9752].ReinterpretCast<int32_t>();auto ci=mem[10008].ReinterpretCast<uint32_t>();auto l11=mem[10024];auto ri=mem[10280].ReinterpretCast<uint32_t>();auto incoming=mem[16384];
        DataCopyExtParams load{N,16*4,(N-16)*4,0,0};RxRows(incoming,shared[kOS_S+p],load);
        for(uint32_t q=0;q<16;++q){AscendC::Gather(panel[q*PC],incoming,ri,q*4,N);PipeBarrier<PIPE_V>();}
        for(uint32_t j=0;j<16;++j){
            DataCopy(coef,panel[j*PC],N);PipeBarrier<PIPE_V>();Abs(ab,coef,N);PipeBarrier<PIPE_V>();Mul(ab,ab,active,N);PipeBarrier<PIPE_V>();
            WholeReduceMax(maximum,ab,int32_t(64),uint8_t(4),1,1,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
            uint32_t maximumBits,pivot;GetReduceRepeatMaxMinSpr<float>(maximumBits,pivot);const float diagonal=coef.GetValue(pivot);perm.SetValue(p+j,int32_t(pivot));
            Fence<HardEvent::S_V>();Muls(coef,coef,1.0f/diagonal,N);PipeBarrier<PIPE_V>();Mul(coef,coef,active,N);PipeBarrier<PIPE_V>();DataCopy(lc[j*N],coef,N);PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();active.SetValue(pivot,0.0f);Fence<HardEvent::S_V>();Mul(coef,coef,active,N);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,N);PipeBarrier<PIPE_V>();
            AscendC::Gather(row,panel,ci,pivot*4,16);PipeBarrier<PIPE_V>();Brcb(broad,row,2,{1,8});PipeBarrier<PIPE_V>();
            if(j+1<16){for(uint32_t first=0;first<N;first+=64){MulAddDst(panel[(j+1)*PC+first],coef[first],broad[(j+1)*8],uint64_t(64),uint8_t(15-j),{1,1,0,PC/8,0,1});PipeBarrier<PIPE_V>();}}
        }
        Fence<HardEvent::S_V>();Muls(pb,perm[p],int32_t(4),16);PipeBarrier<PIPE_V>();
        for(uint32_t q=0;q<16;++q){
            AscendC::Gather(row,lc[q*N],pb.ReinterpretCast<uint32_t>(),0,16);PipeBarrier<PIPE_V>();Brcb(broad,row,2,{1,8});PipeBarrier<PIPE_V>();
            uint64_t mask[2]={1ULL<<(q&7),0};Muls(l11[q/8*8],broad,1.0f,mask,uint8_t(16),{1,1,2,1});PipeBarrier<PIPE_V>();
        }
        Tx(shared[kOS_LBlock],lc,N*16);Tx(shared[kOS_L11],l11,16*16);Tx(shared[kOS_Perm+p],perm[p].ReinterpretCast<float>(),16);
    }

    __aicore__ inline void Panel(uint32_t p){
        auto panel=mem;auto lc=mem[4224];auto coef=mem[8320];auto ab=mem[8576];auto maximum=mem[8832];auto row=mem[8840];auto broad=mem[8856];auto active=mem[8984];auto reduce=mem[9240];
        auto perm=mem[9496].ReinterpretCast<int32_t>();auto pb=mem[9752].ReinterpretCast<int32_t>();auto ci=mem[10008].ReinterpretCast<uint32_t>();auto l11=mem[10024];auto ri=mem[10280].ReinterpretCast<uint32_t>();auto incoming=mem[16384];
        DataCopyExtParams load{N,16*4,(N-16)*4,0,0};RxRows(incoming,shared[kOS_S+p],load);
        for(uint32_t q=0;q<16;++q){AscendC::Gather(panel[q*PC],incoming,ri,q*4,N);PipeBarrier<PIPE_V>();}
        for(uint32_t j=0;j<16;++j){
            DataCopy(coef,panel[j*PC],N);PipeBarrier<PIPE_V>();Abs(ab,coef,N);PipeBarrier<PIPE_V>();Mul(ab,ab,active,N);PipeBarrier<PIPE_V>();
            WholeReduceMax(maximum,ab,int32_t(64),uint8_t(4),1,1,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
            uint32_t maximumBits,pivot;GetReduceRepeatMaxMinSpr<float>(maximumBits,pivot);const float diagonal=coef.GetValue(pivot);perm.SetValue(p+j,int32_t(pivot));
            Fence<HardEvent::S_V>();Muls(coef,coef,1.0f/diagonal,N);PipeBarrier<PIPE_V>();Mul(coef,coef,active,N);PipeBarrier<PIPE_V>();DataCopy(lc[j*N],coef,N);PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();active.SetValue(pivot,0.0f);Fence<HardEvent::S_V>();Mul(coef,coef,active,N);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,N);PipeBarrier<PIPE_V>();
            AscendC::Gather(row,panel,ci,pivot*4,16);PipeBarrier<PIPE_V>();Duplicate(row,0.0f,j+1);PipeBarrier<PIPE_V>();Brcb(broad,row,2,{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t first=0;first<N;first+=64){MulAddDst(panel[first],coef[first],broad,uint64_t(64),uint8_t(16),{1,1,0,PC/8,0,1});PipeBarrier<PIPE_V>();}
        }
        Fence<HardEvent::S_V>();Muls(pb,perm[p],int32_t(4),16);PipeBarrier<PIPE_V>();
        for(uint32_t q=0;q<16;++q){
            AscendC::Gather(row,lc[q*N],pb.ReinterpretCast<uint32_t>(),0,16);PipeBarrier<PIPE_V>();Brcb(broad,row,2,{1,8});PipeBarrier<PIPE_V>();
            uint64_t mask[2]={1ULL<<(q&7),0};Muls(l11[q/8*8],broad,1.0f,mask,uint8_t(16),{1,1,2,1});PipeBarrier<PIPE_V>();
        }
        Tx(shared[kOS_LBlock],lc,N*16);Tx(shared[kOS_L11],l11,16*16);Tx(shared[kOS_Perm+p],perm[p].ReinterpretCast<float>(),16);
    }
    __aicore__ inline void U12(uint32_t p){
        const uint32_t columnStart=quad*64;auto u=mem[16384];auto l11=mem[17424];auto coef=mem[17680];auto broad=mem[17744];auto index=mem[23000].ReinterpretCast<uint32_t>();auto perm=mem[23016].ReinterpretCast<int32_t>();
        Rx(perm.ReinterpretCast<float>(),shared[kOS_Perm+p],16);Rx(l11,shared[kOS_L11],256);
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();
        for(uint32_t j=0;j<16;++j){uint32_t original=uint32_t(perm.GetValue(j));DataCopy(u[j*64],shared[kOS_S+original*N+columnStart],64);}
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        for(uint32_t j=0;j<15;++j){
            AscendC::Gather(coef,l11,index,j*4,16);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,16);PipeBarrier<PIPE_V>();Brcb(broad,coef,2,{1,8});PipeBarrier<PIPE_V>();
            MulAddDst(u[(j+1)*64],broad[(j+1)*8],u[j*64],uint64_t(64),uint8_t(15-j),{1,0,1,8,1,0});PipeBarrier<PIPE_V>();
        }
        DataCopyExtParams store{16,64*4,0,(N-64)*4,0};TxRows(shared[kOS_U+p*N+columnStart],u,store);
    }
    __aicore__ inline void PrepareL(uint32_t p){
        auto columns=mem;auto nd=mem[4096];auto index=mem[6144].ReinterpretCast<uint32_t>();
        DataCopyExtParams load{16,128*4,(N-128)*4,1,0};RxRows(columns,shared[kOS_LBlock+r0],load);
        for(uint32_t first=0;first<128;first+=8){AscendC::Gather(nd[first*16],columns,index,first*4,128);PipeBarrier<PIPE_V>();}
        Tx(shared[kOS_Private+quad*4096],nd,128*16);
        if((quad&1)==0){DataCopyExtParams store{128,16*4,0,(N-16)*4,0};TxRows(shared[kOS_L+r0*N+p],nd,store);}
    }
    __aicore__ inline void PrepareNegativeU(uint32_t p){
        auto u=mem[8192];DataCopyExtParams load{16,128*4,(N-128)*4,0,0};RxRows(u,shared[kOS_U+p*N+c0],load);Muls(u,u,-1.0f,16*128);PipeBarrier<PIPE_V>();Tx(shared[kOS_Private+quad*4096+2048],u,16*128);
    }
    __aicore__ inline void InitSolveTransposeIndex(){
        auto q=mem.ReinterpretCast<int32_t>();auto r=mem[8192].ReinterpretCast<int32_t>();
        auto c=mem[16384].ReinterpretCast<int32_t>();auto index=mem[24576].ReinterpretCast<int32_t>();
        CreateVecIndex(q,int32_t(0),8192);Duplicate(index,int32_t(31),8192);PipeBarrier<PIPE_V>();
        And(r.ReinterpretCast<int16_t>(),q.ReinterpretCast<int16_t>(),index.ReinterpretCast<int16_t>(),16384);
        ShiftRight(c,q,int32_t(5),8192);PipeBarrier<PIPE_V>();
        Muls(r,r,int32_t(N*4),8192);Muls(c,c,int32_t(4),8192);PipeBarrier<PIPE_V>();
        Add(index,r,c,8192);PipeBarrier<PIPE_V>();
    }
    template<bool Lower> __aicore__ inline void WideDiagonalInverse(){
        constexpr uint32_t B=64;
        static_assert(kOS_Identity+4*2*B*B==kOS_Total,"Reuse dead identity and Cube operand cache");
        const uint32_t first=quad*B;
        auto lu=mem,inv=mem[4096],coef=mem[8192],broad=mem[8256];
        auto diagonal=mem[8768],inverse=mem[8832],ones=mem[8896];
        auto di=mem[8960].ReinterpretCast<uint32_t>();auto ci=mem[9024].ReinterpretCast<uint32_t>();
        DataCopyExtParams load{B,B*4,(N-B)*4,0,0};RxRows(lu,shared[kOS_U+first*N+first],load);
        Duplicate(inv,0.0f,B*B);Duplicate(ones,1.0f,B);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
        for(uint32_t q=0;q<B;++q){inv.SetValue(q*(B+1),1.0f);di.SetValue(q,q*(B+1)*4);ci.SetValue(q,q*B*4);}
        Fence<HardEvent::S_V>();
        if constexpr(!Lower){
            AscendC::Gather(diagonal,lu,di,0,B);PipeBarrier<PIPE_V>();Div(inverse,ones,diagonal,B);PipeBarrier<PIPE_V>();
            for(uint32_t j=B-1;j>0;--j){
                AscendC::Gather(coef,lu,ci,j*4,B);PipeBarrier<PIPE_V>();Mul(coef,coef,inverse,B);PipeBarrier<PIPE_V>();
                Muls(coef,coef,-1.0f,B);PipeBarrier<PIPE_V>();Brcb(broad,coef,B/8,{1,8});PipeBarrier<PIPE_V>();
                MulAddDst(inv,broad,inv[j*B],uint64_t(B),uint8_t(j),{1,0,1,B/8,1,0});PipeBarrier<PIPE_V>();
            }
        }else{
            for(uint32_t j=0;j<B-1;++j){
                AscendC::Gather(coef,lu,ci,j*4,B);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,B);PipeBarrier<PIPE_V>();
                Brcb(broad,coef,B/8,{1,8});PipeBarrier<PIPE_V>();
                MulAddDst(inv[(j+1)*B],broad[(j+1)*8],inv[j*B],uint64_t(B),uint8_t(B-1-j),{1,0,1,B/8,1,0});PipeBarrier<PIPE_V>();
            }
        }
        Tx(shared[kOS_Identity+quad*2*B*B+(Lower?B*B:0)],inv,B*B);
    }
    __aicore__ inline void PrepareSolveCache(uint32_t first,LocalTensor<float>u){
        auto work=mem[8192],transposed=mem[16384],diagonal=mem[20000],inverse=mem[20032],minus=mem[20064],broad=mem[20096];
        auto index=mem[20400].ReinterpretCast<uint32_t>();auto transposeIndex=mem[24576].ReinterpretCast<uint32_t>();
        for(uint32_t r=0;r<32;++r)index.SetValue(r,(r*(N+1)+first)*4);Fence<HardEvent::S_V>();
        AscendC::Gather(diagonal,u,index,0,32);PipeBarrier<PIPE_V>();Duplicate(minus,1.0f,32);PipeBarrier<PIPE_V>();
        Div(inverse,minus,diagonal,32);PipeBarrier<PIPE_V>();Muls(minus,inverse,-1.0f,32);PipeBarrier<PIPE_V>();
        Brcb(broad,minus,4,{1,8});PipeBarrier<PIPE_V>();
        for(uint32_t c=0;c<N;c+=64){Mul(work[c],u[c],broad,uint64_t(64),uint8_t(32),{1,1,0,N/8,N/8,1});PipeBarrier<PIPE_V>();}
        Brcb(broad,inverse,4,{1,8});PipeBarrier<PIPE_V>();
        for(uint32_t r=0;r<32;++r){
            uint32_t j=first+r;if(j){Muls(work[r*N],u[r*N],-1.0f,j);PipeBarrier<PIPE_V>();}
            uint64_t mask[2]={1ULL<<(j&63),0};Muls(work[r*N+(j/64)*64],broad[r*8],1.0f,mask,uint8_t(1),{1,0,8,1});PipeBarrier<PIPE_V>();
        }
        AscendC::Gather(transposed,work,transposeIndex,0,32*N);PipeBarrier<PIPE_V>();
        DataCopyExtParams store{uint16_t(N),32*4,0,(N-32)*4,0};TxRows(shared[kOS_S+first],transposed,store);
    }
    __aicore__ inline void Assemble(){
        InitSolveTransposeIndex();
        const uint32_t first=(GetBlockIdx()%8)*32;auto u=mem;auto lower=mem[8192];auto perm=mem[22000].ReinterpretCast<int32_t>();
        Rx(perm.ReinterpretCast<float>(),shared[kOS_Perm],N);Rx(u,shared[kOS_U+first*N],32*N);
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();
        for(uint32_t j=0;j<32;++j){uint32_t original=uint32_t(perm.GetValue(first+j));DataCopy(lower[j*N],shared[kOS_L+original*N],N);}
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        for(uint32_t j=0;j<32;++j){
            if(first+j){Muls(u[j*N],lower[j*N],1.0f,first+j);PipeBarrier<PIPE_V>();}
        }
        Tx(shared[kOS_U+first*N],u,32*N);
        PrepareSolveCache(first,u);
    }
    __aicore__ inline void PrivateFactor(){
        for(uint32_t first=0;first<N;first+=32){Rx(mem,shared[kOS_U+first*N],32*N);DataCopyExtParams store{32,N*4,0,(RP-N)*4,0};TxRows(privateWg[first*RP],mem,store);}
        auto perm=mem[22000].ReinterpretCast<int32_t>();Rx(perm.ReinterpretCast<float>(),shared[kOS_Perm],N);Tx(privateWg[N*RP],perm.ReinterpretCast<float>(),N);
        Rx(mem,bg[bo],N*t.k);auto index=mem[18000].ReinterpretCast<int32_t>();Muls(index,perm,int32_t(t.k*4),N);PipeBarrier<PIPE_V>();
        auto compact=mem[1024];AscendC::Gather(compact,mem,index.ReinterpretCast<uint32_t>(),quad*4,N);PipeBarrier<PIPE_V>();auto broad=mem[2048];Brcb(broad,compact,N/8,{1,8});PipeBarrier<PIPE_V>();
        DataCopyExtParams store{N,8*4,0,N*4,0};TxRows(privateWg[N],broad,store);
    }
    TPipe pipe;LinalgSolveTilingData t;GlobalTensor<float>ag,bg,shared,privateWg;LocalTensor<float>mem;
    uint32_t matrix,quad,r0,c0;uint64_t ao,bo;
};

// Lanen256_: local16 triangular transforms; offdiagonal factor propagation on Cube.
// Shared L is negative and remains in original block-row order. T=invL*P.
class Lanen256_CubeLUT16 {
    static constexpr uint32_t N=256,B=16,RP=40,CRP=24,CS=16*B*CRP;
public:
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR Bv,GM_ADDR workspace,const LinalgSolveTilingData &data){
        t=data;if ASCEND_IS_AIC{matrix=GetBlockIdx()/4;quad=GetBlockIdx()%4;}else{matrix=GetBlockIdx()/8;quad=(GetBlockIdx()/2)%4;}
        r0=quad/2*128;c0=quad%2*128;
        shared.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace))+t.schurOffset/4+uint64_t(matrix)*kOS_Total);
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));uint32_t coords=matrix,ai=0;
        for(int d=int(t.batchRank)-1;d>=0;--d){uint32_t v=coords%t.batchDims[d];coords/=t.batchDims[d];ai+=v*t.aStrides[d];}ao=uint64_t(ai)*N*N;
        if ASCEND_IS_AIV{mem=LocalTensor<float>(TPosition::VECCALC,0,32768);}
    }
    __aicore__ inline void Process(){
        if ASCEND_IS_AIC{CubeProcess();}
        else{
            if(GetBlockIdx()%8==0){
                Duplicate(mem,0.0f,128*128);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();for(uint32_t j=0;j<128;++j)mem.SetValue(j*129,1.0f);Tx(shared[kOS_Identity],mem,128*128);
                auto perm=mem[23000].ReinterpretCast<int32_t>();CreateVecIndex(perm,int32_t(0),N);PipeBarrier<PIPE_V>();Tx(shared[kOS_Perm],perm.ReinterpretCast<float>(),N);
                InitD16();
            }
            SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(0);
            for(uint32_t p=0;p<N;p+=B){
                CrossCoreWaitFlag<2>(1);SyncAll<true>();
                if(GetBlockIdx()%8==0){if(factorGood)D16<true>(p);else D16Identity(p);}
                SyncAll<true>();
                if(p+B<N){
                    CrossCoreSetFlag<2,PIPE_MTE3>(0);CrossCoreWaitFlag<2>(1);SyncAll<true>();
                    CrossCoreSetFlag<2,PIPE_MTE3>(0);
                }
            }
            if(GetBlockIdx()%8==0){auto status=mem[2800].ReinterpretCast<uint32_t>();Duplicate(status,0U,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();status.SetValue(0,factorGood?1U:0U);Tx(shared[kOS_L11],status.ReinterpretCast<float>(),8);}
            Assemble();SyncAll<true>();Rx(mem[2800],shared[kOS_L11],8);factorGood=mem[2800].ReinterpretCast<uint32_t>().GetValue(0)!=0;
        }
    }
    __aicore__ inline bool FactorOK(){return factorGood;}
private:
    __aicore__ inline void Tx(GlobalTensor<float>d,LocalTensor<float>s,uint32_t count){Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();DataCopy(d,s,count);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();}
    __aicore__ inline void Rx(LocalTensor<float>d,GlobalTensor<float>s,uint32_t count){Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopy(d,s,count);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();}
    __aicore__ inline void RowsIn(LocalTensor<float>d,GlobalTensor<float>s,DataCopyExtParams p){Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopyPadExtParams<float> pad{false,0,0,0};DataCopyPad(d,s,p,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();}
    __aicore__ inline void InitD16(){
        auto ci=mem[856].ReinterpretCast<uint32_t>(),upci=mem[2744].ReinterpretCast<uint32_t>(),di=mem[2760].ReinterpretCast<uint32_t>();auto cols=mem[1272].ReinterpretCast<int32_t>();
        Duplicate(mem[4000],0.0f,256);Duplicate(mem[4256],0.0f,384);Duplicate(mem[4640],-1.0f,16);Duplicate(mem[4656],1.0f,16);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
        for(uint32_t j=0;j<B;++j){ci.SetValue(j,j*RP*4);upci.SetValue(j,j*B*4);di.SetValue(j,j*(B+1)*4);cols.SetValue(j,j*4);mem[4000].SetValue(j*(B+1),1.0f);mem[4256].SetValue(j*(CRP+1),1.0f);}Fence<HardEvent::S_V>();
    }
    __aicore__ inline void D16Identity(uint32_t p){
        // Keep the exact AIC stage/flag counts. No invalid diagonal or index is used.
        factorGood=false;Tx(shared[kOS_Identity+(p/B)*B*CRP],mem[4256],B*CRP);Tx(shared[kOS_Identity+CS+(p/B)*B*CRP],mem[4256],B*CRP);
        Muls(mem[1288],mem[4000],-1.0f,256);PipeBarrier<PIPE_V>();Tx(shared[kOS_LBlock],mem[1288],256);Tx(shared[kOS_LBlock+256],mem[4000],256);
        if(p+B==N){Fence<HardEvent::V_MTE3>();DataCopyExtParams st{B,B*4,0,(N-B)*4,0};DataCopyPad(shared[kOS_U+p*N+p],mem[4000],st);Fence<HardEvent::MTE3_V>();}
    }
    template<bool Pivot> __aicore__ inline void D16(uint32_t p){
        auto mat=mem,coef=mem[640],ab=mem[656],maximum=mem[672],active=mem[680],row=mem[696],broad=mem[728];
        auto index=mem[856].ReinterpretCast<uint32_t>(),perm=mem[872].ReinterpretCast<uint32_t>();auto full=mem[888].ReinterpretCast<int32_t>(),bases=mem[1144].ReinterpretCast<int32_t>(),cols=mem[1272].ReinterpretCast<int32_t>();
        auto u=mem[1288],tp=mem[1544],tc=mem[1800],up=mem[2184],diag=mem[2568],iv=mem[2584],ib=mem[2616];auto ci=mem[2744].ReinterpretCast<uint32_t>(),di=mem[2760].ReinterpretCast<uint32_t>();auto hist=mem[3000];
        Duplicate(mat,0.0f,B*RP);Duplicate(tc,0.0f,B*CRP);Duplicate(active,1.0f,B);PipeBarrier<PIPE_V>();
        DataCopyExtParams load{B,B*4,(N-B)*4,(RP-B)/8,0};RowsIn(mat,shared[kOS_S+p*N+p],load);
        DataCopy(mat[B],mem[4000],DataCopyParams{B,2,0,(RP-B)/8});DataCopy(up,mem[4256],B*CRP);PipeBarrier<PIPE_V>();
        if constexpr(!Pivot){Muls(u,mat,1.0f,uint64_t(B),uint8_t(B),{1,1,B/8,RP/8});PipeBarrier<PIPE_V>();Tx(shared[kOS_Private+(p/B)*256],u,256);}
        uint64_t availableRows=(1ULL<<B)-1;
        for(uint32_t j=0;j<B;++j){
            Gather(coef,mat,index,j*4,B);PipeBarrier<PIPE_V>();
            if constexpr(Pivot){
                Abs(ab,coef,B);PipeBarrier<PIPE_V>();uint64_t enabled[2]={availableRows,0};WholeReduceMax(maximum,ab,enabled,uint8_t(1),1,1,2);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
                uint32_t bits,pivot;GetReduceRepeatMaxMinSpr<float>(bits,pivot);if(pivot>=B){D16Identity(p);return;}float d=coef.GetValue(pivot);if(!(d!=0.0f && d<=3.402823466e38f && d>=-3.402823466e38f)){D16Identity(p);return;}perm.SetValue(j,pivot*RP*4);
                availableRows&=~(1ULL<<pivot);
                if(j+1<B){
                    auto selected=mem[5000];Duplicate(selected,0.0f,B);PipeBarrier<PIPE_V>();uint64_t remaining[2]={availableRows,0};Muls(selected,coef,-1.0f/d,remaining,uint8_t(1),{1,1,2,2});PipeBarrier<PIPE_V>();
                    DataCopy(row,mat[pivot*RP],32);PipeBarrier<PIPE_V>();Brcb(broad,selected,2,{1,8});PipeBarrier<PIPE_V>();MulAddDst(mat,broad,row,uint64_t(32),uint8_t(B),{1,0,1,RP/8,1,0});PipeBarrier<PIPE_V>();
                }
            }else{
                Brcb(diag,coef[(j/8)*8],1,{1,8});PipeBarrier<PIPE_V>();Div(iv,mem[4640],diag[(j%8)*8],uint64_t(8),uint8_t(1),{1,1,1,1,1,1});PipeBarrier<PIPE_V>();
                Mul(coef,coef,iv,uint64_t(B),uint8_t(1),{1,1,0,2,2,0});PipeBarrier<PIPE_V>();Duplicate(coef,0.0f,j+1);PipeBarrier<PIPE_V>();DataCopy(hist[j*B],coef,B);DataCopy(row,mat[j*RP],32);PipeBarrier<PIPE_V>();
                if(j+1<B){Brcb(broad,coef,2,{1,8});PipeBarrier<PIPE_V>();MulAddDst(mat[(j+1)*RP],broad[(j+1)*8],row,uint64_t(32),uint8_t(B-j-1),{1,0,1,RP/8,1,0});PipeBarrier<PIPE_V>();}
            }
        }
        if constexpr(!Pivot){
            Abs(hist,hist,256);PipeBarrier<PIPE_V>();WholeReduceMax(maximum,hist,int32_t(64),uint8_t(4),1,1,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
            uint32_t bits,at;GetReduceRepeatMaxMinSpr<float>(bits,at);union Lanen256_Multiplier{uint32_t u;float f;}v;v.u=bits;if(!(v.f<=256.0f)){D16<true>(p);return;}
            Muls(u,mat,1.0f,uint64_t(B),uint8_t(B),{1,1,B/8,RP/8});Muls(tp,mat[B],1.0f,uint64_t(B),uint8_t(B),{1,1,B/8,RP/8});PipeBarrier<PIPE_V>();
        }else{
            Fence<HardEvent::S_V>();Brcb(bases.ReinterpretCast<float>(),perm.ReinterpretCast<float>(),2,{1,8});PipeBarrier<PIPE_V>();Add(full,bases,cols,uint64_t(B),uint8_t(B),{1,0,1,B/8,1,0});PipeBarrier<PIPE_V>();Gather(u,mat,full.ReinterpretCast<uint32_t>(),0,256);Gather(tp,mat,full.ReinterpretCast<uint32_t>(),B*4,256);PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();auto pg=mem[3600].ReinterpretCast<uint32_t>();for(uint32_t j=0;j<B;++j)pg.SetValue(j,p+perm.GetValue(j)/(RP*4));Fence<HardEvent::S_V>();Tx(shared[kOS_Perm+p],pg.ReinterpretCast<float>(),B);
        }
        Muls(tc,tp,1.0f,uint64_t(B),uint8_t(B),{1,1,CRP/8,B/8});PipeBarrier<PIPE_V>();Gather(diag,u,di,0,B);PipeBarrier<PIPE_V>();Div(iv,mem[4656],diag,B);PipeBarrier<PIPE_V>();Brcb(ib,iv,2,{1,8});PipeBarrier<PIPE_V>();
        for(int j=B-1;j>=0;--j){Mul(up[uint32_t(j)*CRP],up[uint32_t(j)*CRP],ib[uint32_t(j)*8],uint64_t(B),uint8_t(1),{1,1,0,CRP/8,CRP/8,1});PipeBarrier<PIPE_V>();if(j){Gather(coef,u,ci,uint32_t(j)*4,B);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,B);PipeBarrier<PIPE_V>();Brcb(broad,coef,2,{1,8});PipeBarrier<PIPE_V>();MulAddDst(up,broad,up[uint32_t(j)*CRP],uint64_t(B),uint8_t(j),{1,0,1,CRP/8,1,0});PipeBarrier<PIPE_V>();}}
        Tx(shared[kOS_Identity+(p/B)*B*CRP],up,B*CRP);Tx(shared[kOS_Identity+CS+(p/B)*B*CRP],tc,B*CRP);
        Muls(tc,up,-1.0f,uint64_t(B),uint8_t(B),{1,1,B/8,CRP/8});PipeBarrier<PIPE_V>();Tx(shared[kOS_LBlock],tc,256);Tx(shared[kOS_LBlock+256],tp,256);
        if(p+B==N){Fence<HardEvent::V_MTE3>();DataCopyExtParams st{B,B*4,0,(N-B)*4,0};DataCopyPad(shared[kOS_U+p*N+p],u,st);Fence<HardEvent::MTE3_V>();}
    }
    __aicore__ inline void Assemble(){
        uint32_t first=(GetBlockIdx()%8)*32;Rx(mem,shared[kOS_U+first*N],32*N);Rx(mem[8192],shared[kOS_L+first*N],32*N);
        for(uint32_t j=0;j<32;++j)if((first+j)/B*B){Muls(mem[j*N],mem[8192+j*N],-1.0f,(first+j)/B*B);PipeBarrier<PIPE_V>();}
        Tx(shared[kOS_U+first*N],mem,32*N);
    }
        __aicore__ inline void CubeInitial(uint32_t p,bool identity,bool initialize){
        LocalTensor<float>a1(TPosition::A1,0,128*16),b1(TPosition::B1,128*16*4,128*16);
        LocalTensor<float>a2(TPosition::A2,0,128*16),b2(TPosition::B2,0,128*16),acc(TPosition::CO1,0,128*128);
        Fence<HardEvent::M_MTE1>();Fence<HardEvent::MTE1_MTE2>();
        Nd2NzParams pa{1,128,16,0,uint16_t(identity?N:16),128,1,0},pb{1,16,128,0,128,16,1,0};
        if(identity){DataCopy(a1,ag[ao+r0*N+c0+p],pa);DataCopy(b1,shared[kOS_Identity+p*128],pb);}
        else{DataCopy(a1,shared[kOS_Private+quad*4096],pa);DataCopy(b1,shared[kOS_Private+quad*4096+2048],pb);}
        Fence<HardEvent::MTE2_MTE1>();
        LoadData2DParams la;la.startIndex=0;la.repeatTimes=2;la.srcStride=8;la.dstGap=0;la.ifTranspose=false;
        for(uint32_t i=0;i<8;++i)LoadData(a2[i*256],a1[i*128],la);
        LoadData2dTransposeParams lb;lb.startIndex=0;lb.repeatTimes=8;lb.srcStride=1;lb.dstGap=0;lb.dstFracGap=7;LoadDataWithTranspose(b2,b1,lb);Fence<HardEvent::MTE1_M>();
        MmadParams mm;mm.m=128;mm.n=128;mm.k=16;mm.cmatrixInitVal=initialize;Mmad(acc,a2,b2,mm);
    }
    __aicore__ inline void CubePublish(uint32_t slice){
        LocalTensor<float> acc(TPosition::CO1,0,128*128);Fence<HardEvent::M_FIX>();
        FixpipeParamsV220 fp;fp.ndNum=1;fp.srcNdStride=0;fp.dstNdStride=0;fp.srcStride=128;fp.dstStride=N;
        // CO1 NZ is [N/16][M][16]: columnTile*M*16 + row*16.
        if(slice>=r0 && slice<r0+128){fp.mSize=16;fp.nSize=128;Fixpipe(shared[kOS_S+slice*N+c0],acc[(slice-r0)*16],fp);}
        if(slice>=c0 && slice<c0+128){fp.mSize=128;fp.nSize=16;Fixpipe(shared[kOS_S+r0*N+slice],acc[(slice-c0)*128],fp);}
        CrossCoreSetFlag<2,PIPE_FIX>(1);Fence<HardEvent::FIX_MTE2>();
    }
    __aicore__ inline void CubeProduct(GlobalTensor<float>dst,uint32_t dp,GlobalTensor<float>aa,uint32_t ap,GlobalTensor<float>bb,uint32_t bp,uint32_t m,uint32_t n){
        const uint32_t mp=(m+15)/16*16,np=(n+15)/16*16;
        LocalTensor<float>a1(TPosition::A1,0,128*16),b1(TPosition::B1,128*16*4,128*16);
        LocalTensor<float>a2(TPosition::A2,0,128*16),b2(TPosition::B2,0,128*16),cc(TPosition::CO1,128*128*4,128*16);
        Fence<HardEvent::M_MTE1>();Fence<HardEvent::MTE1_MTE2>();
        Nd2NzParams pa{1,uint16_t(m),16,0,uint16_t(ap),uint16_t(mp),1,0},pb{1,16,uint16_t(n),0,uint16_t(bp),16,1,0};
        DataCopy(a1,aa,pa);DataCopy(b1,bb,pb);Fence<HardEvent::MTE2_MTE1>();
        LoadData2DParams la;la.startIndex=0;la.repeatTimes=2;la.srcStride=mp/16;la.dstGap=0;la.ifTranspose=false;
        for(uint32_t i=0;i<mp/16;++i)LoadData(a2[i*256],a1[i*128],la);
        LoadData2dTransposeParams lb;lb.startIndex=0;lb.repeatTimes=np/16;lb.srcStride=1;lb.dstGap=0;lb.dstFracGap=np/16-1;LoadDataWithTranspose(b2,b1,lb);Fence<HardEvent::MTE1_M>();
        MmadParams mm;mm.m=m;mm.n=n;mm.k=16;mm.cmatrixInitVal=true;Mmad(cc,a2,b2,mm);Fence<HardEvent::M_FIX>();
        FixpipeParamsV220 fp;fp.ndNum=1;fp.srcNdStride=0;fp.dstNdStride=0;fp.mSize=m;fp.nSize=n;fp.srcStride=mp;fp.dstStride=dp;Fixpipe(dst,cc,fp);Fence<HardEvent::FIX_MTE2>();
    }
    __aicore__ inline void CubeUL(uint32_t p){
        if(quad<2){uint32_t row=quad*128;CubeProduct(shared[kOS_L+row*N+p],N,shared[kOS_S+row*N+p],N,shared[kOS_LBlock],B,128,B);}
        else{uint32_t col=(quad-2)*128;CubeProduct(shared[kOS_U+p*N+col],N,shared[kOS_LBlock+256],B,shared[kOS_S+p*N+col],N,B,128);}
    }
    __aicore__ inline void CubeSchur(uint32_t p){
        LocalTensor<float>a1(TPosition::A1,0,128*16),b1(TPosition::B1,128*16*4,128*16);
        LocalTensor<float>a2(TPosition::A2,0,128*16),b2(TPosition::B2,0,128*16),acc(TPosition::CO1,0,128*128);
        Fence<HardEvent::M_MTE1>();Fence<HardEvent::MTE1_MTE2>();
        Nd2NzParams pa{1,128,16,0,N,128,1,0},pb{1,16,128,0,N,16,1,0};DataCopy(a1,shared[kOS_L+r0*N+p],pa);DataCopy(b1,shared[kOS_U+p*N+c0],pb);Fence<HardEvent::MTE2_MTE1>();
        LoadData2DParams la;la.startIndex=0;la.repeatTimes=2;la.srcStride=8;la.dstGap=0;la.ifTranspose=false;for(uint32_t i=0;i<8;++i)LoadData(a2[i*256],a1[i*128],la);
        LoadData2dTransposeParams lb;lb.startIndex=0;lb.repeatTimes=8;lb.srcStride=1;lb.dstGap=0;lb.dstFracGap=7;LoadDataWithTranspose(b2,b1,lb);Fence<HardEvent::MTE1_M>();
        MmadParams mm;mm.m=128;mm.n=128;mm.k=16;mm.cmatrixInitVal=false;Mmad(acc,a2,b2,mm);
    }
    __aicore__ inline void CubeProcess(){
        SetHF32Mode(HF32Mode::DISABLE);CrossCoreWaitFlag<2>(0);
        for(uint32_t q=0;q<128;q+=16)CubeInitial(q,true,q==0);CubePublish(0);
        for(uint32_t p=0;p+B<N;p+=B){
            CrossCoreWaitFlag<2>(0);CubeUL(p);CrossCoreSetFlag<2,PIPE_FIX>(1);
            CrossCoreWaitFlag<2>(0);CubeSchur(p);CubePublish(p+B);
        }
    }

    TPipe pipe;bool factorGood=true;LinalgSolveTilingData t;GlobalTensor<float>ag,shared;LocalTensor<float>mem;uint32_t matrix,quad,r0,c0;uint64_t ao;
};

// Lanen256_: independent block16 triangular caches, one resident triangle per AIV.
// Lanen256_: all eight AIV share the fixed original-A accurate residual.
class Lanen256_SharedDot2 {
    static constexpr uint32_t N=256,PS=69632,TOTAL=49152,CS=6144;
    static constexpr uint32_t AP=0,WORK=8192,XFULL=32768,BFULL=33536,XX=34304,BB=34560,RHS=34816,INDEX=35072;
public:
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR Bv,GM_ADDR workspace,const LinalgSolveTilingData&data,bool enabled){
        t=data;active=enabled;valid=enabled;worker=GetBlockIdx()%8;matrix=GetBlockIdx()/8;mem=LocalTensor<float>(TPosition::VECCALC,0,TOTAL);
        auto u=reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace));cache.SetGlobalBuffer(u+uint64_t(matrix)*8*PS+5*PS);shared.SetGlobalBuffer(u+t.schurOffset/4+uint64_t(matrix)*kOS_Total);
        debug.SetGlobalBuffer(u+uint64_t(t.blocks-1)*PS);ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Bv));uint32_t coords=matrix,ai=0,bi=0;for(int j=int(t.batchRank)-1;j>=0;--j){auto c=coords%t.batchDims[j];coords/=t.batchDims[j];ai+=c*t.aStrides[j];bi+=c*t.bStrides[j];}ao=uint64_t(ai)*N*N;bo=uint64_t(bi)*N*3;
    }
    __aicore__ inline void Process(){
        PipeBarrier<PIPE_ALL>();int64_t z=GetSystemCycle();
        if(active){
            Rx(mem[AP],ag[ao+worker*32*N],32*N);Rx(mem[BFULL],bg[bo],N*3);for(uint32_t rhs=0;rhs<3;++rhs)Rx(mem[XFULL+rhs*N],shared[kOS_Identity+2*CS+rhs*512+N],N);
            auto ix=mem[INDEX].ReinterpretCast<int32_t>();CreateVecIndex(ix,int32_t(0),N);PipeBarrier<PIPE_V>();Muls(ix,ix,int32_t(12),N);PipeBarrier<PIPE_V>();
        }
        for(uint32_t rhs=0;rhs<3;++rhs){
            if(active){DataCopy(mem[XX],mem[XFULL+rhs*N],N);Gather(mem[BB],mem[BFULL],mem[INDEX].ReinterpretCast<uint32_t>(),rhs*4,N);PipeBarrier<PIPE_V>();ResidualDot2();}
            else{Duplicate(mem[RHS+worker*32],0.0f,32);PipeBarrier<PIPE_V>();}
            Tx(cache[32768+rhs*N+worker*32],mem[RHS+worker*32],32);
        }
        PipeBarrier<PIPE_ALL>();uint32_t ticks=uint32_t(GetSystemCycle()-z);auto pk=mem[48000].ReinterpretCast<uint32_t>();Duplicate(pk,0U,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();pk.SetValue(0,uint32_t(valid));pk.SetValue(1,0x53445231U);pk.SetValue(2,GetBlockIdx());pk.SetValue(3,uint32_t(active));pk.SetValue(4,tilesDone);pk.SetValue(5,ticks);Tx(debug[42000+GetBlockIdx()*8],pk.ReinterpretCast<float>(),8);SyncAll<true>();
    }
private:
    __aicore__ inline void Rx(LocalTensor<float>d,GlobalTensor<float>s,uint32_t n){Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopy(d,s,n);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();}
    __aicore__ inline void Lanen256_ExtractSum(LocalTensor<float> dst,LocalTensor<float> src,LocalTensor<float> blocks){
        BlockReduceSum(blocks,src,int32_t(32),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();
        WholeReduceSum(dst,blocks,int32_t(32),uint8_t(8),1,1,4);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ResidualDot2(){
        // The lower cache is dead until the next Solve. Reuse it plus old solve scratch,
        // save X/B outside six Tile8 buffers, then reload this cache after the residual.
        constexpr uint32_t Tile=8,Count=Tile*N;
        constexpr uint32_t XH=20992,XL=21248,SH=21504,SL=21568,SE=21632,RT=21696,SX=21760,SB=22016,RR=22272;
        static_assert(RR+N<=TOTAL,"Dot2 temporal lower-cache alias bound");
        auto aa=mem[WORK],pp=aa[Count],ee=pp[Count],ss=ee[Count],zz=ss[Count],tt=zz[Count];
        auto xh=mem[XH],xl=mem[XL],high=mem[SH],low=mem[SL],error=mem[SE],rhsTerms=mem[RT],x=mem[SX],b=mem[SB],rr=mem[RR];
        DataCopy(x,mem[XX],N);DataCopy(b,mem[BB],N);PipeBarrier<PIPE_V>();
        Muls(xh,x,4097.0f,N);PipeBarrier<PIPE_V>();Sub(xl,xh,x,N);PipeBarrier<PIPE_V>();Sub(xh,xh,xl,N);PipeBarrier<PIPE_V>();Sub(xl,x,xh,N);PipeBarrier<PIPE_V>();
        for(uint32_t first=worker*32;first<(worker+1)*32;first+=Tile){++tilesDone;
            DataCopy(aa,mem[AP+(first-worker*32)*N],Count);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<N;c+=64)Mul(pp[c],aa[c],x[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();
            Muls(tt,aa,4097.0f,Count);PipeBarrier<PIPE_V>();Sub(zz,tt,aa,Count);PipeBarrier<PIPE_V>();Sub(ss,tt,zz,Count);PipeBarrier<PIPE_V>();Sub(zz,aa,ss,Count);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<N;c+=64)Mul(ee[c],ss[c],xh[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();Sub(ee,pp,ee,Count);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<N;c+=64)Mul(tt[c],zz[c],xh[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();Sub(ee,ee,tt,Count);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<N;c+=64)Mul(tt[c],ss[c],xl[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();Sub(ee,ee,tt,Count);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<N;c+=64)Mul(tt[c],zz[c],xl[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();Sub(ee,tt,ee,Count);PipeBarrier<PIPE_V>();
            // Exact-unit extraction uses one common normal power of two per Tile8.
            // The maximum over all 2048 products is cheap and bounds every row sum.
            Abs(aa,pp,Count);PipeBarrier<PIPE_V>();WholeReduceMax(high,aa,int32_t(64),uint8_t(32),1,1,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
            uint32_t maxBits,maxAt;GetReduceRepeatMaxMinSpr<float>(maxBits,maxAt);
            uint32_t exponent=((maxBits>>23)&255)+10;
            if(!(exponent>13 && exponent<254)){
                valid=false;Duplicate(rr,0.0f,N);PipeBarrier<PIPE_V>();break;
            }
            union Lanen256_Bits{uint32_t u;float f;}scale;scale.u=exponent<<23;
            Duplicate(ss,scale.f,Count);PipeBarrier<PIPE_V>();
            Add(zz,pp,ss,Count);PipeBarrier<PIPE_V>();Sub(zz,zz,ss,Count);PipeBarrier<PIPE_V>();Sub(tt,pp,zz,Count);PipeBarrier<PIPE_V>();
            Lanen256_ExtractSum(high,zz,aa);
            // Lanen256_: exact high-unit sum plus rounded small remainder.
            // Preserve full TwoProduct compensation; no original-A prepass.
            Lanen256_ExtractSum(low,tt,ss);Lanen256_ExtractSum(rhsTerms,ee,ss);
            Add(low,low,rhsTerms,64);PipeBarrier<PIPE_V>();
            Sub(ss,b[first],high,Tile);PipeBarrier<PIPE_V>();Sub(zz,b[first],ss,Tile);PipeBarrier<PIPE_V>();
            Add(tt,ss,zz,Tile);PipeBarrier<PIPE_V>();Sub(tt,b[first],tt,Tile);PipeBarrier<PIPE_V>();Sub(zz,zz,high,Tile);PipeBarrier<PIPE_V>();Add(tt,tt,zz,Tile);PipeBarrier<PIPE_V>();Sub(tt,tt,low,Tile);PipeBarrier<PIPE_V>();Add(rr[first],ss,tt,Tile);PipeBarrier<PIPE_V>();
        }
        DataCopy(mem[RHS+worker*32],rr[worker*32],32);PipeBarrier<PIPE_V>();
    }
    template<typename T>__aicore__ inline void Tx(GlobalTensor<T>d,LocalTensor<T>s,uint32_t n){Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();DataCopy(d,s,n);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();}
    TPipe pipe;LinalgSolveTilingData t;LocalTensor<float>mem;GlobalTensor<float>ag,bg,cache,shared,debug;uint32_t worker,matrix,tilesDone=0;bool active=true,valid=true;uint64_t ao,bo;
};
class Lanen256_SharedResidualSolve16 {
    static constexpr uint32_t N=256,B=16,NB=16,RP=24,CS=NB*B*RP,TS=NB*(NB+1)/2*B*B;
    static constexpr uint32_t CACHE=TS,RHS=CACHE+CS,XX=RHS+N,BB=XX+N,TEMP=BB+N;
    static constexpr uint32_t SOL=TEMP+16*N,SUM=SOL+64,INDEX=SUM+64,PACK=INDEX+N,TOTAL=48640;
    static_assert(TOTAL*4<=196608,"N256 resident triangle and cache budget");
public:
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR Bv,GM_ADDR X,GM_ADDR workspace,const LinalgSolveTilingData &data,bool enabled){
        inputA=A;inputB=Bv;workspaceBase=workspace;t=data;factorEnabled=enabled;valid=enabled;worker=GetBlockIdx()%8;pair=worker/2;lower=(worker&1);matrix=GetBlockIdx()/8;
        mem=LocalTensor<float>(TPosition::VECCALC,0,TOTAL);
        auto user=reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace));
        shared.SetGlobalBuffer(user+t.schurOffset/4+uint64_t(matrix)*kOS_Total);
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Bv));xg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(X));
        uint32_t coords=matrix,ai=0,bi=0;
        for(int d=int(t.batchRank)-1;d>=0;--d){uint32_t v=coords%t.batchDims[d];coords/=t.batchDims[d];ai+=v*t.aStrides[d];bi+=v*t.bStrides[d];}
        ao=uint64_t(ai)*N*N;bo=uint64_t(bi)*N*3;
    }
    __aicore__ inline bool Process(){
        auto rhs=mem[RHS],x=mem[XX],b=mem[BB];
        if(pair<3){
            if(factorEnabled){LoadTriangle();Rx(mem[CACHE],shared[kOS_Identity+(lower?CS:0)],CS);}
            if(lower){
                Lanen256_LoadB();DataCopy(rhs,b,N);PipeBarrier<PIPE_V>();if(factorEnabled)Solve();else{Duplicate(rhs,0.0f,N);PipeBarrier<PIPE_V>();}
                Tx(shared[kOS_Identity+2*CS+pair*512],rhs,N);PairSync();PairSync();
            }else{
                PairSync();Rx(rhs,shared[kOS_Identity+2*CS+pair*512],N);if(factorEnabled)Solve();else{Duplicate(rhs,0.0f,N);PipeBarrier<PIPE_V>();}
                DataCopy(x,rhs,N);PipeBarrier<PIPE_V>();GlobalTensor<float>initial;initial.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspaceBase))+uint64_t(t.blocks-1)*N*272+1024+GetBlockIdx()*N);Tx(initial,x,N);Tx(shared[kOS_Identity+2*CS+pair*512+N],x,N);PairSync();
            }
        }
        SyncAll<true>();
        {Lanen256_SharedDot2 residual;residual.Init(inputA,inputB,workspaceBase,t,factorEnabled);residual.Process();}
        if(pair>=3)return factorEnabled;
        // Residual temporaries intentionally retire both resident triangles.
        // Reload all consumers before using them again; include this in whole cost.
        if(factorEnabled){LoadTriangle();Rx(mem[CACHE],shared[kOS_Identity+(lower?CS:0)],CS);}
        Lanen256_CheckResidualFlags();
        if(lower){
            Lanen256_LoadB();GlobalTensor<float>rg;rg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspaceBase))+uint64_t(matrix)*8*N*272+5*N*272+32768+pair*N);Rx(rhs,rg,N);
            if(valid)Solve();else{Duplicate(rhs,0.0f,N);PipeBarrier<PIPE_V>();}Tx(shared[kOS_Identity+2*CS+pair*512],rhs,N);PairSync();PairSync();
            Rx(x,shared[kOS_Identity+2*CS+pair*512+N],N);Residual();bool good=Lanen256_Guard();
            for(uint32_t extra=0;extra<3;++extra){
                bool more=valid&&!good;auto flag=mem[SOL].ReinterpretCast<uint32_t>();Duplicate(flag,0U,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();flag.SetValue(0,more?1U:0U);flag.SetValue(1,extra);Tx(shared[kOS_Identity+2*CS+3*512+pair*8],flag.ReinterpretCast<float>(),8);PairSync();if(!more)break;
                Solve();Tx(shared[kOS_Identity+2*CS+pair*512],rhs,N);PairSync();PairSync();Rx(x,shared[kOS_Identity+2*CS+pair*512+N],N);Residual();good=Lanen256_Guard();
            }return good;
        }else{
            Rx(x,shared[kOS_Identity+2*CS+pair*512+N],N);PairSync();Rx(rhs,shared[kOS_Identity+2*CS+pair*512],N);if(valid)Solve();else{Duplicate(rhs,0.0f,N);PipeBarrier<PIPE_V>();}Add(x,x,rhs,N);PipeBarrier<PIPE_V>();Tx(shared[kOS_Identity+2*CS+pair*512+N],x,N);PairSync();
            for(uint32_t extra=0;extra<3;++extra){PairSync();Rx(mem[SOL],shared[kOS_Identity+2*CS+3*512+pair*8],8);bool more=mem[SOL].ReinterpretCast<uint32_t>().GetValue(0)!=0;if(!more)break;PairSync();Rx(rhs,shared[kOS_Identity+2*CS+pair*512],N);Solve();Add(x,x,rhs,N);PipeBarrier<PIPE_V>();Tx(shared[kOS_Identity+2*CS+pair*512+N],x,N);PairSync();}
            GlobalTensor<float>finalSnapshot;finalSnapshot.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspaceBase))+uint64_t(t.blocks-1)*N*272+16384+GetBlockIdx()*N);Tx(finalSnapshot,x,N);auto broad=mem[TEMP];Brcb(broad,x,N/8,{1,8});PipeBarrier<PIPE_V>();Fence<HardEvent::V_MTE3>();DataCopyExtParams px{N,4,0,2*4,0};DataCopyPad(xg[uint64_t(matrix)*N*3+pair],broad,px);Fence<HardEvent::MTE3_V>();return true;
        }
    }

private:
    __aicore__ inline void Lanen256_LoadB(){
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();DataCopyExtParams pb{N,12,0,0,0};DataCopyPadExtParams<float>pad{true,0,5,0.0f};DataCopyPad(mem[PACK],bg[bo],pb,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();auto ix=mem[INDEX].ReinterpretCast<uint32_t>();for(uint32_t i=0;i<N;++i)ix.SetValue(i,i*8*4);Fence<HardEvent::S_V>();Gather(mem[BB],mem[PACK],ix,pair*4,N);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Lanen256_CheckResidualFlags(){
        GlobalTensor<float>flags;flags.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspaceBase))+uint64_t(t.blocks-1)*N*272+42000+matrix*8*8);Rx(mem[SOL],flags,64);auto pk=mem[SOL].ReinterpretCast<uint32_t>();for(uint32_t w=0;w<8;++w)valid=valid&&(pk.GetValue(w*8)!=0);
    }
    __aicore__ inline void Rx(LocalTensor<float>d,GlobalTensor<float>s,uint32_t count){Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopy(d,s,count);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();}
    __aicore__ inline void Tx(GlobalTensor<float>d,LocalTensor<float>s,uint32_t count){Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();DataCopy(d,s,count);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();}
    __aicore__ inline void PairSync(){CrossCoreSetFlag<1,PIPE_MTE3>(4);CrossCoreWaitFlag<1>(4);}
    __aicore__ inline uint32_t Offset(uint32_t block){return lower?B*B*block*(block+1)/2:B*B*(NB*block-block*(block-1)/2);}
    __aicore__ inline void LoadTriangle(){
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();
        for(uint32_t block=0;block<NB;++block){
            uint32_t first=block*B,width=lower?first+B:N-first,column=lower?0:first;
            DataCopyExtParams p{B,width*4,(N-width)*4,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};DataCopyPad(mem[Offset(block)],shared[kOS_U+first*N+column],p,pad);
        }
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline bool Lanen256_Guard(){
        auto tmp=mem[TEMP],sum=mem[SUM];Mul(tmp,mem[RHS],mem[RHS],N);Mul(tmp[N],mem[BB],mem[BB],N);PipeBarrier<PIPE_V>();
        WholeReduceSum(sum,tmp,int32_t(64),uint8_t(8),1,1,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
        float rn=0.0f,bn=0.0f;for(uint32_t j=0;j<4;++j){rn+=sum.GetValue(j);bn+=sum.GetValue(j+4);}
        return valid && rn>=0.0f && bn>=0.0f && rn<=2.5e-9f*(bn+1.0e-20f);
    }
    __aicore__ inline void Lanen256_PermuteRHS(){
        DataCopy(mem[TEMP],mem[RHS],N);PipeBarrier<PIPE_V>();Gather(mem[RHS],mem[TEMP],mem.ReinterpretCast<uint32_t>(),0,N);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Solve(){
        auto rhs=mem[RHS],product=mem[TEMP],sol=mem[SOL],sum=mem[SUM];
        for(uint32_t step=0;step<NB;++step){
            uint32_t block=lower?step:NB-1-step,first=block*B,stride=lower?first+B:N-first;
            uint32_t start=lower?0:first+B,length=lower?first:N-first-B,local=lower?0:B,offset=Offset(block);
            if(length){
                for(uint32_t col=0;col<length;col+=64){uint32_t mask=length-col<64?length-col:64;Mul(product[col],mem[offset+local+col],rhs[start+col],uint64_t(mask),uint8_t(B),{1,1,1,N/8,uint8_t(stride/8),0});}
                PipeBarrier<PIPE_V>();
                uint32_t m=length<64?length:64;WholeReduceSum(sum,product,int32_t(m),uint8_t(B),1,1,N/8);PipeBarrier<PIPE_V>();
                for(uint32_t col=64;col<length;col+=64){uint32_t mask=length-col<64?length-col:64;WholeReduceSum(sol,product[col],int32_t(mask),uint8_t(B),1,1,N/8);PipeBarrier<PIPE_V>();Add(sum,sum,sol,B);PipeBarrier<PIPE_V>();}
                Sub(rhs[first],rhs[first],sum,B);PipeBarrier<PIPE_V>();
            }
            Mul(product,mem[CACHE+block*B*RP],rhs[first],uint64_t(B),uint8_t(B),{1,1,1,RP/8,RP/8,0});PipeBarrier<PIPE_V>();
            WholeReduceSum(sol,product,int32_t(B),uint8_t(B),1,1,RP/8);PipeBarrier<PIPE_V>();DataCopy(rhs[first],sol,B);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Lanen256_ExtractSum(LocalTensor<float> dst,LocalTensor<float> src,LocalTensor<float> blocks){
        BlockReduceSum(blocks,src,int32_t(32),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();
        WholeReduceSum(dst,blocks,int32_t(32),uint8_t(8),1,1,4);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Lanen256_ResidualDot2(){
        // The lower cache is dead until the next Solve. Reuse it plus old solve scratch,
        // save X/B outside six Tile8 buffers, then reload this cache after the residual.
        constexpr uint32_t Tile=8,Count=Tile*N;
        constexpr uint32_t WORK=CACHE,XH=WORK+6*Count,XL=XH+N,SH=XL+N,SL=SH+64,SE=SL+64,RT=SE+64,SX=RT+64,SB=SX+N,RR=SB+N;
        static_assert(RR+N<=TOTAL,"Dot2 temporal lower-cache alias bound");
        auto aa=mem[WORK],pp=aa[Count],ee=pp[Count],ss=ee[Count],zz=ss[Count],tt=zz[Count];
        auto xh=mem[XH],xl=mem[XL],high=mem[SH],low=mem[SL],error=mem[SE],rhsTerms=mem[RT],x=mem[SX],b=mem[SB],rr=mem[RR];
        DataCopy(x,mem[XX],N);DataCopy(b,mem[BB],N);PipeBarrier<PIPE_V>();
        Muls(xh,x,4097.0f,N);PipeBarrier<PIPE_V>();Sub(xl,xh,x,N);PipeBarrier<PIPE_V>();Sub(xh,xh,xl,N);PipeBarrier<PIPE_V>();Sub(xl,x,xh,N);PipeBarrier<PIPE_V>();
        for(uint32_t first=0;first<N;first+=Tile){
            Rx(aa,ag[ao+uint64_t(first)*N],Count);
            for(uint32_t c=0;c<N;c+=64)Mul(pp[c],aa[c],x[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();
            Muls(tt,aa,4097.0f,Count);PipeBarrier<PIPE_V>();Sub(zz,tt,aa,Count);PipeBarrier<PIPE_V>();Sub(ss,tt,zz,Count);PipeBarrier<PIPE_V>();Sub(zz,aa,ss,Count);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<N;c+=64)Mul(ee[c],ss[c],xh[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();Sub(ee,pp,ee,Count);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<N;c+=64)Mul(tt[c],zz[c],xh[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();Sub(ee,ee,tt,Count);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<N;c+=64)Mul(tt[c],ss[c],xl[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();Sub(ee,ee,tt,Count);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<N;c+=64)Mul(tt[c],zz[c],xl[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();Sub(ee,tt,ee,Count);PipeBarrier<PIPE_V>();
            // Exact-unit extraction uses one common normal power of two per Tile8.
            // The maximum over all 2048 products is cheap and bounds every row sum.
            Abs(aa,pp,Count);PipeBarrier<PIPE_V>();WholeReduceMax(high,aa,int32_t(64),uint8_t(32),1,1,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
            uint32_t maxBits,maxAt;GetReduceRepeatMaxMinSpr<float>(maxBits,maxAt);
            uint32_t exponent=((maxBits>>23)&255)+10;
            if(!(exponent>13 && exponent<254)){
                valid=false;Duplicate(rr,0.0f,N);PipeBarrier<PIPE_V>();break;
            }
            union Lanen256_Bits{uint32_t u;float f;}scale;scale.u=exponent<<23;
            Duplicate(ss,scale.f,Count);PipeBarrier<PIPE_V>();
            Add(zz,pp,ss,Count);PipeBarrier<PIPE_V>();Sub(zz,zz,ss,Count);PipeBarrier<PIPE_V>();Sub(tt,pp,zz,Count);PipeBarrier<PIPE_V>();
            Lanen256_ExtractSum(high,zz,aa);
            // Lanen256_: exact high-unit sum plus rounded small remainder.
            // Preserve full TwoProduct compensation; no original-A prepass.
            Lanen256_ExtractSum(low,tt,ss);Lanen256_ExtractSum(rhsTerms,ee,ss);
            Add(low,low,rhsTerms,64);PipeBarrier<PIPE_V>();
            Sub(ss,b[first],high,Tile);PipeBarrier<PIPE_V>();Sub(zz,b[first],ss,Tile);PipeBarrier<PIPE_V>();
            Add(tt,ss,zz,Tile);PipeBarrier<PIPE_V>();Sub(tt,b[first],tt,Tile);PipeBarrier<PIPE_V>();Sub(zz,zz,high,Tile);PipeBarrier<PIPE_V>();Add(tt,tt,zz,Tile);PipeBarrier<PIPE_V>();Sub(tt,tt,low,Tile);PipeBarrier<PIPE_V>();Add(rr[first],ss,tt,Tile);PipeBarrier<PIPE_V>();
        }
        DataCopy(mem[RHS],rr,N);DataCopy(mem[XX],x,N);DataCopy(mem[BB],b,N);PipeBarrier<PIPE_V>();Rx(mem[CACHE],shared[kOS_Identity+CS],CS);
    }

    __aicore__ inline void Residual(){
        constexpr uint32_t Tile=8,Count=Tile*N;auto aa=mem[TEMP],pp=aa[Count],sum=mem[SUM],term=mem[SOL],rhs=mem[RHS],x=mem[XX],b=mem[BB];
        for(uint32_t first=0;first<N;first+=Tile){
            Rx(aa,ag[ao+uint64_t(first)*N],Count);
            for(uint32_t c=0;c<N;c+=64)Mul(pp[c],aa[c],x[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();
            WholeReduceSum(sum,pp,int32_t(64),uint8_t(Tile),1,1,N/8);PipeBarrier<PIPE_V>();
            for(uint32_t c=64;c<N;c+=64){WholeReduceSum(term,pp[c],int32_t(64),uint8_t(Tile),1,1,N/8);PipeBarrier<PIPE_V>();Add(sum,sum,term,Tile);PipeBarrier<PIPE_V>();}
            Sub(rhs[first],b[first],sum,Tile);PipeBarrier<PIPE_V>();
        }
    }
    TPipe pipe;GM_ADDR inputA;GM_ADDR inputB;GM_ADDR workspaceBase;LinalgSolveTilingData t;LocalTensor<float>mem;GlobalTensor<float>ag,bg,xg,shared;uint32_t worker,pair,matrix;bool lower;bool valid=true,factorEnabled=true;uint64_t ao,bo;
};

class Lanen256_NoPPSolve16 {
    static constexpr uint32_t N=256,B=16,NB=16,RP=24,CS=NB*B*RP,TS=NB*(NB+1)/2*B*B;
    static constexpr uint32_t CACHE=TS,RHS=CACHE+CS,XX=RHS+N,BB=XX+N,TEMP=BB+N;
    static constexpr uint32_t SOL=TEMP+16*N,SUM=SOL+64,INDEX=SUM+64,PACK=INDEX+N,TOTAL=48640;
    static_assert(TOTAL*4<=196608,"N256 resident triangle and cache budget");
public:
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR Bv,GM_ADDR X,GM_ADDR workspace,const LinalgSolveTilingData &data){
        workspaceBase=workspace;t=data;worker=GetBlockIdx()%8;pair=worker/2;lower=(worker&1);matrix=GetBlockIdx()/8;
        mem=LocalTensor<float>(TPosition::VECCALC,0,TOTAL);
        auto user=reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace));
        shared.SetGlobalBuffer(user+t.schurOffset/4+uint64_t(matrix)*kOS_Total);
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Bv));xg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(X));
        uint32_t coords=matrix,ai=0,bi=0;
        for(int d=int(t.batchRank)-1;d>=0;--d){uint32_t v=coords%t.batchDims[d];coords/=t.batchDims[d];ai+=v*t.aStrides[d];bi+=v*t.bStrides[d];}
        ao=uint64_t(ai)*N*N;bo=uint64_t(bi)*N*3;
    }
    __aicore__ inline bool Process(){
        if(pair>=3)return true;
        LoadTriangle();Rx(mem[CACHE],shared[kOS_Identity+(lower?CS:0)],CS);
        auto rhs=mem[RHS],x=mem[XX],b=mem[BB];
        if(lower){
            Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
            DataCopyExtParams pb{N,3*4,0,0,0};DataCopyPadExtParams<float> pad{true,0,5,0.0f};DataCopyPad(mem[PACK],bg[bo],pb,pad);
            Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
            auto index=mem[INDEX].ReinterpretCast<uint32_t>();for(uint32_t i=0;i<N;++i)index.SetValue(i,i*8*4);Fence<HardEvent::S_V>();
            Gather(b,mem[PACK],index,pair*4,N);PipeBarrier<PIPE_V>();DataCopy(rhs,b,N);PipeBarrier<PIPE_V>();
            Solve();Tx(shared[kOS_Identity+2*CS+pair*512],rhs,N);PairSync();PairSync();
            for(uint32_t ir=0;ir<1;++ir){
                Rx(x,shared[kOS_Identity+2*CS+pair*512+N],N);Lanen256_ResidualDot2();Solve();Tx(shared[kOS_Identity+2*CS+pair*512],rhs,N);PairSync();PairSync();
            }
            Rx(x,shared[kOS_Identity+2*CS+pair*512+N],N);Residual();
            bool good=Lanen256_Guard();
            for(uint32_t extra=0;extra<3;++extra){
                bool more=valid && !good;
                auto flag=mem[SOL].ReinterpretCast<uint32_t>();Duplicate(flag,0U,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();flag.SetValue(0,more?1U:0U);flag.SetValue(1,extra);Tx(shared[kOS_Identity+2*CS+3*512+pair*8],flag.ReinterpretCast<float>(),8);PairSync();
                if(!more)break;
                // Reuse the already checked FP32 residual. No additional Dot2 product.
                Solve();Tx(shared[kOS_Identity+2*CS+pair*512],rhs,N);PairSync();PairSync();
                Rx(x,shared[kOS_Identity+2*CS+pair*512+N],N);Residual();good=Lanen256_Guard();
            }
            return good;
        }else{
            PairSync();Rx(rhs,shared[kOS_Identity+2*CS+pair*512],N);Solve();DataCopy(x,rhs,N);PipeBarrier<PIPE_V>();GlobalTensor<float> initial;initial.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspaceBase))+uint64_t(t.blocks-1)*N*272+1024+GetBlockIdx()*N);Tx(initial,x,N);Tx(shared[kOS_Identity+2*CS+pair*512+N],x,N);PairSync();
            for(uint32_t ir=0;ir<1;++ir){
                PairSync();Rx(rhs,shared[kOS_Identity+2*CS+pair*512],N);Solve();Add(x,x,rhs,N);PipeBarrier<PIPE_V>();Tx(shared[kOS_Identity+2*CS+pair*512+N],x,N);PairSync();
            }
            for(uint32_t extra=0;extra<3;++extra){
                PairSync();Rx(mem[SOL],shared[kOS_Identity+2*CS+3*512+pair*8],8);bool more=mem[SOL].ReinterpretCast<uint32_t>().GetValue(0)!=0;if(!more)break;
                PairSync();Rx(rhs,shared[kOS_Identity+2*CS+pair*512],N);Solve();Add(x,x,rhs,N);PipeBarrier<PIPE_V>();Tx(shared[kOS_Identity+2*CS+pair*512+N],x,N);PairSync();
            }
            GlobalTensor<float> finalSnapshot;finalSnapshot.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspaceBase))+uint64_t(t.blocks-1)*N*272+16384+GetBlockIdx()*N);Tx(finalSnapshot,x,N);
            auto broad=mem[TEMP];Brcb(broad,x,N/8,{1,8});PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_MTE3>();DataCopyExtParams px{N,4,0,2*4,0};DataCopyPad(xg[uint64_t(matrix)*N*3+pair],broad,px);Fence<HardEvent::MTE3_V>();
            return true;
        }
    }
private:
    __aicore__ inline void Rx(LocalTensor<float>d,GlobalTensor<float>s,uint32_t count){Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopy(d,s,count);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();}
    __aicore__ inline void Tx(GlobalTensor<float>d,LocalTensor<float>s,uint32_t count){Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();DataCopy(d,s,count);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();}
    __aicore__ inline void PairSync(){CrossCoreSetFlag<1,PIPE_MTE3>(4);CrossCoreWaitFlag<1>(4);}
    __aicore__ inline uint32_t Offset(uint32_t block){return lower?B*B*block*(block+1)/2:B*B*(NB*block-block*(block-1)/2);}
    __aicore__ inline void LoadTriangle(){
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();
        for(uint32_t block=0;block<NB;++block){
            uint32_t first=block*B,width=lower?first+B:N-first,column=lower?0:first;
            DataCopyExtParams p{B,width*4,(N-width)*4,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};DataCopyPad(mem[Offset(block)],shared[kOS_U+first*N+column],p,pad);
        }
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline bool Lanen256_Guard(){
        auto tmp=mem[TEMP],sum=mem[SUM];Mul(tmp,mem[RHS],mem[RHS],N);Mul(tmp[N],mem[BB],mem[BB],N);PipeBarrier<PIPE_V>();
        WholeReduceSum(sum,tmp,int32_t(64),uint8_t(8),1,1,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
        float rn=0.0f,bn=0.0f;for(uint32_t j=0;j<4;++j){rn+=sum.GetValue(j);bn+=sum.GetValue(j+4);}
        return valid && rn>=0.0f && bn>=0.0f && rn<=2.5e-9f*(bn+1.0e-20f);
    }
    __aicore__ inline void Lanen256_PermuteRHS(){
        DataCopy(mem[TEMP],mem[RHS],N);PipeBarrier<PIPE_V>();Gather(mem[RHS],mem[TEMP],mem.ReinterpretCast<uint32_t>(),0,N);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Solve(){
        auto rhs=mem[RHS],product=mem[TEMP],sol=mem[SOL],sum=mem[SUM];
        for(uint32_t step=0;step<NB;++step){
            uint32_t block=lower?step:NB-1-step,first=block*B,stride=lower?first+B:N-first;
            uint32_t start=lower?0:first+B,length=lower?first:N-first-B,local=lower?0:B,offset=Offset(block);
            if(length){
                for(uint32_t col=0;col<length;col+=64){uint32_t mask=length-col<64?length-col:64;Mul(product[col],mem[offset+local+col],rhs[start+col],uint64_t(mask),uint8_t(B),{1,1,1,N/8,uint8_t(stride/8),0});}
                PipeBarrier<PIPE_V>();
                uint32_t m=length<64?length:64;WholeReduceSum(sum,product,int32_t(m),uint8_t(B),1,1,N/8);PipeBarrier<PIPE_V>();
                for(uint32_t col=64;col<length;col+=64){uint32_t mask=length-col<64?length-col:64;WholeReduceSum(sol,product[col],int32_t(mask),uint8_t(B),1,1,N/8);PipeBarrier<PIPE_V>();Add(sum,sum,sol,B);PipeBarrier<PIPE_V>();}
                Sub(rhs[first],rhs[first],sum,B);PipeBarrier<PIPE_V>();
            }
            Mul(product,mem[CACHE+block*B*RP],rhs[first],uint64_t(B),uint8_t(B),{1,1,1,RP/8,RP/8,0});PipeBarrier<PIPE_V>();
            WholeReduceSum(sol,product,int32_t(B),uint8_t(B),1,1,RP/8);PipeBarrier<PIPE_V>();DataCopy(rhs[first],sol,B);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Lanen256_ExtractSum(LocalTensor<float> dst,LocalTensor<float> src,LocalTensor<float> blocks){
        BlockReduceSum(blocks,src,int32_t(32),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();
        WholeReduceSum(dst,blocks,int32_t(32),uint8_t(8),1,1,4);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Lanen256_ResidualDot2(){
        // The lower cache is dead until the next Solve. Reuse it plus old solve scratch,
        // save X/B outside six Tile8 buffers, then reload this cache after the residual.
        constexpr uint32_t Tile=8,Count=Tile*N;
        constexpr uint32_t WORK=CACHE,XH=WORK+6*Count,XL=XH+N,SH=XL+N,SL=SH+64,SE=SL+64,RT=SE+64,SX=RT+64,SB=SX+N,RR=SB+N;
        static_assert(RR+N<=TOTAL,"Dot2 temporal lower-cache alias bound");
        auto aa=mem[WORK],pp=aa[Count],ee=pp[Count],ss=ee[Count],zz=ss[Count],tt=zz[Count];
        auto xh=mem[XH],xl=mem[XL],high=mem[SH],low=mem[SL],error=mem[SE],rhsTerms=mem[RT],x=mem[SX],b=mem[SB],rr=mem[RR];
        DataCopy(x,mem[XX],N);DataCopy(b,mem[BB],N);PipeBarrier<PIPE_V>();
        Muls(xh,x,4097.0f,N);PipeBarrier<PIPE_V>();Sub(xl,xh,x,N);PipeBarrier<PIPE_V>();Sub(xh,xh,xl,N);PipeBarrier<PIPE_V>();Sub(xl,x,xh,N);PipeBarrier<PIPE_V>();
        for(uint32_t first=0;first<N;first+=Tile){
            Rx(aa,ag[ao+uint64_t(first)*N],Count);
            for(uint32_t c=0;c<N;c+=64)Mul(pp[c],aa[c],x[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();
            Muls(tt,aa,4097.0f,Count);PipeBarrier<PIPE_V>();Sub(zz,tt,aa,Count);PipeBarrier<PIPE_V>();Sub(ss,tt,zz,Count);PipeBarrier<PIPE_V>();Sub(zz,aa,ss,Count);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<N;c+=64)Mul(ee[c],ss[c],xh[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();Sub(ee,pp,ee,Count);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<N;c+=64)Mul(tt[c],zz[c],xh[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();Sub(ee,ee,tt,Count);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<N;c+=64)Mul(tt[c],ss[c],xl[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();Sub(ee,ee,tt,Count);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<N;c+=64)Mul(tt[c],zz[c],xl[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();Sub(ee,tt,ee,Count);PipeBarrier<PIPE_V>();
            // Exact-unit extraction uses one common normal power of two per Tile8.
            // The maximum over all 2048 products is cheap and bounds every row sum.
            Abs(aa,pp,Count);PipeBarrier<PIPE_V>();WholeReduceMax(high,aa,int32_t(64),uint8_t(32),1,1,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
            uint32_t maxBits,maxAt;GetReduceRepeatMaxMinSpr<float>(maxBits,maxAt);
            uint32_t exponent=((maxBits>>23)&255)+10;
            if(!(exponent>13 && exponent<254)){
                valid=false;Duplicate(rr,0.0f,N);PipeBarrier<PIPE_V>();break;
            }
            union Lanen256_Bits{uint32_t u;float f;}scale;scale.u=exponent<<23;
            Duplicate(ss,scale.f,Count);PipeBarrier<PIPE_V>();
            Add(zz,pp,ss,Count);PipeBarrier<PIPE_V>();Sub(zz,zz,ss,Count);PipeBarrier<PIPE_V>();Sub(tt,pp,zz,Count);PipeBarrier<PIPE_V>();
            Lanen256_ExtractSum(high,zz,aa);
            Muls(ss,ss,0.000244140625f,Count);PipeBarrier<PIPE_V>();Add(aa,tt,ss,Count);PipeBarrier<PIPE_V>();Sub(aa,aa,ss,Count);PipeBarrier<PIPE_V>();Sub(zz,tt,aa,Count);PipeBarrier<PIPE_V>();
            // tt is dead; use its first256 values as the row-block reduction scratch.
            Lanen256_ExtractSum(low,aa,tt);Lanen256_ExtractSum(error,zz,tt);Lanen256_ExtractSum(rhsTerms,ee,tt);
            Add(low,low,error,64);PipeBarrier<PIPE_V>();Add(low,low,rhsTerms,64);PipeBarrier<PIPE_V>();
            Sub(ss,b[first],high,Tile);PipeBarrier<PIPE_V>();Sub(zz,b[first],ss,Tile);PipeBarrier<PIPE_V>();
            Add(tt,ss,zz,Tile);PipeBarrier<PIPE_V>();Sub(tt,b[first],tt,Tile);PipeBarrier<PIPE_V>();Sub(zz,zz,high,Tile);PipeBarrier<PIPE_V>();Add(tt,tt,zz,Tile);PipeBarrier<PIPE_V>();Sub(tt,tt,low,Tile);PipeBarrier<PIPE_V>();Add(rr[first],ss,tt,Tile);PipeBarrier<PIPE_V>();
        }
        DataCopy(mem[RHS],rr,N);DataCopy(mem[XX],x,N);DataCopy(mem[BB],b,N);PipeBarrier<PIPE_V>();Rx(mem[CACHE],shared[kOS_Identity+CS],CS);
    }

    __aicore__ inline void Residual(){
        constexpr uint32_t Tile=8,Count=Tile*N;auto aa=mem[TEMP],pp=aa[Count],sum=mem[SUM],term=mem[SOL],rhs=mem[RHS],x=mem[XX],b=mem[BB];
        for(uint32_t first=0;first<N;first+=Tile){
            Rx(aa,ag[ao+uint64_t(first)*N],Count);
            for(uint32_t c=0;c<N;c+=64)Mul(pp[c],aa[c],x[c],uint64_t(64),uint8_t(Tile),{1,1,1,N/8,N/8,0});PipeBarrier<PIPE_V>();
            WholeReduceSum(sum,pp,int32_t(64),uint8_t(Tile),1,1,N/8);PipeBarrier<PIPE_V>();
            for(uint32_t c=64;c<N;c+=64){WholeReduceSum(term,pp[c],int32_t(64),uint8_t(Tile),1,1,N/8);PipeBarrier<PIPE_V>();Add(sum,sum,term,Tile);PipeBarrier<PIPE_V>();}
            Sub(rhs[first],b[first],sum,Tile);PipeBarrier<PIPE_V>();
        }
    }
    TPipe pipe;GM_ADDR workspaceBase;LinalgSolveTilingData t;LocalTensor<float>mem;GlobalTensor<float>ag,bg,xg,shared;uint32_t worker,pair,matrix;bool lower;bool valid=true;uint64_t ao,bo;
};

__aicore__ inline bool Lanen256_TryCubeLUT16(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR workspace,const LinalgSolveTilingData &data){
    if(data.blocks>32)return false;
    bool factorOK=true;{Lanen256_CubeLUT16 factor;factor.Init(A,B,workspace,data);factor.Process();if ASCEND_IS_AIV{factorOK=factor.FactorOK();}}
    GlobalTensor<uint32_t> decision;decision.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(GetUserWorkspace(workspace))+uint64_t(data.blocks-1)*256*(data.nPad+2*data.kPad));
    if ASCEND_IS_AIC{
        CrossCoreWaitFlag<2>(7);DataCacheCleanAndInvalid<uint32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(decision);return decision.GetValue(0)!=0;
    }else{
        if(GetBlockIdx()%8==0){LocalTensor<uint32_t>fs(TPosition::VECCALC,0,8);Duplicate(fs,0U,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();fs.SetValue(0,factorOK?1U:0U);Fence<HardEvent::S_MTE3>();DataCopy(decision[512+(GetBlockIdx()/8)*8],fs,8);Fence<HardEvent::MTE3_S>();}
        bool ok=false;{Lanen256_SharedResidualSolve16 solver;solver.Init(A,B,X,workspace,data,factorOK);ok=solver.Process();}
        LocalTensor<uint32_t> record(TPosition::VECCALC,0,264);Duplicate(record,0U,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();record.SetValue(0,ok?1:0);Fence<HardEvent::S_MTE3>();DataCopy(decision[8+GetBlockIdx()*8],record,8);Fence<HardEvent::MTE3_S>();SyncAll<true>();
        if(GetBlockIdx()==0){Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopy(record,decision[8],data.blocks*8);Fence<HardEvent::MTE2_S>();bool all=true;for(uint32_t i=0;i<data.blocks;++i)all=all && record.GetValue(i*8)!=0;record.SetValue(0,all?1:0);Fence<HardEvent::S_MTE3>();DataCopy(decision,record,8);Fence<HardEvent::MTE3_S>();}
        SyncAll<true>();Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopy(record,decision,8);Fence<HardEvent::MTE2_S>();bool all=record.GetValue(0)!=0;CrossCoreSetFlag<2,PIPE_MTE3>(7);return all;
    }
}


// Partial-pivot LU, retained for iterative refinement. Each vector core owns
// independent systems; all arithmetic, including the correction, runs on NPU.
template <bool OnChip, bool Lanen200_UseExtractedResidual=false, bool Lanen200_BlockTri16=false> class Solver {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR workspace,
                                const LinalgSolveTilingData &data) {
        userWorkspace=GetUserWorkspace(workspace);
        t = data; n = t.n; k = t.k; np = t.nPad; kp = t.kPad; rp = t.rowPitch;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
        const uint64_t elements = uint64_t(n) * (np + 2 * kp);
        if (!OnChip || (t.dotSize && n >= 128 && n <= 256)) {
            wg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace))
                               + uint64_t(GetBlockIdx()) * elements);
        }
        pipe.InitBuffer(storage, OnChip ? elements * 4 : (t.dotSize ? 2 * n * kp * 4 : 32));
        mat = storage.Get<float>();
        uint32_t scratchFloats = 2 * rp + 9 * np + 8 * kp;
        if (17 * t.dotSize > scratchFloats) scratchFloats = 17 * t.dotSize;
        pipe.InitBuffer(scratch, scratchFloats * 4);
        row = scratch.Get<float>(); other = row[rp]; column = other[rp];
        original = column[8 * np]; v0 = original[np];
        v1 = v0[kp]; v2 = v1[kp]; v3 = v2[kp];
        v4 = v3[kp]; v5 = v4[kp]; v6 = v5[kp]; v7 = v6[kp];
        pipe.InitBuffer(pivotBuffer, np * 4); pivots = pivotBuffer.Get<int32_t>();
        if ((t.paired && !(GetBlockIdx() & 1)) || t.blocked) {
            GlobalTensor<int32_t> pg;
            pg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(GetUserWorkspace(workspace))
                               + uint64_t(GetBlockIdx()) * elements + n * rp);
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            DataCopy(pivots, pg, n);
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        }
        const uint32_t residualRows = n >= 128 ? 16 : 8;
        const uint32_t tileBytes = (6 * residualRows * t.dotSize + 2 * n * kp + 3 * t.dotSize + 5 * residualRows * 8) * 4;
        uint32_t triangleBytes = (!OnChip && t.dotSize && n <= 256) ? n * (np + 8) * 2 : 32;
        if (t.dotSize && n > 16 && n <= 64) triangleBytes = tileBytes;
        if (!OnChip && t.dotSize && n >= 128 && n <= 256 && tileBytes > triangleBytes) triangleBytes = tileBytes;
        pipe.InitBuffer(triangleBuffer, triangleBytes);
        triangle = triangleBuffer.Get<float>();
        if(t.originalSchur){
            static_assert(2*256*8*4+17*256*4+256*4+256*(256+8)*2+16*256*4<=184*1024,"Block solve UB");
            pipe.InitBuffer(blockSolveBuffer,16*256*4);
        }
        if (t.dotSize) {
            dotIndex = scratch.Get<uint32_t>()[12 * t.dotSize];
            for (uint32_t i = 0; i < t.dotSize; ++i) dotIndex.SetValue(i, i * 4);
        }
    }

    __aicore__ inline void Process(uint32_t Lanen200_forcedBatch=0xffffffffU) {
        const bool splitBlocked = t.blocked && t.rhsSplit;
        const bool splitPaired=t.paired && t.rhsSplit;
        const uint32_t firstBatch = Lanen200_forcedBatch!=0xffffffffU ? Lanen200_forcedBatch : splitPaired ? GetBlockIdx()/(2*PairSlots()) : splitBlocked ? GetBlockIdx()/k : t.paired ? GetBlockIdx() / 2 : GetBlockIdx();
        const uint32_t batchStep = Lanen200_forcedBatch!=0xffffffffU ? t.batches : splitPaired ? t.blocks/(2*PairSlots()) : splitBlocked ? t.blocks/k : t.paired ? t.blocks / 2 : t.blocks;
        for (uint32_t batch = firstBatch; batch < t.batches; batch += batchStep) {
            uint32_t coords = batch, ai = 0, bi = 0;
            for (int d = static_cast<int>(t.batchRank) - 1; d >= 0; --d) {
                uint32_t c = coords % t.batchDims[d]; coords /= t.batchDims[d];
                ai += c * t.aStrides[d]; bi += c * t.bStrides[d];
            }
            ao = uint64_t(ai) * n * n; bo = uint64_t(bi) * n * k;
            if constexpr(Lanen200_BlockTri16)Lanen200_refinePass=0;
            xo = uint64_t(batch) * n * k;
            if (t.blocked) FromGlobal(mat, wg, n * rp);
            else if (!t.paired) { Initialize(); Factor(); }
            if(t.originalSchur)InitialForwardRows();
            BackSolve(false);
            if (t.paired && k >= 2) SolveSync();
            Refine();
            if constexpr(Lanen200_BlockTri16){Lanen200_extraIR=Lanen200_NeedExtraCorrection();if(Lanen200_extraIR)Refine();}
            if (t.dotSize && n <= 256) {
                Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
                DataCopyExtParams out{static_cast<uint16_t>(n), k * 4, OnChip ? np / 8 : 0, 0, 0};
                if(splitPaired){
                    uint32_t c=(GetBlockIdx()/2)%PairSlots();for(uint32_t i=0;i<n;++i)dotIndex.SetValue(i,i*kp*4);Fence<HardEvent::S_V>();
                    AscendC::Gather(original,mat,dotIndex,c*4,n);PipeBarrier<PIPE_V>();Brcb(column,original,uint8_t(n/8),{1,8});PipeBarrier<PIPE_V>();Fence<HardEvent::V_MTE3>();
                    DataCopyExtParams single{uint16_t(n),4,0,(k-1)*4,0};DataCopyPad(xg[xo+c],column,single);
                }else if constexpr (OnChip) {
                    if (splitBlocked) {
                        const uint32_t c=GetBlockIdx()%k;
                        for(uint32_t i=0;i<n;++i) dotIndex.SetValue(i,i*rp*4);
                        Fence<HardEvent::S_V>();
                        AscendC::Gather(original,mat,dotIndex,(np+c)*4,n); PipeBarrier<PIPE_V>();
                        Brcb(column,original,uint8_t(n/8),{1,8}); PipeBarrier<PIPE_V>();
                        Fence<HardEvent::V_MTE3>();
                        DataCopyExtParams single{uint16_t(n),4,0,(k-1)*4,0};
                        DataCopyPad(xg[xo+c],column,single);
                    } else DataCopyPad(xg[xo], mat[np], out);
                }
                else DataCopyPad(xg[xo], mat, out);
                Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
            } else {
                for (uint32_t i = 0; i < n; ++i) {
                    Load(row, i * rp + np, kp);
                    ToGlobal(xg[xo + uint64_t(i) * k], row, k);
                }
            }
            if constexpr(Lanen200_BlockTri16){auto pk=scratch.Get<uint32_t>();for(uint32_t i=0;i<8;++i)pk.SetValue(i,0);pk.SetValue(0,0x4e324952U);pk.SetValue(1,uint32_t(Lanen200_extraIR));ToGlobal(wg[n*rp+208],pk.ReinterpretCast<float>(),8);}
        }
    }

    __aicore__ inline void ResidualWorker(GM_ADDR workspace) {
        uint32_t coords=t.rhsSplit?GetBlockIdx()/(2*PairSlots()):GetBlockIdx()/2,ai=0,bi=0;
        for(int d=int(t.batchRank)-1;d>=0;--d) { uint32_t c=coords%t.batchDims[d];coords/=t.batchDims[d];ai+=c*t.aStrides[d];bi+=c*t.bStrides[d]; }
        ao=uint64_t(ai)*n*n;bo=uint64_t(bi)*n*k;
        SolveSync();
        GlobalTensor<float> solved;
        const uint64_t elements=uint64_t(n)*(np+2*kp);
        solved.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace))+uint64_t(GetBlockIdx()-1)*elements+np);
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
        DataCopyExtParams p{uint16_t(n),kp*4,np*4,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(mat,solved,p,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        TiledResidual();SolveSync();
    }

private:
    __aicore__ inline uint32_t PairSlots(){return t.originalSchur?t.pairsPerMatrix:k;}
    __aicore__ inline void SolveSync(){
        if(t.originalSchur){CrossCoreSetFlag<1,PIPE_MTE3>(4);CrossCoreWaitFlag<1>(4);}else SyncAll<true>();
    }
    __aicore__ inline void InitialForwardRows(){
        if constexpr(!OnChip){
            LoadRightCache();DataCopy(mat[n*kp],mat,n*kp);PipeBarrier<PIPE_V>();CompactForwardSolve();DataCopy(mat,mat[n*kp],n*kp);PipeBarrier<PIPE_V>();StoreRightCache();
        }
    }
    __aicore__ inline void FromGlobal(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t count) {
        Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
        Fence<HardEvent::MTE3_MTE2>();
        DataCopyExtParams p{1, count * 4, 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, 0, 0, 0};
        DataCopyPad(dst, src, p, pad);
        Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline void ToGlobal(GlobalTensor<float> dst, LocalTensor<float> src, uint32_t count) {
        Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
        DataCopyExtParams p{1, count * 4, 0, 0, 0};
        DataCopyPad(dst, src, p);
        Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
    }
    __aicore__ inline void LocalCopy(LocalTensor<float> dst, LocalTensor<float> src, uint32_t count) {
        Fence<HardEvent::S_V>();
        DataCopy(dst, src, count);
        PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
    }
    __aicore__ inline void Load(LocalTensor<float> dst, uint32_t offset, uint32_t count) {
        if constexpr (OnChip) LocalCopy(dst, mat[offset], count);
        else FromGlobal(dst, wg[offset], count);
    }
    __aicore__ inline void Store(uint32_t offset, LocalTensor<float> src, uint32_t count) {
        if constexpr (OnChip) LocalCopy(mat[offset], src, count);
        else ToGlobal(wg[offset], src, count);
    }
    __aicore__ inline void LoadR(LocalTensor<float> dst, uint32_t i) { Load(dst, n * rp + i * kp, kp); }
    __aicore__ inline void StoreR(uint32_t i, LocalTensor<float> src) { Store(n * rp + i * kp, src, kp); }
    __aicore__ inline void Ax(LocalTensor<float> dst, LocalTensor<float> src, float a, uint32_t count) {
        Fence<HardEvent::S_V>();
        Axpy(dst, src, a, count); PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
    }
    __aicore__ inline void Scale(LocalTensor<float> dst, float a, uint32_t count) {
        Fence<HardEvent::S_V>();
        Muls(dst, dst, a, count); PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
    }
    __aicore__ inline void Gather(uint32_t j, uint32_t start, uint32_t end) {
        if constexpr (!OnChip) {
            if (end == start) return;
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            Fence<HardEvent::MTE3_MTE2>();
            DataCopyExtParams p{static_cast<uint16_t>(end - start), 4, (rp - 1) * 4, 0, 0};
            DataCopyPadExtParams<float> pad{true, 0, 0, 0};
            DataCopyPad(column, wg[start * rp + j], p, pad);
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        }
    }
    __aicore__ inline float Coeff(uint32_t i, uint32_t j, uint32_t start) {
        if constexpr (OnChip) return mat.GetValue(i * rp + j);
        else return column.GetValue((i - start) * 8);
    }
    __aicore__ inline void Initialize() {
        if constexpr (OnChip) {
            if (t.dotSize && n <= 256) {
                Duplicate(mat, 0.0f, n * (np + 2 * kp)); PipeBarrier<PIPE_V>();
                Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
                DataCopyExtParams pa{static_cast<uint16_t>(n), n * 4, 0, kp / 8, 0};
                DataCopyExtParams pb{static_cast<uint16_t>(n), k * 4, 0, np / 8, 0};
                DataCopyPadExtParams<float> pad{true, 0, 0, 0};
                DataCopyPad(mat, ag[ao], pa, pad); DataCopyPad(mat[np], bg[bo], pb, pad);
                Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>(); return;
            }
        }
        for (uint32_t i = 0; i < n; ++i) {
            Duplicate(row, 0.0f, rp); PipeBarrier<PIPE_V>();
            FromGlobal(row, ag[ao + uint64_t(i) * n], n);
            FromGlobal(row[np], bg[bo + uint64_t(i) * k], k);
            Store(i * rp, row, rp);
        }
    }
    __aicore__ inline void EliminateRows(LocalTensor<float> target, uint32_t start,
                                         uint32_t rows, uint32_t pivot) {
        BinaryRepeatParams update{1, 0, 1, static_cast<uint8_t>(rp / 8), 1, 0};
        for (uint32_t offset = 0; offset < rp; offset += 64) {
            const uint64_t mask = rp - offset < 64 ? rp - offset : 64;
            MulAddDst(target[offset], column[start * 8], row[offset], mask,
                      static_cast<uint8_t>(rows), update);
            PipeBarrier<PIPE_V>();
        }
        uint64_t mask[2] = {1ULL << (pivot & 7), 0};
        UnaryRepeatParams retain{1, 1, static_cast<uint8_t>(rp / 8), 1};
        Muls(target[(pivot / 8) * 8], column[start * 8], -1.0f, mask,
             static_cast<uint8_t>(rows), retain);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void FactorRows() {
        auto compact = scratch.Get<float>()[13 * t.dotSize];
        auto maximum = scratch.Get<float>()[14 * t.dotSize];
        for (uint32_t i = 0; i < n; ++i)
            dotIndex.SetValue(i, i * (OnChip ? rp * 4 : 32));
        Fence<HardEvent::S_V>();
        for (uint32_t j = 0; j < n; ++j) {
            Duplicate(compact, 0.0f, np); PipeBarrier<PIPE_V>();
            if constexpr (OnChip) {
                AscendC::Gather(compact, mat, dotIndex, j * 4, n);
            } else {
                Gather(j, 0, n);
                AscendC::Gather(compact, column, dotIndex, 0, n);
            }
            PipeBarrier<PIPE_V>();
            Abs(original, compact, np); PipeBarrier<PIPE_V>();
            if (j) { Duplicate(original, 0.0f, j); PipeBarrier<PIPE_V>(); }
            AscendC::ReduceMax(maximum, original, column, static_cast<int32_t>(n), true);
            PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
            const uint32_t pivot = maximum.ReinterpretCast<uint32_t>().GetValue(1);
            const float diagonal = compact.GetValue(pivot);
            if (pivot != j) {
                compact.SetValue(pivot, compact.GetValue(j));
                compact.SetValue(j, diagonal);
            }
            pivots.SetValue(j, pivot);
            Load(row, j * rp, rp);
            if (pivot != j) {
                Load(other, pivot * rp, rp);
                Store(pivot * rp, row, rp); Store(j * rp, other, rp);
                LocalCopy(row, other, rp);
            }
            Fence<HardEvent::S_V>();
            Duplicate(row, 0.0f, j + 1); PipeBarrier<PIPE_V>();
            Muls(compact, compact, -1.0f / diagonal, np); PipeBarrier<PIPE_V>();
            Brcb(column, compact, static_cast<uint8_t>(np / 8), {1, 8});
            PipeBarrier<PIPE_V>();
            if constexpr (OnChip) {
                if (j + 1 < n) EliminateRows(mat[(j + 1) * rp], j + 1, n - j - 1, j);
            } else {
                const uint32_t tileRows = (n * (np + 8) / 2) / rp;
                for (uint32_t first = j + 1; first < n; first += tileRows) {
                    const uint32_t count = n - first < tileRows ? n - first : tileRows;
                    FromGlobal(triangle, wg[first * rp], count * rp);
                    EliminateRows(triangle, first, count, j);
                    ToGlobal(wg[first * rp], triangle, count * rp);
                }
            }
        }
    }

    __aicore__ inline void Factor() {
        if (t.dotSize && n <= 256) { FactorRows(); return; }
        for (uint32_t j = 0; j < n; ++j) {
            Gather(j, j, n);
            uint32_t pivot = j; float largest = -1.0f;
            for (uint32_t i = j; i < n; ++i) {
                float value = Coeff(i, j, j); float ab = value < 0.0f ? -value : value;
                if (ab > largest) { largest = ab; pivot = i; }
            }
            pivots.SetValue(j, pivot);
            Load(row, j * rp, rp);
            if (pivot != j) {
                Load(other, pivot * rp, rp);
                Store(pivot * rp, row, rp); Store(j * rp, other, rp);
                LocalCopy(row, other, rp);
            }
            const float inverse = 1.0f / row.GetValue(j);
            // Keep previous L columns, and write the new multiplier explicitly.
            Fence<HardEvent::S_V>();
            Duplicate(row, 0.0f, j + 1); PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
            for (uint32_t i = j + 1; i < n; ++i) {
                Load(other, i * rp, rp);
                const float multiplier = other.GetValue(j) * inverse;
                Ax(other, row, -multiplier, rp);
                other.SetValue(j, multiplier);
                Store(i * rp, other, rp);
            }
        }
    }
    __aicore__ inline void BackSolve(bool residual) {
        if (t.dotSize && n <= 256) { BackSolveRows(residual); return; }
        for (int j = static_cast<int>(n) - 1; j >= 0; --j) {
            Gather(j, 0, j + 1);
            const float inverse = 1.0f / Coeff(j, j, 0);
            if (residual) LoadR(row, j); else Load(row, j * rp + np, kp);
            Scale(row, inverse, kp);
            if (residual) StoreR(j, row); else Store(j * rp + np, row, kp);
            for (uint32_t i = 0; i < static_cast<uint32_t>(j); ++i) {
                const float coefficient = Coeff(i, j, 0);
                if (residual) LoadR(other, i); else Load(other, i * rp + np, kp);
                Ax(other, row, -coefficient, kp);
                if (residual) StoreR(i, other); else Store(i * rp + np, other, kp);
            }
        }
    }
    // Pack only the needed triangular coefficients when the full LU exceeds UB.
    __aicore__ inline void PrepareTriangle(bool lower) {
        if constexpr(!OnChip){if(t.originalSchur){PrepareBlockTriangle(lower);return;}}
        if constexpr (OnChip) {
            for (uint32_t i = 0; i < n; ++i) dotIndex.SetValue(i, i * rp * 4);
            Fence<HardEvent::S_V>();
        } else {
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            Fence<HardEvent::MTE3_MTE2>();
            uint32_t offset = 0;
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t length = lower ? i : n - i;
                dotIndex.SetValue(i, (offset - (lower ? 0 : i)) * 4);
                if (length) {
                    DataCopyExtParams p{1, length * 4, 0, 0, 0};
                    DataCopyPadExtParams<float> pad{true, 0, 0, 0};
                    DataCopyPad(triangle[offset], wg[i * rp + (lower ? 0 : i)], p, pad);
                    offset += (length + 7) / 8 * 8;
                }
            }
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
            Fence<HardEvent::S_V>();
        }
    }
    __aicore__ inline LocalTensor<float> TriangleColumn(uint32_t j) {
        auto compact = scratch.Get<float>()[13 * t.dotSize];
        Duplicate(compact, 0.0f, np); PipeBarrier<PIPE_V>();
        if constexpr (OnChip) AscendC::Gather(compact, mat, dotIndex, j * 4, n);
        else AscendC::Gather(compact, triangle, dotIndex, j * 4, n);
        PipeBarrier<PIPE_V>();
        return compact;
    }
    __aicore__ inline void BroadcastNegative(LocalTensor<float> compact) {
        Muls(compact, compact, -1.0f, np); PipeBarrier<PIPE_V>();
        Brcb(column, compact, static_cast<uint8_t>(np / 8), {1, 8});
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void UpdateRows(LocalTensor<float> rhs, uint32_t stride,
                                      uint32_t start, uint32_t rows, uint32_t pivot) {
        // Eight RHS rows fill the 64 FP32 lanes. The pivot RHS is reused per block.
        const uint32_t groups = rows / 8, tail = rows % 8;
        const uint8_t blocks = static_cast<uint8_t>(stride / 8);
        BinaryRepeatParams p{blocks, 1, 0, static_cast<uint8_t>(8 * blocks), 8, 0};
        if (groups) {
            MulAddDst(rhs[start * stride], column[start * 8], rhs[pivot * stride],
                      uint64_t(64), static_cast<uint8_t>(groups), p);
            PipeBarrier<PIPE_V>();
        }
        if (tail) {
            const uint32_t first = start + groups * 8;
            MulAddDst(rhs[first * stride], column[first * 8], rhs[pivot * stride],
                      uint64_t(tail * 8), uint8_t(1), p);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void LoadRightCache() {
        if constexpr (!OnChip) {
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            Fence<HardEvent::MTE3_MTE2>();
            DataCopyExtParams p{static_cast<uint16_t>(n), kp * 4, np * 4, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(mat, wg[np], p, pad);
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        }
    }
    __aicore__ inline void StoreRightCache() {
        if constexpr (!OnChip) {
            Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
            DataCopyExtParams p{static_cast<uint16_t>(n), kp * 4, 0, np * 4, 0};
            DataCopyPad(wg[np], mat, p);
            Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
        }
    }
    __aicore__ inline void BackSolveRows(bool residual) {
        if(t.rhsSplit && (t.paired || t.blocked)){CompactBackSolve(residual);return;}
        if (!residual) LoadRightCache();
        LocalTensor<float> rhs = mat;
        uint32_t stride = kp;
        if constexpr (OnChip) { rhs = residual ? mat[n * rp] : mat[np]; stride = residual ? kp : rp; }
        else if (residual) rhs = mat[n * kp];
        PrepareTriangle(false);
        for (int j = static_cast<int>(n) - 1; j >= 0; --j) {
            auto compact = TriangleColumn(j); Fence<HardEvent::V_S>();
            const float inverse = 1.0f / compact.GetValue(j);
            Muls(rhs[j * stride], rhs[j * stride], inverse, kp); PipeBarrier<PIPE_V>();
            BroadcastNegative(compact);
            UpdateRows(rhs, stride, 0, j, j);
        }
        if constexpr (!OnChip) {
            if (residual) ToGlobal(wg[n * rp], rhs, n * kp);
            else StoreRightCache();
        }
    }
    __aicore__ inline LocalTensor<float> ResidualCache() {
        if constexpr (OnChip) return mat[n * rp];
        else return mat[n * kp];
    }
    __aicore__ inline void PermuteResidualRows() {
        auto rhs = ResidualCache();
        if(t.originalSchur){
            auto index=scratch.Get<int32_t>()[14*t.dotSize];auto compact=scratch.Get<float>();
            Muls(index,pivots,int32_t(kp*4),n);PipeBarrier<PIPE_V>();
            AscendC::Gather(compact,rhs,index.ReinterpretCast<uint32_t>(),((GetBlockIdx()/2)%PairSlots())*4,n);PipeBarrier<PIPE_V>();CompactScatter(rhs,kp,compact);return;
        }
        for (uint32_t j = 0; j < n; ++j) {
            uint32_t pivot = pivots.GetValue(j);
            if (pivot != j) {
                DataCopy(row, rhs[j * kp], kp); PipeBarrier<PIPE_V>();
                DataCopy(other, rhs[pivot * kp], kp); PipeBarrier<PIPE_V>();
                DataCopy(rhs[pivot * kp], row, kp); PipeBarrier<PIPE_V>();
                DataCopy(rhs[j * kp], other, kp); PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void ForwardSolveRows() {
        if(t.rhsSplit && (t.paired || t.blocked)){CompactForwardSolve();return;}
        auto rhs = ResidualCache(); PrepareTriangle(true);
        for (uint32_t j = 0; j + 1 < n; ++j) {
            auto compact = TriangleColumn(j); BroadcastNegative(compact);
            UpdateRows(rhs, kp, j + 1, n - j - 1, j);
        }
    }

    __aicore__ inline void CompactScatter(LocalTensor<float> rhs,uint32_t stride,LocalTensor<float> compact){
        const uint32_t c=t.paired?(GetBlockIdx()/2)%PairSlots():GetBlockIdx()%k;
        Brcb(column,compact,uint8_t(np/8),{1,8});PipeBarrier<PIPE_V>();
        uint64_t mask[2]={1ULL<<c,0};
        for(uint32_t first=0;first<n;first+=128){
            const uint32_t rows=n-first<128?n-first:128;
            Muls(rhs[first*stride],column[first*8],1.0f,mask,uint8_t(rows),{1,1,uint8_t(stride/8),1});PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void PrepareBlockTriangle(bool lower){
        constexpr uint32_t N=256;const uint32_t matrix=GetBlockIdx()/8;
        GlobalTensor<float>tg;tg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace)+t.schurOffset/4+uint64_t(matrix)*kOS_Total+kOS_S);
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();
        uint32_t offset=0;
        for(uint32_t j=0;j<N;++j){
            uint32_t first=lower?(j+1)/8*8:0,length=lower?N-first:j+1;
            dotIndex.SetValue(j,(offset-(lower?first:0))*4);
            if(length){DataCopyExtParams p{1,length*4,0,0,0};DataCopyPadExtParams<float>pad{true,0,0,0};DataCopyPad(triangle[offset],tg[j*N+first],p,pad);offset+=(length+7)/8*8;}
        }
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();Fence<HardEvent::S_V>();
    }
    __aicore__ inline void BlockTriangleSolve(bool residual,bool lower){
        constexpr uint32_t N=256,D=256,B=64;
        if(!residual && !lower)LoadRightCache();
        auto rhs=residual?ResidualCache():mat;const uint32_t c=(GetBlockIdx()/2)%PairSlots();
        auto compact=scratch.Get<float>(),saved=scratch.Get<float>()[D];auto rowBytes=scratch.Get<int32_t>()[11*D];
        auto values=scratch.Get<float>()[13*D];auto index=scratch.Get<int32_t>()[14*D];auto inverse=scratch.Get<float>()[15*D];
        auto tile=blockSolveBuffer.Get<float>();auto tileIndex=mat.ReinterpretCast<int32_t>();
        PrepareBlockTriangle(lower);
        CreateVecIndex(index,int32_t(0),N);PipeBarrier<PIPE_V>();Muls(index,index,int32_t(kp*4),N);PipeBarrier<PIPE_V>();
        AscendC::Gather(compact,rhs,index.ReinterpretCast<uint32_t>(),c*4,N);PipeBarrier<PIPE_V>();
        AscendC::Gather(saved,mat,index.ReinterpretCast<uint32_t>(),c*4,N);PipeBarrier<PIPE_V>();
        if(!lower){
            CreateVecIndex(index,int32_t(0),N);PipeBarrier<PIPE_V>();Muls(index,index,int32_t(4),N);PipeBarrier<PIPE_V>();Add(index,index,dotIndex.ReinterpretCast<int32_t>(),N);PipeBarrier<PIPE_V>();
            AscendC::Gather(inverse,triangle,index.ReinterpretCast<uint32_t>(),0,N);PipeBarrier<PIPE_V>();Mul(compact,compact,inverse,N);PipeBarrier<PIPE_V>();
        }
        CreateVecIndex(rowBytes,int32_t(0),N);PipeBarrier<PIPE_V>();Muls(rowBytes,rowBytes,int32_t(4),N);PipeBarrier<PIPE_V>();
        GlobalTensor<float>dg;dg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace)+t.schurOffset/4+uint64_t(GetBlockIdx()/8)*kOS_Total+kOS_Identity);
        for(uint32_t step=0;step<N/B;++step){
            uint32_t block=lower?step:N/B-1-step,first=block*B;
            FromGlobal(tile,dg[block*2*B*B+(lower?B*B:0)],B*B);
            Mul(tile,tile,compact[first],uint64_t(B),uint8_t(B),{1,1,1,B/8,B/8,0});PipeBarrier<PIPE_V>();
            WholeReduceSum(values,tile,int32_t(B),uint8_t(B),1,1,B/8);PipeBarrier<PIPE_V>();
            DataCopy(compact[first],values,B);DataCopy(inverse,values,B);PipeBarrier<PIPE_V>();
            const uint32_t rowStart=lower?first+B:0,rowEnd=lower?N:first;
            for(uint32_t rowFirst=rowStart;rowFirst<rowEnd;rowFirst+=64){
                const uint32_t rows=rowEnd-rowFirst<64?rowEnd-rowFirst:64;
                Muls(tileIndex,dotIndex[first].ReinterpretCast<int32_t>(),int32_t(1),uint64_t(B),uint8_t(rows),{1,1,B/8,0});PipeBarrier<PIPE_V>();
                Brcb(column,rowBytes[rowFirst].ReinterpretCast<float>(),uint8_t(rows/8),{1,8});PipeBarrier<PIPE_V>();
                Add(tileIndex,tileIndex,column.ReinterpretCast<int32_t>(),uint64_t(B),uint8_t(rows),{1,1,0,B/8,B/8,1});PipeBarrier<PIPE_V>();
                AscendC::Gather(tile,triangle,tileIndex.ReinterpretCast<uint32_t>(),0,rows*B);PipeBarrier<PIPE_V>();
                Mul(tile,tile,inverse,uint64_t(B),uint8_t(rows),{1,1,1,B/8,B/8,0});PipeBarrier<PIPE_V>();
                WholeReduceSum(values,tile,int32_t(B),uint8_t(rows),1,1,B/8);PipeBarrier<PIPE_V>();Add(compact[rowFirst],compact[rowFirst],values,rows);PipeBarrier<PIPE_V>();
            }
        }
        CompactScatter(mat,kp,saved);CompactScatter(rhs,kp,compact);
        if(!lower){if(residual)ToGlobal(wg[n*rp],rhs,n*kp);else StoreRightCache();}
    }
    // Same per-element tolerance as acceptance; large first correction requests one more exact-residual IR.
    __aicore__ inline bool Lanen200_NeedExtraCorrection(){
        static_assert(OnChip,"N200 correction guard needsresidentX/delta");constexpr uint32_t D=256;auto xc=scratch.Get<float>(),dc=scratch.Get<float>()[D],error=scratch.Get<float>()[13*D],maximum=scratch.Get<float>()[15*D];auto bytes=scratch.Get<int32_t>()[14*D];
        CreateVecIndex(bytes,int32_t(0),n);PipeBarrier<PIPE_V>();Muls(bytes,bytes,int32_t(rp*4),n);PipeBarrier<PIPE_V>();AscendC::Gather(xc,mat[np],bytes.ReinterpretCast<uint32_t>(),(GetBlockIdx()%4)*4,n);PipeBarrier<PIPE_V>();
        CreateVecIndex(bytes,int32_t(0),n);PipeBarrier<PIPE_V>();Muls(bytes,bytes,int32_t(8*4),n);PipeBarrier<PIPE_V>();AscendC::Gather(dc,ResidualCache(),bytes.ReinterpretCast<uint32_t>(),(GetBlockIdx()%4)*4,n);PipeBarrier<PIPE_V>();
        Duplicate(error,-1.0f,D);Abs(xc,xc,n);Abs(dc,dc,n);PipeBarrier<PIPE_V>();Adds(xc,xc,1.0f,n);PipeBarrier<PIPE_V>();Muls(xc,xc,0.0001f,n);PipeBarrier<PIPE_V>();Sub(error,dc,xc,n);PipeBarrier<PIPE_V>();WholeReduceMax(maximum,error,int32_t(64),uint8_t(4),1,1,8);Fence<HardEvent::V_S>();uint32_t bits,index;GetReduceRepeatMaxMinSpr<float>(bits,index);union Lanen200_GuardBits{uint32_t u;float f;}check;check.u=bits;return !(check.f<=0.0f);
    }
    // Lanen200 ownership: complete MODE15 cached block16 triangle, old instances defaultfalse.
    __aicore__ inline void Lanen200_CachedTriangle(bool residual,bool lower){
        static_assert(OnChip,"N200 cachedtriangle needsresidentLU");
        constexpr uint32_t N=200,D=256,B=16,CP=24,NB=13,CS=NB*B*CP;
        auto compact=scratch.Get<float>(),cache=scratch.Get<float>()[D],product=scratch.Get<float>()[D+B*CP];
        auto sol=scratch.Get<float>()[D+B*CP+32*CP],sum=scratch.Get<float>()[D+B*CP+32*CP+64];
        auto index=scratch.Get<uint32_t>()[D+B*CP+32*CP+128];
        const uint32_t c=GetBlockIdx()%4,stride=residual?8:rp;auto rhs=residual?ResidualCache():mat[N];
        Duplicate(compact,0.0f,D);PipeBarrier<PIPE_V>();
        for(uint32_t i=0;i<D;++i)index.SetValue(i,i*stride*4);Fence<HardEvent::S_V>();
        AscendC::Gather(compact,rhs,index,c*4,N);PipeBarrier<PIPE_V>();
        GlobalTensor<float> cached;const uint64_t elements=N*216;
        cached.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWorkspace)+uint64_t(t.blocks)*elements+uint64_t(GetBlockIdx()/4*4)*N*rp+N*rp+(lower?CS:0));
        for(uint32_t step=0;step<NB;++step){
            uint32_t block=lower?step:NB-1-step,first=block*B,w=N-first<B?N-first:B;
            FromGlobal(cache,cached[block*B*CP],B*CP);
            Mul(product,cache,compact[first],uint64_t(B),uint8_t(B),{1,1,1,CP/8,CP/8,0});PipeBarrier<PIPE_V>();
            WholeReduceSum(sol,product,int32_t(B),uint8_t(B),1,1,CP/8);PipeBarrier<PIPE_V>();DataCopy(compact[first],sol,B);PipeBarrier<PIPE_V>();
            const uint32_t begin=lower?first+w:0,end=lower?N:first;
            for(uint32_t rowFirst=begin;rowFirst<end;rowFirst+=32){
                uint32_t rows=end-rowFirst<32?end-rowFirst:32;
                Mul(product,mat[rowFirst*rp+first],sol,uint64_t(w),uint8_t(rows),{1,1,1,CP/8,uint8_t(rp/8),0});PipeBarrier<PIPE_V>();
                WholeReduceSum(sum,product,int32_t(w),uint8_t(rows),1,1,CP/8);PipeBarrier<PIPE_V>();Sub(compact[rowFirst],compact[rowFirst],sum,rows);PipeBarrier<PIPE_V>();
            }
        }
        CompactScatter(rhs,stride,compact);
    }
    // End Lanen200 cachedtriangle.
    __aicore__ inline void CompactBackSolve(bool residual){
        if constexpr(Lanen200_BlockTri16){Lanen200_CachedTriangle(residual,false);return;}
        if(t.originalSchur){BlockTriangleSolve(residual,false);return;}
        if(!residual)LoadRightCache();
        LocalTensor<float>rhs=mat;uint32_t stride=kp;
        if constexpr(OnChip){rhs=residual?mat[n*rp]:mat[np];stride=residual?kp:rp;}
        else if(residual)rhs=mat[n*kp];
        const uint32_t c=t.paired?(GetBlockIdx()/2)%PairSlots():GetBlockIdx()%k,d=t.dotSize;
        auto compact=scratch.Get<float>();auto coeff=scratch.Get<float>()[13*d];
        auto index=scratch.Get<int32_t>()[14*d];auto inverse=scratch.Get<float>()[15*d];auto broadcast=scratch.Get<float>()[16*d];
        PrepareTriangle(false);
        CreateVecIndex(index,int32_t(0),n);PipeBarrier<PIPE_V>();Muls(index,index,int32_t(4),n);PipeBarrier<PIPE_V>();
        Add(index,index,dotIndex.ReinterpretCast<int32_t>(),n);PipeBarrier<PIPE_V>();
        if constexpr(OnChip)AscendC::Gather(inverse,mat,index.ReinterpretCast<uint32_t>(),0,n);
        else AscendC::Gather(inverse,triangle,index.ReinterpretCast<uint32_t>(),0,n);
        PipeBarrier<PIPE_V>();Duplicate(broadcast,-1.0f,n);PipeBarrier<PIPE_V>();Div(inverse,broadcast,inverse,n);PipeBarrier<PIPE_V>();
        CreateVecIndex(index,int32_t(0),n);PipeBarrier<PIPE_V>();Muls(index,index,int32_t(stride*4),n);PipeBarrier<PIPE_V>();
        AscendC::Gather(compact,rhs,index.ReinterpretCast<uint32_t>(),c*4,n);PipeBarrier<PIPE_V>();
        Duplicate(index,int32_t(0),n);PipeBarrier<PIPE_V>();
        // Uraw = D*Uunit. Scale all equations before unit-upper substitution.
        Mul(compact,compact,inverse,n);PipeBarrier<PIPE_V>();Muls(compact,compact,-1.0f,n);PipeBarrier<PIPE_V>();
        for(int j=int(n)-1;j>0;--j){
            if constexpr(OnChip)AscendC::Gather(coeff,mat,dotIndex,uint32_t(j)*4,n);
            else AscendC::Gather(coeff,triangle,dotIndex,uint32_t(j)*4,n);
            PipeBarrier<PIPE_V>();Mul(coeff,coeff,inverse,n);PipeBarrier<PIPE_V>();
            AscendC::Gather(broadcast,compact,index.ReinterpretCast<uint32_t>(),uint32_t(j)*4,n);PipeBarrier<PIPE_V>();
            MulAddDst(compact,coeff,broadcast,int32_t(j));PipeBarrier<PIPE_V>();
        }
        CompactScatter(rhs,stride,compact);
        if constexpr(!OnChip){if(residual)ToGlobal(wg[n*rp],rhs,n*kp);else StoreRightCache();}
    }
    __aicore__ inline void CompactForwardSolve(){
        if constexpr(Lanen200_BlockTri16){Lanen200_CachedTriangle(true,true);return;}
        if(t.originalSchur){BlockTriangleSolve(true,true);return;}
        auto rhs=ResidualCache();PrepareTriangle(true);
        const uint32_t c=t.paired?(GetBlockIdx()/2)%PairSlots():GetBlockIdx()%k,d=t.dotSize;
        auto compact=scratch.Get<float>();auto coeff=scratch.Get<float>()[13*d];
        auto index=scratch.Get<int32_t>()[14*d];auto broadcast=scratch.Get<float>()[16*d];
        CreateVecIndex(index,int32_t(0),n);PipeBarrier<PIPE_V>();Muls(index,index,int32_t(kp*4),n);PipeBarrier<PIPE_V>();
        AscendC::Gather(compact,rhs,index.ReinterpretCast<uint32_t>(),c*4,n);PipeBarrier<PIPE_V>();
        Duplicate(index,int32_t(0),n);PipeBarrier<PIPE_V>();
        for(uint32_t j=0;j+1<n;++j){
            if constexpr(OnChip)AscendC::Gather(coeff,mat,dotIndex,j*4,n);
            else AscendC::Gather(coeff,triangle,dotIndex,j*4,n);
            PipeBarrier<PIPE_V>();Muls(coeff,coeff,-1.0f,n);PipeBarrier<PIPE_V>();Duplicate(coeff,0.0f,j+1);PipeBarrier<PIPE_V>();
            AscendC::Gather(broadcast,compact,index.ReinterpretCast<uint32_t>(),j*4,n);PipeBarrier<PIPE_V>();
            MulAddDst(compact,coeff,broadcast,int32_t(n));PipeBarrier<PIPE_V>();
        }
        CompactScatter(rhs,kp,compact);
    }

    __aicore__ inline void CompensatedScalar(float term, float &sum, float &error) {
        volatile float next = sum + term;
        volatile float other = next - sum;
        volatile float restored = next - other;
        volatile float aerror = sum - restored;
        volatile float berror = term - other;
        volatile float roundoff = aerror + berror;
        volatile float low = error + roundoff;
        sum = next; error = low;
    }

    __aicore__ inline void TileMul(LocalTensor<float> dst, LocalTensor<float> a,
                                   LocalTensor<float> b, uint32_t d, uint32_t tile) {
        BinaryRepeatParams p{1, 1, 1, static_cast<uint8_t>(d / 8),
                             static_cast<uint8_t>(d / 8), 0};
        for (uint32_t offset = 0; offset < d; offset += 64) {
            Mul(dst[offset], a[offset], b[offset], uint64_t(d - offset < 64 ? d - offset : 64), static_cast<uint8_t>(tile), p);
        }
        PipeBarrier<PIPE_V>();
    }
    // Lanen256_: tolerance-matched FP32 residual for the N256 Schur family.
    __aicore__ inline void Lanen256_FastResidual() {
        constexpr uint32_t D=256,Tile=16,Count=Tile*D;
        auto aa=triangle;auto pp=aa[Count];auto xc=pp[Count];auto bb=xc[D*8];auto xx=bb[D*8];
        auto sums=xx[D];auto terms=sums[Tile];auto rhsTerms=terms[Tile];
        auto rr=ResidualCache();
        Duplicate(rr,0.0f,D*8);DataCopy(xc,mat,D*8);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();
        DataCopyExtParams pb{D,k*4,0,0,0};DataCopyPadExtParams<float>pad{true,0,0,0};
        DataCopyPad(bb,bg[bo],pb,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        for(uint32_t i=0;i<D;++i)dotIndex.SetValue(i,i*kp*4);Fence<HardEvent::S_V>();
        const uint32_t c=(GetBlockIdx()/2)%PairSlots(),begin=(GetBlockIdx()&1)*(D/2),end=begin+D/2;
        AscendC::Gather(xx,xc,dotIndex,c*4,D);PipeBarrier<PIPE_V>();
        for(uint32_t first=begin;first<end;first+=Tile){
            Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
            DataCopy(aa,ag[ao+uint64_t(first)*D],Count);Fence<HardEvent::MTE2_V>();
            TileMul(pp,aa,xx,D,Tile);
            WholeReduceSum(sums,pp,int32_t(64),uint8_t(Tile),1,1,D/8);PipeBarrier<PIPE_V>();
            for(uint32_t off=64;off<D;off+=64){
                WholeReduceSum(terms,pp[off],int32_t(64),uint8_t(Tile),1,1,D/8);PipeBarrier<PIPE_V>();
                Add(sums,sums,terms,Tile);PipeBarrier<PIPE_V>();
            }
            AscendC::Gather(rhsTerms,bb[first*kp],dotIndex,c*4,Tile);PipeBarrier<PIPE_V>();
            Sub(terms,rhsTerms,sums,Tile);PipeBarrier<PIPE_V>();Brcb(column,terms,uint8_t(Tile/8),{1,8});PipeBarrier<PIPE_V>();
            uint64_t mask[2]={1ULL<<c,0};Muls(rr[first*kp],column,1.0f,mask,uint8_t(Tile),{1,1,uint8_t(kp/8),1});PipeBarrier<PIPE_V>();
        }
        ToGlobal(wg[n*rp],rr,D*8);
    }

    // MODE15 first pass only. Requested later pass retains the full error-free product/extracted sum.
    __aicore__ inline void Lanen200_FirstFP32Residual(){
        static_assert(OnChip,"N200 firstresidual needscanonicalLUinWG");constexpr uint32_t N=200,D=256,Tile=32,Count=Tile*D;
        Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();DataCopyExtParams ps{N,8*4,uint32_t((rp-8)/8),uint32_t((rp-8)*4),0};DataCopyPad(wg[N],mat[N],ps);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
        auto aa=mat,pp=aa[Count],bc=pp[Count],xc=bc[N*8],xx=xc[N*8],blocks=xx[D],sum=blocks[Tile*D/8],values=sum[64];auto rr=ResidualCache();
        static_assert((2*Count+2*N*8+D+Tile*D/8+2*64)*4<N*208*4,"N200 firstresidual workexcludesDelta");
        Duplicate(rr,0.0f,N*8);Duplicate(xx,0.0f,D);Duplicate(sum,0.0f,64);PipeBarrier<PIPE_V>();for(uint32_t i=0;i<D;++i)dotIndex.SetValue(i,i*8*4);Fence<HardEvent::S_V>();
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopyPadExtParams<float>pad{true,0,0,0};DataCopyExtParams px{N,8*4,N*4,0,0},pb{N,4*4,0,0,0};DataCopyPad(xc,wg[N],px,pad);DataCopyPad(bc,bg[bo],pb,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();const uint32_t c=GetBlockIdx()%4;
        AscendC::Gather(xx,xc,dotIndex,c*4,N);PipeBarrier<PIPE_V>();
        for(uint32_t first=0;first<N;first+=Tile){uint32_t rows=N-first<Tile?N-first:Tile;Duplicate(aa,0.0f,Count);PipeBarrier<PIPE_V>();Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();DataCopyExtParams pa{uint16_t(rows),N*4,0,(D-N)/8,0};DataCopyPad(aa,ag[ao+uint64_t(first)*N],pa,pad);Fence<HardEvent::MTE2_V>();TileMul(pp,aa,xx,D,Tile);
            BlockReduceSum(blocks,pp,int32_t(Count/64),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();WholeReduceSum(sum,blocks,int32_t(D/8),uint8_t(Tile),1,1,D/64);PipeBarrier<PIPE_V>();Duplicate(values,0.0f,64);PipeBarrier<PIPE_V>();AscendC::Gather(values,bc[first*8],dotIndex,c*4,rows);PipeBarrier<PIPE_V>();Sub(sum,values,sum,64);PipeBarrier<PIPE_V>();Brcb(column,sum,uint8_t(Tile/8),{1,8});PipeBarrier<PIPE_V>();uint64_t mask[2]={1ULL<<c,0};Muls(rr[first*8],column,1.0f,mask,uint8_t(rows),{1,1,1,1});PipeBarrier<PIPE_V>();
        }
        FromGlobal(mat,wg,N*rp);
    }
    // Lanen200 ownership: TwoProduct plus exact-unit extraction sums, MODE15 only.
    __aicore__ inline void Lanen200_ExtractedRowSum(LocalTensor<float> dst,
                                                  LocalTensor<float> src,LocalTensor<float> blocks){
        BlockReduceSum(blocks,src,int32_t(64),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();
        WholeReduceSum(dst,blocks,int32_t(32),uint8_t(16),1,1,4);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool Lanen200_TryExtractedResidual(){
        static_assert(OnChip,"Lanen200 extracted residual retains OnChip LU");
        constexpr uint32_t N=200,D=256,Tile=16,Count=Tile*D;
        if constexpr(Lanen200_BlockTri16){
            Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();
            DataCopyExtParams pxsave{N,8*4,uint32_t((rp-8)/8),uint32_t((rp-8)*4),0};DataCopyPad(wg[N],mat[N],pxsave);
            Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
        }else ToGlobal(wg,mat,N*rp);
        auto aa=mat;auto pp=aa[Count];auto ee=pp[Count];auto ss=ee[Count];auto zz=ss[Count];auto tt=zz[Count];
        auto xc=tt[Count];auto bc=xc[N*8];auto xx=bc[N*8];auto xh=xx[D];auto xl=xh[D];
        auto blocks=xl[D];auto maxPairs=blocks[Tile*D/8];auto sigma=maxPairs[64];
        auto expMask=sigma[64];auto sumHigh=expMask[64];auto sumLow=sumHigh[64];
        auto sumTiny=sumLow[64];auto sumErr=sumTiny[64];auto rhsValues=sumErr[64];auto lowTotal=rhsValues[64];
        auto guard=lowTotal[64];auto maxIndex=guard[16].ReinterpretCast<uint32_t>();auto sigmaBroad=guard[80];
        static_assert((6*Count+2*N*8+3*D+Tile*D/8+9*64+16+64+Tile*8)*4<N*208*4,"Lanen200 extracted work excludes Delta");
        auto rr=ResidualCache();
        Duplicate(rr,0.0f,N*8);Duplicate(xx,0.0f,D);Duplicate(xh,0.0f,D);Duplicate(xl,0.0f,D);
        Duplicate(maxPairs,0.0f,64);Duplicate(sumHigh,0.0f,64);Duplicate(sumLow,0.0f,64);
        Duplicate(sumTiny,0.0f,64);Duplicate(sumErr,0.0f,64);
        Duplicate(expMask.ReinterpretCast<int32_t>(),int32_t(0x7f800000),64);PipeBarrier<PIPE_V>();
        for(uint32_t i=0;i<D;++i)dotIndex.SetValue(i,i*8*4);
        for(uint32_t i=0;i<64;++i)maxIndex.SetValue(i,i*8);
        Fence<HardEvent::S_V>();
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();
        DataCopyPadExtParams<float> pad{true,0,0,0};DataCopyExtParams px{N,8*4,N*4,0,0};
        DataCopyPad(xc,wg[N],px,pad);DataCopyExtParams pb{N,4*4,0,0,0};DataCopyPad(bc,bg[bo],pb,pad);
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        const uint32_t c=GetBlockIdx()%4;
        AscendC::Gather(xx,xc,dotIndex,c*4,N);PipeBarrier<PIPE_V>();
        Muls(xh,xx,4097.0f,D);PipeBarrier<PIPE_V>();Sub(xl,xh,xx,D);PipeBarrier<PIPE_V>();
        Sub(xh,xh,xl,D);PipeBarrier<PIPE_V>();Sub(xl,xx,xh,D);PipeBarrier<PIPE_V>();
        for(uint32_t first=0;first<N;first+=Tile){
            const uint32_t rows=N-first<Tile?N-first:Tile;
            Duplicate(aa,0.0f,Count);PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
            DataCopyExtParams pa{uint16_t(rows),N*4,0,(D-N)/8,0};DataCopyPad(aa,ag[ao+uint64_t(first)*N],pa,pad);Fence<HardEvent::MTE2_V>();
            // Preserve the full error-free product sequence used by the Dot2 baseline.
            TileMul(pp,aa,xx,D,Tile);
            Muls(tt,aa,4097.0f,Count);PipeBarrier<PIPE_V>();Sub(zz,tt,aa,Count);PipeBarrier<PIPE_V>();
            Sub(ss,tt,zz,Count);PipeBarrier<PIPE_V>();Sub(zz,aa,ss,Count);PipeBarrier<PIPE_V>();
            TileMul(ee,ss,xh,D,Tile);Sub(ee,pp,ee,Count);PipeBarrier<PIPE_V>();
            TileMul(tt,zz,xh,D,Tile);Sub(ee,ee,tt,Count);PipeBarrier<PIPE_V>();
            TileMul(tt,ss,xl,D,Tile);Sub(ee,ee,tt,Count);PipeBarrier<PIPE_V>();
            TileMul(tt,zz,xl,D,Tile);Sub(ee,tt,ee,Count);PipeBarrier<PIPE_V>();
            Abs(aa,pp,Count);PipeBarrier<PIPE_V>();
            BlockReduceMax(blocks,aa,int32_t(64),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();
            WholeReduceMax(maxPairs,blocks,int32_t(32),uint8_t(Tile),1,1,4);PipeBarrier<PIPE_V>();
            Duplicate(sigma,1.0f,64);PipeBarrier<PIPE_V>();
            AscendC::Gather(sigma,maxPairs,maxIndex,0,Tile);PipeBarrier<PIPE_V>();
            WholeReduceMax(guard,sigma,int32_t(rows),uint8_t(1),1,1,8);
            WholeReduceMin(guard[8],sigma,int32_t(rows),uint8_t(1),1,1,8);Fence<HardEvent::V_S>();
            union Lanen200_Bits{float f;uint32_t u;}largest,smallest;
            largest.f=guard.GetValue(0);smallest.f=guard.GetValue(8);
            const uint32_t highExp=((largest.u>>23)&255)+10,lowExp=((smallest.u>>23)&255)+10;
            // Exact extraction requires finite normal sigma and sigma*2^-12.
            // These IEEE exponent limits match the existing RowResidual fallback domain.
            if(!(lowExp>13 && highExp<254)){FromGlobal(mat,wg,N*rp);return false;}
            And(sigma.ReinterpretCast<int16_t>(),sigma.ReinterpretCast<int16_t>(),expMask.ReinterpretCast<int16_t>(),128);PipeBarrier<PIPE_V>();
            Adds(sigma.ReinterpretCast<int32_t>(),sigma.ReinterpretCast<int32_t>(),int32_t(10U<<23),64);PipeBarrier<PIPE_V>();
            Brcb(sigmaBroad,sigma,uint8_t(Tile/8),{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t off=0;off<D;off+=64)
                Muls(ss[off],sigmaBroad,1.0f,uint64_t(64),uint8_t(Tile),{1,0,D/8,1});
            PipeBarrier<PIPE_V>();
            // High and low extracted components have fixed binary units; their sums are exact.
            Add(zz,pp,ss,Count);PipeBarrier<PIPE_V>();Sub(zz,zz,ss,Count);PipeBarrier<PIPE_V>();
            Sub(tt,pp,zz,Count);PipeBarrier<PIPE_V>();
            Lanen200_ExtractedRowSum(sumHigh,zz,blocks);
            Muls(ss,ss,0.000244140625f,Count);PipeBarrier<PIPE_V>();
            Add(aa,tt,ss,Count);PipeBarrier<PIPE_V>();Sub(aa,aa,ss,Count);PipeBarrier<PIPE_V>();
            Sub(zz,tt,aa,Count);PipeBarrier<PIPE_V>();
            Lanen200_ExtractedRowSum(sumLow,aa,blocks);
            Lanen200_ExtractedRowSum(sumTiny,zz,blocks);
            Lanen200_ExtractedRowSum(sumErr,ee,blocks);
            Add(lowTotal,sumLow,sumTiny,64);PipeBarrier<PIPE_V>();Add(lowTotal,lowTotal,sumErr,64);PipeBarrier<PIPE_V>();
            Duplicate(rhsValues,0.0f,64);PipeBarrier<PIPE_V>();
            AscendC::Gather(rhsValues,bc[first*8],dotIndex,c*4,rows);PipeBarrier<PIPE_V>();
            Sub(ss,rhsValues,sumHigh,64);PipeBarrier<PIPE_V>();Sub(zz,rhsValues,ss,64);PipeBarrier<PIPE_V>();
            Add(tt,ss,zz,64);PipeBarrier<PIPE_V>();Sub(tt,rhsValues,tt,64);PipeBarrier<PIPE_V>();
            Sub(zz,zz,sumHigh,64);PipeBarrier<PIPE_V>();Add(tt,tt,zz,64);PipeBarrier<PIPE_V>();
            Sub(tt,tt,lowTotal,64);PipeBarrier<PIPE_V>();Add(ss,ss,tt,64);PipeBarrier<PIPE_V>();
            Brcb(column,ss,uint8_t(Tile/8),{1,8});PipeBarrier<PIPE_V>();
            uint64_t mask[2]={1ULL<<c,0};Muls(rr[first*8],column,1.0f,mask,uint8_t(rows),{1,1,1,1});PipeBarrier<PIPE_V>();
        }
        FromGlobal(mat,wg,N*rp);return true;
    }
    // End Lanen200 extracted residual ownership.

    __aicore__ inline void TiledResidual() {
        if constexpr(Lanen200_UseExtractedResidual){if(Lanen200_TryExtractedResidual())return;}
        if(t.originalSchur){Lanen256_FastResidual();return;}
        const uint32_t tile = n >= 128 ? 16 : 8;
        const uint32_t d = t.dotSize, count = tile * d, compactCount = tile * 8;
        const bool spill = OnChip && n >= 128;
        auto work = triangle;
        if (spill) {
            ToGlobal(wg, mat, n * rp);
            work = mat;
        }
        auto aa = work; auto pp = work[count]; auto ee = work[2 * count];
        auto ss = work[3 * count]; auto zz = work[4 * count]; auto tt = work[5 * count];
        auto ll = aa; auto hh = pp; auto hl = ee;
        auto xc = work[6 * count]; auto bc = xc[n * kp];
        auto xx = bc[n * kp]; auto xh = xx[d]; auto xl = xh[d];
        auto smallP = xl[d]; auto smallE = smallP[compactCount];
        auto shifts = smallE[compactCount].ReinterpretCast<uint32_t>();
        auto rr = ResidualCache();
        Duplicate(rr, 0.0f, n * kp); PipeBarrier<PIPE_V>();
        if (spill) {
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>(); Fence<HardEvent::MTE3_MTE2>();
            DataCopyExtParams p{static_cast<uint16_t>(n), kp * 4, np * 4, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(xc, wg[np], p, pad);
            Fence<HardEvent::MTE2_V>();
        } else if constexpr (OnChip) {
            DataCopyParams p{static_cast<uint16_t>(n), static_cast<uint16_t>(kp / 8),
                             static_cast<uint16_t>(np / 8), 0};
            DataCopy(xc, mat[np], p); PipeBarrier<PIPE_V>();
        } else {
            DataCopy(xc, mat, n * kp); PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>(); Fence<HardEvent::MTE3_MTE2>();
        DataCopyExtParams pb{static_cast<uint16_t>(n), k * 4, 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, 0, 0, 0};
        DataCopyPad(bc, bg[bo], pb, pad);
        Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        for (uint32_t i = 0; i < d; ++i) dotIndex.SetValue(i, i * kp * 4);
        for (uint32_t level = 0; level < 3; ++level) {
            const uint32_t half = 4 >> level;
            for (uint32_t i = 0; i < tile; ++i) for (uint32_t q = 0; q < 8; ++q)
                shifts.SetValue(level * compactCount + i * 8 + q, (i * 8 + ((q + half) & 7)) * 4);
        }
        Fence<HardEvent::S_V>();
        const bool splitPaired=t.paired && t.rhsSplit;
        const uint32_t rowBegin=splitPaired?(GetBlockIdx()&1)*(n/2):0,rowEnd=splitPaired?rowBegin+n/2:n;
        for (uint32_t c = splitPaired ? (GetBlockIdx()/2)%PairSlots() : (t.blocked && t.rhsSplit) ? GetBlockIdx()%k : (t.paired && k >= 2) ? (GetBlockIdx() & 1) : 0; c < k; c += splitPaired ? k : (t.blocked && t.rhsSplit) ? k : (t.paired && k >= 2) ? 2 : 1) {
            Duplicate(xx, 0.0f, d); PipeBarrier<PIPE_V>();
            AscendC::Gather(xx, xc, dotIndex, c * 4, n); PipeBarrier<PIPE_V>();
            Muls(xh, xx, 4097.0f, d); PipeBarrier<PIPE_V>();
            Sub(xl, xh, xx, d); PipeBarrier<PIPE_V>();
            Sub(xh, xh, xl, d); PipeBarrier<PIPE_V>();
            Sub(xl, xx, xh, d); PipeBarrier<PIPE_V>();
            for (uint32_t first = rowBegin; first < rowEnd; first += tile) {
                const uint32_t rows = n - first < tile ? n - first : tile;
                Duplicate(aa, 0.0f, count); PipeBarrier<PIPE_V>();
                Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
                DataCopyExtParams pa{static_cast<uint16_t>(rows), n * 4, 0, (d - np) / 8, 0};
                DataCopyPad(aa, ag[ao + uint64_t(first) * n], pa, pad);
                Fence<HardEvent::MTE2_V>();
                TileMul(pp, aa, xx, d, tile);
                Muls(tt, aa, 4097.0f, count); PipeBarrier<PIPE_V>();
                Sub(zz, tt, aa, count); PipeBarrier<PIPE_V>();
                Sub(ss, tt, zz, count); PipeBarrier<PIPE_V>();
                Sub(zz, aa, ss, count); PipeBarrier<PIPE_V>();
                TileMul(ee, ss, xh, d, tile);
                Sub(ee, pp, ee, count); PipeBarrier<PIPE_V>();
                TileMul(tt, zz, xh, d, tile);
                Sub(ee, ee, tt, count); PipeBarrier<PIPE_V>();
                TileMul(tt, ss, xl, d, tile);
                Sub(ee, ee, tt, count); PipeBarrier<PIPE_V>();
                TileMul(tt, zz, xl, d, tile);
                Sub(ee, tt, ee, count); PipeBarrier<PIPE_V>();
                BinaryRepeatParams reduce{1, 1, 1, static_cast<uint8_t>(d / 8),
                                         static_cast<uint8_t>(d / 8), static_cast<uint8_t>(d / 8)};
                UnaryRepeatParams copy{1, 1, static_cast<uint8_t>(d / 8), static_cast<uint8_t>(d / 8)};
                for (uint32_t span = d; span > 8; span >>= 1) {
                    const uint32_t half = span >> 1;
                    for (uint32_t offset = 0; offset < half; offset += 64) {
                        const uint64_t mask = half - offset < 64 ? half - offset : 64;
                        Add(ss[offset], pp[offset], pp[half + offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Sub(zz[offset], ss[offset], pp[offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Sub(tt[offset], ss[offset], zz[offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Sub(tt[offset], pp[offset], tt[offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Sub(zz[offset], pp[half + offset], zz[offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Add(tt[offset], tt[offset], zz[offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Add(ll[offset], ee[offset], ee[half + offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Add(ll[offset], ll[offset], tt[offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Muls(pp[offset], ss[offset], 1.0f, mask, static_cast<uint8_t>(tile), copy); PipeBarrier<PIPE_V>();
                        Muls(ee[offset], ll[offset], 1.0f, mask, static_cast<uint8_t>(tile), copy); PipeBarrier<PIPE_V>();
                    }
                }
                DataCopyParams pack{static_cast<uint16_t>(tile), 1, static_cast<uint16_t>(d / 8 - 1), 0};
                DataCopy(smallP, pp, pack); DataCopy(smallE, ee, pack); PipeBarrier<PIPE_V>();
                for (uint32_t level = 0; level < 3; ++level) {
                    AscendC::Gather(hh, smallP, shifts[level * compactCount], 0, compactCount); PipeBarrier<PIPE_V>();
                    AscendC::Gather(hl, smallE, shifts[level * compactCount], 0, compactCount); PipeBarrier<PIPE_V>();
                    Add(ss, smallP, hh, compactCount); PipeBarrier<PIPE_V>();
                    Sub(zz, ss, smallP, compactCount); PipeBarrier<PIPE_V>();
                    Sub(tt, ss, zz, compactCount); PipeBarrier<PIPE_V>();
                    Sub(tt, smallP, tt, compactCount); PipeBarrier<PIPE_V>();
                    Sub(zz, hh, zz, compactCount); PipeBarrier<PIPE_V>();
                    Add(tt, tt, zz, compactCount); PipeBarrier<PIPE_V>();
                    Add(ll, smallE, hl, compactCount); PipeBarrier<PIPE_V>();
                    Add(ll, ll, tt, compactCount); PipeBarrier<PIPE_V>();
                    DataCopy(smallP, ss, compactCount); DataCopy(smallE, ll, compactCount); PipeBarrier<PIPE_V>();
                }
                AscendC::Gather(hh, smallP, dotIndex, 0, tile); PipeBarrier<PIPE_V>();
                AscendC::Gather(hl, smallE, dotIndex, 0, tile); PipeBarrier<PIPE_V>();
                Duplicate(ll, 0.0f, tile); PipeBarrier<PIPE_V>();
                AscendC::Gather(ll, bc[first * kp], dotIndex, c * 4, rows); PipeBarrier<PIPE_V>();
                Sub(ss, ll, hh, tile); PipeBarrier<PIPE_V>();
                Sub(zz, ll, ss, tile); PipeBarrier<PIPE_V>();
                Add(tt, ss, zz, tile); PipeBarrier<PIPE_V>();
                Sub(tt, ll, tt, tile); PipeBarrier<PIPE_V>();
                Sub(zz, zz, hh, tile); PipeBarrier<PIPE_V>();
                Add(tt, tt, zz, tile); PipeBarrier<PIPE_V>();
                Sub(tt, tt, hl, tile); PipeBarrier<PIPE_V>();
                Add(ss, ss, tt, tile); PipeBarrier<PIPE_V>();
                Brcb(column, ss, static_cast<uint8_t>(tile / 8), {1, 8}); PipeBarrier<PIPE_V>();
                uint64_t mask[2] = {1ULL << c, 0};
                Muls(rr[first * kp], column, 1.0f, mask, static_cast<uint8_t>(rows), {1, 1, static_cast<uint8_t>(kp / 8), 1});
                PipeBarrier<PIPE_V>();
            }
        }
        if (spill) FromGlobal(mat, wg, n * rp);
        if constexpr (!OnChip) ToGlobal(wg[n * rp], rr, n * kp);
    }

    __aicore__ inline void RowResidual() {
        const uint32_t d = t.dotSize;
        uint32_t logd = 0; for (uint32_t q = d; q > 1; q >>= 1) ++logd;
        for (uint32_t i = 0; i < d; ++i) dotIndex.SetValue(i, i * 4);
        auto base = scratch.Get<float>();
        auto aa = base; auto xx = base[d]; auto xh = base[2*d]; auto xl = base[3*d];
        auto pp = base[4*d]; auto ee = base[5*d];
        auto ss = base[6*d]; auto zz = base[7*d]; auto tt = base[8*d]; auto ll = base[9*d];
        auto hh = base[10*d]; auto hl = base[11*d]; auto bb = base[13*d];
        LocalTensor<float> rr = mat[n * kp];
        if constexpr (OnChip) rr = mat[n * rp];
        Duplicate(rr, 0.0f, n * kp); PipeBarrier<PIPE_V>();
        FromGlobal(bb, bg[bo], n * k);
        for (uint32_t c = 0; c < k; ++c) {
            Duplicate(xx, 0.0f, d); PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < n; ++j) {
                Load(aa, j * rp + np, kp);
                xx.SetValue(j, aa.GetValue(c));
            }
            Fence<HardEvent::S_V>();
            Muls(xh, xx, 4097.0f, d); PipeBarrier<PIPE_V>();
            Sub(xl, xh, xx, d); PipeBarrier<PIPE_V>();
            Sub(xh, xh, xl, d); PipeBarrier<PIPE_V>();
            Sub(xl, xx, xh, d); PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < n; ++i) {
                Duplicate(aa, 0.0f, d); PipeBarrier<PIPE_V>();
                FromGlobal(aa, ag[ao + uint64_t(i) * n], n);
                // Error-free vector TwoProduct, followed by compensated pairs.
                Mul(pp, aa, xx, d); PipeBarrier<PIPE_V>();
                Muls(tt, aa, 4097.0f, d); PipeBarrier<PIPE_V>();
                Sub(zz, tt, aa, d); PipeBarrier<PIPE_V>();
                Sub(ss, tt, zz, d); PipeBarrier<PIPE_V>();
                Sub(zz, aa, ss, d); PipeBarrier<PIPE_V>();
                Mul(ee, ss, xh, d); PipeBarrier<PIPE_V>();
                Sub(ee, pp, ee, d); PipeBarrier<PIPE_V>();
                Mul(tt, zz, xh, d); PipeBarrier<PIPE_V>();
                Sub(ee, ee, tt, d); PipeBarrier<PIPE_V>();
                Mul(tt, ss, xl, d); PipeBarrier<PIPE_V>();
                Sub(ee, ee, tt, d); PipeBarrier<PIPE_V>();
                Mul(tt, zz, xl, d); PipeBarrier<PIPE_V>();
                Sub(ee, tt, ee, d); PipeBarrier<PIPE_V>();
                if (d > 16) {
                    Abs(tt, pp, d); PipeBarrier<PIPE_V>();
                    AscendC::ReduceMax(ss, tt, hh, static_cast<int32_t>(d), false);
                    PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
                    union FloatBits { float value; uint32_t bits; } scale;
                    scale.value = ss.GetValue(0);
                    const uint32_t exponent = ((scale.bits >> 23) & 255) + logd + 2;
                    if (exponent > 13 && exponent < 254) {
                        scale.bits = exponent << 23; const float sigma = scale.value;
                        scale.bits = (exponent - 12) << 23; const float sigmaLow = scale.value;
                        // Both extracted sums are exact: common integer units and
                        // fewer than24 significant sum bits. The remainder is tiny.
                        Adds(ss, pp, sigma, d); PipeBarrier<PIPE_V>();
                        Adds(ss, ss, -sigma, d); PipeBarrier<PIPE_V>();
                        Sub(zz, pp, ss, d); PipeBarrier<PIPE_V>();
                        Adds(tt, zz, sigmaLow, d); PipeBarrier<PIPE_V>();
                        Adds(tt, tt, -sigmaLow, d); PipeBarrier<PIPE_V>();
                        Sub(ll, zz, tt, d); PipeBarrier<PIPE_V>();
                        AscendC::ReduceSum(pp, ss, hh, static_cast<int32_t>(d)); PipeBarrier<PIPE_V>();
                        AscendC::ReduceSum(pp[8], tt, hh, static_cast<int32_t>(d)); PipeBarrier<PIPE_V>();
                        AscendC::ReduceSum(zz, ll, hh, static_cast<int32_t>(d)); PipeBarrier<PIPE_V>();
                        AscendC::ReduceSum(ss, ee, hh, static_cast<int32_t>(d)); PipeBarrier<PIPE_V>();
                        Fence<HardEvent::V_S>();
                        float value = bb.GetValue(i * k + c), error = 0.0f;
                        CompensatedScalar(-pp.GetValue(0), value, error);
                        CompensatedScalar(-pp.GetValue(8), value, error);
                        CompensatedScalar(-zz.GetValue(0), value, error);
                        CompensatedScalar(-ss.GetValue(0), value, error);
                        rr.SetValue(i * kp + c, value + error);
                        continue;
                    }
                }
                for (uint32_t span = d; span > 1; span >>= 1) {
                    const uint32_t half = span >> 1;
                    AscendC::Gather(hh, pp, dotIndex, half * 4, half); PipeBarrier<PIPE_V>();
                    AscendC::Gather(hl, ee, dotIndex, half * 4, half); PipeBarrier<PIPE_V>();
                    Add(ss, pp, hh, half); PipeBarrier<PIPE_V>();
                    Sub(zz, ss, pp, half); PipeBarrier<PIPE_V>();
                    Sub(tt, ss, zz, half); PipeBarrier<PIPE_V>();
                    Sub(tt, pp, tt, half); PipeBarrier<PIPE_V>();
                    Sub(zz, hh, zz, half); PipeBarrier<PIPE_V>();
                    Add(tt, tt, zz, half); PipeBarrier<PIPE_V>();
                    Add(ll, ee, hl, half); PipeBarrier<PIPE_V>();
                    Add(ll, ll, tt, half); PipeBarrier<PIPE_V>();
                    const uint32_t copied = half < 8 ? 8 : half;
                    DataCopy(pp, ss, copied); PipeBarrier<PIPE_V>();
                    DataCopy(ee, ll, copied); PipeBarrier<PIPE_V>();
                }
                Fence<HardEvent::V_S>();
                const float high = pp.GetValue(0), low = ee.GetValue(0), b = bb.GetValue(i * k + c);
                // Volatile enforces the error-free scalar TwoDiff order.
                volatile float value = b - high;
                volatile float bvirtual = b - value;
                volatile float avirtual = value + bvirtual;
                volatile float berror = bvirtual - high;
                volatile float aerror = b - avirtual;
                volatile float error = aerror + berror;
                volatile float correction = error - low;
                rr.SetValue(i * kp + c, value + correction);
            }
        }
        Fence<HardEvent::S_V>();
        if constexpr (!OnChip) ToGlobal(wg[n * rp], rr, n * kp);
    }

    __aicore__ inline void Refine() {
        bool Lanen200_cheapFirst=false;
        if constexpr(Lanen200_BlockTri16){Lanen200_cheapFirst=(Lanen200_refinePass++==0);if(Lanen200_cheapFirst)Lanen200_FirstFP32Residual();}
        if(!Lanen200_cheapFirst){
        if (t.dotSize) {
            if ((n > 16 && n <= 64) || (n >= 128 && n <= 256 && (!OnChip || 6 * 16 * t.dotSize + 2 * n * kp + 3 * t.dotSize + 5 * 16 * 8 <= n * rp))) TiledResidual();
            else RowResidual();
        } else {
            // Error-free two-product and TwoSum, accumulating both errors (Dot2).
            // 910B vector multiply-add is not assumed to be a true FMA.
            for (uint32_t i = 0; i < n; ++i) {
                FromGlobal(original, ag[ao + uint64_t(i) * n], n);
                FromGlobal(v0, bg[bo + uint64_t(i) * k], k);
                Duplicate(v1, 0.0f, kp); PipeBarrier<PIPE_V>();
                for (uint32_t j = 0; j < n; ++j) {
                    const float a = -original.GetValue(j);
                    union Bits { float f; uint32_t u; } bits;
                    bits.f = a; bits.u &= 0xfffff000U;
                    const float ahi = bits.f, alo = a - ahi;
                    Load(row, j * rp + np, kp);
                    Muls(v2, row, a, kp); PipeBarrier<PIPE_V>();
                    Muls(v4, row, 4097.0f, kp); PipeBarrier<PIPE_V>();
                    Sub(v5, v4, row, kp); PipeBarrier<PIPE_V>();
                    Sub(v4, v4, v5, kp); PipeBarrier<PIPE_V>();
                    Sub(v5, row, v4, kp); PipeBarrier<PIPE_V>();
                    Muls(v6, v4, ahi, kp); PipeBarrier<PIPE_V>();
                    Sub(v6, v6, v2, kp); PipeBarrier<PIPE_V>();
                    Muls(v7, v5, ahi, kp); PipeBarrier<PIPE_V>();
                    Add(v6, v6, v7, kp); PipeBarrier<PIPE_V>();
                    Muls(v7, v4, alo, kp); PipeBarrier<PIPE_V>();
                    Add(v6, v6, v7, kp); PipeBarrier<PIPE_V>();
                    Muls(v7, v5, alo, kp); PipeBarrier<PIPE_V>();
                    Add(v6, v6, v7, kp); PipeBarrier<PIPE_V>();
                    Add(v3, v0, v2, kp); PipeBarrier<PIPE_V>();
                    Sub(v4, v3, v0, kp); PipeBarrier<PIPE_V>();
                    Sub(v5, v3, v4, kp); PipeBarrier<PIPE_V>();
                    Sub(v5, v0, v5, kp); PipeBarrier<PIPE_V>();
                    Sub(v4, v2, v4, kp); PipeBarrier<PIPE_V>();
                    Add(v5, v5, v4, kp); PipeBarrier<PIPE_V>();
                    Add(v6, v6, v5, kp); PipeBarrier<PIPE_V>();
                    Add(v1, v1, v6, kp); PipeBarrier<PIPE_V>();
                    LocalCopy(v0, v3, kp);
                }
                Add(v0, v0, v1, kp); PipeBarrier<PIPE_V>();
                StoreR(i, v0);
            }
        }
        }
        if (t.dotSize && n <= 256) {
            if (t.paired && k >= 2) {
                SolveSync();
                GlobalTensor<float> workerR;
                const uint64_t elements=uint64_t(n)*(np+2*kp);
                workerR.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace)+uint64_t(GetBlockIdx()+1)*elements+n*rp);
                FromGlobal(triangle,workerR,n*kp);
                auto rhs=ResidualCache();
                if(t.rhsSplit){
                    uint32_t c=(GetBlockIdx()/2)%PairSlots(),first=n/2;uint64_t mask[2]={1ULL<<c,0};
                    Muls(rhs[first*kp],triangle[first*kp],1.0f,mask,uint8_t(n/2),{1,1,uint8_t(kp/8),uint8_t(kp/8)});PipeBarrier<PIPE_V>();
                }else for(uint32_t c=1;c<k;c+=2) for(uint32_t first=0;first<n;first+=128) {
                    uint64_t mask[2]={1ULL<<c,0};uint32_t rows=n-first<128?n-first:128;
                    Muls(rhs[first*kp],triangle[first*kp],1.0f,mask,uint8_t(rows),{1,1,uint8_t(kp/8),uint8_t(kp/8)});PipeBarrier<PIPE_V>();
                }
            }
            PermuteResidualRows(); ForwardSolveRows();
        } else {
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t pivot = pivots.GetValue(j);
                if (pivot != j) {
                    LoadR(row, j); LoadR(other, pivot);
                    StoreR(pivot, row); StoreR(j, other);
                }
            }
            for (uint32_t j = 0; j < n; ++j) {
                Gather(j, j + 1, n); LoadR(row, j);
                for (uint32_t i = j + 1; i < n; ++i) {
                    const float coefficient = Coeff(i, j, j + 1);
                    LoadR(other, i); Ax(other, row, -coefficient, kp); StoreR(i, other);
                }
            }
        }
        BackSolve(true);
        if (t.dotSize && n <= 256) {
            if constexpr (OnChip) {
                const uint8_t blocks = rp / 8;
                BinaryRepeatParams add{blocks, blocks, 1, static_cast<uint8_t>(8 * blocks),
                                       static_cast<uint8_t>(8 * blocks), 8};
                if (n / 8) Add(mat[np], mat[np], mat[n * rp], uint64_t(64), static_cast<uint8_t>(n / 8), add);
                if (n % 8) {
                    uint32_t first = n / 8 * 8;
                    Add(mat[first * rp + np], mat[first * rp + np], mat[n * rp + first * kp],
                        uint64_t((n % 8) * 8), uint8_t(1), add);
                }
                PipeBarrier<PIPE_V>();
            } else {
                Add(mat, mat, mat[n * kp], n * kp); PipeBarrier<PIPE_V>();
            }
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                Load(row, i * rp + np, kp); LoadR(other, i);
                Add(row, row, other, kp); PipeBarrier<PIPE_V>();
                Store(i * rp + np, row, kp);
            }
        }
    }

    uint32_t Lanen200_refinePass=0;
    bool Lanen200_extraIR=false;
    GM_ADDR userWorkspace;
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage, scratch, pivotBuffer, triangleBuffer, blockSolveBuffer;
    LocalTensor<float> triangle;
    LocalTensor<float> mat, row, other, column, original, v0, v1, v2, v3, v4, v5, v6, v7;
    LocalTensor<int32_t> pivots;
    LocalTensor<uint32_t> dotIndex;
    GlobalTensor<float> ag, bg, xg, wg;
    LinalgSolveTilingData t;
    uint32_t n, k, np, kp, rp;
    uint64_t ao, bo, xo;
};

// BEGIN Lanen64 Tiny16 direct solve.
class Lanen64_Tiny16 {
    static constexpr uint32_t N=16, CP=24, COLS=17, AL=24;
    static constexpr uint32_t AB=AL*CP, MM=AB+N, RW=MM+8;
    static constexpr uint32_t BC=RW+AL, XX=BC+AL*8, TOTAL=XX+N;
    static constexpr uint32_t CI=TOTAL, OI=CI+AL, AA=OI+N, BB=AA+N*N, END=BB+N;
    static constexpr uint32_t CB=END, FINAL=CB+AL*8;
    static_assert(FINAL*4<=196608,"Lanen64 Tiny16 direct broadcast UB");
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR out){
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        mem=LocalTensor<float>(TPosition::VECCALC,0,TOTAL);
        mat=mem;ab=mem[AB];maximum=mem[MM];row=mem[RW];broad=mem[BC];x=mem[XX];
        ci=LocalTensor<uint32_t>(TPosition::VECCALC,CI*4,AL);
        oi=LocalTensor<uint32_t>(TPosition::VECCALC,OI*4,N);
        aa=LocalTensor<float>(TPosition::VECCALC,AA*4,N*N);
        bb=LocalTensor<float>(TPosition::VECCALC,BB*4,N);
        cb=LocalTensor<uint32_t>(TPosition::VECCALC,CB*4,AL*8);
        const uint32_t batch=GetBlockIdx();
        DataCopyPadExtParams<float> pad{true,0,0,0};
        DataCopyExtParams pa{1,N*N*4,0,0,0},pb{1,N*4,0,0,0};
        DataCopyPad(aa,ag[uint64_t(batch)*N*N],pa,pad);
        DataCopyPad(bb,bg[uint64_t(batch)*N],pb,pad);
        auto positions=row.ReinterpretCast<int32_t>();
        for(uint32_t q=0;q<AL;++q)positions.SetValue(q,int32_t(q));
        Fence<HardEvent::S_V>();
        Muls(ci.ReinterpretCast<int32_t>(),positions,int32_t(CP*4),AL);PipeBarrier<PIPE_V>();
        Brcb(cb,ci,AL/8,{1,8});PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process(){
        Fence<HardEvent::MTE2_V>();
        Duplicate(mat[(N+1)*CP],0.0f,(AL-N-1)*CP);PipeBarrier<PIPE_V>();
        uint64_t src[16],dst[16];
        for(uint32_t r=0;r<N;++r)src[r]=reinterpret_cast<uint64_t>(aa[r*N].GetPhyAddr());
        for(uint32_t c=0;c<8;++c)
            for(uint32_t half=0;half<2;++half)
                dst[2*c+half]=reinterpret_cast<uint64_t>(mat[c*CP+half*8].GetPhyAddr());
        TransDataTo5HDParams transpose{false,false,2,CP,1};
        TransDataTo5HD<float>(dst,src,transpose);
        DataCopy(mat[N*CP],bb,N);PipeBarrier<PIPE_V>();
        uint64_t active=0xffff;
        for(uint32_t first=0;first<N;first+=8){
            const uint32_t width=AL-first;
            for(uint32_t q=0;q<8;++q)Pivot(first+q,first,width,active);
        }
        Fence<HardEvent::S_V>();
        AscendC::Gather(x,mat,oi,N*CP*4,N);PipeBarrier<PIPE_V>();
        Muls(x,x,-1.0f,N);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams storeParams{1,N*4,0,0,0};
        DataCopyPad(xg[uint64_t(GetBlockIdx())*N],x,storeParams);
        Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
    }
private:
    __aicore__ inline void Pivot(uint32_t j,uint32_t first,uint32_t width,uint64_t &active){
        auto coef=mat[j*CP];
        uint32_t pivot=0;float diagonal=0.0f;
        bool search=j<(3*N)/8;
        if(!search){
            pivot=uint32_t(GetSFFValue<1>(active));
            Fence<HardEvent::V_S>();
            diagonal=coef.GetValue(pivot);search=(diagonal==0.0f);
        }
        if(search){
            Abs(ab,coef,N);PipeBarrier<PIPE_V>();
            uint64_t available[2]={active,0};
            WholeReduceMax(maximum,ab,available,uint8_t(1),1,1,8);
            Fence<HardEvent::V_S>();
            uint32_t bits;GetReduceRepeatMaxMinSpr<float>(bits,pivot);
            diagonal=coef.GetValue(pivot);
        }
        oi.SetValue(j,pivot*4);
        uint64_t chosen[2]={uint64_t(1)<<pivot,0};
        AscendC::Gather(broad,mat,cb[first*8],pivot*4,width*8);PipeBarrier<PIPE_V>();
        Muls(broad,broad,-1.0f/diagonal,width*8);
        Adds(coef,coef,1.0f,chosen,uint8_t(1),{1,1,N/8,N/8});PipeBarrier<PIPE_V>();
        const uint32_t begin=j+1,count=COLS-j-1;
        MulAddDst(mat[begin*CP],coef,broad[(begin-first)*8],uint64_t(N),uint8_t(count),{1,1,0,CP/8,0,1});PipeBarrier<PIPE_V>();
        active&=~(uint64_t(1)<<pivot);
    }
    GlobalTensor<float> ag,bg,xg;
    LocalTensor<float> mem,mat,ab,maximum,row,broad,x,aa,bb;
    LocalTensor<uint32_t> ci,oi,cb;
};
// END Lanen64 Tiny16 direct solve.

template <uint32_t N> class TinyNegativePivot {
    static constexpr uint32_t D = 16, RP = 24, ND = N * D;
    static constexpr uint32_t MA = 0, AA = N * RP, PP = AA + ND, EE = PP + ND;
    static constexpr uint32_t SS = EE + ND, ZZ = SS + ND, TT = ZZ + ND, LL = TT + ND;
    static constexpr uint32_t HH = LL + ND, HL = HH + ND, AH = HL + ND, AL = AH + ND;
    static constexpr uint32_t XT = AL + ND, XH = XT + ND, XL = XH + ND, IV = XL + ND;
    static constexpr uint32_t XX = IV + ND, RR = XX + D, BB = RR + D, DD = BB + D;
    static constexpr uint32_t CC = DD + D, AB = CC + D, MM = AB + D, RW = MM + 8;
    static constexpr uint32_t BC = RW + RP, TOTAL = BC + D * 8;
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR out) {
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        mem = LocalTensor<float>(TPosition::VECCALC, 0, TOTAL);
        mat = mem[MA]; aa = mem[AA]; p = mem[PP]; e = mem[EE]; s = mem[SS]; z = mem[ZZ];
        temp = mem[TT]; low = mem[LL]; h = mem[HH]; hl = mem[HL]; ah = mem[AH]; al = mem[AL];
        xt = mem[XT]; xh = mem[XH]; xl = mem[XL]; inverse = mem[IV];
        x = mem[XX]; r = mem[RR]; bvalues = mem[BB]; delta = mem[DD];
        coef = mem[CC]; ab = mem[AB]; maximum = mem[MM]; row = mem[RW]; broad = mem[BC];
        colIndex = LocalTensor<uint32_t>(TPosition::VECCALC, TOTAL * 4, D);
        resultIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + D) * 4, D);
        tileIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D) * 4, ND);
        inverseIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D + ND) * 4, ND);
        shiftIndex = LocalTensor<uint32_t>(TPosition::VECCALC, (TOTAL + 2 * D + 2 * ND) * 4, 128);
        // AA and B are disjoint from every index-initialization writer.
        DataCopyPadExtParams<float> pad{true,0,0,0};
        DataCopyExtParams pa{1,ND*4,0,0,0},pb{1,D*4,0,0,0};
        const uint32_t batch=GetBlockIdx();
        DataCopyPad(aa,ag[uint64_t(batch)*N*N],pa,pad);
        DataCopyPad(bvalues,bg[uint64_t(batch)*N],pb,pad);
        auto positions=coef.ReinterpretCast<int32_t>();
        for(uint32_t i=0;i<D;++i)positions.SetValue(i,int32_t(i));
        Fence<HardEvent::S_V>();
        auto ci=colIndex.ReinterpretCast<int32_t>();
        Muls(ci,positions,int32_t(RP*4),D);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process(){
        const uint32_t batch=GetBlockIdx();
        Fence<HardEvent::MTE2_V>();
        Brcb(broad,bvalues,D/8,{1,8});PipeBarrier<PIPE_V>();
        // Fill the entire 8-lane B/padding block before any row operation reads it.
        Muls(mat[D],broad,1.0f,uint64_t(8),uint8_t(N),{1,1,RP/8,1});
            Muls(mat, aa, 1.0f, uint64_t(D), uint8_t(N), {1, 1, RP / 8, D / 8});
            PipeBarrier<PIPE_V>();
            GaussJordan();
            Fence<HardEvent::S_V>();
            AscendC::Gather(x,mat,resultIndex,D*4,N);PipeBarrier<PIPE_V>();
            Muls(x,x,-1.0f,D);PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_MTE3>();
            DataCopyExtParams out{1, N * 4, 0, 0, 0};
            DataCopyPad(xg[uint64_t(batch) * N], x, out);
            Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
    }
private:
    __aicore__ inline void GaussJordan(){
        uint64_t active=0xffff;
        for(uint32_t j=0;j<N;++j){
            AscendC::Gather(coef,mat,colIndex,j*4,N);PipeBarrier<PIPE_V>();
            uint32_t pivot=0;
            float diagonal=0.0f;
            bool search=j<(3*N)/8;
            if(!search){
                pivot=uint32_t(GetSFFValue<1>(active));
                Fence<HardEvent::V_S>();
                diagonal=coef.GetValue(pivot);
                search=(diagonal==0.0f);
            }
            if(search){
                Abs(ab,coef,D);PipeBarrier<PIPE_V>();
                uint64_t available[2]={active,0};
                WholeReduceMax(maximum,ab,available,uint8_t(1),1,1,8);
                Fence<HardEvent::V_S>();
                uint32_t bits;GetReduceRepeatMaxMinSpr<float>(bits,pivot);
                diagonal=coef.GetValue(pivot);
            }
            resultIndex.SetValue(j,pivot*RP*4);
            uint64_t chosen[2]={uint64_t(1)<<pivot,0};
            Muls(row,mat[pivot*RP],-1.0f/diagonal,RP);
            Duplicate(coef,0.0f,chosen,uint8_t(1),1,D/8);PipeBarrier<PIPE_V>();
            DataCopy(mat[pivot*RP],row,RP);
            Brcb(broad,coef,D/8,{1,8});
            MulAddDst(mat,broad,row,uint64_t(RP),uint8_t(N),{1,0,1,RP/8,1,0});PipeBarrier<PIPE_V>();
            active&=~(uint64_t(1)<<pivot);
        }
    }
    __aicore__ inline void Residual() {
        AscendC::Gather(xt, x, tileIndex, 0, ND); PipeBarrier<PIPE_V>();
        Muls(xh,xt,4097.0f,ND);Muls(temp,aa,4097.0f,ND);Mul(p,aa,xt,ND);PipeBarrier<PIPE_V>();
        Sub(xl,xh,xt,ND);Sub(al,temp,aa,ND);PipeBarrier<PIPE_V>();
        Sub(xh,xh,xl,ND);Sub(ah,temp,al,ND);PipeBarrier<PIPE_V>();
        Sub(xl,xt,xh,ND);Sub(al,aa,ah,ND);PipeBarrier<PIPE_V>();
        Mul(e,ah,xh,ND);Mul(xt,al,xh,ND);Mul(z,ah,xl,ND);Mul(low,al,xl,ND);PipeBarrier<PIPE_V>();
        Sub(e,p,e,ND);PipeBarrier<PIPE_V>();Sub(e,e,xt,ND);PipeBarrier<PIPE_V>();
        Sub(e,e,z,ND);PipeBarrier<PIPE_V>();Sub(e,low,e,ND);PipeBarrier<PIPE_V>();
        auto sums=p;auto errors=e;uint32_t count=ND/2;
        for(uint32_t level=0;level<4;++level){
            AscendC::Gather(h,sums,shiftIndex,0,count);AscendC::Gather(xh,sums,shiftIndex,4,count);
            AscendC::Gather(hl,errors,shiftIndex,0,count);AscendC::Gather(xl,errors,shiftIndex,4,count);PipeBarrier<PIPE_V>();
            Add(s,h,xh,count);Add(low,hl,xl,count);PipeBarrier<PIPE_V>();
            Sub(z,s,h,count);PipeBarrier<PIPE_V>();
            Sub(temp,s,z,count);Sub(ah,xh,z,count);PipeBarrier<PIPE_V>();
            Sub(temp,h,temp,count);PipeBarrier<PIPE_V>();Add(temp,temp,ah,count);PipeBarrier<PIPE_V>();
            Add(low,low,temp,count);PipeBarrier<PIPE_V>();sums=s;errors=low;count/=2;
        }
        DataCopy(coef,s,D);DataCopy(ab,low,D);PipeBarrier<PIPE_V>();
        Sub(r, bvalues, coef, D); PipeBarrier<PIPE_V>();
        Sub(z, bvalues, r, D); PipeBarrier<PIPE_V>();
        Add(temp, r, z, D); PipeBarrier<PIPE_V>();
        Sub(temp, bvalues, temp, D); PipeBarrier<PIPE_V>();
        Sub(z, z, coef, D); PipeBarrier<PIPE_V>();
        Add(temp, temp, z, D); PipeBarrier<PIPE_V>();
        Sub(temp, temp, ab, D); PipeBarrier<PIPE_V>();
        Add(r, r, temp, D); PipeBarrier<PIPE_V>();
    }

    GlobalTensor<float> ag, bg, xg;
    LocalTensor<float> mem, mat, aa, p, e, s, z, temp, low, h, hl, ah, al, xt, xh, xl, inverse;
    LocalTensor<float> x, r, bvalues, delta, coef, ab, maximum, row, broad;
    LocalTensor<uint32_t> colIndex, resultIndex, tileIndex, inverseIndex, shiftIndex;
};



template <uint32_t N> class JointNarrowSolve {
    static constexpr uint32_t D=64, C=56, RP=72, W=64, NS=N*8;
    static constexpr uint32_t CO=N*RP, AB=CO+D, MM=AB+D, RW=MM+8;
    static constexpr uint32_t BC=RW+RP, XX=BC+D*8, TOTAL=XX+NS;
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR out){
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        mem=LocalTensor<float>(TPosition::VECCALC,0,TOTAL);
        mat=mem;coef=mem[CO];ab=mem[AB];maximum=mem[MM];row=mem[RW];broad=mem[BC];packed=mem[XX];
        ci=LocalTensor<uint32_t>(TPosition::VECCALC,TOTAL*4,D);
        ri=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+D)*4,D);
        oi=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+2*D)*4,NS);
        cols=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+2*D+NS)*4,8);
        const uint32_t batch=GetBlockIdx();
        DataCopyPadExtParams<float> pad{true,0,0,0};
        DataCopyExtParams pa{N,N*4,0,(RP-(N+7)/8*8)/8,0};
        DataCopyExtParams pb{N,8,0,(RP-8)/8,0};
        DataCopyPad(mat,ag[uint64_t(batch)*N*N],pa,pad);
        DataCopyPad(mat[C],bg[uint64_t(batch)*N*2],pb,pad);
        auto positions=coef.ReinterpretCast<int32_t>();
        for(uint32_t q=0;q<D;++q)positions.SetValue(q,int32_t(q));
        Fence<HardEvent::S_V>();
        Muls(ci.ReinterpretCast<int32_t>(),positions,int32_t(RP*4),D);
        Muls(cols.ReinterpretCast<int32_t>(),positions,int32_t(4),8);
        Duplicate(ri.ReinterpretCast<int32_t>(),int32_t(0),D);PipeBarrier<PIPE_V>();
        Adds(cols.ReinterpretCast<int32_t>(),cols.ReinterpretCast<int32_t>(),int32_t(C*4),8);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process(){
        Fence<HardEvent::MTE2_V>();
        uint64_t active=(uint64_t(1)<<N)-1;
        for(uint32_t j=0;j<N;++j){
            AscendC::Gather(coef,mat,ci,j*4,N);PipeBarrier<PIPE_V>();
            Abs(ab,coef,D);PipeBarrier<PIPE_V>();
            uint64_t available[2]={active,0};
            WholeReduceMax(maximum,ab,available,uint8_t(1),1,1,8);
            Fence<HardEvent::V_S>();
            uint32_t bits,pivot;GetReduceRepeatMaxMinSpr<float>(bits,pivot);
            const float diagonal=coef.GetValue(pivot);
            ri.SetValue(j,pivot*RP*4);
            uint64_t chosen[2]={uint64_t(1)<<pivot,0};
            Muls(row,mat[pivot*RP],-1.0f/diagonal,W);
            Duplicate(coef,0.0f,chosen,uint8_t(1),1,D/8);PipeBarrier<PIPE_V>();
            DataCopy(mat[pivot*RP],row,W);
            Brcb(broad,coef,D/8,{1,8});PipeBarrier<PIPE_V>();
            MulAddDst(mat,broad,row,uint64_t(W),uint8_t(N),{1,0,1,RP/8,1,0});PipeBarrier<PIPE_V>();
            active&=~(uint64_t(1)<<pivot);
        }
        Fence<HardEvent::S_V>();
        Brcb(broad,ri.ReinterpretCast<float>(),D/8,{1,8});PipeBarrier<PIPE_V>();
        Add(oi.ReinterpretCast<int32_t>(),broad.ReinterpretCast<int32_t>(),cols.ReinterpretCast<int32_t>(),uint64_t(8),uint8_t(N),{1,1,1,1,1,0});PipeBarrier<PIPE_V>();
        AscendC::Gather(packed,mat,oi,0,NS);PipeBarrier<PIPE_V>();
        Muls(packed,packed,-1.0f,NS);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams storeParams{N,8,0,0,0};
        DataCopyPad(xg[uint64_t(GetBlockIdx())*N*2],packed,storeParams);
        Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
    }
private:
    GlobalTensor<float> ag,bg,xg;
    LocalTensor<float> mem,mat,coef,ab,maximum,row,broad,packed;
    LocalTensor<uint32_t> ci,ri,oi,cols;
};

// Lanen256_: matched-row half Jacobi preconditioner with FP32 residuals.
// E has an odd17-block row stride; the original input remains in GM for residuals.
class Lanen256_MatchedJacobi {
    static constexpr uint32_t N=256,RP=272,T=32,RT=8;
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,const LinalgSolveTilingData &data){
        t=data;matrix=GetBlockIdx()/8;c=(GetBlockIdx()/2)%4;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
        uint32_t coords=matrix,ai=0,bi=0;
        for(int d=int(t.batchRank)-1;d>=0;--d){uint32_t q=coords%t.batchDims[d];coords/=t.batchDims[d];ai+=q*t.aStrides[d];bi+=q*t.bStrides[d];}
        ao=uint64_t(ai)*N*N;bo=uint64_t(bi)*N*t.k;xo=uint64_t(matrix)*N*t.k;
        LocalTensor<half> h(TPosition::VECCALC,0,81920);hm=h;hp=h[N*RP];hx=hp[T*N];hn=hx[N];hb=hn[N];hs0=hb[N];hs1=hs0[T];
        LocalTensor<float> f(TPosition::VECCALC,160*1024,8192);aa=f;pp=aa[RT*N];xv=pp[RT*N];rr=xv[N];bv=rr[N];diag=bv[N];v0=diag[N];v1=v0[N];maximum=v1[N];
        perm=f[5648].ReinterpretCast<uint32_t>();index=f[5904].ReinterpretCast<uint32_t>();s0=f[6160];s1=f[6224];
    }
    __aicore__ inline bool Process(){
        if(!Prepare())return false;
        if(!Apply(bv))return false;DataCopy(xv,v0,N);PipeBarrier<PIPE_V>();
        for(uint32_t ir=0;ir<1;++ir){Residual();if(!Apply(rr))return false;Add(xv,xv,v0,N);PipeBarrier<PIPE_V>();}
        Residual();
        Abs(v1,rr,N);PipeBarrier<PIPE_V>();const float rmax=Maximum(v1);
        if(!(rmax>=0.0f && rmax<=5.0e-5f*(bmax+1.0e-20f)))return false;
        Abs(v1,xv,N);PipeBarrier<PIPE_V>();WholeReduceSum(s0,v1,int32_t(64),uint8_t(4),1,1,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
        for(uint32_t i=0;i<4;++i){const float sum=s0.GetValue(i);if(!(sum>=0.0f && sum<1.0e30f))return false;}
        Brcb(aa,xv,N/8,{1,8});PipeBarrier<PIPE_V>();Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();
        DataCopyExtParams store{N,4,0,(t.k-1)*4,0};DataCopyPad(xg[xo+c],aa,store);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
        return true;
    }
private:
    __aicore__ inline float Maximum(LocalTensor<float> value){
        WholeReduceMax(maximum,value,int32_t(64),uint8_t(4),1,1,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();uint32_t bits,which;GetReduceRepeatMaxMinSpr<float>(bits,which);union{uint32_t u;float f;}m;m.u=bits;return m.f;
    }
    __aicore__ inline bool Prepare(){
        constexpr uint32_t Rows=16;
        uint32_t seen[8]={};uint32_t columns[Rows];float factors[Rows];
        auto absRows=hp.ReinterpretCast<float>();auto pairs=s1[64];
        for(uint32_t first=0;first<N;first+=Rows){
            Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();DataCopy(aa,ag[ao+uint64_t(first)*N],Rows*N);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
            Abs(absRows,aa,Rows*N);PipeBarrier<PIPE_V>();
            for(uint32_t r=0;r<Rows;++r)WholeReduceMax(pairs[r*8],absRows[r*N],int32_t(64),uint8_t(4),1,1,8);
            PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
            for(uint32_t r=0;r<Rows;++r){
                float peak=-1.0f;uint32_t column=0;
                for(uint32_t q=0;q<4;++q){const float value=pairs.GetValue(r*8+q*2);const uint32_t chosen=pairs.ReinterpretCast<uint32_t>().GetValue(r*8+q*2+1)%64+q*64;if(value>peak){peak=value;column=chosen;}}
                if(!(peak>1.0e-30f && peak<1.0e30f) || (seen[column/32]&(1U<<(column%32))))return false;
                seen[column/32]|=1U<<(column%32);const float inv=1.0f/aa.GetValue(r*N+column);diag.SetValue(column,inv);perm.SetValue(column,(first+r)*4);columns[r]=column;factors[r]=inv;
            }
            Fence<HardEvent::S_V>();
            for(uint32_t r=0;r<Rows;++r)Muls(aa[r*N],aa[r*N],-factors[r],N);
            PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
            for(uint32_t r=0;r<Rows;++r)aa.SetValue(r*N+columns[r],0.0f);
            Fence<HardEvent::S_V>();
            for(uint32_t r=0;r<Rows;++r){Cast(hm[columns[r]*RP],aa[r*N],RoundMode::CAST_RINT,N);Duplicate(hm[columns[r]*RP+N],half(0.0f),RP-N);}
            PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();DataCopyExtParams load{N,4,(t.k-1)*4,0,0};DataCopyPadExtParams<float>pad{false,0,0,0};DataCopyPad(aa,bg[bo+c],load,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        for(uint32_t i=0;i<N;++i)index.SetValue(i,i*8*4);Fence<HardEvent::S_V>();AscendC::Gather(bv,aa,index,0,N);PipeBarrier<PIPE_V>();Abs(v1,bv,N);PipeBarrier<PIPE_V>();bmax=Maximum(v1);
        return bmax>=0.0f && bmax<1.0e30f;
    }
    __aicore__ inline bool Apply(LocalTensor<float> source){
        AscendC::Gather(v0,source,perm,0,N);PipeBarrier<PIPE_V>();Mul(v0,v0,diag,N);PipeBarrier<PIPE_V>();Abs(v1,v0,N);PipeBarrier<PIPE_V>();const float peak=Maximum(v1);
        if(peak==0.0f){Duplicate(v0,0.0f,N);PipeBarrier<PIPE_V>();return true;}
        if(!(peak>1.0e-30f && peak<1.0e30f))return false;
        union{uint32_t u;float f;}value,scale;value.f=peak;const uint32_t exponent=(value.u>>23)&255U;scale.u=(254U-exponent)<<23;const float unscale=1.0f/scale.f;
        Muls(v0,v0,scale.f,N);PipeBarrier<PIPE_V>();Cast(hb,v0,RoundMode::CAST_RINT,N);Duplicate(hx,half(0.0f),N);PipeBarrier<PIPE_V>();
        for(uint32_t iteration=0;iteration<8;++iteration){
            for(uint32_t first=0;first<N;first+=T){
                Mul(hp,hm[first*RP],hx,uint64_t(128),uint8_t(T),{1,1,1,N/16,RP/16,0});
                Mul(hp[128],hm[first*RP+128],hx[128],uint64_t(128),uint8_t(T),{1,1,1,N/16,RP/16,0});PipeBarrier<PIPE_V>();
                WholeReduceSum(hs0,hp,int32_t(128),uint8_t(T),1,1,N/16);WholeReduceSum(hs1,hp[128],int32_t(128),uint8_t(T),1,1,N/16);PipeBarrier<PIPE_V>();
                Add(hn[first],hs0,hs1,T);PipeBarrier<PIPE_V>();Add(hn[first],hn[first],hb[first],T);PipeBarrier<PIPE_V>();
            }
            DataCopy(hx,hn,N);PipeBarrier<PIPE_V>();
        }
        Cast(v0,hx,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();Muls(v0,v0,unscale,N);PipeBarrier<PIPE_V>();return true;
    }
    __aicore__ inline void Residual(){
        for(uint32_t first=0;first<N;first+=RT){
            Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();DataCopy(aa,ag[ao+uint64_t(first)*N],RT*N);Fence<HardEvent::MTE2_V>();
            for(uint32_t off=0;off<N;off+=64){Mul(pp[off],aa[off],xv[off],uint64_t(64),uint8_t(RT),{1,1,1,N/8,N/8,0});}PipeBarrier<PIPE_V>();
            WholeReduceSum(s0,pp,int32_t(64),uint8_t(RT),1,1,N/8);PipeBarrier<PIPE_V>();
            for(uint32_t off=64;off<N;off+=64){WholeReduceSum(s1,pp[off],int32_t(64),uint8_t(RT),1,1,N/8);PipeBarrier<PIPE_V>();Add(s0,s0,s1,RT);PipeBarrier<PIPE_V>();}
            Sub(rr[first],bv[first],s0,RT);PipeBarrier<PIPE_V>();
        }
    }
    LinalgSolveTilingData t;GlobalTensor<float>ag,bg,xg;LocalTensor<half>hm,hp,hx,hn,hb,hs0,hs1;
    LocalTensor<float>aa,pp,xv,rr,bv,diag,v0,v1,maximum,s0,s1;LocalTensor<uint32_t>perm,index;
    uint32_t matrix,c;uint64_t ao,bo,xo;float bmax;
};

// All AIVs make one common decision before the original MIX fallback runs.
// Status occupies the dead identity scratch and is overwritten by fallback init.
__aicore__ inline bool Lanen256_TryMatchedJacobi(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR workspace,const LinalgSolveTilingData &data){
    if(data.blocks>32)return false;
    GlobalTensor<uint32_t> status;status.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(GetUserWorkspace(workspace))+data.schurOffset/4+kOS_Identity);
    // The last (quad3, odd) private slot has no Solver or PrivateFactor writer.
    GlobalTensor<uint32_t> decision;decision.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(GetUserWorkspace(workspace))+uint64_t(data.blocks-1)*256*(data.nPad+2*data.kPad));
    if ASCEND_IS_AIC{
        CrossCoreWaitFlag<2>(7);DataCacheCleanAndInvalid<uint32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(decision);return decision.GetValue(0)!=0;
    }else{
        TPipe pipe;bool ok=true;
        if(!(GetBlockIdx()&1) && (GetBlockIdx()/2)%4<data.k){Lanen256_MatchedJacobi op;op.Init(A,B,X,data);ok=op.Process();}
        LocalTensor<uint32_t>record(TPosition::VECCALC,184*1024,264);Duplicate(record,0U,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();record.SetValue(0,ok?1:0);Fence<HardEvent::S_MTE3>();DataCopy(status[GetBlockIdx()*8],record,8);Fence<HardEvent::MTE3_S>();SyncAll<true>();
        if(GetBlockIdx()==0){
            Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopy(record,status,data.blocks*8);Fence<HardEvent::MTE2_S>();
            bool all=true;for(uint32_t i=0;i<data.blocks;++i)all=all && record.GetValue(i*8)!=0;
            record.SetValue(0,all?1:0);Fence<HardEvent::S_MTE3>();DataCopy(decision,record,8);Fence<HardEvent::MTE3_S>();
        }
        SyncAll<true>();Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopy(record,decision,8);Fence<HardEvent::MTE2_S>();const bool all=record.GetValue(0)!=0;CrossCoreSetFlag<2,PIPE_MTE3>(7);return all;
    }
}

// Lanen15_ direct augmented solve for matching-batch N15 K1 matrices.
// Rows are [15 A values, zero padding, 1 B value, initialized B padding].
// Six complete pivot searches precede first-active-row selection; exact zero
// pivots still trigger a complete search of the remaining rows.
class Lanen15_NarrowSolve {
    static constexpr uint32_t N=15,D=16,RP=24;
    static constexpr uint32_t BV=N*RP,CF=BV+D,AB=CF+D,MX=AB+D;
    static constexpr uint32_t RW=MX+8,BC=RW+RP,XX=BC+D*8,TOTAL=XX+D;
public:
    __aicore__ inline void Init(GM_ADDR aAddr,GM_ADDR bAddr,GM_ADDR xAddr){
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aAddr));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bAddr));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(xAddr));
        mem=LocalTensor<float>(TPosition::VECCALC,0,TOTAL);
        mat=mem; bvalues=mem[BV];coef=mem[CF];ab=mem[AB];maximum=mem[MX];
        row=mem[RW];broad=mem[BC];x=mem[XX];
        colIndex=LocalTensor<uint32_t>(TPosition::VECCALC,TOTAL*4,D);
        resultIndex=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+D)*4,D);
        const uint32_t batch=GetBlockIdx();
        DataCopyPadExtParams<float> pad{true,0,1,0};
        DataCopyExtParams pa{N,N*4,0,(RP-D)/8,0},pb{1,N*4,0,0,0};
        DataCopyPad(mat,ag[uint64_t(batch)*N*N],pa,pad);
        DataCopyPad(bvalues,bg[uint64_t(batch)*N],pb,pad);
        auto packedIndex=colIndex.ReinterpretCast<uint64_t>();
        for(uint32_t i=0;i<D/2;++i){
            const uint64_t low=uint64_t(2*i*RP*4);
            const uint64_t high=uint64_t((2*i+1)*RP*4);
            packedIndex.SetValue(i,low|(high<<32));
        }
        coef.SetValue(N,0.0f);
        Fence<HardEvent::S_V>();
    }
    __aicore__ inline void Process(){
        Fence<HardEvent::MTE2_V>();
        Brcb(broad,bvalues,D/8,{1,8});PipeBarrier<PIPE_V>();
        Muls(mat[D],broad,-1.0f,uint64_t(8),uint8_t(N),{1,1,RP/8,1});
        PipeBarrier<PIPE_V>();
        Solve();
        Fence<HardEvent::S_V>();
        AscendC::Gather(x,mat,resultIndex,D*4,N);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams out{1,N*4,0,0,0};
        DataCopyPad(xg[uint64_t(GetBlockIdx())*N],x,out);
        Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
    }
private:
    __aicore__ inline void Solve(){
        uint64_t active=(uint64_t(1)<<N)-1;
        for(uint32_t j=0;j<N;++j){
            AscendC::Gather(coef,mat,colIndex,j*4,N);PipeBarrier<PIPE_V>();
            uint32_t pivot=0;float diagonal=0.0f;
            bool search=j<6;
            if(!search){
                pivot=uint32_t(GetSFFValue<1>(active));
                Fence<HardEvent::V_S>();
                diagonal=coef.GetValue(pivot);search=(diagonal==0.0f);
            }
            if(search){
                Abs(ab,coef,N);PipeBarrier<PIPE_V>();
                uint64_t available[2]={active,0};
                WholeReduceMax(maximum,ab,available,uint8_t(1),1,1,8);
                Fence<HardEvent::V_S>();
                uint32_t bits;GetReduceRepeatMaxMinSpr<float>(bits,pivot);
                diagonal=coef.GetValue(pivot);
            }
            resultIndex.SetValue(j,pivot*RP*4);
            uint64_t chosen[2]={uint64_t(1)<<pivot,0};
            Muls(row,mat[pivot*RP],-1.0f/diagonal,RP);
            Duplicate(coef,0.0f,chosen,uint8_t(1),1,D/8);PipeBarrier<PIPE_V>();
            DataCopy(mat[pivot*RP],row,RP);
            Brcb(broad,coef,D/8,{1,8});
            MulAddDst(mat,broad,row,uint64_t(RP),uint8_t(N),{1,0,1,RP/8,1,0});
            PipeBarrier<PIPE_V>();
            active&=~(uint64_t(1)<<pivot);
        }
    }
    GlobalTensor<float> ag,bg,xg;
    LocalTensor<float> mem,mat,bvalues,coef,ab,maximum,row,broad,x;
    LocalTensor<uint32_t> colIndex,resultIndex;
};

// BEGIN Lanen64 joint RHS direct solve.
class Lanen64_JointNarrow64 {
    static constexpr uint32_t N=64, CP=72, COLS=66, AL=72;
    static constexpr uint32_t AB=AL*CP, MM=AB+N, RW=MM+8;
    static constexpr uint32_t BC=RW+AL, XX=BC+AL*8, TOTAL=XX+2*N;
    static constexpr uint32_t CI=TOTAL, OI=CI+AL, BI=OI+2*N;
    static constexpr uint32_t AA=BI+N, BB=AA+N*N, END=BB+2*N;
    static_assert(END*4<=196608,"Lanen64 column-major live UB");
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR out){
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        mem=LocalTensor<float>(TPosition::VECCALC,0,TOTAL);
        mat=mem;ab=mem[AB];maximum=mem[MM];row=mem[RW];broad=mem[BC];packed=mem[XX];
        ci=LocalTensor<uint32_t>(TPosition::VECCALC,CI*4,AL);
        oi=LocalTensor<uint32_t>(TPosition::VECCALC,OI*4,2*N);
        bi=LocalTensor<uint32_t>(TPosition::VECCALC,BI*4,N);
        aa=LocalTensor<float>(TPosition::VECCALC,AA*4,N*N);
        bb=LocalTensor<float>(TPosition::VECCALC,BB*4,2*N);
        const uint32_t batch=GetBlockIdx();
        DataCopyPadExtParams<float> pad{true,0,0,0};
        // A and B each enter UB as one contiguous transfer; expansion has no GM accesses.
        DataCopyExtParams pa{1,N*N*4,0,0,0};
        DataCopyExtParams pb{1,2*N*4,0,0,0};
        DataCopyPad(aa,ag[uint64_t(batch)*N*N],pa,pad);
        DataCopyPad(bb,bg[uint64_t(batch)*N*2],pb,pad);
        auto positions=row.ReinterpretCast<int32_t>();
        for(uint32_t q=0;q<AL;++q)positions.SetValue(q,int32_t(q));
        Fence<HardEvent::S_V>();
        Muls(ci.ReinterpretCast<int32_t>(),positions,int32_t(CP*4),AL);
        Muls(bi.ReinterpretCast<int32_t>(),positions,int32_t(8),N);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process(){
        Fence<HardEvent::MTE2_V>();
        Duplicate(mat,0.0f,AL*CP);PipeBarrier<PIPE_V>();
        TransDataTo5HDParams transpose{false,false,8,CP,1};
        for(uint32_t block=0;block<N/16;++block){
            uint64_t src[16],dst[16];
            for(uint32_t r=0;r<16;++r)
                src[r]=reinterpret_cast<uint64_t>(aa[(block*16+r)*N].GetPhyAddr());
            for(uint32_t c=0;c<8;++c)
                for(uint32_t half=0;half<2;++half)
                    dst[2*c+half]=reinterpret_cast<uint64_t>(mat[c*CP+block*16+half*8].GetPhyAddr());
            TransDataTo5HD<float>(dst,src,transpose);
        }
        AscendC::Gather(mat[N*CP],bb,bi,0,N);
        AscendC::Gather(mat[(N+1)*CP],bb,bi,4,N);PipeBarrier<PIPE_V>();
        // Original AA is dead after all four transposes. Keep every other
        // live buffer at its original address; reuse two disjoint 128-word
        // intervals for padded indices and row data.
        auto Lanen200_paddedCI=aa.ReinterpretCast<uint32_t>();
        DataCopy(Lanen200_paddedCI,ci,AL);
        Duplicate(Lanen200_paddedCI[AL].ReinterpretCast<int32_t>(),int32_t(0),128-AL);PipeBarrier<PIPE_V>();
        ci=Lanen200_paddedCI;row=aa[128];
        uint64_t active=~uint64_t(0);
        for(uint32_t first=0;first<N;first+=8){
            const uint32_t width=AL-first;
            uint64_t Lanen200_rowMask[2]={width>=64?~uint64_t(0):(uint64_t(1)<<width)-1,0};
            const uint8_t Lanen200_rowRepeats=first==0?2:1;
            for(uint32_t q=0;q<8;++q)Pivot(first+q,first,width,active,Lanen200_rowMask,Lanen200_rowRepeats);
        }
        Fence<HardEvent::S_V>();
        AscendC::Gather(packed,mat,oi,0,2*N);PipeBarrier<PIPE_V>();
        Muls(packed,packed,-1.0f,2*N);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams storeParams{1,2*N*4,0,0,0};
        DataCopyPad(xg[uint64_t(GetBlockIdx())*N*2],packed,storeParams);
        Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
    }
private:
    __aicore__ inline void Pivot(uint32_t j,uint32_t first,uint32_t width,uint64_t &active,uint64_t *Lanen200_rowMask,uint8_t Lanen200_rowRepeats){
            auto coef=mat[j*CP];
            // The pivot rule is signed maximum, not magnitude maximum.
            // If every active entry is nonpositive, select the most negative.
            // SPR contains the original signed float; no dependent UB read or
            // quantized sign encoding is needed.
            uint64_t available[2]={active,0};
            WholeReduceMax(maximum,coef,available,uint8_t(1),1,1,8);
            Fence<HardEvent::V_S>();
            uint32_t bits,pivot;GetReduceRepeatMaxMinSpr<float>(bits,pivot);
            union Lanen200_SignedPivot {uint32_t u;float f;} Lanen200_value;
            Lanen200_value.u=bits;
            if(Lanen200_value.f<=0.0f && j+1<N){
                WholeReduceMin(maximum,coef,available,uint8_t(1),1,1,8);
                Fence<HardEvent::V_S>();
                GetReduceRepeatMaxMinSpr<float>(bits,pivot);Lanen200_value.u=bits;
            }
            const float diagonal=Lanen200_value.f;
            oi.SetValue(2*j,(N*CP+pivot)*4);
            oi.SetValue(2*j+1,((N+1)*CP+pivot)*4);
            uint64_t chosen[2]={uint64_t(1)<<pivot,0};
            AscendC::Gather(row,mat,ci[first],pivot*4,Lanen200_rowMask,Lanen200_rowRepeats,uint16_t(8));PipeBarrier<PIPE_V>();
            Muls(row,row,-1.0f/diagonal,Lanen200_rowMask,Lanen200_rowRepeats,{1,1,8,8});
            // Unify selected-row normalization with the ordinary outer update.
            // The selected coefficient becomes d+1; only its result has changed rounding.
            Adds(coef,coef,1.0f,chosen,uint8_t(1),{1,1,N/8,N/8});PipeBarrier<PIPE_V>();
            Brcb(broad,row,width/8,{1,8});PipeBarrier<PIPE_V>();
            const uint32_t count=COLS-j-1,begin=j+1;
            MulAddDst(mat[begin*CP],coef,broad[(begin-first)*8],uint64_t(N),uint8_t(count),{1,1,0,CP/8,0,1});
            PipeBarrier<PIPE_V>();
            active&=~(uint64_t(1)<<pivot);
    }
    GlobalTensor<float> ag,bg,xg;
    LocalTensor<float> mem,mat,ab,maximum,row,broad,packed,aa,bb;
    LocalTensor<uint32_t> ci,oi,bi;
};
// END Lanen64 joint RHS direct solve.

// Lanen200 ownership: one standard LU per system, four refined RHS workers.
class Lanen200_SharedFactor {
    static constexpr uint32_t N=200,D=256,RP=208,ELEMENTS=N*216;
    static constexpr uint32_t CO=N*RP,AB=CO+D,MM=AB+D,RW=MM+8;
    static constexpr uint32_t BC=RW+RP,CI=BC+D*8,PI=CI+D,TOTAL=PI+D;
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR workspace){
        static_assert(TOTAL*4==179552,"Lanen200 factor phase UB");
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        wg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace)));
        mem=LocalTensor<float>(TPosition::VECCALC,0,TOTAL);
        mat=mem;coef=mem[CO];ab=mem[AB];maximum=mem[MM];row=mem[RW];broad=mem[BC];
        ci=mem[CI].ReinterpretCast<uint32_t>();piv=mem[PI].ReinterpretCast<int32_t>();
        for(uint32_t q=0;q<D;++q)ci.SetValue(q,q*RP*4);
        Fence<HardEvent::S_V>();Duplicate(coef,0.0f,D);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
        const uint32_t batch=GetBlockIdx()/4;
        DataCopyPadExtParams<float> pad{true,0,0,0};
        DataCopyExtParams pa{N,N*4,0,1,0};
        DataCopyExtParams pb{N,16,0,(RP-8)/8,0};
        DataCopyPad(mat,ag[uint64_t(batch)*N*N],pa,pad);
        DataCopyPad(mat[N],bg[uint64_t(batch)*N*4],pb,pad);
    }
    __aicore__ inline void Process(){
        Fence<HardEvent::MTE2_V>();
        for(uint32_t j=0;j<N;++j){
            AscendC::Gather(coef,mat,ci,j*4,N);PipeBarrier<PIPE_V>();
            Abs(ab,coef,D);PipeBarrier<PIPE_V>();
            if(j){Duplicate(ab,0.0f,j);PipeBarrier<PIPE_V>();}
            WholeReduceMax(maximum,ab,int32_t(64),uint8_t(4),1,1,8);
            Fence<HardEvent::V_S>();
            uint32_t bits,pivot;GetReduceRepeatMaxMinSpr<float>(bits,pivot);
            const float diagonal=coef.GetValue(pivot);
            piv.SetValue(j,int32_t(pivot));
            if(pivot!=j)coef.SetValue(pivot,coef.GetValue(j));
            Fence<HardEvent::S_V>();
            if(pivot!=j){
                DataCopy(row,mat[j*RP],RP);PipeBarrier<PIPE_V>();
                DataCopy(mat[j*RP],mat[pivot*RP],RP);PipeBarrier<PIPE_V>();
                DataCopy(mat[pivot*RP],row,RP);PipeBarrier<PIPE_V>();
            }
            Muls(coef,coef,-1.0f/diagonal,N);PipeBarrier<PIPE_V>();
            Brcb(broad,coef,D/8,{1,8});PipeBarrier<PIPE_V>();
            if(j+1<N){
                const uint32_t start=j/8*8;
                const uint32_t width=RP-start<64?RP-start:64;
                const uint64_t upto=width==64?~0ULL:(1ULL<<width)-1;
                uint64_t tailMask[2]={upto&(~0ULL<<(j&7)),0};
                MulAddDst(mat[(j+1)*RP+start],broad[(j+1)*8],mat[j*RP+start],tailMask,uint8_t(N-j-1),{1,0,1,RP/8,1,0});
                for(uint32_t c=start+64;c<RP;c+=64)
                    MulAddDst(mat[(j+1)*RP+c],broad[(j+1)*8],mat[j*RP+c],uint64_t(RP-c<64?RP-c:64),uint8_t(N-j-1),{1,0,1,RP/8,1,0});
                PipeBarrier<PIPE_V>();
                // Retain standard L below the diagonal. Later swaps move it too.
                const uint32_t rows=N-j-1,groups=rows/8,tail=rows%8;
                const uint64_t packedMask=0x0101010101010101ULL<<(j&7);
                UnaryRepeatParams packedStride{RP/8,1,RP,8};
                if(groups){
                    uint64_t columnMask[2]={packedMask,0};
                    Muls(mat[(j+1)*RP+start],broad[(j+1)*8],-1.0f,columnMask,uint8_t(groups),packedStride);
                }
                if(tail){
                    const uint32_t first=j+1+groups*8;
                    uint64_t columnMask[2]={packedMask&((1ULL<<(tail*8))-1),0};
                    Muls(mat[first*RP+start],broad[first*8],-1.0f,columnMask,uint8_t(1),packedStride);
                }
                PipeBarrier<PIPE_V>();
            }
        }
        // Four workspaces receive identical LU and original pivot sequence.
        Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();
        for(uint32_t lane=0;lane<4;++lane){
            const uint64_t offset=uint64_t(GetBlockIdx()+lane)*ELEMENTS;
            DataCopy(wg[offset],mat,N*RP);
            DataCopy(wg[offset+N*RP],piv.ReinterpretCast<float>(),N);
        }
        Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
    }
private:
    GlobalTensor<float>ag,bg,wg;
    LocalTensor<float>mem,mat,coef,ab,maximum,row,broad;
    LocalTensor<uint32_t>ci;LocalTensor<int32_t>piv;
};
// Lanen200 ownership: shared Cube blocked LU, MODE15 ownerpair per four workers.
class Lanen200_SharedCubeFactor {
    static constexpr uint32_t N=200, RP=208, ELEMENTS=N*216;
    static constexpr uint32_t ROW=N*RP, OTHER=ROW+RP, COEF=OTHER+RP;
    static constexpr uint32_t ABS=COEF+N, MAX=ABS+256, COL=MAX+8;
    static constexpr uint32_t IDX=COL+8*N, PIV=IDX+N, TEMP=PIV+N;
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR workspace,const LinalgSolveTilingData &data) {
        static_assert((TEMP+16*RP)*4<=192*1024,"N200 padded argmax fits UB withoutSelect");
        t=data;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a)); bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        auto base=reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace));
        wg.SetGlobalBuffer(base+uint64_t(GetBlockIdx())*ELEMENTS);
        product.SetGlobalBuffer(base+uint64_t(t.blocks)*ELEMENTS+uint64_t(GetBlockIdx())*N*RP);
        if ASCEND_IS_AIV {
            mem=LocalTensor<float>(TPosition::VECCALC,0,TEMP+16*RP); mat=mem; row=mem[ROW]; other=mem[OTHER];
            coef=mem[COEF]; ab=mem[ABS]; maximum=mem[MAX]; column=mem[COL]; temp=mem[TEMP];
            index=mem[IDX].ReinterpretCast<uint32_t>(); piv=mem[PIV].ReinterpretCast<int32_t>();
            for(uint32_t i=0;i<N;++i) index.SetValue(i,i*RP*4);
            Fence<HardEvent::S_V>(); Duplicate(mat,0.0f,N*RP);Duplicate(ab,0.0f,256); PipeBarrier<PIPE_V>();
            uint32_t coords=t.rhsSplit ? GetBlockIdx()/t.k : GetBlockIdx(),ai=0,bi=0;
            for(int d=int(t.batchRank)-1;d>=0;--d) { uint32_t q=coords%t.batchDims[d]; coords/=t.batchDims[d];ai+=q*t.aStrides[d];bi+=q*t.bStrides[d]; }
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            DataCopyPadExtParams<float> pad{true,0,0,0};
            DataCopyExtParams pa{N,N*4,0,1,0},pb{N,t.k*4,0,N/8,0};
            DataCopyPad(mat,ag[uint64_t(ai)*N*N],pa,pad); DataCopyPad(mat[N],bg[uint64_t(bi)*N*t.k],pb,pad);
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        }
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC {
            SetHF32Mode(HF32Mode::DISABLE);
            for(uint32_t p=0;p+16<N;p+=16) {
                uint32_t end=p+16; CrossCoreWaitFlag<2>(0);
                for(uint32_t r=end;r<N;r+=128) for(uint32_t c=end;c<RP;c+=128)
                    Multiply(p,r,c,N-r<128?N-r:128,RP-c<128?RP-c:128);
                CrossCoreSetFlag<2,PIPE_FIX>(1);
            }
        } else {
            for(uint32_t p=0;p<N;p+=16) {
                uint32_t end=p+16<N?p+16:N;
                for(uint32_t j=p;j<end;++j) PanelPivot(p,end,j);
                // Solve the panel's U12 block; RHS columns are part of the right tail.
                for(uint32_t j=p;j<end;++j) {
                    AscendC::Gather(coef,mat,index,j*4,N); PipeBarrier<PIPE_V>();
                    Muls(coef,coef,-1.0f,N); PipeBarrier<PIPE_V>(); Brcb(column,coef,N/8,{1,8}); PipeBarrier<PIPE_V>();
                    DataCopy(row,mat[j*RP],RP); PipeBarrier<PIPE_V>();
                    if(j+1<end) for(uint32_t c=end;c<RP;c+=64) {
                        MulAddDst(mat[(j+1)*RP+c],column[(j+1)*8],row[c],uint64_t(RP-c<64?RP-c:64),uint8_t(end-j-1),{1,0,1,RP/8,1,0});
                        PipeBarrier<PIPE_V>();
                    }
                }
                if(end<N) {
                    Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
                    DataCopyExtParams pl{uint16_t(N-end),16*4,(RP-16)/8,(RP-16)*4,0};
                    DataCopyExtParams pu{16,(RP-end)*4,end/8,end*4,0};
                    DataCopyPad(wg[end*RP+p],mat[end*RP+p],pl); DataCopyPad(wg[p*RP+end],mat[p*RP+end],pu);
                    CrossCoreSetFlag<2,PIPE_MTE3>(0); CrossCoreWaitFlag<2>(1);
                    for(uint32_t first=end;first<N;first+=16) {
                        uint32_t rows=N-first<16?N-first:16;
                        Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
                        DataCopyExtParams pc{uint16_t(rows),(RP-end)*4,end*4,end/8,0};
                        DataCopyPadExtParams<float> pad{false,0,0,0};
                        DataCopyPad(temp,product[first*RP+end],pc,pad); Fence<HardEvent::MTE2_V>();
                        for(uint32_t c=0;c<RP-end;c+=64) {
                            Sub(mat[first*RP+end+c],mat[first*RP+end+c],temp[c],uint64_t(RP-end-c<64?RP-end-c:64),uint8_t(rows),{1,1,1,RP/8,RP/8,RP/8});
                            PipeBarrier<PIPE_V>();
                        }
                    }
                }
            }
            Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
            for(uint32_t lane=0;lane<4;++lane){
                DataCopy(wg[uint64_t(lane)*ELEMENTS],mat,N*RP);
                DataCopy(wg[uint64_t(lane)*ELEMENTS+N*RP],piv.ReinterpretCast<float>(),N);
            }
            Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
        }
    }
private:
    __aicore__ inline void PanelPivot(uint32_t p,uint32_t end,uint32_t j) {
        AscendC::Gather(coef,mat,index,j*4,N); PipeBarrier<PIPE_V>(); Abs(ab,coef,N); PipeBarrier<PIPE_V>();
        if(j) { Duplicate(ab,0.0f,j); PipeBarrier<PIPE_V>(); }
        WholeReduceMax(maximum,ab,int32_t(64),uint8_t(4),1,1,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
        uint32_t maximumBits,pivot;GetReduceRepeatMaxMinSpr<float>(maximumBits,pivot);float diagonal=coef.GetValue(pivot);
        if(pivot!=j) { coef.SetValue(pivot,coef.GetValue(j));coef.SetValue(j,diagonal); }
        piv.SetValue(j,pivot); Fence<HardEvent::S_V>();
        DataCopy(row,mat[j*RP],RP); PipeBarrier<PIPE_V>();
        if(pivot!=j) {
            DataCopy(other,mat[pivot*RP],RP); PipeBarrier<PIPE_V>();
            DataCopy(mat[pivot*RP],row,RP); DataCopy(mat[j*RP],other,RP); PipeBarrier<PIPE_V>();
            DataCopy(row,other,RP); PipeBarrier<PIPE_V>();
        }
        Duplicate(row[p],0.0f,j-p+1); PipeBarrier<PIPE_V>();
        Muls(coef,coef,-1.0f/diagonal,N); PipeBarrier<PIPE_V>(); Brcb(column,coef,N/8,{1,8}); PipeBarrier<PIPE_V>();
        if(j+1<N) {
            MulAddDst(mat[(j+1)*RP+p],column[(j+1)*8],row[p],uint64_t(end-p),uint8_t(N-j-1),{1,0,1,RP/8,1,0}); PipeBarrier<PIPE_V>();
            const uint32_t rows=N-j-1,groups=rows/8,tail=rows%8,start=(j/8)*8;
            const uint64_t packedMask=0x0101010101010101ULL<<(j&7);
            UnaryRepeatParams packedStride{RP/8,1,RP,8};
            if(groups){uint64_t mask[2]={packedMask,0};
                Muls(mat[(j+1)*RP+start],column[(j+1)*8],-1.0f,mask,uint8_t(groups),packedStride);}
            if(tail){const uint32_t first=j+1+groups*8;uint64_t mask[2]={packedMask&((1ULL<<(tail*8))-1),0};
                Muls(mat[first*RP+start],column[first*8],-1.0f,mask,uint8_t(1),packedStride);}
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Multiply(uint32_t p,uint32_t r,uint32_t c,uint32_t m,uint32_t n) {
        uint32_t mp=(m+15)/16*16,np=(n+15)/16*16;
        LocalTensor<float> a1(TPosition::A1,0,128*16), b1(TPosition::B1,128*16*4,128*16);
        LocalTensor<float> a2(TPosition::A2,0,128*16), b2(TPosition::B2,0,128*16), cc(TPosition::CO1,0,128*128);
        Nd2NzParams pa{1,uint16_t(m),16,0,RP,uint16_t(mp),1,0};
        Nd2NzParams pb{1,16,uint16_t(n),0,RP,16,1,0};
        DataCopy(a1,wg[r*RP+p],pa); DataCopy(b1,wg[p*RP+c],pb);
        Fence<HardEvent::MTE2_MTE1>();
        LoadData2DParams la;la.startIndex=0;la.repeatTimes=2;la.srcStride=mp/16;la.dstGap=0;la.ifTranspose=false;
        for(uint32_t i=0;i<mp/16;++i) LoadData(a2[i*256],a1[i*128],la);
        LoadData2dTransposeParams lb;lb.startIndex=0;lb.repeatTimes=np/16;lb.srcStride=1;lb.dstGap=0;lb.dstFracGap=np/16-1;
        LoadDataWithTranspose(b2,b1,lb); Fence<HardEvent::MTE1_M>();
        MmadParams mm;mm.m=m;mm.n=n;mm.k=16;mm.cmatrixInitVal=true;Mmad(cc,a2,b2,mm);
        Fence<HardEvent::M_FIX>();
        FixpipeParamsV220 fp;fp.ndNum=1;fp.srcNdStride=0;fp.dstNdStride=0;fp.mSize=m;fp.nSize=n;fp.srcStride=mp;fp.dstStride=RP;
        Fixpipe(product[r*RP+c],cc,fp); Fence<HardEvent::FIX_MTE2>();
    }
    TPipe pipe;
    LinalgSolveTilingData t;
    GlobalTensor<float> ag,bg,wg,product;
    LocalTensor<float> mem,mat,row,other,coef,ab,maximum,column,temp;
    LocalTensor<uint32_t> index; LocalTensor<int32_t> piv;
};

// Lanen200 globalPP32 panel ownership; B16solver caches remain independent.
class Lanen200_GlobalPP32Factor {
    static constexpr uint32_t N=200, RP=208, ELEMENTS=N*216,PB=32;
    static constexpr uint32_t ROW=N*RP, OTHER=ROW+RP, COEF=OTHER+RP;
    static constexpr uint32_t ABS=COEF+N, MAX=ABS+256, COL=MAX+8;
    static constexpr uint32_t IDX=COL+8*N, PIV=IDX+N, TEMP=PIV+N;
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR workspace,const LinalgSolveTilingData &data) {
        static_assert((TEMP+16*RP)*4<=192*1024,"N200 padded argmax fits UB withoutSelect");
        t=data;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a)); bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        auto base=reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace));
        wg.SetGlobalBuffer(base+uint64_t(GetBlockIdx())*ELEMENTS);
        product.SetGlobalBuffer(base+uint64_t(t.blocks)*ELEMENTS+uint64_t(GetBlockIdx())*N*RP);
        if ASCEND_IS_AIV {
            mem=LocalTensor<float>(TPosition::VECCALC,0,TEMP+16*RP); mat=mem; row=mem[ROW]; other=mem[OTHER];
            coef=mem[COEF]; ab=mem[ABS]; maximum=mem[MAX]; column=mem[COL]; temp=mem[TEMP];
            index=mem[IDX].ReinterpretCast<uint32_t>(); piv=mem[PIV].ReinterpretCast<int32_t>();
            for(uint32_t i=0;i<N;++i) index.SetValue(i,i*RP*4);
            Fence<HardEvent::S_V>(); Duplicate(mat,0.0f,N*RP);Duplicate(ab,0.0f,256); PipeBarrier<PIPE_V>();
            uint32_t coords=t.rhsSplit ? GetBlockIdx()/t.k : GetBlockIdx(),ai=0,bi=0;
            for(int d=int(t.batchRank)-1;d>=0;--d) { uint32_t q=coords%t.batchDims[d]; coords/=t.batchDims[d];ai+=q*t.aStrides[d];bi+=q*t.bStrides[d]; }
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            DataCopyPadExtParams<float> pad{true,0,0,0};
            DataCopyExtParams pa{N,N*4,0,1,0},pb{N,t.k*4,0,N/8,0};
            DataCopyPad(mat,ag[uint64_t(ai)*N*N],pa,pad); DataCopyPad(mat[N],bg[uint64_t(bi)*N*t.k],pb,pad);
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        }
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC {
            SetHF32Mode(HF32Mode::DISABLE);
            for(uint32_t p=0;p+PB<N;p+=PB) {
                uint32_t end=p+PB; CrossCoreWaitFlag<2>(0);
                for(uint32_t r=end;r<N;r+=128) for(uint32_t c=end;c<RP;c+=128)
                    Multiply(p,r,c,N-r<128?N-r:128,RP-c<128?RP-c:128);
                CrossCoreSetFlag<2,PIPE_FIX>(1);
            }
        } else {
            for(uint32_t p=0;p<N;p+=PB) {
                uint32_t end=p+PB<N?p+PB:N;
                for(uint32_t j=p;j<end;++j) PanelPivot(p,end,j);
                // Solve the panel's U12 block; RHS columns are part of the right tail.
                for(uint32_t j=p;j<end;++j) {
                    AscendC::Gather(coef,mat,index,j*4,N); PipeBarrier<PIPE_V>();
                    Muls(coef,coef,-1.0f,N); PipeBarrier<PIPE_V>(); Brcb(column,coef,N/8,{1,8}); PipeBarrier<PIPE_V>();
                    DataCopy(row,mat[j*RP],RP); PipeBarrier<PIPE_V>();
                    if(j+1<end) for(uint32_t c=end;c<RP;c+=64) {
                        MulAddDst(mat[(j+1)*RP+c],column[(j+1)*8],row[c],uint64_t(RP-c<64?RP-c:64),uint8_t(end-j-1),{1,0,1,RP/8,1,0});
                        PipeBarrier<PIPE_V>();
                    }
                }
                if(end<N) {
                    Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
                    DataCopyExtParams pl{uint16_t(N-end),PB*4,(RP-PB)/8,(RP-PB)*4,0};
                    DataCopyExtParams pu{PB,(RP-end)*4,end/8,end*4,0};
                    DataCopyPad(wg[end*RP+p],mat[end*RP+p],pl); DataCopyPad(wg[p*RP+end],mat[p*RP+end],pu);
                    CrossCoreSetFlag<2,PIPE_MTE3>(0); CrossCoreWaitFlag<2>(1);
                    for(uint32_t first=end;first<N;first+=16) {
                        uint32_t rows=N-first<16?N-first:16;
                        Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
                        DataCopyExtParams pc{uint16_t(rows),(RP-end)*4,end*4,end/8,0};
                        DataCopyPadExtParams<float> pad{false,0,0,0};
                        DataCopyPad(temp,product[first*RP+end],pc,pad); Fence<HardEvent::MTE2_V>();
                        for(uint32_t c=0;c<RP-end;c+=64) {
                            Sub(mat[first*RP+end+c],mat[first*RP+end+c],temp[c],uint64_t(RP-end-c<64?RP-end-c:64),uint8_t(rows),{1,1,1,RP/8,RP/8,RP/8});
                            PipeBarrier<PIPE_V>();
                        }
                    }
                }
            }
            Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
            for(uint32_t lane=0;lane<4;++lane){
                DataCopy(wg[uint64_t(lane)*ELEMENTS],mat,N*RP);
                DataCopy(wg[uint64_t(lane)*ELEMENTS+N*RP],piv.ReinterpretCast<float>(),N);
            }
            Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
        }
    }
private:
    __aicore__ inline void PanelPivot(uint32_t p,uint32_t end,uint32_t j) {
        AscendC::Gather(coef,mat,index,j*4,N); PipeBarrier<PIPE_V>(); Abs(ab,coef,N); PipeBarrier<PIPE_V>();
        if(j) { Duplicate(ab,0.0f,j); PipeBarrier<PIPE_V>(); }
        WholeReduceMax(maximum,ab,int32_t(64),uint8_t(4),1,1,8);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
        uint32_t maximumBits,pivot;GetReduceRepeatMaxMinSpr<float>(maximumBits,pivot);float diagonal=coef.GetValue(pivot);
        if(pivot!=j) { coef.SetValue(pivot,coef.GetValue(j));coef.SetValue(j,diagonal); }
        piv.SetValue(j,pivot); Fence<HardEvent::S_V>();
        DataCopy(row,mat[j*RP],RP); PipeBarrier<PIPE_V>();
        if(pivot!=j) {
            DataCopy(other,mat[pivot*RP],RP); PipeBarrier<PIPE_V>();
            DataCopy(mat[pivot*RP],row,RP); DataCopy(mat[j*RP],other,RP); PipeBarrier<PIPE_V>();
            DataCopy(row,other,RP); PipeBarrier<PIPE_V>();
        }
        Duplicate(row[p],0.0f,j-p+1); PipeBarrier<PIPE_V>();
        Muls(coef,coef,-1.0f/diagonal,N); PipeBarrier<PIPE_V>(); Brcb(column,coef,N/8,{1,8}); PipeBarrier<PIPE_V>();
        if(j+1<N) {
            MulAddDst(mat[(j+1)*RP+p],column[(j+1)*8],row[p],uint64_t(end-p),uint8_t(N-j-1),{1,0,1,RP/8,1,0}); PipeBarrier<PIPE_V>();
            const uint32_t rows=N-j-1,groups=rows/8,tail=rows%8,start=(j/8)*8;
            const uint64_t packedMask=0x0101010101010101ULL<<(j&7);
            UnaryRepeatParams packedStride{RP/8,1,RP,8};
            if(groups){uint64_t mask[2]={packedMask,0};
                Muls(mat[(j+1)*RP+start],column[(j+1)*8],-1.0f,mask,uint8_t(groups),packedStride);}
            if(tail){const uint32_t first=j+1+groups*8;uint64_t mask[2]={packedMask&((1ULL<<(tail*8))-1),0};
                Muls(mat[first*RP+start],column[first*8],-1.0f,mask,uint8_t(1),packedStride);}
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Multiply(uint32_t p,uint32_t r,uint32_t c,uint32_t m,uint32_t n){
        uint32_t mp=(m+15)/16*16,np=(n+15)/16*16;
        LocalTensor<float>a1(TPosition::A1,0,128*16),b1(TPosition::B1,128*16*4,128*16),a2(TPosition::A2,0,128*16),b2(TPosition::B2,0,128*16),cc(TPosition::CO1,0,128*128);
        for(uint32_t q=0;q<PB;q+=16){
            Fence<HardEvent::M_MTE1>();Fence<HardEvent::MTE1_MTE2>();
            Nd2NzParams pa{1,uint16_t(m),16,0,RP,uint16_t(mp),1,0},pb{1,16,uint16_t(n),0,RP,16,1,0};DataCopy(a1,wg[r*RP+p+q],pa);DataCopy(b1,wg[(p+q)*RP+c],pb);Fence<HardEvent::MTE2_MTE1>();
            LoadData2DParams la;la.startIndex=0;la.repeatTimes=2;la.srcStride=mp/16;la.dstGap=0;la.ifTranspose=false;for(uint32_t i=0;i<mp/16;++i)LoadData(a2[i*256],a1[i*128],la);
            LoadData2dTransposeParams lb;lb.startIndex=0;lb.repeatTimes=np/16;lb.srcStride=1;lb.dstGap=0;lb.dstFracGap=np/16-1;LoadDataWithTranspose(b2,b1,lb);Fence<HardEvent::MTE1_M>();MmadParams mm;mm.m=m;mm.n=n;mm.k=16;mm.cmatrixInitVal=q==0;Mmad(cc,a2,b2,mm);
        }
        Fence<HardEvent::M_FIX>();FixpipeParamsV220 fp;fp.ndNum=1;fp.srcNdStride=0;fp.dstNdStride=0;fp.mSize=m;fp.nSize=n;fp.srcStride=mp;fp.dstStride=RP;Fixpipe(product[r*RP+c],cc,fp);Fence<HardEvent::FIX_MTE2>();
    }
    TPipe pipe;
    LinalgSolveTilingData t;
    GlobalTensor<float> ag,bg,wg,product;
    LocalTensor<float> mem,mat,row,other,coef,ab,maximum,column,temp;
    LocalTensor<uint32_t> index; LocalTensor<int32_t> piv;
};


// Lanen200 ownership: factor remains sealed-A106 global-PP; four workers share diagonal caches.
class Lanen200_GlobalPPCaches {
    static constexpr uint32_t N=200,B=16,RP=208,CP=24,NB=13,CS=NB*B*CP,ELEMENTS=N*216;
public:
    __aicore__ inline void Init(GM_ADDR workspace,const LinalgSolveTilingData&d){t=d;auto user=reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace));wg.SetGlobalBuffer(user+uint64_t(GetBlockIdx())*ELEMENTS);cache.SetGlobalBuffer(user+uint64_t(t.blocks)*ELEMENTS+uint64_t(GetBlockIdx()/4*4)*N*RP+N*RP);mem=LocalTensor<float>(TPosition::VECCALC,0,2048);}
    __aicore__ inline void Process(){for(uint32_t block=GetBlockIdx()%4;block<NB;block+=4)Make(block);}
private:
    __aicore__ inline void Make(uint32_t block){
        auto lu=mem,up=mem[B*CP],lo=mem[2*B*CP],coef=mem[3*B*CP],bc=mem[3*B*CP+B],diag=mem[1296],iv=mem[1312],ib=mem[1328],ones=mem[1456];auto ci=mem[1472].ReinterpretCast<uint32_t>();auto di=mem[1488].ReinterpretCast<uint32_t>();const uint32_t first=block*B,w=N-first<B?N-first:B;
        Duplicate(lu,0.0f,3*B*CP);Duplicate(ones,1.0f,B);PipeBarrier<PIPE_V>();Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();DataCopyExtParams p{uint16_t(w),w*4,(RP-w)*4,(CP-w)/8,0};DataCopyPadExtParams<float>pad{false,0,0,0};DataCopyPad(lu,wg[first*RP+first],p,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        for(uint32_t i=0;i<B;++i){up.SetValue(i*CP+i,1.0f);lo.SetValue(i*CP+i,1.0f);if(i>=w)lu.SetValue(i*CP+i,1.0f);ci.SetValue(i,i*CP*4);di.SetValue(i,i*(CP+1)*4);}Fence<HardEvent::S_V>();Gather(diag,lu,di,0,B);PipeBarrier<PIPE_V>();Div(iv,ones,diag,B);PipeBarrier<PIPE_V>();Brcb(ib,iv,2,{1,8});PipeBarrier<PIPE_V>();
        for(int j=int(B)-1;j>=0;--j){Mul(up[uint32_t(j)*CP],up[uint32_t(j)*CP],ib[uint32_t(j)*8],uint64_t(B),uint8_t(1),{1,1,0,CP/8,CP/8,1});PipeBarrier<PIPE_V>();if(j){Gather(coef,lu,ci,uint32_t(j)*4,B);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,B);PipeBarrier<PIPE_V>();Brcb(bc,coef,2,{1,8});PipeBarrier<PIPE_V>();MulAddDst(up,bc,up[uint32_t(j)*CP],uint64_t(B),uint8_t(j),{1,0,1,CP/8,1,0});PipeBarrier<PIPE_V>();}}
        for(uint32_t j=0;j+1<B;++j){Gather(coef,lu,ci,j*4,B);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,B);PipeBarrier<PIPE_V>();Brcb(bc,coef,2,{1,8});PipeBarrier<PIPE_V>();MulAddDst(lo[(j+1)*CP],bc[(j+1)*8],lo[j*CP],uint64_t(B),uint8_t(B-j-1),{1,0,1,CP/8,1,0});PipeBarrier<PIPE_V>();}
        Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();DataCopy(cache[block*B*CP],up,B*CP);DataCopy(cache[CS+block*B*CP],lo,B*CP);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
    }
    TPipe pipe;LinalgSolveTilingData t;LocalTensor<float>mem;GlobalTensor<float>wg,cache;
};
// End Lanen200 ownership.


// RootN50 contiguousstore ownership.
template <uint32_t N> class RootN50_ContiguousStore {
    static constexpr uint32_t D=64, C=56, RP=72, W=64, NS=N*8;
    static constexpr uint32_t CO=N*RP, AB=CO+D, MM=AB+D, RW=MM+8;
    static constexpr uint32_t BC=RW+RP, XX=BC+D*8, TOTAL=XX+NS;
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR out){
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        mem=LocalTensor<float>(TPosition::VECCALC,0,TOTAL);
        mat=mem;coef=mem[CO];ab=mem[AB];maximum=mem[MM];row=mem[RW];broad=mem[BC];packed=mem[XX];
        ci=LocalTensor<uint32_t>(TPosition::VECCALC,TOTAL*4,D);
        oi=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+2*D)*4,NS);
        const uint32_t batch=GetBlockIdx();
        DataCopyPadExtParams<float> pad{true,0,0,0};
        DataCopyExtParams pa{N,N*4,0,(RP-(N+7)/8*8)/8,0};
        DataCopyExtParams pb{N,8,0,(RP-8)/8,0};
        DataCopyPad(mat,ag[uint64_t(batch)*N*N],pa,pad);
        DataCopyPad(mat[C],bg[uint64_t(batch)*N*2],pb,pad);
        auto positions=coef.ReinterpretCast<int32_t>();
        for(uint32_t q=0;q<D;++q)positions.SetValue(q,int32_t(q));
        Fence<HardEvent::S_V>();
        Muls(ci.ReinterpretCast<int32_t>(),positions,int32_t(RP*4),D);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process(){
        Fence<HardEvent::MTE2_V>();
        uint64_t active=(uint64_t(1)<<N)-1;
        for(uint32_t j=0;j<N;++j){
            AscendC::Gather(coef,mat,ci,j*4,N);PipeBarrier<PIPE_V>();
            Abs(ab,coef,D);PipeBarrier<PIPE_V>();
            uint64_t available[2]={active,0};
            WholeReduceMax(maximum,ab,available,uint8_t(1),1,1,8);
            Fence<HardEvent::V_S>();
            uint32_t bits,pivot;GetReduceRepeatMaxMinSpr<float>(bits,pivot);
            const float diagonal=coef.GetValue(pivot);
            oi.SetValue(2*j,pivot*RP*4+C*4);
            oi.SetValue(2*j+1,pivot*RP*4+(C+1)*4);
            uint64_t chosen[2]={uint64_t(1)<<pivot,0};
            Muls(row,mat[pivot*RP],-1.0f/diagonal,W);
            Duplicate(coef,0.0f,chosen,uint8_t(1),1,D/8);PipeBarrier<PIPE_V>();
            DataCopy(mat[pivot*RP],row,W);
            Brcb(broad,coef,D/8,{1,8});PipeBarrier<PIPE_V>();
            MulAddDst(mat,broad,row,uint64_t(W),uint8_t(N),{1,0,1,RP/8,1,0});PipeBarrier<PIPE_V>();
            active&=~(uint64_t(1)<<pivot);
        }
        Fence<HardEvent::S_V>();
        AscendC::Gather(packed,mat,oi,0,N*2);PipeBarrier<PIPE_V>();
        Muls(packed,packed,-1.0f,N*2);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams storeParams{1,N*2*4,0,0,0};
        DataCopyPad(xg[uint64_t(GetBlockIdx())*N*2],packed,storeParams);
        Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
    }
private:
    GlobalTensor<float> ag,bg,xg;
    LocalTensor<float> mem,mat,coef,ab,maximum,row,broad,packed;
    LocalTensor<uint32_t> ci,oi;
};
// End RootN50 contiguousstore ownership.

// RootN50 contiguousIO ownership.
template <uint32_t N> class RootN50_ContiguousIO {
    static constexpr uint32_t D=64, C=56, RP=72, W=64, NS=N*8;
    static constexpr uint32_t CO=N*RP, AB=CO+D, MM=AB+D, RW=MM+8;
    static constexpr uint32_t BC=RW+RP, XX=BC+D*8, RAW=XX+NS;
    static constexpr uint32_t RA=(N*N+7)/8*8,RB=(N*2+7)/8*8,FM=RAW+RA+RB,BX=FM+N*D,TOTAL=BX+RB*8;
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR out){
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        mem=LocalTensor<float>(TPosition::VECCALC,0,TOTAL);
        mat=mem;coef=mem[CO];ab=mem[AB];maximum=mem[MM];row=mem[RW];broad=mem[BC];packed=mem[XX];
        ci=LocalTensor<uint32_t>(TPosition::VECCALC,TOTAL*4,D);
        oi=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+2*D)*4,NS);
        raw=mem[RAW];fmt=mem[FM];expanded=mem[BX];
        ti=LocalTensor<uint32_t>(TPosition::VECCALC,(TOTAL+2*D+NS)*4,N*D);
        pattern=LocalTensor<int32_t>(TPosition::VECCALC,(TOTAL+2*D+NS+N*D)*4,D);
        static_assert((TOTAL+2*D+NS+N*D+D)*4<=196608,"RootN50 contiguousIO UB bound");
        Duplicate(mat,0.0f,N*RP);
        const uint32_t batch=GetBlockIdx();
        DataCopyPadExtParams<float> padA{true,0,uint8_t(RA-N*N),0},padB{true,0,uint8_t(RB-N*2),0};
        DataCopyExtParams pa{1,N*N*4,0,0,0},pb{1,N*2*4,0,0,0};
        DataCopyPad(raw,ag[uint64_t(batch)*N*N],pa,padA);
        DataCopyPad(raw[RA],bg[uint64_t(batch)*N*2],pb,padB);
        auto positions=coef.ReinterpretCast<int32_t>();
        for(uint32_t q=0;q<D;++q)positions.SetValue(q,int32_t(q));
        Fence<HardEvent::S_V>();
        Muls(ci.ReinterpretCast<int32_t>(),positions,int32_t(N*4),D);
        Muls(pattern,positions,int32_t(4),D);PipeBarrier<PIPE_V>();
        Brcb(broad,ci.ReinterpretCast<float>(),D/8,{1,8});PipeBarrier<PIPE_V>();
        Add(ti.ReinterpretCast<int32_t>(),broad.ReinterpretCast<int32_t>(),pattern,uint64_t(D),uint8_t(N),{1,0,1,D/8,1,0});
        Muls(ci.ReinterpretCast<int32_t>(),positions,int32_t(RP*4),D);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process(){
        Fence<HardEvent::MTE2_V>();
        AscendC::Gather(fmt,raw,ti,0,N*D);
        Brcb(expanded,raw[RA],RB/8,{1,8});PipeBarrier<PIPE_V>();
        uint64_t lane0[2]={1,0},lane1[2]={2,0};
        Muls(mat,fmt,1.0f,uint64_t(N),uint8_t(N),{1,1,RP/8,D/8});
        Muls(mat[C],expanded,1.0f,lane0,uint8_t(N),{1,1,RP/8,2});
        Muls(mat[C],expanded[8],1.0f,lane1,uint8_t(N),{1,1,RP/8,2});PipeBarrier<PIPE_V>();
        uint64_t active=(uint64_t(1)<<N)-1;
        for(uint32_t j=0;j<N;++j){
            AscendC::Gather(coef,mat,ci,j*4,N);PipeBarrier<PIPE_V>();
            Abs(ab,coef,D);PipeBarrier<PIPE_V>();
            uint64_t available[2]={active,0};
            WholeReduceMax(maximum,ab,available,uint8_t(1),1,1,8);
            Fence<HardEvent::V_S>();
            uint32_t bits,pivot;GetReduceRepeatMaxMinSpr<float>(bits,pivot);
            const float diagonal=coef.GetValue(pivot);
            oi.SetValue(2*j,pivot*RP*4+C*4);
            oi.SetValue(2*j+1,pivot*RP*4+(C+1)*4);
            uint64_t chosen[2]={uint64_t(1)<<pivot,0};
            coef.SetValue(pivot,diagonal+1.0f);
            Muls(row,mat[pivot*RP],-1.0f/diagonal,W);
            Fence<HardEvent::S_V>();
            Brcb(broad,coef,D/8,{1,8});PipeBarrier<PIPE_V>();
            MulAddDst(mat,broad,row,uint64_t(W),uint8_t(N),{1,0,1,RP/8,1,0});PipeBarrier<PIPE_V>();
            active&=~(uint64_t(1)<<pivot);
        }
        Fence<HardEvent::S_V>();
        AscendC::Gather(packed,mat,oi,0,N*2);PipeBarrier<PIPE_V>();
        Muls(packed,packed,-1.0f,N*2);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams storeParams{1,N*2*4,0,0,0};
        DataCopyPad(xg[uint64_t(GetBlockIdx())*N*2],packed,storeParams);
        Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
    }
private:
    GlobalTensor<float> ag,bg,xg;
    LocalTensor<float> mem,mat,coef,ab,maximum,row,broad,packed,raw,fmt,expanded;
    LocalTensor<uint32_t> ci,oi,ti;
    LocalTensor<int32_t>pattern;
};
// End RootN50 contiguousIO ownership.

// Root N50: padded64 rows, CP72 column-major, complete full-PP joint RHS solve.
class RootN50_ColumnMajor50 {
    static constexpr uint32_t N=50,D=64,CP=72,COLS=N+2,AL=56,RB=104,RA=2504;
    static constexpr uint32_t AB=AL*CP,MM=AB+D,RW=MM+8,BC=RW+D,XX=BC+D*8,TOTAL=XX+RB;
    static constexpr uint32_t CI=TOTAL,OI=CI+D,BI=OI+RB,AA=BI+D,BB=AA+RA;
    static constexpr uint32_t FM=BB+RB,TI=FM+N*D,END=TI+N*D;
    static_assert(END*4<=196608,"RootN50 column-major UB bound");
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR out){
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        mem=LocalTensor<float>(TPosition::VECCALC,0,TOTAL);
        mat=mem;ab=mem[AB];maximum=mem[MM];row=mem[RW];broad=mem[BC];packed=mem[XX];
        ci=LocalTensor<uint32_t>(TPosition::VECCALC,CI*4,D);
        oi=LocalTensor<uint32_t>(TPosition::VECCALC,OI*4,RB);
        bi=LocalTensor<uint32_t>(TPosition::VECCALC,BI*4,D);
        aa=LocalTensor<float>(TPosition::VECCALC,AA*4,RA);
        bb=LocalTensor<float>(TPosition::VECCALC,BB*4,RB);
        fmt=LocalTensor<float>(TPosition::VECCALC,FM*4,N*D);
        ti=LocalTensor<uint32_t>(TPosition::VECCALC,TI*4,N*D);
        const uint32_t batch=GetBlockIdx();
        DataCopyPadExtParams<float> padA{true,0,4,0},padB{true,0,4,0};
        DataCopyExtParams pa{1,N*N*4,0,0,0},pb{1,2*N*4,0,0,0};
        DataCopyPad(aa,ag[uint64_t(batch)*N*N],pa,padA);
        DataCopyPad(bb,bg[uint64_t(batch)*N*2],pb,padB);
        auto positions=row.ReinterpretCast<int32_t>();
        for(uint32_t q=0;q<D;++q)positions.SetValue(q,int32_t(q));
        Fence<HardEvent::S_V>();
        Muls(ci.ReinterpretCast<int32_t>(),positions,int32_t(CP*4),D);
        Muls(bi.ReinterpretCast<int32_t>(),positions,int32_t(8),D);
        Muls(ab.ReinterpretCast<int32_t>(),positions,int32_t(N*4),D);PipeBarrier<PIPE_V>();
        uint64_t paddedRows[2]={~((uint64_t(1)<<N)-1),0};
        Duplicate(ab,0.0f,paddedRows,uint8_t(1),1,D/8);
        Muls(positions,positions,int32_t(4),D);PipeBarrier<PIPE_V>();
        Brcb(broad,positions.ReinterpretCast<float>(),D/8,{1,8});PipeBarrier<PIPE_V>();
        Add(ti.ReinterpretCast<int32_t>(),broad.ReinterpretCast<int32_t>(),ab.ReinterpretCast<int32_t>(),
            uint64_t(D),uint8_t(N),{1,0,1,D/8,1,0});PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process(){
        Fence<HardEvent::MTE2_V>();
        Duplicate(mat,0.0f,AL*CP);PipeBarrier<PIPE_V>();
        Gather(fmt,aa,ti,0,N*D);PipeBarrier<PIPE_V>();
        DataCopyParams layout{N,D/8,0,(CP-D)/8};DataCopy(mat,fmt,layout);PipeBarrier<PIPE_V>();
        uint64_t paddedRows[2]={0xfffc,0};
        Duplicate(mat[48],0.0f,paddedRows,uint8_t(N),1,CP/8);PipeBarrier<PIPE_V>();
        Gather(mat[N*CP],bb,bi,0,N);
        Gather(mat[(N+1)*CP],bb,bi,4,N);PipeBarrier<PIPE_V>();
        // Match the validated N64 normal-mask row path. Gather may read
        // all repeat indices even with masked output; pad CI to 128 words.
        // Original AA is dead after the full transpose and both RHS gathers.
        auto RootN50_paddedCI=aa.ReinterpretCast<uint32_t>();
        DataCopy(RootN50_paddedCI,ci,D);
        Duplicate(RootN50_paddedCI[D].ReinterpretCast<int32_t>(),int32_t(0),D);
        PipeBarrier<PIPE_V>();ci=RootN50_paddedCI;
        uint64_t active=(uint64_t(1)<<N)-1;
        for(uint32_t first=0;first<N;first+=8){
            const uint32_t width=AL-first,steps=N-first<8?N-first:8;
            uint64_t RootN50_rowMask[2]={(uint64_t(1)<<width)-1,0};
            for(uint32_t q=0;q<steps;++q)Pivot(first+q,first,width,active,RootN50_rowMask);
        }
        Fence<HardEvent::S_V>();
        Gather(packed,mat,oi,0,2*N);PipeBarrier<PIPE_V>();
        Muls(packed,packed,-1.0f,2*N);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams storeParams{1,2*N*4,0,0,0};
        DataCopyPad(xg[uint64_t(GetBlockIdx())*N*2],packed,storeParams);
        Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
    }
private:
    __aicore__ inline void Pivot(uint32_t j,uint32_t first,uint32_t width,uint64_t &active,uint64_t *RootN50_rowMask){
        auto coef=mat[j*CP];
        Abs(ab,coef,D);PipeBarrier<PIPE_V>();
        uint64_t available[2]={active,0};
        WholeReduceMax(maximum,ab,available,uint8_t(1),1,1,8);Fence<HardEvent::V_S>();
        uint32_t bits,pivot;GetReduceRepeatMaxMinSpr<float>(bits,pivot);
        const float diagonal=coef.GetValue(pivot);
        oi.SetValue(2*j,(N*CP+pivot)*4);oi.SetValue(2*j+1,((N+1)*CP+pivot)*4);
        uint64_t chosen[2]={uint64_t(1)<<pivot,0};
        Gather(row,mat,ci[first],pivot*4,RootN50_rowMask,uint8_t(1),uint16_t(8));PipeBarrier<PIPE_V>();
        Muls(row,row,-1.0f/diagonal,RootN50_rowMask,uint8_t(1),{1,1,8,8});
        Adds(coef,coef,1.0f,chosen,uint8_t(1),{1,1,D/8,D/8});PipeBarrier<PIPE_V>();
        Brcb(broad,row,width/8,{1,8});PipeBarrier<PIPE_V>();
        const uint32_t count=COLS-j-1,begin=j+1;
        MulAddDst(mat[begin*CP],coef,broad[(begin-first)*8],uint64_t(D),uint8_t(count),{1,1,0,CP/8,0,1});
        PipeBarrier<PIPE_V>();active&=~(uint64_t(1)<<pivot);
    }
    GlobalTensor<float> ag,bg,xg;
    LocalTensor<float> mem,mat,ab,maximum,row,broad,packed,aa,bb,fmt;
    LocalTensor<uint32_t> ci,oi,bi,ti;
};

// BEGIN Lanen64 N200 split-LU GJ16 full solve.
static constexpr uint32_t Lanen64_GJ_C=41600,Lanen64_GJ_UP=83200,Lanen64_GJ_LP=88192,Lanen64_GJ_NEGUP=93184,Lanen64_GJ_RAW=98176,Lanen64_GJ_TMP=101504,Lanen64_GJ_NORM=104832,Lanen64_GJ_FINAL=108160,Lanen64_GJ_ID=111360,Lanen64_GJ_RAW1=154624,Lanen64_GJ_TOTAL=157952;
class Lanen64_SplitGJ16Factor {
    static constexpr uint32_t N=200,RP=208,ELEMENTS=N*216,B=16,CRP=24,CS=13*B*CRP;
    static_assert(Lanen64_GJ_TOTAL<=4*N*RP,"N200 existing four product slots fit rawC/LP/UP/rows/identity");
    static constexpr uint32_t ROW=N*RP,OTHER=ROW+RP,COEF=OTHER+RP,ABS=COEF+N,MAX=ABS+256,COL=MAX+8,IDX=COL+8*N,PIV=IDX+N,TEMP=PIV+N;
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR workspace,const LinalgSolveTilingData &data) {
        static_assert((TEMP+16*RP+RP+256)*4<=192*1024,"N200 padded argmax fits UB withoutSelect");
        t=data;quad=GetBlockIdx()%4;r0=quad/2*96;c0=quad%2*96;rows=quad/2?104:96;cols=quad%2?112:96;mp=(rows+15)/16*16;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a)); bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        auto base=reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace));
        wg.SetGlobalBuffer(base+uint64_t(GetBlockIdx()/4*4)*ELEMENTS);
        product.SetGlobalBuffer(base+uint64_t(t.blocks)*ELEMENTS+uint64_t(GetBlockIdx()/4*4)*N*RP);
        if ASCEND_IS_AIV {if(quad==0){
            mem=LocalTensor<float>(TPosition::VECCALC,0,TEMP+16*RP); mat=mem; row=mem[ROW]; other=mem[OTHER];
            coef=mem[COEF]; ab=mem[ABS]; maximum=mem[MAX]; column=mem[COL]; temp=mem[TEMP];
            index=mem[IDX].ReinterpretCast<uint32_t>(); piv=mem[PIV].ReinterpretCast<int32_t>();
            colids=LocalTensor<int32_t>(TPosition::VECCALC,(TEMP+16*RP)*4,RP);eye=LocalTensor<float>(TPosition::VECCALC,(TEMP+16*RP+RP)*4,256);
            for(uint32_t i=0;i<B;++i){index.SetValue(i,i*40*4);index.SetValue(16+i,i*16*4);index.SetValue(32+i,i*8*4);}
            for(uint32_t i=0;i<RP;++i)colids.SetValue(i,int32_t(i*4));
            Fence<HardEvent::S_V>(); Duplicate(mat,0.0f,N*RP);Duplicate(ab,-1.0f,256);Duplicate(eye,0.0f,256); PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();for(uint32_t i=0;i<16;++i)eye.SetValue(i*17,1.0f);Fence<HardEvent::S_V>();
            uint32_t coords=t.rhsSplit ? GetBlockIdx()/t.k : GetBlockIdx(),ai=0,bi=0;
            for(int d=int(t.batchRank)-1;d>=0;--d) { uint32_t q=coords%t.batchDims[d]; coords/=t.batchDims[d];ai+=q*t.aStrides[d];bi+=q*t.bStrides[d]; }
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            DataCopyPadExtParams<float> pad{true,0,0,0};
            DataCopyExtParams pa{N,N*4,0,1,0},pb{N,t.k*4,0,N/8,0};
            DataCopyPad(mat,ag[uint64_t(ai)*N*N],pa,pad); DataCopyPad(mat[N],bg[uint64_t(bi)*N*t.k],pb,pad);
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        }}
    }
    __aicore__ inline void Process(){
        if ASCEND_IS_AIC{
            SetHF32Mode(HF32Mode::DISABLE);CrossCoreWaitFlag<2>(0);CubeInitialize();PublishPanel(0);CrossCoreSetFlag<2,PIPE_FIX>(1);
            for(uint32_t p=0;p<N;p+=B){
                CrossCoreWaitFlag<2>(0);if(quad<2)NormalizeRows(p);CrossCoreSetFlag<2,PIPE_FIX>(1);
                CrossCoreWaitFlag<2>(2);UpdateAll(p);
                if(p+B<N)PublishPanel(p+B);else PublishFinal();
                CrossCoreSetFlag<2,PIPE_FIX>(3);
            }
        }else{
            if(quad==0){
                Muls(mat,mat,-1.0f,N*RP);PipeBarrier<PIPE_V>();Tx(product,mat,N*RP);
                Duplicate(mat,0.0f,208*208);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();for(uint32_t i=0;i<208;++i)mat.SetValue(i*209,-1.0f);Tx(product[Lanen64_GJ_ID],mat,208*208);
            }
            SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(0);CrossCoreWaitFlag<2>(1);SyncAll<true>();
            for(uint32_t p=0;p<N;p+=B){const uint32_t w=N-p<B?N-p:B;
                if(quad==0){
                    DataCopyExtParams pd{uint16_t(w),w*4,(RP-w)*4,(RP-w)/8,0};RxRows(mat[p*RP+p],product[RawOffset(p)+p],pd);
                    FactorBlock(p,p+w);
                    // C is the raw S-column block. Its selected rows are explicitly restored.
                    Duplicate(temp,0.0f,B*B);PipeBarrier<PIPE_V>();Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();DataCopyExtParams z{uint16_t(w),B*4,0,(RP-B)*4,0};DataCopyPad(product[Lanen64_GJ_C+p*RP+p],temp,z);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
                    if(w<B){Duplicate(temp,0.0f,(B-w)*RP);PipeBarrier<PIPE_V>();Tx(product[RawOffset(p)+w*RP],temp,(B-w)*RP);}
                }
                SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(0);CrossCoreWaitFlag<2>(1);SyncAll<true>();
                CrossCoreSetFlag<2,PIPE_MTE3>(2);CrossCoreWaitFlag<2>(3);SyncAll<true>();
            }
            if(quad==0){
                Rx(mat,product[Lanen64_GJ_C],N*RP);Rx(temp,product[Lanen64_GJ_FINAL],N*16);
                DataCopyParams xb{N,1,1,uint16_t(RP/8-1)};DataCopy(mat[N],temp[8],xb);CreateVecIndex(piv,int32_t(0),N);PipeBarrier<PIPE_V>();
                for(uint32_t lane=0;lane<4;++lane){Tx(wg[uint64_t(lane)*ELEMENTS],mat,N*RP);Tx(wg[uint64_t(lane)*ELEMENTS+N*RP],piv.ReinterpretCast<float>(),N);}
                auto domain=temp.ReinterpretCast<uint32_t>();for(uint32_t q=0;q<8;++q)domain.SetValue(q,0);domain.SetValue(0,0x53474a46U);domain.SetValue(1,uint32_t(factorBad));Tx(wg[N*RP+200],domain.ReinterpretCast<float>(),8);
            }
            SyncAll<true>();
        }
    }
private:
    __aicore__ inline uint32_t RawOffset(uint32_t p){return ((p/B)&1)?Lanen64_GJ_RAW1:Lanen64_GJ_RAW;}
    __aicore__ inline void Tx(GlobalTensor<float>d,LocalTensor<float>s,uint32_t n){Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();DataCopy(d,s,n);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();}
    __aicore__ inline void Rx(LocalTensor<float>d,GlobalTensor<float>s,uint32_t n){Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopy(d,s,n);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();}
    __aicore__ inline void RxRows(LocalTensor<float>d,GlobalTensor<float>s,DataCopyExtParams p){Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopyPadExtParams<float>pad{false,0,0,0};DataCopyPad(d,s,p,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();}
    __aicore__ inline void FactorBlock(uint32_t p,uint32_t end){
        // Natural row order CP=[U | invL*P], only below-pivot rows update. P is folded in the right identity.
        constexpr uint32_t R=40,PREFIX=6;const uint32_t w=end-p;
        auto cp=temp,up=temp[640],lo=temp[1024],recip=temp[1408],ones=temp[1536];auto zi=temp[1544].ReinterpretCast<uint32_t>();auto dg=temp[1552],negative=temp[1560],iv=temp[1568],absolute=temp[1584];auto exponent=temp[1600].ReinterpretCast<int32_t>();auto ef=temp[1616],limits=temp[1632];
        Duplicate(temp,0.0f,1664);PipeBarrier<PIPE_V>();Duplicate(recip,1.0f,B*8);Duplicate(ones,1.0f,8);PipeBarrier<PIPE_V>();
        DataCopyParams toCP{B,B/8,0,(R-B)/8},toUP{B,B/8,0,(CRP-B)/8};DataCopy(cp,eye,toCP);DataCopy(cp[B],eye,toCP);DataCopy(up,eye,toUP);PipeBarrier<PIPE_V>();DataCopyParams raw{uint16_t(w),uint16_t(w/8),uint16_t((RP-w)/8),uint16_t((R-w)/8)};DataCopy(cp,mat[p*RP+p],raw);PipeBarrier<PIPE_V>();
        for(uint32_t j=0;j<w;++j){
            if(j<PREFIX){
                Gather(coef,cp,index,j*4,B);PipeBarrier<PIPE_V>();Abs(row,coef,B);PipeBarrier<PIPE_V>();if(j)Duplicate(row,0.0f,j);if(w<B)Duplicate(row[w],0.0f,B-w);PipeBarrier<PIPE_V>();WholeReduceMax(maximum,row,int32_t(B),uint8_t(1),1,1,8);Fence<HardEvent::V_S>();uint32_t bits,choice;GetReduceRepeatMaxMinSpr<float>(bits,choice);if(choice<j||choice>=w)choice=j;const float value=1.0f/coef.GetValue(choice);factorBad=factorBad|(value==0.0f)|!(value<=3.402823466e38f&&value>=-3.402823466e38f);Duplicate(recip[j*8],value,8);PipeBarrier<PIPE_V>();
                if(choice!=j){DataCopy(row,cp[j*R],2*B);DataCopy(other,cp[choice*R],2*B);PipeBarrier<PIPE_V>();DataCopy(cp[j*R],other,2*B);DataCopy(cp[choice*R],row,2*B);PipeBarrier<PIPE_V>();}
            }else{Gather(dg,cp,zi,(j*R+j)*4,8);PipeBarrier<PIPE_V>();Div(recip[j*8],ones,dg,8);PipeBarrier<PIPE_V>();}
            if(j+1<w){Gather(coef,cp,index,j*4,B);PipeBarrier<PIPE_V>();Muls(negative,recip[j*8],-1.0f,8);PipeBarrier<PIPE_V>();Mul(coef,coef,negative,uint64_t(B),uint8_t(1),{1,1,0,2,2,0});PipeBarrier<PIPE_V>();Brcb(column,coef,2,{1,8});PipeBarrier<PIPE_V>();MulAddDst(cp[(j+1)*R],column[(j+1)*8],cp[j*R],uint64_t(2*B),uint8_t(w-j-1),{1,0,1,R/8,1,0});PipeBarrier<PIPE_V>();}
        }
        // One finite/range decision per block; no scalar diagonal load for the fixed tail.
        Gather(iv,recip,index[32],0,B);PipeBarrier<PIPE_V>();Abs(absolute,iv,B);PipeBarrier<PIPE_V>();ShiftRight(exponent,absolute.ReinterpretCast<int32_t>(),int32_t(23),B);PipeBarrier<PIPE_V>();Cast(ef,exponent,RoundMode::CAST_NONE,B);PipeBarrier<PIPE_V>();WholeReduceMax(limits,ef,int32_t(B),uint8_t(1),1,1,8,ReduceOrder::ORDER_ONLY_VALUE);WholeReduceMin(limits[8],ef,int32_t(B),uint8_t(1),1,1,8,ReduceOrder::ORDER_ONLY_VALUE);Fence<HardEvent::V_S>();factorBad=factorBad|(limits.GetValue(0)>=255.0f)|(limits.GetValue(8)<=0.0f);
        for(int j=int(w)-1;j>=0;--j){Mul(up[uint32_t(j)*CRP],up[uint32_t(j)*CRP],recip[uint32_t(j)*8],uint64_t(B),uint8_t(1),{1,1,0,CRP/8,CRP/8,1});PipeBarrier<PIPE_V>();if(j){Gather(coef,cp,index,uint32_t(j)*4,B);PipeBarrier<PIPE_V>();Muls(coef,coef,-1.0f,B);PipeBarrier<PIPE_V>();Brcb(column,coef,2,{1,8});PipeBarrier<PIPE_V>();MulAddDst(up,column,up[uint32_t(j)*CRP],uint64_t(B),uint8_t(j),{1,0,1,CRP/8,1,0});PipeBarrier<PIPE_V>();}}
        DataCopyParams saveL{B,B/8,(R-B)/8,(CRP-B)/8};DataCopy(lo,cp[B],saveL);PipeBarrier<PIPE_V>();Tx(product[Lanen64_GJ_UP+(p/B)*B*CRP],up,B*CRP);Tx(product[Lanen64_GJ_LP+(p/B)*B*CRP],lo,B*CRP);Muls(up,up,-1.0f,B*CRP);PipeBarrier<PIPE_V>();Tx(product[Lanen64_GJ_NEGUP+(p/B)*B*CRP],up,B*CRP);
    }
    __aicore__ inline void CubeLoad(GlobalTensor<float>aa,uint32_t ap,GlobalTensor<float>bb,uint32_t bp,uint32_t m,uint32_t n){
        uint32_t mpp=(m+15)/16*16,npp=(n+15)/16*16;
        LocalTensor<float>a1(TPosition::A1,0,128*16),b1(TPosition::B1,128*16*4,128*16),a2(TPosition::A2,0,128*16),b2(TPosition::B2,0,128*16);
        Fence<HardEvent::M_MTE1>();Fence<HardEvent::MTE1_MTE2>();Nd2NzParams pa{1,uint16_t(m),16,0,uint16_t(ap),uint16_t(mpp),1,0},pb{1,16,uint16_t(n),0,uint16_t(bp),16,1,0};DataCopy(a1,aa,pa);DataCopy(b1,bb,pb);Fence<HardEvent::MTE2_MTE1>();
        LoadData2DParams la;la.startIndex=0;la.repeatTimes=2;la.srcStride=mpp/16;la.dstGap=0;la.ifTranspose=false;for(uint32_t i=0;i<mpp/16;++i)LoadData(a2[i*256],a1[i*128],la);
        LoadData2dTransposeParams lb;lb.startIndex=0;lb.repeatTimes=npp/16;lb.srcStride=1;lb.dstGap=0;lb.dstFracGap=npp/16-1;LoadDataWithTranspose(b2,b1,lb);Fence<HardEvent::MTE1_M>();
    }
    __aicore__ inline void CubeTmp(GlobalTensor<float>out,uint32_t op,GlobalTensor<float>aa,uint32_t ap,GlobalTensor<float>bb,uint32_t bp,uint32_t m,uint32_t n){
        CubeLoad(aa,ap,bb,bp,m,n);LocalTensor<float>a2(TPosition::A2,0,128*16),b2(TPosition::B2,0,128*16),cc(TPosition::CO1,65536,128*128);MmadParams mm;mm.m=m;mm.n=n;mm.k=16;mm.cmatrixInitVal=true;Mmad(cc,a2,b2,mm);Fence<HardEvent::M_FIX>();FixpipeParamsV220 fp;fp.ndNum=1;fp.srcNdStride=0;fp.dstNdStride=0;fp.mSize=m;fp.nSize=n;fp.srcStride=(m+15)/16*16;fp.dstStride=op;Fixpipe(out,cc,fp);Fence<HardEvent::FIX_MTE2>();
    }
    __aicore__ inline void CubeInitialize(){
        LocalTensor<float>a2(TPosition::A2,0,128*16),b2(TPosition::B2,0,128*16),acc(TPosition::CO1,0,112*112);
        for(uint32_t q=0;q<cols;q+=16){CubeLoad(product[r0*RP+c0+q],RP,product[Lanen64_GJ_ID+(c0+q)*RP+c0],RP,rows,cols);MmadParams mm;mm.m=rows;mm.n=cols;mm.k=16;mm.cmatrixInitVal=q==0;Mmad(acc,a2,b2,mm);}
    }
    __aicore__ inline void NormalizeRows(uint32_t p){
        CubeTmp(product[Lanen64_GJ_TMP+c0],RP,product[Lanen64_GJ_LP+(p/B)*B*CRP],CRP,product[RawOffset(p)+c0],RP,B,cols);
        CubeTmp(product[Lanen64_GJ_NORM+c0],RP,product[Lanen64_GJ_NEGUP+(p/B)*B*CRP],CRP,product[Lanen64_GJ_TMP+c0],RP,B,cols);
    }
    __aicore__ inline void Accumulate(GlobalTensor<float>aa,uint32_t ap,GlobalTensor<float>bb,uint32_t bp){
        CubeLoad(aa,ap,bb,bp,rows,cols);LocalTensor<float>a2(TPosition::A2,0,128*16),b2(TPosition::B2,0,128*16),acc(TPosition::CO1,0,112*112);MmadParams mm;mm.m=rows;mm.n=cols;mm.k=B;mm.cmatrixInitVal=false;Mmad(acc,a2,b2,mm);
    }
    __aicore__ inline void UpdateAll(uint32_t p){
        // Identity is negative. Norm is negative. Clear selected old rows exactly, update others, restore selected N.
        Accumulate(product[Lanen64_GJ_ID+r0*RP+p],RP,product[RawOffset(p)+c0],RP);
        Accumulate(product[Lanen64_GJ_C+r0*RP+p],RP,product[Lanen64_GJ_NORM+c0],RP);
        Accumulate(product[Lanen64_GJ_ID+r0*RP+p],RP,product[Lanen64_GJ_NORM+c0],RP);
    }
    __aicore__ inline void PublishPanel(uint32_t p){
        LocalTensor<float>acc(TPosition::CO1,0,112*112);Fence<HardEvent::M_FIX>();FixpipeParamsV220 fp;fp.ndNum=1;fp.srcNdStride=0;fp.dstNdStride=0;fp.srcStride=mp;fp.dstStride=RP;
        if(p>=r0&&p<r0+rows){fp.mSize=N-p<B?N-p:B;fp.nSize=cols;Fixpipe(product[RawOffset(p)+c0],acc[(p-r0)*16],fp);}
        if(p>=c0&&p<c0+cols){fp.mSize=rows;fp.nSize=B;Fixpipe(product[Lanen64_GJ_C+r0*RP+p],acc[(p-c0)*mp],fp);}
        Fence<HardEvent::FIX_MTE2>();
    }
    __aicore__ inline void PublishFinal(){
        if((quad&1)==0)return;LocalTensor<float>acc(TPosition::CO1,0,112*112);Fence<HardEvent::M_FIX>();FixpipeParamsV220 fp;fp.ndNum=1;fp.srcNdStride=0;fp.dstNdStride=0;fp.srcStride=mp;fp.dstStride=16;fp.mSize=rows;fp.nSize=16;Fixpipe(product[Lanen64_GJ_FINAL+r0*16],acc[(192-c0)*mp],fp);Fence<HardEvent::FIX_MTE2>();
    }
    bool factorBad=false;
    TPipe pipe;LinalgSolveTilingData t;uint32_t quad,r0,c0,rows,cols,mp;GlobalTensor<float>ag,bg,wg,product;
    LocalTensor<float>mem,mat,row,other,coef,ab,maximum,column,temp,eye;LocalTensor<uint32_t>index;LocalTensor<int32_t>piv,colids;
};

template <bool OnChip, bool Lanen200_UseExtractedResidual=false, bool Lanen200_BlockTri16=false> class Lanen64_SplitGJSolver {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR workspace,
                                const LinalgSolveTilingData &data) {
        userWorkspace=GetUserWorkspace(workspace);
        t = data; n = t.n; k = t.k; np = t.nPad; kp = t.kPad; rp = t.rowPitch;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
        const uint64_t elements = uint64_t(n) * (np + 2 * kp);
        if (!OnChip || (t.dotSize && n >= 128 && n <= 256)) {
            wg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace))
                               + uint64_t(GetBlockIdx()) * elements);
        }
        pipe.InitBuffer(storage, OnChip ? elements * 4 : (t.dotSize ? 2 * n * kp * 4 : 32));
        mat = storage.Get<float>();
        uint32_t scratchFloats = 2 * rp + 9 * np + 8 * kp;
        if (17 * t.dotSize > scratchFloats) scratchFloats = 17 * t.dotSize;
        pipe.InitBuffer(scratch, scratchFloats * 4);
        row = scratch.Get<float>(); other = row[rp]; column = other[rp];
        original = column[8 * np]; v0 = original[np];
        v1 = v0[kp]; v2 = v1[kp]; v3 = v2[kp];
        v4 = v3[kp]; v5 = v4[kp]; v6 = v5[kp]; v7 = v6[kp];
        pipe.InitBuffer(pivotBuffer, np * 4); pivots = pivotBuffer.Get<int32_t>();
        if ((t.paired && !(GetBlockIdx() & 1)) || t.blocked) {
            GlobalTensor<int32_t> pg;
            pg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(GetUserWorkspace(workspace))
                               + uint64_t(GetBlockIdx()) * elements + n * rp);
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            DataCopy(pivots, pg, n);
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        }
        const uint32_t residualRows = n >= 128 ? 16 : 8;
        const uint32_t tileBytes = (6 * residualRows * t.dotSize + 2 * n * kp + 3 * t.dotSize + 5 * residualRows * 8) * 4;
        uint32_t triangleBytes = (!OnChip && t.dotSize && n <= 256) ? n * (np + 8) * 2 : 32;
        if (t.dotSize && n > 16 && n <= 64) triangleBytes = tileBytes;
        if (!OnChip && t.dotSize && n >= 128 && n <= 256 && tileBytes > triangleBytes) triangleBytes = tileBytes;
        pipe.InitBuffer(triangleBuffer, triangleBytes);
        triangle = triangleBuffer.Get<float>();
        if(t.originalSchur){
            static_assert(2*256*8*4+17*256*4+256*4+256*(256+8)*2+16*256*4<=184*1024,"Block solve UB");
            pipe.InitBuffer(blockSolveBuffer,16*256*4);
        }
        if (t.dotSize) {
            dotIndex = scratch.Get<uint32_t>()[12 * t.dotSize];
            for (uint32_t i = 0; i < t.dotSize; ++i) dotIndex.SetValue(i, i * 4);
        }
    }

    __aicore__ inline void Process(uint32_t Lanen200_forcedBatch=0xffffffffU) {
        const bool splitBlocked = t.blocked && t.rhsSplit;
        const bool splitPaired=t.paired && t.rhsSplit;
        const uint32_t firstBatch = Lanen200_forcedBatch!=0xffffffffU ? Lanen200_forcedBatch : splitPaired ? GetBlockIdx()/(2*PairSlots()) : splitBlocked ? GetBlockIdx()/k : t.paired ? GetBlockIdx() / 2 : GetBlockIdx();
        const uint32_t batchStep = Lanen200_forcedBatch!=0xffffffffU ? t.batches : splitPaired ? t.blocks/(2*PairSlots()) : splitBlocked ? t.blocks/k : t.paired ? t.blocks / 2 : t.blocks;
        for (uint32_t batch = firstBatch; batch < t.batches; batch += batchStep) {
            uint32_t coords = batch, ai = 0, bi = 0;
            for (int d = static_cast<int>(t.batchRank) - 1; d >= 0; --d) {
                uint32_t c = coords % t.batchDims[d]; coords /= t.batchDims[d];
                ai += c * t.aStrides[d]; bi += c * t.bStrides[d];
            }
            ao = uint64_t(ai) * n * n; bo = uint64_t(bi) * n * k;
            if constexpr(Lanen200_BlockTri16)Lanen200_refinePass=0;
            xo = uint64_t(batch) * n * k;
            if (t.blocked) FromGlobal(mat, wg, n * rp);
            else if (!t.paired) { Initialize(); Factor(); }
            if(t.originalSchur)InitialForwardRows();
            BackSolve(false);
            if (t.paired && k >= 2) SolveSync();
            Refine();
            if constexpr(Lanen200_BlockTri16){Lanen200_extraIR=Lanen200_NeedExtraCorrection();if(Lanen200_extraIR)Refine();}
            if (t.dotSize && n <= 256) {
                Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
                DataCopyExtParams out{static_cast<uint16_t>(n), k * 4, OnChip ? np / 8 : 0, 0, 0};
                if(splitPaired){
                    uint32_t c=(GetBlockIdx()/2)%PairSlots();for(uint32_t i=0;i<n;++i)dotIndex.SetValue(i,i*kp*4);Fence<HardEvent::S_V>();
                    AscendC::Gather(original,mat,dotIndex,c*4,n);PipeBarrier<PIPE_V>();Brcb(column,original,uint8_t(n/8),{1,8});PipeBarrier<PIPE_V>();Fence<HardEvent::V_MTE3>();
                    DataCopyExtParams single{uint16_t(n),4,0,(k-1)*4,0};DataCopyPad(xg[xo+c],column,single);
                }else if constexpr (OnChip) {
                    if (splitBlocked) {
                        const uint32_t c=GetBlockIdx()%k;
                        for(uint32_t i=0;i<n;++i) dotIndex.SetValue(i,i*rp*4);
                        Fence<HardEvent::S_V>();
                        AscendC::Gather(original,mat,dotIndex,(np+c)*4,n); PipeBarrier<PIPE_V>();
                        Brcb(column,original,uint8_t(n/8),{1,8}); PipeBarrier<PIPE_V>();
                        Fence<HardEvent::V_MTE3>();
                        DataCopyExtParams single{uint16_t(n),4,0,(k-1)*4,0};
                        DataCopyPad(xg[xo+c],column,single);
                    } else DataCopyPad(xg[xo], mat[np], out);
                }
                else DataCopyPad(xg[xo], mat, out);
                Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
            } else {
                for (uint32_t i = 0; i < n; ++i) {
                    Load(row, i * rp + np, kp);
                    ToGlobal(xg[xo + uint64_t(i) * k], row, k);
                }
            }
            if constexpr(Lanen200_BlockTri16){auto pk=scratch.Get<uint32_t>();for(uint32_t i=0;i<8;++i)pk.SetValue(i,0);pk.SetValue(0,0x4e324952U);pk.SetValue(1,uint32_t(Lanen200_extraIR));ToGlobal(wg[n*rp+208],pk.ReinterpretCast<float>(),8);}
        }
    }

    __aicore__ inline void ResidualWorker(GM_ADDR workspace) {
        uint32_t coords=t.rhsSplit?GetBlockIdx()/(2*PairSlots()):GetBlockIdx()/2,ai=0,bi=0;
        for(int d=int(t.batchRank)-1;d>=0;--d) { uint32_t c=coords%t.batchDims[d];coords/=t.batchDims[d];ai+=c*t.aStrides[d];bi+=c*t.bStrides[d]; }
        ao=uint64_t(ai)*n*n;bo=uint64_t(bi)*n*k;
        SolveSync();
        GlobalTensor<float> solved;
        const uint64_t elements=uint64_t(n)*(np+2*kp);
        solved.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace))+uint64_t(GetBlockIdx()-1)*elements+np);
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
        DataCopyExtParams p{uint16_t(n),kp*4,np*4,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(mat,solved,p,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        TiledResidual();SolveSync();
    }

private:
    __aicore__ inline void Lanen64_Replay(){
        constexpr uint32_t N=200,D=256,B=16,CP=24,CS=13*B*CP;
        auto compact=scratch.Get<float>(),lc=compact[D],uc=lc[B*CP],prod=uc[B*CP],sh=prod[N*8],sl=sh[D],ql=sl[D],qu=ql[64];auto ix=scratch.Get<int32_t>()[14*D];
        auto rr=ResidualCache();const uint32_t rhs=GetBlockIdx()%4;Duplicate(compact,0.0f,D);PipeBarrier<PIPE_V>();CreateVecIndex(ix,int32_t(0),N);PipeBarrier<PIPE_V>();Muls(ix,ix,int32_t(8*4),N);PipeBarrier<PIPE_V>();AscendC::Gather(compact,rr,ix.ReinterpretCast<uint32_t>(),rhs*4,N);PipeBarrier<PIPE_V>();
        GlobalTensor<float>shared;shared.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWorkspace)+uint64_t(t.blocks)*N*216+uint64_t(GetBlockIdx()/4*4)*N*208);
        for(uint32_t p=0;p<N;p+=B){uint32_t width=N-p<B?N-p:B;FromGlobal(lc,shared[Lanen64_GJ_LP+(p/B)*B*CP],B*CP);FromGlobal(uc,shared[Lanen64_GJ_UP+(p/B)*B*CP],B*CP);
            Mul(prod,lc,compact[p],uint64_t(64),uint8_t(2),{1,CP/8,0,8,CP,0});PipeBarrier<PIPE_V>();BlockReduceSum(sh,prod,int32_t(2),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();Mul(prod,lc[8],compact[p+8],uint64_t(64),uint8_t(2),{1,CP/8,0,8,CP,0});PipeBarrier<PIPE_V>();BlockReduceSum(sl,prod,int32_t(2),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();Add(ql,sh,sl,B);PipeBarrier<PIPE_V>();
            Mul(prod,uc,ql,uint64_t(64),uint8_t(2),{1,CP/8,0,8,CP,0});PipeBarrier<PIPE_V>();BlockReduceSum(sh,prod,int32_t(2),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();Mul(prod,uc[8],ql[8],uint64_t(64),uint8_t(2),{1,CP/8,0,8,CP,0});PipeBarrier<PIPE_V>();BlockReduceSum(sl,prod,int32_t(2),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();Add(qu,sh,sl,B);PipeBarrier<PIPE_V>();
            Mul(prod,mat[p],qu,uint64_t(64),uint8_t(N/8),{1,208/8,0,8,208,0});PipeBarrier<PIPE_V>();BlockReduceSum(sh,prod,int32_t(N/8),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();
            if(width==16){Mul(prod,mat[p+8],qu[8],uint64_t(64),uint8_t(N/8),{1,208/8,0,8,208,0});PipeBarrier<PIPE_V>();BlockReduceSum(sl,prod,int32_t(N/8),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();Add(sh,sh,sl,N);PipeBarrier<PIPE_V>();}
            Sub(compact,compact,sh,N);PipeBarrier<PIPE_V>();DataCopy(compact[p],qu,B);PipeBarrier<PIPE_V>();
        }
        CompactScatter(rr,8,compact);
    }
    __aicore__ inline uint32_t PairSlots(){return t.originalSchur?t.pairsPerMatrix:k;}
    __aicore__ inline void SolveSync(){
        if(t.originalSchur){CrossCoreSetFlag<1,PIPE_MTE3>(4);CrossCoreWaitFlag<1>(4);}else SyncAll<true>();
    }
    __aicore__ inline void InitialForwardRows(){
        if constexpr(!OnChip){
            LoadRightCache();DataCopy(mat[n*kp],mat,n*kp);PipeBarrier<PIPE_V>();CompactForwardSolve();DataCopy(mat,mat[n*kp],n*kp);PipeBarrier<PIPE_V>();StoreRightCache();
        }
    }
    __aicore__ inline void FromGlobal(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t count) {
        Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
        Fence<HardEvent::MTE3_MTE2>();
        DataCopyExtParams p{1, count * 4, 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, 0, 0, 0};
        DataCopyPad(dst, src, p, pad);
        Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline void ToGlobal(GlobalTensor<float> dst, LocalTensor<float> src, uint32_t count) {
        Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
        DataCopyExtParams p{1, count * 4, 0, 0, 0};
        DataCopyPad(dst, src, p);
        Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
    }
    __aicore__ inline void LocalCopy(LocalTensor<float> dst, LocalTensor<float> src, uint32_t count) {
        Fence<HardEvent::S_V>();
        DataCopy(dst, src, count);
        PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
    }
    __aicore__ inline void Load(LocalTensor<float> dst, uint32_t offset, uint32_t count) {
        if constexpr (OnChip) LocalCopy(dst, mat[offset], count);
        else FromGlobal(dst, wg[offset], count);
    }
    __aicore__ inline void Store(uint32_t offset, LocalTensor<float> src, uint32_t count) {
        if constexpr (OnChip) LocalCopy(mat[offset], src, count);
        else ToGlobal(wg[offset], src, count);
    }
    __aicore__ inline void LoadR(LocalTensor<float> dst, uint32_t i) { Load(dst, n * rp + i * kp, kp); }
    __aicore__ inline void StoreR(uint32_t i, LocalTensor<float> src) { Store(n * rp + i * kp, src, kp); }
    __aicore__ inline void Ax(LocalTensor<float> dst, LocalTensor<float> src, float a, uint32_t count) {
        Fence<HardEvent::S_V>();
        Axpy(dst, src, a, count); PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
    }
    __aicore__ inline void Scale(LocalTensor<float> dst, float a, uint32_t count) {
        Fence<HardEvent::S_V>();
        Muls(dst, dst, a, count); PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
    }
    __aicore__ inline void Gather(uint32_t j, uint32_t start, uint32_t end) {
        if constexpr (!OnChip) {
            if (end == start) return;
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            Fence<HardEvent::MTE3_MTE2>();
            DataCopyExtParams p{static_cast<uint16_t>(end - start), 4, (rp - 1) * 4, 0, 0};
            DataCopyPadExtParams<float> pad{true, 0, 0, 0};
            DataCopyPad(column, wg[start * rp + j], p, pad);
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        }
    }
    __aicore__ inline float Coeff(uint32_t i, uint32_t j, uint32_t start) {
        if constexpr (OnChip) return mat.GetValue(i * rp + j);
        else return column.GetValue((i - start) * 8);
    }
    __aicore__ inline void Initialize() {
        if constexpr (OnChip) {
            if (t.dotSize && n <= 256) {
                Duplicate(mat, 0.0f, n * (np + 2 * kp)); PipeBarrier<PIPE_V>();
                Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
                DataCopyExtParams pa{static_cast<uint16_t>(n), n * 4, 0, kp / 8, 0};
                DataCopyExtParams pb{static_cast<uint16_t>(n), k * 4, 0, np / 8, 0};
                DataCopyPadExtParams<float> pad{true, 0, 0, 0};
                DataCopyPad(mat, ag[ao], pa, pad); DataCopyPad(mat[np], bg[bo], pb, pad);
                Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>(); return;
            }
        }
        for (uint32_t i = 0; i < n; ++i) {
            Duplicate(row, 0.0f, rp); PipeBarrier<PIPE_V>();
            FromGlobal(row, ag[ao + uint64_t(i) * n], n);
            FromGlobal(row[np], bg[bo + uint64_t(i) * k], k);
            Store(i * rp, row, rp);
        }
    }
    __aicore__ inline void EliminateRows(LocalTensor<float> target, uint32_t start,
                                         uint32_t rows, uint32_t pivot) {
        BinaryRepeatParams update{1, 0, 1, static_cast<uint8_t>(rp / 8), 1, 0};
        for (uint32_t offset = 0; offset < rp; offset += 64) {
            const uint64_t mask = rp - offset < 64 ? rp - offset : 64;
            MulAddDst(target[offset], column[start * 8], row[offset], mask,
                      static_cast<uint8_t>(rows), update);
            PipeBarrier<PIPE_V>();
        }
        uint64_t mask[2] = {1ULL << (pivot & 7), 0};
        UnaryRepeatParams retain{1, 1, static_cast<uint8_t>(rp / 8), 1};
        Muls(target[(pivot / 8) * 8], column[start * 8], -1.0f, mask,
             static_cast<uint8_t>(rows), retain);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void FactorRows() {
        auto compact = scratch.Get<float>()[13 * t.dotSize];
        auto maximum = scratch.Get<float>()[14 * t.dotSize];
        for (uint32_t i = 0; i < n; ++i)
            dotIndex.SetValue(i, i * (OnChip ? rp * 4 : 32));
        Fence<HardEvent::S_V>();
        for (uint32_t j = 0; j < n; ++j) {
            Duplicate(compact, 0.0f, np); PipeBarrier<PIPE_V>();
            if constexpr (OnChip) {
                AscendC::Gather(compact, mat, dotIndex, j * 4, n);
            } else {
                Gather(j, 0, n);
                AscendC::Gather(compact, column, dotIndex, 0, n);
            }
            PipeBarrier<PIPE_V>();
            Abs(original, compact, np); PipeBarrier<PIPE_V>();
            if (j) { Duplicate(original, 0.0f, j); PipeBarrier<PIPE_V>(); }
            AscendC::ReduceMax(maximum, original, column, static_cast<int32_t>(n), true);
            PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
            const uint32_t pivot = maximum.ReinterpretCast<uint32_t>().GetValue(1);
            const float diagonal = compact.GetValue(pivot);
            if (pivot != j) {
                compact.SetValue(pivot, compact.GetValue(j));
                compact.SetValue(j, diagonal);
            }
            pivots.SetValue(j, pivot);
            Load(row, j * rp, rp);
            if (pivot != j) {
                Load(other, pivot * rp, rp);
                Store(pivot * rp, row, rp); Store(j * rp, other, rp);
                LocalCopy(row, other, rp);
            }
            Fence<HardEvent::S_V>();
            Duplicate(row, 0.0f, j + 1); PipeBarrier<PIPE_V>();
            Muls(compact, compact, -1.0f / diagonal, np); PipeBarrier<PIPE_V>();
            Brcb(column, compact, static_cast<uint8_t>(np / 8), {1, 8});
            PipeBarrier<PIPE_V>();
            if constexpr (OnChip) {
                if (j + 1 < n) EliminateRows(mat[(j + 1) * rp], j + 1, n - j - 1, j);
            } else {
                const uint32_t tileRows = (n * (np + 8) / 2) / rp;
                for (uint32_t first = j + 1; first < n; first += tileRows) {
                    const uint32_t count = n - first < tileRows ? n - first : tileRows;
                    FromGlobal(triangle, wg[first * rp], count * rp);
                    EliminateRows(triangle, first, count, j);
                    ToGlobal(wg[first * rp], triangle, count * rp);
                }
            }
        }
    }

    __aicore__ inline void Factor() {
        if (t.dotSize && n <= 256) { FactorRows(); return; }
        for (uint32_t j = 0; j < n; ++j) {
            Gather(j, j, n);
            uint32_t pivot = j; float largest = -1.0f;
            for (uint32_t i = j; i < n; ++i) {
                float value = Coeff(i, j, j); float ab = value < 0.0f ? -value : value;
                if (ab > largest) { largest = ab; pivot = i; }
            }
            pivots.SetValue(j, pivot);
            Load(row, j * rp, rp);
            if (pivot != j) {
                Load(other, pivot * rp, rp);
                Store(pivot * rp, row, rp); Store(j * rp, other, rp);
                LocalCopy(row, other, rp);
            }
            const float inverse = 1.0f / row.GetValue(j);
            // Keep previous L columns, and write the new multiplier explicitly.
            Fence<HardEvent::S_V>();
            Duplicate(row, 0.0f, j + 1); PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
            for (uint32_t i = j + 1; i < n; ++i) {
                Load(other, i * rp, rp);
                const float multiplier = other.GetValue(j) * inverse;
                Ax(other, row, -multiplier, rp);
                other.SetValue(j, multiplier);
                Store(i * rp, other, rp);
            }
        }
    }
    __aicore__ inline void BackSolve(bool residual){return;}
    // Pack only the needed triangular coefficients when the full LU exceeds UB.
    __aicore__ inline void PrepareTriangle(bool lower) {
        if constexpr(!OnChip){if(t.originalSchur){PrepareBlockTriangle(lower);return;}}
        if constexpr (OnChip) {
            for (uint32_t i = 0; i < n; ++i) dotIndex.SetValue(i, i * rp * 4);
            Fence<HardEvent::S_V>();
        } else {
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            Fence<HardEvent::MTE3_MTE2>();
            uint32_t offset = 0;
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t length = lower ? i : n - i;
                dotIndex.SetValue(i, (offset - (lower ? 0 : i)) * 4);
                if (length) {
                    DataCopyExtParams p{1, length * 4, 0, 0, 0};
                    DataCopyPadExtParams<float> pad{true, 0, 0, 0};
                    DataCopyPad(triangle[offset], wg[i * rp + (lower ? 0 : i)], p, pad);
                    offset += (length + 7) / 8 * 8;
                }
            }
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
            Fence<HardEvent::S_V>();
        }
    }
    __aicore__ inline LocalTensor<float> TriangleColumn(uint32_t j) {
        auto compact = scratch.Get<float>()[13 * t.dotSize];
        Duplicate(compact, 0.0f, np); PipeBarrier<PIPE_V>();
        if constexpr (OnChip) AscendC::Gather(compact, mat, dotIndex, j * 4, n);
        else AscendC::Gather(compact, triangle, dotIndex, j * 4, n);
        PipeBarrier<PIPE_V>();
        return compact;
    }
    __aicore__ inline void BroadcastNegative(LocalTensor<float> compact) {
        Muls(compact, compact, -1.0f, np); PipeBarrier<PIPE_V>();
        Brcb(column, compact, static_cast<uint8_t>(np / 8), {1, 8});
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void UpdateRows(LocalTensor<float> rhs, uint32_t stride,
                                      uint32_t start, uint32_t rows, uint32_t pivot) {
        // Eight RHS rows fill the 64 FP32 lanes. The pivot RHS is reused per block.
        const uint32_t groups = rows / 8, tail = rows % 8;
        const uint8_t blocks = static_cast<uint8_t>(stride / 8);
        BinaryRepeatParams p{blocks, 1, 0, static_cast<uint8_t>(8 * blocks), 8, 0};
        if (groups) {
            MulAddDst(rhs[start * stride], column[start * 8], rhs[pivot * stride],
                      uint64_t(64), static_cast<uint8_t>(groups), p);
            PipeBarrier<PIPE_V>();
        }
        if (tail) {
            const uint32_t first = start + groups * 8;
            MulAddDst(rhs[first * stride], column[first * 8], rhs[pivot * stride],
                      uint64_t(tail * 8), uint8_t(1), p);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void LoadRightCache() {
        if constexpr (!OnChip) {
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
            Fence<HardEvent::MTE3_MTE2>();
            DataCopyExtParams p{static_cast<uint16_t>(n), kp * 4, np * 4, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(mat, wg[np], p, pad);
            Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        }
    }
    __aicore__ inline void StoreRightCache() {
        if constexpr (!OnChip) {
            Fence<HardEvent::V_MTE3>(); Fence<HardEvent::S_MTE3>();
            DataCopyExtParams p{static_cast<uint16_t>(n), kp * 4, 0, np * 4, 0};
            DataCopyPad(wg[np], mat, p);
            Fence<HardEvent::MTE3_V>(); Fence<HardEvent::MTE3_S>();
        }
    }
    __aicore__ inline void BackSolveRows(bool residual) {
        if(t.rhsSplit && (t.paired || t.blocked)){CompactBackSolve(residual);return;}
        if (!residual) LoadRightCache();
        LocalTensor<float> rhs = mat;
        uint32_t stride = kp;
        if constexpr (OnChip) { rhs = residual ? mat[n * rp] : mat[np]; stride = residual ? kp : rp; }
        else if (residual) rhs = mat[n * kp];
        PrepareTriangle(false);
        for (int j = static_cast<int>(n) - 1; j >= 0; --j) {
            auto compact = TriangleColumn(j); Fence<HardEvent::V_S>();
            const float inverse = 1.0f / compact.GetValue(j);
            Muls(rhs[j * stride], rhs[j * stride], inverse, kp); PipeBarrier<PIPE_V>();
            BroadcastNegative(compact);
            UpdateRows(rhs, stride, 0, j, j);
        }
        if constexpr (!OnChip) {
            if (residual) ToGlobal(wg[n * rp], rhs, n * kp);
            else StoreRightCache();
        }
    }
    __aicore__ inline LocalTensor<float> ResidualCache() {
        if constexpr (OnChip) return mat[n * rp];
        else return mat[n * kp];
    }
    __aicore__ inline void PermuteResidualRows() {
        auto rhs = ResidualCache();
        if(t.originalSchur){
            auto index=scratch.Get<int32_t>()[14*t.dotSize];auto compact=scratch.Get<float>();
            Muls(index,pivots,int32_t(kp*4),n);PipeBarrier<PIPE_V>();
            AscendC::Gather(compact,rhs,index.ReinterpretCast<uint32_t>(),((GetBlockIdx()/2)%PairSlots())*4,n);PipeBarrier<PIPE_V>();CompactScatter(rhs,kp,compact);return;
        }
        for (uint32_t j = 0; j < n; ++j) {
            uint32_t pivot = pivots.GetValue(j);
            if (pivot != j) {
                DataCopy(row, rhs[j * kp], kp); PipeBarrier<PIPE_V>();
                DataCopy(other, rhs[pivot * kp], kp); PipeBarrier<PIPE_V>();
                DataCopy(rhs[pivot * kp], row, kp); PipeBarrier<PIPE_V>();
                DataCopy(rhs[j * kp], other, kp); PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void ForwardSolveRows(){Lanen64_Replay();}

    __aicore__ inline void CompactScatter(LocalTensor<float> rhs,uint32_t stride,LocalTensor<float> compact){
        const uint32_t c=t.paired?(GetBlockIdx()/2)%PairSlots():GetBlockIdx()%k;
        Brcb(column,compact,uint8_t(np/8),{1,8});PipeBarrier<PIPE_V>();
        uint64_t mask[2]={1ULL<<c,0};
        for(uint32_t first=0;first<n;first+=128){
            const uint32_t rows=n-first<128?n-first:128;
            Muls(rhs[first*stride],column[first*8],1.0f,mask,uint8_t(rows),{1,1,uint8_t(stride/8),1});PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void PrepareBlockTriangle(bool lower){
        constexpr uint32_t N=256;const uint32_t matrix=GetBlockIdx()/8;
        GlobalTensor<float>tg;tg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace)+t.schurOffset/4+uint64_t(matrix)*kOS_Total+kOS_S);
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();
        uint32_t offset=0;
        for(uint32_t j=0;j<N;++j){
            uint32_t first=lower?(j+1)/8*8:0,length=lower?N-first:j+1;
            dotIndex.SetValue(j,(offset-(lower?first:0))*4);
            if(length){DataCopyExtParams p{1,length*4,0,0,0};DataCopyPadExtParams<float>pad{true,0,0,0};DataCopyPad(triangle[offset],tg[j*N+first],p,pad);offset+=(length+7)/8*8;}
        }
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();Fence<HardEvent::S_V>();
    }
    __aicore__ inline void BlockTriangleSolve(bool residual,bool lower){
        constexpr uint32_t N=256,D=256,B=64;
        if(!residual && !lower)LoadRightCache();
        auto rhs=residual?ResidualCache():mat;const uint32_t c=(GetBlockIdx()/2)%PairSlots();
        auto compact=scratch.Get<float>(),saved=scratch.Get<float>()[D];auto rowBytes=scratch.Get<int32_t>()[11*D];
        auto values=scratch.Get<float>()[13*D];auto index=scratch.Get<int32_t>()[14*D];auto inverse=scratch.Get<float>()[15*D];
        auto tile=blockSolveBuffer.Get<float>();auto tileIndex=mat.ReinterpretCast<int32_t>();
        PrepareBlockTriangle(lower);
        CreateVecIndex(index,int32_t(0),N);PipeBarrier<PIPE_V>();Muls(index,index,int32_t(kp*4),N);PipeBarrier<PIPE_V>();
        AscendC::Gather(compact,rhs,index.ReinterpretCast<uint32_t>(),c*4,N);PipeBarrier<PIPE_V>();
        AscendC::Gather(saved,mat,index.ReinterpretCast<uint32_t>(),c*4,N);PipeBarrier<PIPE_V>();
        if(!lower){
            CreateVecIndex(index,int32_t(0),N);PipeBarrier<PIPE_V>();Muls(index,index,int32_t(4),N);PipeBarrier<PIPE_V>();Add(index,index,dotIndex.ReinterpretCast<int32_t>(),N);PipeBarrier<PIPE_V>();
            AscendC::Gather(inverse,triangle,index.ReinterpretCast<uint32_t>(),0,N);PipeBarrier<PIPE_V>();Mul(compact,compact,inverse,N);PipeBarrier<PIPE_V>();
        }
        CreateVecIndex(rowBytes,int32_t(0),N);PipeBarrier<PIPE_V>();Muls(rowBytes,rowBytes,int32_t(4),N);PipeBarrier<PIPE_V>();
        GlobalTensor<float>dg;dg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace)+t.schurOffset/4+uint64_t(GetBlockIdx()/8)*kOS_Total+kOS_Identity);
        for(uint32_t step=0;step<N/B;++step){
            uint32_t block=lower?step:N/B-1-step,first=block*B;
            FromGlobal(tile,dg[block*2*B*B+(lower?B*B:0)],B*B);
            Mul(tile,tile,compact[first],uint64_t(B),uint8_t(B),{1,1,1,B/8,B/8,0});PipeBarrier<PIPE_V>();
            WholeReduceSum(values,tile,int32_t(B),uint8_t(B),1,1,B/8);PipeBarrier<PIPE_V>();
            DataCopy(compact[first],values,B);DataCopy(inverse,values,B);PipeBarrier<PIPE_V>();
            const uint32_t rowStart=lower?first+B:0,rowEnd=lower?N:first;
            for(uint32_t rowFirst=rowStart;rowFirst<rowEnd;rowFirst+=64){
                const uint32_t rows=rowEnd-rowFirst<64?rowEnd-rowFirst:64;
                Muls(tileIndex,dotIndex[first].ReinterpretCast<int32_t>(),int32_t(1),uint64_t(B),uint8_t(rows),{1,1,B/8,0});PipeBarrier<PIPE_V>();
                Brcb(column,rowBytes[rowFirst].ReinterpretCast<float>(),uint8_t(rows/8),{1,8});PipeBarrier<PIPE_V>();
                Add(tileIndex,tileIndex,column.ReinterpretCast<int32_t>(),uint64_t(B),uint8_t(rows),{1,1,0,B/8,B/8,1});PipeBarrier<PIPE_V>();
                AscendC::Gather(tile,triangle,tileIndex.ReinterpretCast<uint32_t>(),0,rows*B);PipeBarrier<PIPE_V>();
                Mul(tile,tile,inverse,uint64_t(B),uint8_t(rows),{1,1,1,B/8,B/8,0});PipeBarrier<PIPE_V>();
                WholeReduceSum(values,tile,int32_t(B),uint8_t(rows),1,1,B/8);PipeBarrier<PIPE_V>();Add(compact[rowFirst],compact[rowFirst],values,rows);PipeBarrier<PIPE_V>();
            }
        }
        CompactScatter(mat,kp,saved);CompactScatter(rhs,kp,compact);
        if(!lower){if(residual)ToGlobal(wg[n*rp],rhs,n*kp);else StoreRightCache();}
    }
    // Same per-element tolerance as acceptance; large first correction requests one more exact-residual IR.
    __aicore__ inline bool Lanen200_NeedExtraCorrection(){
        static_assert(OnChip,"N200 correction guard needsresidentX/delta");constexpr uint32_t D=256;auto xc=scratch.Get<float>(),dc=scratch.Get<float>()[D],error=scratch.Get<float>()[13*D],maximum=scratch.Get<float>()[15*D];auto bytes=scratch.Get<int32_t>()[14*D];
        CreateVecIndex(bytes,int32_t(0),n);PipeBarrier<PIPE_V>();Muls(bytes,bytes,int32_t(rp*4),n);PipeBarrier<PIPE_V>();AscendC::Gather(xc,mat[np],bytes.ReinterpretCast<uint32_t>(),(GetBlockIdx()%4)*4,n);PipeBarrier<PIPE_V>();
        CreateVecIndex(bytes,int32_t(0),n);PipeBarrier<PIPE_V>();Muls(bytes,bytes,int32_t(8*4),n);PipeBarrier<PIPE_V>();AscendC::Gather(dc,ResidualCache(),bytes.ReinterpretCast<uint32_t>(),(GetBlockIdx()%4)*4,n);PipeBarrier<PIPE_V>();
        Duplicate(error,-1.0f,D);Abs(xc,xc,n);Abs(dc,dc,n);PipeBarrier<PIPE_V>();Adds(xc,xc,1.0f,n);PipeBarrier<PIPE_V>();Muls(xc,xc,0.0001f,n);PipeBarrier<PIPE_V>();Sub(error,dc,xc,n);PipeBarrier<PIPE_V>();WholeReduceMax(maximum,error,int32_t(64),uint8_t(4),1,1,8);Fence<HardEvent::V_S>();uint32_t bits,index;GetReduceRepeatMaxMinSpr<float>(bits,index);union Lanen200_GuardBits{uint32_t u;float f;}check;check.u=bits;return !(check.f<=0.0f);
    }
    // Lanen200 ownership: complete MODE15 cached block16 triangle, old instances defaultfalse.
    __aicore__ inline void Lanen200_CachedTriangle(bool residual,bool lower){
        static_assert(OnChip,"N200 cachedtriangle needsresidentLU");
        constexpr uint32_t N=200,D=256,B=16,CP=24,NB=13,CS=NB*B*CP;
        auto compact=scratch.Get<float>(),cache=scratch.Get<float>()[D],product=scratch.Get<float>()[D+B*CP];
        auto sol=scratch.Get<float>()[D+B*CP+32*CP],sum=scratch.Get<float>()[D+B*CP+32*CP+64];
        auto index=scratch.Get<uint32_t>()[D+B*CP+32*CP+128];
        const uint32_t c=GetBlockIdx()%4,stride=residual?8:rp;auto rhs=residual?ResidualCache():mat[N];
        Duplicate(compact,0.0f,D);PipeBarrier<PIPE_V>();
        for(uint32_t i=0;i<D;++i)index.SetValue(i,i*stride*4);Fence<HardEvent::S_V>();
        AscendC::Gather(compact,rhs,index,c*4,N);PipeBarrier<PIPE_V>();
        GlobalTensor<float> cached;const uint64_t elements=N*216;
        cached.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWorkspace)+uint64_t(t.blocks)*elements+uint64_t(GetBlockIdx()/4*4)*N*rp+N*rp+(lower?CS:0));
        for(uint32_t step=0;step<NB;++step){
            uint32_t block=lower?step:NB-1-step,first=block*B,w=N-first<B?N-first:B;
            FromGlobal(cache,cached[block*B*CP],B*CP);
            Mul(product,cache,compact[first],uint64_t(64),uint8_t(2),{1,CP/8,0,8,CP,0});PipeBarrier<PIPE_V>();
            BlockReduceSum(sum,product,int32_t(2),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();
            Mul(product,cache[8],compact[first+8],uint64_t(64),uint8_t(2),{1,CP/8,0,8,CP,0});PipeBarrier<PIPE_V>();
            BlockReduceSum(sum[32],product,int32_t(2),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();
            Add(sol,sum,sum[32],B);PipeBarrier<PIPE_V>();DataCopy(compact[first],sol,B);PipeBarrier<PIPE_V>();
            const uint32_t begin=lower?first+w:0,end=lower?N:first;
            for(uint32_t rowFirst=begin;rowFirst<end;rowFirst+=32){
                uint32_t rows=end-rowFirst<32?end-rowFirst:32;
                Mul(product,mat[rowFirst*rp+first],sol,uint64_t(64),uint8_t(rows/8),{1,uint8_t(rp/8),0,8,uint8_t(rp),0});PipeBarrier<PIPE_V>();
                BlockReduceSum(sum,product,int32_t(rows/8),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();
                if(w==16){Mul(product,mat[rowFirst*rp+first+8],sol[8],uint64_t(64),uint8_t(rows/8),{1,uint8_t(rp/8),0,8,uint8_t(rp),0});PipeBarrier<PIPE_V>();BlockReduceSum(sum[32],product,int32_t(rows/8),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();Add(sum,sum,sum[32],rows);PipeBarrier<PIPE_V>();}
                Sub(compact[rowFirst],compact[rowFirst],sum,rows);PipeBarrier<PIPE_V>();
            }
        }
        CompactScatter(rhs,stride,compact);
    }
    // End Lanen200 cachedtriangle.
    __aicore__ inline void CompactBackSolve(bool residual){
        if constexpr(Lanen200_BlockTri16){Lanen200_CachedTriangle(residual,false);return;}
        if(t.originalSchur){BlockTriangleSolve(residual,false);return;}
        if(!residual)LoadRightCache();
        LocalTensor<float>rhs=mat;uint32_t stride=kp;
        if constexpr(OnChip){rhs=residual?mat[n*rp]:mat[np];stride=residual?kp:rp;}
        else if(residual)rhs=mat[n*kp];
        const uint32_t c=t.paired?(GetBlockIdx()/2)%PairSlots():GetBlockIdx()%k,d=t.dotSize;
        auto compact=scratch.Get<float>();auto coeff=scratch.Get<float>()[13*d];
        auto index=scratch.Get<int32_t>()[14*d];auto inverse=scratch.Get<float>()[15*d];auto broadcast=scratch.Get<float>()[16*d];
        PrepareTriangle(false);
        CreateVecIndex(index,int32_t(0),n);PipeBarrier<PIPE_V>();Muls(index,index,int32_t(4),n);PipeBarrier<PIPE_V>();
        Add(index,index,dotIndex.ReinterpretCast<int32_t>(),n);PipeBarrier<PIPE_V>();
        if constexpr(OnChip)AscendC::Gather(inverse,mat,index.ReinterpretCast<uint32_t>(),0,n);
        else AscendC::Gather(inverse,triangle,index.ReinterpretCast<uint32_t>(),0,n);
        PipeBarrier<PIPE_V>();Duplicate(broadcast,-1.0f,n);PipeBarrier<PIPE_V>();Div(inverse,broadcast,inverse,n);PipeBarrier<PIPE_V>();
        CreateVecIndex(index,int32_t(0),n);PipeBarrier<PIPE_V>();Muls(index,index,int32_t(stride*4),n);PipeBarrier<PIPE_V>();
        AscendC::Gather(compact,rhs,index.ReinterpretCast<uint32_t>(),c*4,n);PipeBarrier<PIPE_V>();
        Duplicate(index,int32_t(0),n);PipeBarrier<PIPE_V>();
        // Uraw = D*Uunit. Scale all equations before unit-upper substitution.
        Mul(compact,compact,inverse,n);PipeBarrier<PIPE_V>();Muls(compact,compact,-1.0f,n);PipeBarrier<PIPE_V>();
        for(int j=int(n)-1;j>0;--j){
            if constexpr(OnChip)AscendC::Gather(coeff,mat,dotIndex,uint32_t(j)*4,n);
            else AscendC::Gather(coeff,triangle,dotIndex,uint32_t(j)*4,n);
            PipeBarrier<PIPE_V>();Mul(coeff,coeff,inverse,n);PipeBarrier<PIPE_V>();
            AscendC::Gather(broadcast,compact,index.ReinterpretCast<uint32_t>(),uint32_t(j)*4,n);PipeBarrier<PIPE_V>();
            MulAddDst(compact,coeff,broadcast,int32_t(j));PipeBarrier<PIPE_V>();
        }
        CompactScatter(rhs,stride,compact);
        if constexpr(!OnChip){if(residual)ToGlobal(wg[n*rp],rhs,n*kp);else StoreRightCache();}
    }
    __aicore__ inline void CompactForwardSolve(){
        if constexpr(Lanen200_BlockTri16){Lanen200_CachedTriangle(true,true);return;}
        if(t.originalSchur){BlockTriangleSolve(true,true);return;}
        auto rhs=ResidualCache();PrepareTriangle(true);
        const uint32_t c=t.paired?(GetBlockIdx()/2)%PairSlots():GetBlockIdx()%k,d=t.dotSize;
        auto compact=scratch.Get<float>();auto coeff=scratch.Get<float>()[13*d];
        auto index=scratch.Get<int32_t>()[14*d];auto broadcast=scratch.Get<float>()[16*d];
        CreateVecIndex(index,int32_t(0),n);PipeBarrier<PIPE_V>();Muls(index,index,int32_t(kp*4),n);PipeBarrier<PIPE_V>();
        AscendC::Gather(compact,rhs,index.ReinterpretCast<uint32_t>(),c*4,n);PipeBarrier<PIPE_V>();
        Duplicate(index,int32_t(0),n);PipeBarrier<PIPE_V>();
        for(uint32_t j=0;j+1<n;++j){
            if constexpr(OnChip)AscendC::Gather(coeff,mat,dotIndex,j*4,n);
            else AscendC::Gather(coeff,triangle,dotIndex,j*4,n);
            PipeBarrier<PIPE_V>();Muls(coeff,coeff,-1.0f,n);PipeBarrier<PIPE_V>();Duplicate(coeff,0.0f,j+1);PipeBarrier<PIPE_V>();
            AscendC::Gather(broadcast,compact,index.ReinterpretCast<uint32_t>(),j*4,n);PipeBarrier<PIPE_V>();
            MulAddDst(compact,coeff,broadcast,int32_t(n));PipeBarrier<PIPE_V>();
        }
        CompactScatter(rhs,kp,compact);
    }

    __aicore__ inline void CompensatedScalar(float term, float &sum, float &error) {
        volatile float next = sum + term;
        volatile float other = next - sum;
        volatile float restored = next - other;
        volatile float aerror = sum - restored;
        volatile float berror = term - other;
        volatile float roundoff = aerror + berror;
        volatile float low = error + roundoff;
        sum = next; error = low;
    }

    __aicore__ inline void TileMul(LocalTensor<float> dst, LocalTensor<float> a,
                                   LocalTensor<float> b, uint32_t d, uint32_t tile) {
        BinaryRepeatParams p{1, 1, 1, static_cast<uint8_t>(d / 8),
                             static_cast<uint8_t>(d / 8), 0};
        for (uint32_t offset = 0; offset < d; offset += 64) {
            Mul(dst[offset], a[offset], b[offset], uint64_t(d - offset < 64 ? d - offset : 64), static_cast<uint8_t>(tile), p);
        }
        PipeBarrier<PIPE_V>();
    }
    // Lanen256_: tolerance-matched FP32 residual for the N256 Schur family.
    __aicore__ inline void Lanen256_FastResidual() {
        constexpr uint32_t D=256,Tile=16,Count=Tile*D;
        auto aa=triangle;auto pp=aa[Count];auto xc=pp[Count];auto bb=xc[D*8];auto xx=bb[D*8];
        auto sums=xx[D];auto terms=sums[Tile];auto rhsTerms=terms[Tile];
        auto rr=ResidualCache();
        Duplicate(rr,0.0f,D*8);DataCopy(xc,mat,D*8);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();
        DataCopyExtParams pb{D,k*4,0,0,0};DataCopyPadExtParams<float>pad{true,0,0,0};
        DataCopyPad(bb,bg[bo],pb,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        for(uint32_t i=0;i<D;++i)dotIndex.SetValue(i,i*kp*4);Fence<HardEvent::S_V>();
        const uint32_t c=(GetBlockIdx()/2)%PairSlots(),begin=(GetBlockIdx()&1)*(D/2),end=begin+D/2;
        AscendC::Gather(xx,xc,dotIndex,c*4,D);PipeBarrier<PIPE_V>();
        for(uint32_t first=begin;first<end;first+=Tile){
            Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
            DataCopy(aa,ag[ao+uint64_t(first)*D],Count);Fence<HardEvent::MTE2_V>();
            TileMul(pp,aa,xx,D,Tile);
            WholeReduceSum(sums,pp,int32_t(64),uint8_t(Tile),1,1,D/8);PipeBarrier<PIPE_V>();
            for(uint32_t off=64;off<D;off+=64){
                WholeReduceSum(terms,pp[off],int32_t(64),uint8_t(Tile),1,1,D/8);PipeBarrier<PIPE_V>();
                Add(sums,sums,terms,Tile);PipeBarrier<PIPE_V>();
            }
            AscendC::Gather(rhsTerms,bb[first*kp],dotIndex,c*4,Tile);PipeBarrier<PIPE_V>();
            Sub(terms,rhsTerms,sums,Tile);PipeBarrier<PIPE_V>();Brcb(column,terms,uint8_t(Tile/8),{1,8});PipeBarrier<PIPE_V>();
            uint64_t mask[2]={1ULL<<c,0};Muls(rr[first*kp],column,1.0f,mask,uint8_t(Tile),{1,1,uint8_t(kp/8),1});PipeBarrier<PIPE_V>();
        }
        ToGlobal(wg[n*rp],rr,D*8);
    }

    // MODE15 first pass only. Requested later pass retains the full error-free product/extracted sum.
    __aicore__ inline void Lanen200_FirstFP32Residual(){
        static_assert(OnChip,"N200 firstresidual needscanonicalLUinWG");constexpr uint32_t N=200,D=256,Tile=32,Count=Tile*D;
        Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();DataCopyExtParams ps{N,8*4,uint32_t((rp-8)/8),uint32_t((rp-8)*4),0};DataCopyPad(wg[N],mat[N],ps);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
        auto aa=mat,pp=aa[Count],bc=pp[Count],xc=bc[N*8],xx=xc[N*8],blocks=xx[D],sum=blocks[Tile*D/8],values=sum[64];auto rr=ResidualCache();
        static_assert((2*Count+2*N*8+D+Tile*D/8+2*64)*4<N*208*4,"N200 firstresidual workexcludesDelta");
        Duplicate(rr,0.0f,N*8);Duplicate(xx,0.0f,D);Duplicate(sum,0.0f,64);PipeBarrier<PIPE_V>();for(uint32_t i=0;i<D;++i)dotIndex.SetValue(i,i*8*4);Fence<HardEvent::S_V>();
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopyPadExtParams<float>pad{true,0,0,0};DataCopyExtParams px{N,8*4,N*4,0,0},pb{N,4*4,0,0,0};DataCopyPad(xc,wg[N],px,pad);DataCopyPad(bc,bg[bo],pb,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();const uint32_t c=GetBlockIdx()%4;
        AscendC::Gather(xx,xc,dotIndex,c*4,N);PipeBarrier<PIPE_V>();
        for(uint32_t first=0;first<N;first+=Tile){uint32_t rows=N-first<Tile?N-first:Tile;Duplicate(aa,0.0f,Count);PipeBarrier<PIPE_V>();Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();DataCopyExtParams pa{uint16_t(rows),N*4,0,(D-N)/8,0};DataCopyPad(aa,ag[ao+uint64_t(first)*N],pa,pad);Fence<HardEvent::MTE2_V>();TileMul(pp,aa,xx,D,Tile);
            BlockReduceSum(blocks,pp,int32_t(Count/64),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();WholeReduceSum(sum,blocks,int32_t(D/8),uint8_t(Tile),1,1,D/64);PipeBarrier<PIPE_V>();Duplicate(values,0.0f,64);PipeBarrier<PIPE_V>();AscendC::Gather(values,bc[first*8],dotIndex,c*4,rows);PipeBarrier<PIPE_V>();Sub(sum,values,sum,64);PipeBarrier<PIPE_V>();Brcb(column,sum,uint8_t(Tile/8),{1,8});PipeBarrier<PIPE_V>();uint64_t mask[2]={1ULL<<c,0};Muls(rr[first*8],column,1.0f,mask,uint8_t(rows),{1,1,1,1});PipeBarrier<PIPE_V>();
        }
        FromGlobal(mat,wg,N*rp);
    }
    // Lanen200 ownership: TwoProduct plus exact-unit extraction sums, MODE15 only.
    __aicore__ inline void Lanen64_ExtractedRowSum32(LocalTensor<float> dst,LocalTensor<float> src,LocalTensor<float> blocks,uint32_t rows){
        BlockReduceSum(blocks,src,int32_t(rows*4),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();
        WholeReduceSum(dst,blocks,int32_t(32),uint8_t(rows),1,1,4);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool Lanen200_TryExtractedResidual(){
        static_assert(OnChip,"Lanen200 extracted residual retains OnChip LU");
        constexpr uint32_t N=200,D=256,Tile=32,Count=Tile*D;
        if constexpr(Lanen200_BlockTri16){
            Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();
            DataCopyExtParams pxsave{N,8*4,uint32_t((rp-8)/8),uint32_t((rp-8)*4),0};DataCopyPad(wg[N],mat[N],pxsave);
            Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();
        }else ToGlobal(wg,mat,N*rp);
        // A is dead after splitting; its buffer becomes the product error. P is dead after extraction high+rest; it becomes extraction low.
        auto aa=mat;auto pp=aa[Count];auto ss=pp[Count];auto zz=ss[Count];auto tt=zz[Count];auto ee=aa;
        auto xx=scratch.Get<float>();auto xh=xx[D];auto xl=xh[D];auto xc=xl[D];auto bc=xc;
        auto blocks=bc[N*8];auto maxPairs=blocks[Tile*D/8];auto sigma=maxPairs[64];auto expMask=sigma[64];
        auto sumHigh=expMask[64];auto sumLow=sumHigh[64];auto sumTiny=sumLow[64];auto sumErr=sumTiny[64];
        auto rhsValues=sumErr[64];auto lowTotal=rhsValues[64];auto guard=lowTotal[64];auto sigmaBroad=guard[16];
        auto localColumn=mat[5*Count];auto dotIndex=pivots.ReinterpretCast<uint32_t>();auto maxIndex=pivots[128].ReinterpretCast<uint32_t>();
        static_assert(5*Count+Tile*8<=N*208,"Fivebuffers+deadcolumn fitbelow immutable rr");
        static_assert(3*D+N*8+Tile*D/8+9*64+16+Tile*8<=17*D,"Full32row exactresidual scratchfits");
        auto rr=ResidualCache();Duplicate(rr,0.0f,N*8);Duplicate(xx,0.0f,D);Duplicate(xh,0.0f,D);Duplicate(xl,0.0f,D);
        Duplicate(maxPairs,0.0f,64);Duplicate(sumHigh,0.0f,64);Duplicate(sumLow,0.0f,64);Duplicate(sumTiny,0.0f,64);Duplicate(sumErr,0.0f,64);Duplicate(expMask.ReinterpretCast<int32_t>(),int32_t(0x7f800000),64);PipeBarrier<PIPE_V>();
        CreateVecIndex(dotIndex.ReinterpretCast<int32_t>(),int32_t(0),N);PipeBarrier<PIPE_V>();Muls(dotIndex.ReinterpretCast<int32_t>(),dotIndex.ReinterpretCast<int32_t>(),int32_t(8*4),N);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();Fence<HardEvent::MTE3_MTE2>();DataCopyPadExtParams<float> pad{true,0,0,0};DataCopyExtParams px{N,8*4,N*4,0,0};DataCopyPad(xc,wg[N],px,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        const uint32_t c=GetBlockIdx()%4;AscendC::Gather(xx,xc,dotIndex,c*4,N);PipeBarrier<PIPE_V>();
        Muls(xh,xx,4097.0f,D);PipeBarrier<PIPE_V>();Sub(xl,xh,xx,D);PipeBarrier<PIPE_V>();Sub(xh,xh,xl,D);PipeBarrier<PIPE_V>();Sub(xl,xx,xh,D);PipeBarrier<PIPE_V>();
        // Initial X gather has finished. Its last72 indices may become the sigma pair indices; B tiles need only first32.
        CreateVecIndex(maxIndex.ReinterpretCast<int32_t>(),int32_t(0),64);PipeBarrier<PIPE_V>();Muls(maxIndex.ReinterpretCast<int32_t>(),maxIndex.ReinterpretCast<int32_t>(),int32_t(8),64);PipeBarrier<PIPE_V>();
        // xc is now dead; MTE2 may reuse exactly the same range for original B.
        Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();DataCopyExtParams pb{N,4*4,0,0,0};DataCopyPad(bc,bg[bo],pb,pad);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        for(uint32_t first=0;first<N;first+=Tile){
            const uint32_t rows=N-first<Tile?N-first:Tile;const uint32_t liveCount=rows*D;
            Duplicate(aa,0.0f,liveCount);PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();
            DataCopyExtParams pa{uint16_t(rows),N*4,0,(D-N)/8,0};DataCopyPad(aa,ag[ao+uint64_t(first)*N],pa,pad);Fence<HardEvent::MTE2_V>();
            // Preserve the full error-free product sequence used by the Dot2 baseline.
            TileMul(pp,aa,xx,D,rows);
            Muls(tt,aa,4097.0f,liveCount);PipeBarrier<PIPE_V>();Sub(zz,tt,aa,liveCount);PipeBarrier<PIPE_V>();
            Sub(ss,tt,zz,liveCount);PipeBarrier<PIPE_V>();Sub(zz,aa,ss,liveCount);PipeBarrier<PIPE_V>();
            TileMul(ee,ss,xh,D,rows);Sub(ee,pp,ee,liveCount);PipeBarrier<PIPE_V>();
            TileMul(tt,zz,xh,D,rows);Sub(ee,ee,tt,liveCount);PipeBarrier<PIPE_V>();
            TileMul(tt,ss,xl,D,rows);Sub(ee,ee,tt,liveCount);PipeBarrier<PIPE_V>();
            TileMul(tt,zz,xl,D,rows);Sub(ee,tt,ee,liveCount);PipeBarrier<PIPE_V>();
            Abs(ss,pp,liveCount);PipeBarrier<PIPE_V>();
            BlockReduceMax(blocks,ss,int32_t(liveCount/64),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();
            WholeReduceMax(maxPairs,blocks,int32_t(32),uint8_t(rows),1,1,4);PipeBarrier<PIPE_V>();
            Duplicate(sigma,1.0f,64);PipeBarrier<PIPE_V>();
            AscendC::Gather(sigma,maxPairs,maxIndex,0,rows);PipeBarrier<PIPE_V>();
            WholeReduceMax(guard,sigma,int32_t(rows),uint8_t(1),1,1,8);
            WholeReduceMin(guard[8],sigma,int32_t(rows),uint8_t(1),1,1,8);Fence<HardEvent::V_S>();
            union Lanen200_Bits{float f;uint32_t u;}largest,smallest;
            largest.f=guard.GetValue(0);smallest.f=guard.GetValue(8);
            const uint32_t highExp=((largest.u>>23)&255)+10,lowExp=((smallest.u>>23)&255)+10;
            // Exact extraction requires finite normal sigma and sigma*2^-12.
            // These IEEE exponent limits match the existing RowResidual fallback domain.
            if(!(lowExp>13 && highExp<254)){FromGlobal(mat,wg,N*rp);return false;}
            And(sigma.ReinterpretCast<int16_t>(),sigma.ReinterpretCast<int16_t>(),expMask.ReinterpretCast<int16_t>(),128);PipeBarrier<PIPE_V>();
            Adds(sigma.ReinterpretCast<int32_t>(),sigma.ReinterpretCast<int32_t>(),int32_t(10U<<23),64);PipeBarrier<PIPE_V>();
            Brcb(sigmaBroad,sigma,uint8_t(rows/8),{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t off=0;off<D;off+=64)
                Muls(ss[off],sigmaBroad,1.0f,uint64_t(64),uint8_t(rows),{1,0,D/8,1});
            PipeBarrier<PIPE_V>();
            // High and low extracted components have fixed binary units; their sums are exact.
            Add(zz,pp,ss,liveCount);PipeBarrier<PIPE_V>();Sub(zz,zz,ss,liveCount);PipeBarrier<PIPE_V>();
            Sub(tt,pp,zz,liveCount);PipeBarrier<PIPE_V>();
            Lanen64_ExtractedRowSum32(sumHigh,zz,blocks,rows);
            Muls(ss,ss,0.000244140625f,liveCount);PipeBarrier<PIPE_V>();
            Add(pp,tt,ss,liveCount);PipeBarrier<PIPE_V>();Sub(pp,pp,ss,liveCount);PipeBarrier<PIPE_V>();
            Sub(zz,tt,pp,liveCount);PipeBarrier<PIPE_V>();
            Lanen64_ExtractedRowSum32(sumLow,pp,blocks,rows);
            Lanen64_ExtractedRowSum32(sumTiny,zz,blocks,rows);
            Lanen64_ExtractedRowSum32(sumErr,ee,blocks,rows);
            Add(lowTotal,sumLow,sumTiny,64);PipeBarrier<PIPE_V>();Add(lowTotal,lowTotal,sumErr,64);PipeBarrier<PIPE_V>();
            Duplicate(rhsValues,0.0f,64);PipeBarrier<PIPE_V>();
            AscendC::Gather(rhsValues,bc[first*8],dotIndex,c*4,rows);PipeBarrier<PIPE_V>();
            Sub(ss,rhsValues,sumHigh,64);PipeBarrier<PIPE_V>();Sub(zz,rhsValues,ss,64);PipeBarrier<PIPE_V>();
            Add(tt,ss,zz,64);PipeBarrier<PIPE_V>();Sub(tt,rhsValues,tt,64);PipeBarrier<PIPE_V>();
            Sub(zz,zz,sumHigh,64);PipeBarrier<PIPE_V>();Add(tt,tt,zz,64);PipeBarrier<PIPE_V>();
            Sub(tt,tt,lowTotal,64);PipeBarrier<PIPE_V>();Add(ss,ss,tt,64);PipeBarrier<PIPE_V>();
            Brcb(localColumn,ss,uint8_t(rows/8),{1,8});PipeBarrier<PIPE_V>();
            uint64_t mask[2]={1ULL<<c,0};Muls(rr[first*8],localColumn,1.0f,mask,uint8_t(rows),{1,1,1,1});PipeBarrier<PIPE_V>();
        }
        FromGlobal(mat,wg,N*rp);return true;
    }
    // End Lanen200 extracted residual ownership.

    __aicore__ inline void TiledResidual() {
        if constexpr(Lanen200_UseExtractedResidual){if(Lanen200_TryExtractedResidual())return;}
        if(t.originalSchur){Lanen256_FastResidual();return;}
        const uint32_t tile = n >= 128 ? 16 : 8;
        const uint32_t d = t.dotSize, count = tile * d, compactCount = tile * 8;
        const bool spill = OnChip && n >= 128;
        auto work = triangle;
        if (spill) {
            ToGlobal(wg, mat, n * rp);
            work = mat;
        }
        auto aa = work; auto pp = work[count]; auto ee = work[2 * count];
        auto ss = work[3 * count]; auto zz = work[4 * count]; auto tt = work[5 * count];
        auto ll = aa; auto hh = pp; auto hl = ee;
        auto xc = work[6 * count]; auto bc = xc[n * kp];
        auto xx = bc[n * kp]; auto xh = xx[d]; auto xl = xh[d];
        auto smallP = xl[d]; auto smallE = smallP[compactCount];
        auto shifts = smallE[compactCount].ReinterpretCast<uint32_t>();
        auto rr = ResidualCache();
        Duplicate(rr, 0.0f, n * kp); PipeBarrier<PIPE_V>();
        if (spill) {
            Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>(); Fence<HardEvent::MTE3_MTE2>();
            DataCopyExtParams p{static_cast<uint16_t>(n), kp * 4, np * 4, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(xc, wg[np], p, pad);
            Fence<HardEvent::MTE2_V>();
        } else if constexpr (OnChip) {
            DataCopyParams p{static_cast<uint16_t>(n), static_cast<uint16_t>(kp / 8),
                             static_cast<uint16_t>(np / 8), 0};
            DataCopy(xc, mat[np], p); PipeBarrier<PIPE_V>();
        } else {
            DataCopy(xc, mat, n * kp); PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>(); Fence<HardEvent::MTE3_MTE2>();
        DataCopyExtParams pb{static_cast<uint16_t>(n), k * 4, 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, 0, 0, 0};
        DataCopyPad(bc, bg[bo], pb, pad);
        Fence<HardEvent::MTE2_V>(); Fence<HardEvent::MTE2_S>();
        for (uint32_t i = 0; i < d; ++i) dotIndex.SetValue(i, i * kp * 4);
        for (uint32_t level = 0; level < 3; ++level) {
            const uint32_t half = 4 >> level;
            for (uint32_t i = 0; i < tile; ++i) for (uint32_t q = 0; q < 8; ++q)
                shifts.SetValue(level * compactCount + i * 8 + q, (i * 8 + ((q + half) & 7)) * 4);
        }
        Fence<HardEvent::S_V>();
        const bool splitPaired=t.paired && t.rhsSplit;
        const uint32_t rowBegin=splitPaired?(GetBlockIdx()&1)*(n/2):0,rowEnd=splitPaired?rowBegin+n/2:n;
        for (uint32_t c = splitPaired ? (GetBlockIdx()/2)%PairSlots() : (t.blocked && t.rhsSplit) ? GetBlockIdx()%k : (t.paired && k >= 2) ? (GetBlockIdx() & 1) : 0; c < k; c += splitPaired ? k : (t.blocked && t.rhsSplit) ? k : (t.paired && k >= 2) ? 2 : 1) {
            Duplicate(xx, 0.0f, d); PipeBarrier<PIPE_V>();
            AscendC::Gather(xx, xc, dotIndex, c * 4, n); PipeBarrier<PIPE_V>();
            Muls(xh, xx, 4097.0f, d); PipeBarrier<PIPE_V>();
            Sub(xl, xh, xx, d); PipeBarrier<PIPE_V>();
            Sub(xh, xh, xl, d); PipeBarrier<PIPE_V>();
            Sub(xl, xx, xh, d); PipeBarrier<PIPE_V>();
            for (uint32_t first = rowBegin; first < rowEnd; first += tile) {
                const uint32_t rows = n - first < tile ? n - first : tile;
                Duplicate(aa, 0.0f, count); PipeBarrier<PIPE_V>();
                Fence<HardEvent::V_MTE2>(); Fence<HardEvent::S_MTE2>();
                DataCopyExtParams pa{static_cast<uint16_t>(rows), n * 4, 0, (d - np) / 8, 0};
                DataCopyPad(aa, ag[ao + uint64_t(first) * n], pa, pad);
                Fence<HardEvent::MTE2_V>();
                TileMul(pp, aa, xx, d, tile);
                Muls(tt, aa, 4097.0f, count); PipeBarrier<PIPE_V>();
                Sub(zz, tt, aa, count); PipeBarrier<PIPE_V>();
                Sub(ss, tt, zz, count); PipeBarrier<PIPE_V>();
                Sub(zz, aa, ss, count); PipeBarrier<PIPE_V>();
                TileMul(ee, ss, xh, d, tile);
                Sub(ee, pp, ee, count); PipeBarrier<PIPE_V>();
                TileMul(tt, zz, xh, d, tile);
                Sub(ee, ee, tt, count); PipeBarrier<PIPE_V>();
                TileMul(tt, ss, xl, d, tile);
                Sub(ee, ee, tt, count); PipeBarrier<PIPE_V>();
                TileMul(tt, zz, xl, d, tile);
                Sub(ee, tt, ee, count); PipeBarrier<PIPE_V>();
                BinaryRepeatParams reduce{1, 1, 1, static_cast<uint8_t>(d / 8),
                                         static_cast<uint8_t>(d / 8), static_cast<uint8_t>(d / 8)};
                UnaryRepeatParams copy{1, 1, static_cast<uint8_t>(d / 8), static_cast<uint8_t>(d / 8)};
                for (uint32_t span = d; span > 8; span >>= 1) {
                    const uint32_t half = span >> 1;
                    for (uint32_t offset = 0; offset < half; offset += 64) {
                        const uint64_t mask = half - offset < 64 ? half - offset : 64;
                        Add(ss[offset], pp[offset], pp[half + offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Sub(zz[offset], ss[offset], pp[offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Sub(tt[offset], ss[offset], zz[offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Sub(tt[offset], pp[offset], tt[offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Sub(zz[offset], pp[half + offset], zz[offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Add(tt[offset], tt[offset], zz[offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Add(ll[offset], ee[offset], ee[half + offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Add(ll[offset], ll[offset], tt[offset], mask, static_cast<uint8_t>(tile), reduce); PipeBarrier<PIPE_V>();
                        Muls(pp[offset], ss[offset], 1.0f, mask, static_cast<uint8_t>(tile), copy); PipeBarrier<PIPE_V>();
                        Muls(ee[offset], ll[offset], 1.0f, mask, static_cast<uint8_t>(tile), copy); PipeBarrier<PIPE_V>();
                    }
                }
                DataCopyParams pack{static_cast<uint16_t>(tile), 1, static_cast<uint16_t>(d / 8 - 1), 0};
                DataCopy(smallP, pp, pack); DataCopy(smallE, ee, pack); PipeBarrier<PIPE_V>();
                for (uint32_t level = 0; level < 3; ++level) {
                    AscendC::Gather(hh, smallP, shifts[level * compactCount], 0, compactCount); PipeBarrier<PIPE_V>();
                    AscendC::Gather(hl, smallE, shifts[level * compactCount], 0, compactCount); PipeBarrier<PIPE_V>();
                    Add(ss, smallP, hh, compactCount); PipeBarrier<PIPE_V>();
                    Sub(zz, ss, smallP, compactCount); PipeBarrier<PIPE_V>();
                    Sub(tt, ss, zz, compactCount); PipeBarrier<PIPE_V>();
                    Sub(tt, smallP, tt, compactCount); PipeBarrier<PIPE_V>();
                    Sub(zz, hh, zz, compactCount); PipeBarrier<PIPE_V>();
                    Add(tt, tt, zz, compactCount); PipeBarrier<PIPE_V>();
                    Add(ll, smallE, hl, compactCount); PipeBarrier<PIPE_V>();
                    Add(ll, ll, tt, compactCount); PipeBarrier<PIPE_V>();
                    DataCopy(smallP, ss, compactCount); DataCopy(smallE, ll, compactCount); PipeBarrier<PIPE_V>();
                }
                AscendC::Gather(hh, smallP, dotIndex, 0, tile); PipeBarrier<PIPE_V>();
                AscendC::Gather(hl, smallE, dotIndex, 0, tile); PipeBarrier<PIPE_V>();
                Duplicate(ll, 0.0f, tile); PipeBarrier<PIPE_V>();
                AscendC::Gather(ll, bc[first * kp], dotIndex, c * 4, rows); PipeBarrier<PIPE_V>();
                Sub(ss, ll, hh, tile); PipeBarrier<PIPE_V>();
                Sub(zz, ll, ss, tile); PipeBarrier<PIPE_V>();
                Add(tt, ss, zz, tile); PipeBarrier<PIPE_V>();
                Sub(tt, ll, tt, tile); PipeBarrier<PIPE_V>();
                Sub(zz, zz, hh, tile); PipeBarrier<PIPE_V>();
                Add(tt, tt, zz, tile); PipeBarrier<PIPE_V>();
                Sub(tt, tt, hl, tile); PipeBarrier<PIPE_V>();
                Add(ss, ss, tt, tile); PipeBarrier<PIPE_V>();
                Brcb(column, ss, static_cast<uint8_t>(tile / 8), {1, 8}); PipeBarrier<PIPE_V>();
                uint64_t mask[2] = {1ULL << c, 0};
                Muls(rr[first * kp], column, 1.0f, mask, static_cast<uint8_t>(rows), {1, 1, static_cast<uint8_t>(kp / 8), 1});
                PipeBarrier<PIPE_V>();
            }
        }
        if (spill) FromGlobal(mat, wg, n * rp);
        if constexpr (!OnChip) ToGlobal(wg[n * rp], rr, n * kp);
    }

    __aicore__ inline void RowResidual() {
        const uint32_t d = t.dotSize;
        uint32_t logd = 0; for (uint32_t q = d; q > 1; q >>= 1) ++logd;
        for (uint32_t i = 0; i < d; ++i) dotIndex.SetValue(i, i * 4);
        auto base = scratch.Get<float>();
        auto aa = base; auto xx = base[d]; auto xh = base[2*d]; auto xl = base[3*d];
        auto pp = base[4*d]; auto ee = base[5*d];
        auto ss = base[6*d]; auto zz = base[7*d]; auto tt = base[8*d]; auto ll = base[9*d];
        auto hh = base[10*d]; auto hl = base[11*d]; auto bb = base[13*d];
        LocalTensor<float> rr = mat[n * kp];
        if constexpr (OnChip) rr = mat[n * rp];
        Duplicate(rr, 0.0f, n * kp); PipeBarrier<PIPE_V>();
        FromGlobal(bb, bg[bo], n * k);
        for (uint32_t c = 0; c < k; ++c) {
            Duplicate(xx, 0.0f, d); PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < n; ++j) {
                Load(aa, j * rp + np, kp);
                xx.SetValue(j, aa.GetValue(c));
            }
            Fence<HardEvent::S_V>();
            Muls(xh, xx, 4097.0f, d); PipeBarrier<PIPE_V>();
            Sub(xl, xh, xx, d); PipeBarrier<PIPE_V>();
            Sub(xh, xh, xl, d); PipeBarrier<PIPE_V>();
            Sub(xl, xx, xh, d); PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < n; ++i) {
                Duplicate(aa, 0.0f, d); PipeBarrier<PIPE_V>();
                FromGlobal(aa, ag[ao + uint64_t(i) * n], n);
                // Error-free vector TwoProduct, followed by compensated pairs.
                Mul(pp, aa, xx, d); PipeBarrier<PIPE_V>();
                Muls(tt, aa, 4097.0f, d); PipeBarrier<PIPE_V>();
                Sub(zz, tt, aa, d); PipeBarrier<PIPE_V>();
                Sub(ss, tt, zz, d); PipeBarrier<PIPE_V>();
                Sub(zz, aa, ss, d); PipeBarrier<PIPE_V>();
                Mul(ee, ss, xh, d); PipeBarrier<PIPE_V>();
                Sub(ee, pp, ee, d); PipeBarrier<PIPE_V>();
                Mul(tt, zz, xh, d); PipeBarrier<PIPE_V>();
                Sub(ee, ee, tt, d); PipeBarrier<PIPE_V>();
                Mul(tt, ss, xl, d); PipeBarrier<PIPE_V>();
                Sub(ee, ee, tt, d); PipeBarrier<PIPE_V>();
                Mul(tt, zz, xl, d); PipeBarrier<PIPE_V>();
                Sub(ee, tt, ee, d); PipeBarrier<PIPE_V>();
                if (d > 16) {
                    Abs(tt, pp, d); PipeBarrier<PIPE_V>();
                    AscendC::ReduceMax(ss, tt, hh, static_cast<int32_t>(d), false);
                    PipeBarrier<PIPE_V>(); Fence<HardEvent::V_S>();
                    union FloatBits { float value; uint32_t bits; } scale;
                    scale.value = ss.GetValue(0);
                    const uint32_t exponent = ((scale.bits >> 23) & 255) + logd + 2;
                    if (exponent > 13 && exponent < 254) {
                        scale.bits = exponent << 23; const float sigma = scale.value;
                        scale.bits = (exponent - 12) << 23; const float sigmaLow = scale.value;
                        // Both extracted sums are exact: common integer units and
                        // fewer than24 significant sum bits. The remainder is tiny.
                        Adds(ss, pp, sigma, d); PipeBarrier<PIPE_V>();
                        Adds(ss, ss, -sigma, d); PipeBarrier<PIPE_V>();
                        Sub(zz, pp, ss, d); PipeBarrier<PIPE_V>();
                        Adds(tt, zz, sigmaLow, d); PipeBarrier<PIPE_V>();
                        Adds(tt, tt, -sigmaLow, d); PipeBarrier<PIPE_V>();
                        Sub(ll, zz, tt, d); PipeBarrier<PIPE_V>();
                        AscendC::ReduceSum(pp, ss, hh, static_cast<int32_t>(d)); PipeBarrier<PIPE_V>();
                        AscendC::ReduceSum(pp[8], tt, hh, static_cast<int32_t>(d)); PipeBarrier<PIPE_V>();
                        AscendC::ReduceSum(zz, ll, hh, static_cast<int32_t>(d)); PipeBarrier<PIPE_V>();
                        AscendC::ReduceSum(ss, ee, hh, static_cast<int32_t>(d)); PipeBarrier<PIPE_V>();
                        Fence<HardEvent::V_S>();
                        float value = bb.GetValue(i * k + c), error = 0.0f;
                        CompensatedScalar(-pp.GetValue(0), value, error);
                        CompensatedScalar(-pp.GetValue(8), value, error);
                        CompensatedScalar(-zz.GetValue(0), value, error);
                        CompensatedScalar(-ss.GetValue(0), value, error);
                        rr.SetValue(i * kp + c, value + error);
                        continue;
                    }
                }
                for (uint32_t span = d; span > 1; span >>= 1) {
                    const uint32_t half = span >> 1;
                    AscendC::Gather(hh, pp, dotIndex, half * 4, half); PipeBarrier<PIPE_V>();
                    AscendC::Gather(hl, ee, dotIndex, half * 4, half); PipeBarrier<PIPE_V>();
                    Add(ss, pp, hh, half); PipeBarrier<PIPE_V>();
                    Sub(zz, ss, pp, half); PipeBarrier<PIPE_V>();
                    Sub(tt, ss, zz, half); PipeBarrier<PIPE_V>();
                    Sub(tt, pp, tt, half); PipeBarrier<PIPE_V>();
                    Sub(zz, hh, zz, half); PipeBarrier<PIPE_V>();
                    Add(tt, tt, zz, half); PipeBarrier<PIPE_V>();
                    Add(ll, ee, hl, half); PipeBarrier<PIPE_V>();
                    Add(ll, ll, tt, half); PipeBarrier<PIPE_V>();
                    const uint32_t copied = half < 8 ? 8 : half;
                    DataCopy(pp, ss, copied); PipeBarrier<PIPE_V>();
                    DataCopy(ee, ll, copied); PipeBarrier<PIPE_V>();
                }
                Fence<HardEvent::V_S>();
                const float high = pp.GetValue(0), low = ee.GetValue(0), b = bb.GetValue(i * k + c);
                // Volatile enforces the error-free scalar TwoDiff order.
                volatile float value = b - high;
                volatile float bvirtual = b - value;
                volatile float avirtual = value + bvirtual;
                volatile float berror = bvirtual - high;
                volatile float aerror = b - avirtual;
                volatile float error = aerror + berror;
                volatile float correction = error - low;
                rr.SetValue(i * kp + c, value + correction);
            }
        }
        Fence<HardEvent::S_V>();
        if constexpr (!OnChip) ToGlobal(wg[n * rp], rr, n * kp);
    }

    __aicore__ inline void Refine() {
        bool Lanen200_cheapFirst=false;
        if constexpr(Lanen200_BlockTri16){Lanen200_cheapFirst=(Lanen200_refinePass++==0);if(Lanen200_cheapFirst)Lanen200_FirstFP32Residual();}
        if(!Lanen200_cheapFirst){
        if (t.dotSize) {
            if ((n > 16 && n <= 64) || (n >= 128 && n <= 256 && (!OnChip || 6 * 16 * t.dotSize + 2 * n * kp + 3 * t.dotSize + 5 * 16 * 8 <= n * rp))) TiledResidual();
            else RowResidual();
        } else {
            // Error-free two-product and TwoSum, accumulating both errors (Dot2).
            // 910B vector multiply-add is not assumed to be a true FMA.
            for (uint32_t i = 0; i < n; ++i) {
                FromGlobal(original, ag[ao + uint64_t(i) * n], n);
                FromGlobal(v0, bg[bo + uint64_t(i) * k], k);
                Duplicate(v1, 0.0f, kp); PipeBarrier<PIPE_V>();
                for (uint32_t j = 0; j < n; ++j) {
                    const float a = -original.GetValue(j);
                    union Bits { float f; uint32_t u; } bits;
                    bits.f = a; bits.u &= 0xfffff000U;
                    const float ahi = bits.f, alo = a - ahi;
                    Load(row, j * rp + np, kp);
                    Muls(v2, row, a, kp); PipeBarrier<PIPE_V>();
                    Muls(v4, row, 4097.0f, kp); PipeBarrier<PIPE_V>();
                    Sub(v5, v4, row, kp); PipeBarrier<PIPE_V>();
                    Sub(v4, v4, v5, kp); PipeBarrier<PIPE_V>();
                    Sub(v5, row, v4, kp); PipeBarrier<PIPE_V>();
                    Muls(v6, v4, ahi, kp); PipeBarrier<PIPE_V>();
                    Sub(v6, v6, v2, kp); PipeBarrier<PIPE_V>();
                    Muls(v7, v5, ahi, kp); PipeBarrier<PIPE_V>();
                    Add(v6, v6, v7, kp); PipeBarrier<PIPE_V>();
                    Muls(v7, v4, alo, kp); PipeBarrier<PIPE_V>();
                    Add(v6, v6, v7, kp); PipeBarrier<PIPE_V>();
                    Muls(v7, v5, alo, kp); PipeBarrier<PIPE_V>();
                    Add(v6, v6, v7, kp); PipeBarrier<PIPE_V>();
                    Add(v3, v0, v2, kp); PipeBarrier<PIPE_V>();
                    Sub(v4, v3, v0, kp); PipeBarrier<PIPE_V>();
                    Sub(v5, v3, v4, kp); PipeBarrier<PIPE_V>();
                    Sub(v5, v0, v5, kp); PipeBarrier<PIPE_V>();
                    Sub(v4, v2, v4, kp); PipeBarrier<PIPE_V>();
                    Add(v5, v5, v4, kp); PipeBarrier<PIPE_V>();
                    Add(v6, v6, v5, kp); PipeBarrier<PIPE_V>();
                    Add(v1, v1, v6, kp); PipeBarrier<PIPE_V>();
                    LocalCopy(v0, v3, kp);
                }
                Add(v0, v0, v1, kp); PipeBarrier<PIPE_V>();
                StoreR(i, v0);
            }
        }
        }
        if (t.dotSize && n <= 256) {
            if (t.paired && k >= 2) {
                SolveSync();
                GlobalTensor<float> workerR;
                const uint64_t elements=uint64_t(n)*(np+2*kp);
                workerR.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace)+uint64_t(GetBlockIdx()+1)*elements+n*rp);
                FromGlobal(triangle,workerR,n*kp);
                auto rhs=ResidualCache();
                if(t.rhsSplit){
                    uint32_t c=(GetBlockIdx()/2)%PairSlots(),first=n/2;uint64_t mask[2]={1ULL<<c,0};
                    Muls(rhs[first*kp],triangle[first*kp],1.0f,mask,uint8_t(n/2),{1,1,uint8_t(kp/8),uint8_t(kp/8)});PipeBarrier<PIPE_V>();
                }else for(uint32_t c=1;c<k;c+=2) for(uint32_t first=0;first<n;first+=128) {
                    uint64_t mask[2]={1ULL<<c,0};uint32_t rows=n-first<128?n-first:128;
                    Muls(rhs[first*kp],triangle[first*kp],1.0f,mask,uint8_t(rows),{1,1,uint8_t(kp/8),uint8_t(kp/8)});PipeBarrier<PIPE_V>();
                }
            }
            ForwardSolveRows();
        } else {
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t pivot = pivots.GetValue(j);
                if (pivot != j) {
                    LoadR(row, j); LoadR(other, pivot);
                    StoreR(pivot, row); StoreR(j, other);
                }
            }
            for (uint32_t j = 0; j < n; ++j) {
                Gather(j, j + 1, n); LoadR(row, j);
                for (uint32_t i = j + 1; i < n; ++i) {
                    const float coefficient = Coeff(i, j, j + 1);
                    LoadR(other, i); Ax(other, row, -coefficient, kp); StoreR(i, other);
                }
            }
        }
        BackSolve(true);
        if (t.dotSize && n <= 256) {
            if constexpr (OnChip) {
                const uint8_t blocks = rp / 8;
                BinaryRepeatParams add{blocks, blocks, 1, static_cast<uint8_t>(8 * blocks),
                                       static_cast<uint8_t>(8 * blocks), 8};
                if (n / 8) Add(mat[np], mat[np], mat[n * rp], uint64_t(64), static_cast<uint8_t>(n / 8), add);
                if (n % 8) {
                    uint32_t first = n / 8 * 8;
                    Add(mat[first * rp + np], mat[first * rp + np], mat[n * rp + first * kp],
                        uint64_t((n % 8) * 8), uint8_t(1), add);
                }
                PipeBarrier<PIPE_V>();
            } else {
                Add(mat, mat, mat[n * kp], n * kp); PipeBarrier<PIPE_V>();
            }
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                Load(row, i * rp + np, kp); LoadR(other, i);
                Add(row, row, other, kp); PipeBarrier<PIPE_V>();
                Store(i * rp + np, row, kp);
            }
        }
    }

    uint32_t Lanen200_refinePass=0;
    bool Lanen200_extraIR=false;
    GM_ADDR userWorkspace;
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage, scratch, pivotBuffer, triangleBuffer, blockSolveBuffer;
    LocalTensor<float> triangle;
    LocalTensor<float> mat, row, other, column, original, v0, v1, v2, v3, v4, v5, v6, v7;
    LocalTensor<int32_t> pivots;
    LocalTensor<uint32_t> dotIndex;
    GlobalTensor<float> ag, bg, xg, wg;
    LinalgSolveTilingData t;
    uint32_t n, k, np, kp, rp;
    uint64_t ao, bo, xo;
};

__aicore__ inline bool Lanen64_SplitNeedsFallback(GM_ADDR workspace){
    GlobalTensor<uint32_t>g;g.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(GetUserWorkspace(workspace))+uint64_t(GetBlockIdx()/4*4)*200*216+200*208+200);LocalTensor<uint32_t>v(TPosition::VECCALC,0,8);Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();DataCopy(v,g,8);Fence<HardEvent::MTE2_S>();return v.GetValue(0)!=0x53474a46U||v.GetValue(1)!=0;
}
__aicore__ inline void Lanen64_MarkFallbackIR(GM_ADDR workspace){
    LocalTensor<uint32_t>v(TPosition::VECCALC,0,8);for(uint32_t i=0;i<8;++i)v.SetValue(i,0);v.SetValue(0,0x4e324952U);for(uint32_t lane=0;lane<4;++lane){GlobalTensor<uint32_t>g;g.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(GetUserWorkspace(workspace))+uint64_t(GetBlockIdx()+lane)*200*216+200*208+208);Fence<HardEvent::S_MTE3>();DataCopy(g,v,8);Fence<HardEvent::MTE3_S>();}
}
// BEGIN Lanen64 cached original-A Jacobi20 with uniform fallback.
// Four existing pairs own56/56/56/32 physical rows. FullK208 A2resident, Xparitybuffers.
class Lanen64_CachedJacobi20 {
    static constexpr uint32_t N=200,RP=208,ELEMENTS=43200,MP=208,XP=16;
    static constexpr uint32_t X0=41600,Y0=44928,X1=48128,FAST_D=56000,FAST_B=57024;
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR ws,const LinalgSolveTilingData&d){
        t=d;core=GetBlockIdx();quad=core%4;sys=core/4;r0=quad*56;rows=quad<3?56:32;mp=(rows+15)/16*16;
        auto base=reinterpret_cast<__gm__ float*>(GetUserWorkspace(ws));product.SetGlobalBuffer(base+uint64_t(t.blocks)*ELEMENTS+uint64_t(sys*4)*N*RP);control.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(base+uint64_t(t.blocks)*ELEMENTS+165888));wg.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(base));
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));xg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        mem=LocalTensor<float>(TPosition::VECCALC,0,49152);mat=mem;xx=mem;yy=mem[3328];rr=mem[6656];delta=mem[9984];temp=mem[13312];reduce=mem[16640];scratch=mem[17664];rev=mem[20000].ReinterpretCast<int32_t>();ix=mem[23200].ReinterpretCast<int32_t>();bb=mem[41600];ib=mem[44800];diag=mem[48000];di=mem[48256].ReinterpretCast<int32_t>();buf=mem[48512];packet=mem[49024].ReinterpretCast<uint32_t>();
        uint32_t coords=sys,ai=0,bi=0;for(int j=int(t.batchRank)-1;j>=0;--j){auto c=coords%t.batchDims[j];coords/=t.batchDims[j];ai+=c*t.aStrides[j];bi+=c*t.bStrides[j];}ao=uint64_t(ai)*N*N;bo=uint64_t(bi)*N*4;xo=uint64_t(sys)*N*4;
    }
    __aicore__ inline bool Process(){
        if ASCEND_IS_AIC{
            CrossCoreWaitFlag<2>(0);bool use=ReadDecision();CrossCoreSetFlag<2,PIPE_FIX>(1);if(!use)return false;
            for(uint32_t step=0;step<21;++step){CrossCoreWaitFlag<2>(2);if(step==0)CacheA();Mvm(step);CrossCoreSetFlag<2,PIPE_FIX>(3);}
            CrossCoreWaitFlag<2>(6);bool okay=ReadDecision();CrossCoreSetFlag<2,PIPE_FIX>(7);return okay;
        }else{
            if(quad==0){LoadOriginal();bool gate=Gate();WriteFlag(gate);}
            if(!InitialDecision())return false;
            if(quad==0)PrepareOwner();SyncAll<true>();PrepareRows();
            for(uint32_t step=0;step<20;++step){Request(step);Sub(rr,bb,yy,rows*XP);PipeBarrier<PIPE_V>();Mul(rr,rr,ib,rows*XP);PipeBarrier<PIPE_V>();if(flip){Gather(delta,rr,rev.ReinterpretCast<uint32_t>(),0,rows*XP);PipeBarrier<PIPE_V>();Add(xx,xx,delta,rows*XP);}else Add(xx,xx,rr,rows*XP);PipeBarrier<PIPE_V>();}
            Request(20);Sub(rr,bb,yy,rows*XP);PipeBarrier<PIPE_V>();Mul(rr,rr,rr,rows*XP);Mul(temp,bb,bb,rows*XP);PipeBarrier<PIPE_V>();AscendC::ReduceSum(reduce,rr,scratch,int32_t(rows*XP));PipeBarrier<PIPE_V>();AscendC::ReduceSum(reduce[8],temp,scratch,int32_t(rows*XP));Fence<HardEvent::V_S>();float rn=reduce.GetValue(0),bn=reduce.GetValue(8);bool okay=FinalDecision(rn,bn);
            if(okay){Fence<HardEvent::V_MTE3>();DataCopyExtParams out{uint16_t(rows),4*4,1,0,0};DataCopyPad(xg[xo+uint64_t(var0)*4],xx,out);Fence<HardEvent::MTE3_V>();if(quad==0)MarkFast();}return okay;
        }
    }
private:
    __aicore__ inline void Tx(GlobalTensor<float>d,LocalTensor<float>s,uint32_t n){Fence<HardEvent::V_MTE3>();Fence<HardEvent::S_MTE3>();DataCopy(d,s,n);Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_S>();}
    __aicore__ inline void Rx(LocalTensor<float>d,GlobalTensor<float>s,uint32_t n){Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();DataCopy(d,s,n);Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();}
    __aicore__ inline void LoadOriginal(){Duplicate(mat,0.0f,N*RP);PipeBarrier<PIPE_V>();Fence<HardEvent::V_MTE2>();DataCopyPadExtParams<float>pd{true,0,0,0};DataCopyExtParams pa{N,N*4,0,1,0},pb{N,4*4,0,N/8,0};DataCopyPad(mat,ag[ao],pa,pd);DataCopyPad(mat[N],bg[bo],pb,pd);Fence<HardEvent::MTE2_V>();}
    __aicore__ inline bool Gate(){
        CreateVecIndex(di,int32_t(0),256);PipeBarrier<PIPE_V>();Muls(di,di,int32_t((RP-1)*4),N);PipeBarrier<PIPE_V>();Adds(di,di,int32_t((N-1)*4),N);PipeBarrier<PIPE_V>();Gather(diag,mat,di.ReinterpretCast<uint32_t>(),0,N);PipeBarrier<PIPE_V>();Mul(buf,diag,diag,N);Duplicate(buf[N],3.402823466e38f,256-N);PipeBarrier<PIPE_V>();WholeReduceMin(mem[48768],buf,uint64_t(64),uint8_t(4),1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();WholeReduceMin(mem[48800],mem[48768],uint64_t(4),uint8_t(1),1,1,8,ReduceOrder::ORDER_ONLY_VALUE);Fence<HardEvent::V_S>();float reverseLeast=mem[48800].GetValue(0);
        // Data-derived orientation: pick the larger minimum diagonal magnitude, then apply the same fixed gate.
        CreateVecIndex(di,int32_t(0),256);PipeBarrier<PIPE_V>();Muls(di,di,int32_t((RP+1)*4),N);PipeBarrier<PIPE_V>();Gather(mem[45824],mat,di.ReinterpretCast<uint32_t>(),0,N);PipeBarrier<PIPE_V>();Mul(buf,mem[45824],mem[45824],N);Duplicate(buf[N],3.402823466e38f,256-N);PipeBarrier<PIPE_V>();WholeReduceMin(mem[48768],buf,uint64_t(64),uint8_t(4),1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();WholeReduceMin(mem[48800],mem[48768],uint64_t(4),uint8_t(1),1,1,8,ReduceOrder::ORDER_ONLY_VALUE);Fence<HardEvent::V_S>();float mainLeast=mem[48800].GetValue(0);flip=reverseLeast>=mainLeast;float least=flip?reverseLeast:mainLeast;if(!flip)DataCopy(diag,mem[45824],N);PipeBarrier<PIPE_V>();Mul(buf,diag,diag,N);PipeBarrier<PIPE_V>();
        // A fixed first eight-row sample is a costed heuristic; final original-A residual and full fallback remain mandatory.
        auto tile=mem[41600];Duplicate(tile,0.0f,8*256);PipeBarrier<PIPE_V>();DataCopyParams cp{8,RP/8,0,(256-RP)/8};DataCopy(tile,mat,cp);PipeBarrier<PIPE_V>();Mul(tile,tile,tile,8*256);PipeBarrier<PIPE_V>();BlockReduceSum(mem[43648],tile,int32_t(32),int32_t(64),1,1,8);PipeBarrier<PIPE_V>();WholeReduceSum(mem[43904],mem[43648],int32_t(32),uint8_t(8),1,1,4);PipeBarrier<PIPE_V>();Sub(mem[43904],mem[43904],buf,8);PipeBarrier<PIPE_V>();WholeReduceMax(mem[43936],mem[43904],uint64_t(8),uint8_t(1),1,1,8);Fence<HardEvent::V_S>();float off=mem[43936].GetValue(0);return least>0.0f&&least>=2.25f*off&&least<=3.402823466e38f;
    }

    __aicore__ inline void WriteFlag(bool value){for(uint32_t i=0;i<8;++i)packet.SetValue(i,0);packet.SetValue(0,uint32_t(value));packet.SetValue(1,uint32_t(flip));Fence<HardEvent::S_MTE3>();DataCopy(control[sys*8],packet,8);Fence<HardEvent::MTE3_S>();}
    __aicore__ inline bool InitialDecision(){SyncAll<true>();if(core==0){Fence<HardEvent::S_MTE2>();DataCopy(packet,control,48);Fence<HardEvent::MTE2_S>();bool all=true;for(uint32_t j=0;j<t.blocks/4;++j)all=all&&(packet.GetValue(j*8)!=0);for(uint32_t i=0;i<8;++i)packet.SetValue(i,0);packet.SetValue(0,uint32_t(all));Fence<HardEvent::S_MTE3>();DataCopy(control[64],packet,8);Fence<HardEvent::MTE3_S>();}SyncAll<true>();Fence<HardEvent::S_MTE2>();DataCopy(packet,control[64],8);Fence<HardEvent::MTE2_S>();bool use=packet.GetValue(0)!=0;CrossCoreSetFlag<2,PIPE_MTE3>(0);CrossCoreWaitFlag<2>(1);return use;}
    __aicore__ inline bool FinalDecision(float rn,float bn){
        for(uint32_t i=0;i<8;++i)packet.SetValue(i,0);packet.ReinterpretCast<float>().SetValue(0,rn);packet.ReinterpretCast<float>().SetValue(1,bn);Fence<HardEvent::S_MTE3>();DataCopy(control[128+core*8],packet,8);Fence<HardEvent::MTE3_S>();SyncAll<true>();
        if(core==0){auto partial=mem[18000].ReinterpretCast<uint32_t>();Fence<HardEvent::V_MTE2>();Fence<HardEvent::S_MTE2>();DataCopy(partial,control[128],t.blocks*8);Fence<HardEvent::MTE2_S>();bool all=true;
            for(uint32_t s=0;s<t.blocks/4;++s){float rs=0.0f,bs=0.0f;for(uint32_t q=0;q<4;++q){rs+=partial.ReinterpretCast<float>().GetValue((s*4+q)*8);bs+=partial.ReinterpretCast<float>().GetValue((s*4+q)*8+1);}bool okay=rs>=0.0f&&rs<=1.0e-10f*bs&&bs<=3.402823466e38f;all=all&&okay;for(uint32_t i=0;i<8;++i)packet.SetValue(i,0);packet.SetValue(0,uint32_t(okay));Fence<HardEvent::S_MTE3>();DataCopy(control[s*8],packet,8);Fence<HardEvent::MTE3_S>();}
            for(uint32_t i=0;i<8;++i)packet.SetValue(i,0);packet.SetValue(0,uint32_t(all));Fence<HardEvent::S_MTE3>();DataCopy(control[64],packet,8);Fence<HardEvent::MTE3_S>();
        }
        SyncAll<true>();Fence<HardEvent::S_MTE2>();DataCopy(packet,control[64],8);Fence<HardEvent::MTE2_S>();bool okay=packet.GetValue(0)!=0;CrossCoreSetFlag<2,PIPE_MTE3>(6);CrossCoreWaitFlag<2>(7);return okay;
    }
    __aicore__ inline bool ReadDecision(){auto g=control[64];DataCacheCleanAndInvalid<uint32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(g);return g.GetValue(0)!=0;}
    __aicore__ inline void PrepareOwner(){
        Tx(product,mat,N*RP);Duplicate(bb,0.0f,N*XP);Duplicate(ib,0.0f,N*XP);PipeBarrier<PIPE_V>();DataCopyParams cp{N,1,RP/8-1,1};DataCopy(bb,mat[N],cp);Duplicate(buf,1.0f,256);PipeBarrier<PIPE_V>();Div(diag,buf,diag,N);PipeBarrier<PIPE_V>();Brcb(ib,diag,25,{2,16});PipeBarrier<PIPE_V>();
        // After the original matrix was exported, its UB range is dead.
        if(flip){CreateVecIndex(rev,int32_t(0),N*XP);PipeBarrier<PIPE_V>();ShiftRight(ix,rev,int32_t(4),N*XP);PipeBarrier<PIPE_V>();Muls(ix,ix,int32_t(-32),N*XP);PipeBarrier<PIPE_V>();Add(rev,rev,ix,N*XP);PipeBarrier<PIPE_V>();Adds(rev,rev,int32_t((N-1)*XP),N*XP);PipeBarrier<PIPE_V>();Muls(rev,rev,int32_t(4),N*XP);}Duplicate(xx,0.0f,MP*XP);PipeBarrier<PIPE_V>();
    Tx(product[FAST_B],bb,N*XP);Tx(product[FAST_D],diag,N);Tx(product[X0],xx,MP*XP);Tx(product[X1],xx,MP*XP);
    }

    __aicore__ inline void PrepareRows(){
        Fence<HardEvent::S_MTE2>();DataCopy(packet,control[sys*8],8);Fence<HardEvent::MTE2_S>();flip=packet.GetValue(1)!=0;var0=flip?N-r0-rows:r0;
        Duplicate(xx,0.0f,mp*XP);Duplicate(bb,0.0f,mp*XP);Duplicate(ib,0.0f,mp*XP);PipeBarrier<PIPE_V>();Rx(bb,product[FAST_B+r0*XP],rows*XP);Rx(diag,product[FAST_D+r0],mp);Brcb(ib,diag,mp/8,{2,16});PipeBarrier<PIPE_V>();
        if(flip){CreateVecIndex(rev,int32_t(0),rows*XP);PipeBarrier<PIPE_V>();ShiftRight(ix,rev,int32_t(4),rows*XP);PipeBarrier<PIPE_V>();Muls(ix,ix,int32_t(-32),rows*XP);PipeBarrier<PIPE_V>();Add(rev,rev,ix,rows*XP);PipeBarrier<PIPE_V>();Adds(rev,rev,int32_t((rows-1)*XP),rows*XP);PipeBarrier<PIPE_V>();Muls(rev,rev,int32_t(4),rows*XP);PipeBarrier<PIPE_V>();}
    }
    __aicore__ inline void Request(uint32_t step){const uint32_t off=(step&1)?X1:X0;Tx(product[off+var0*XP],xx,rows*XP);SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(2);CrossCoreWaitFlag<2>(3);Rx(yy,product[Y0+r0*XP],rows*XP);}
    __aicore__ inline void CacheA(){
        SetHF32Mode(HF32Mode::DISABLE);LocalTensor<float>a1(TPosition::A1,32768,64*MP),a2(TPosition::A2,0,64*MP);Nd2NzParams pa{1,uint16_t(rows),MP,0,RP,uint16_t(mp),1,0};DataCopy(a1,product[r0*RP],pa);Fence<HardEvent::MTE2_MTE1>();LoadData2DParams la;la.startIndex=0;la.repeatTimes=MP/8;la.srcStride=mp/16;la.dstGap=0;la.ifTranspose=false;for(uint32_t i=0;i<mp/16;++i)LoadData(a2[i*16*MP],a1[i*128],la);Fence<HardEvent::MTE1_M>();
    }
    __aicore__ inline void Mvm(uint32_t step){
        LocalTensor<float>b1(TPosition::B1,8192,MP*XP),a2(TPosition::A2,0,64*MP),b2(TPosition::B2,0,MP*XP),cc(TPosition::CO1,0,64*XP);Fence<HardEvent::M_MTE1>();Fence<HardEvent::MTE1_MTE2>();Nd2NzParams pb{13,16,16,256,16,16,1,256};DataCopy(b1,product[(step&1)?X1:X0],pb);Fence<HardEvent::MTE2_MTE1>();LoadData2dTransposeParams lb;lb.startIndex=0;lb.repeatTimes=13;lb.srcStride=1;lb.dstGap=1;lb.dstFracGap=0;LoadDataWithTranspose(b2,b1,lb);Fence<HardEvent::MTE1_M>();MmadParams mm;mm.m=rows;mm.n=16;mm.k=MP;mm.cmatrixInitVal=true;Mmad(cc,a2,b2,mm);Fence<HardEvent::M_MTE1>();Fence<HardEvent::M_FIX>();FixpipeParamsV220 fp;fp.ndNum=1;fp.srcNdStride=0;fp.dstNdStride=0;fp.mSize=rows;fp.nSize=16;fp.srcStride=mp;fp.dstStride=16;Fixpipe(product[Y0+r0*XP],cc,fp);Fence<HardEvent::FIX_MTE2>();
    }
    __aicore__ inline void MarkFast(){for(uint32_t i=0;i<8;++i)packet.SetValue(i,0);packet.SetValue(0,0x53474a46U);Fence<HardEvent::S_MTE3>();DataCopy(wg[uint64_t(sys*4)*ELEMENTS+N*RP+200],packet,8);Fence<HardEvent::MTE3_S>();packet.SetValue(0,0x4e324952U);for(uint32_t j=0;j<4;++j){Fence<HardEvent::S_MTE3>();DataCopy(wg[uint64_t(sys*4+j)*ELEMENTS+N*RP+208],packet,8);Fence<HardEvent::MTE3_S>();}}

    TPipe pipe;bool flip=true;LinalgSolveTilingData t;uint32_t core,quad,sys,r0,rows,mp,var0;uint64_t ao,bo,xo;GlobalTensor<float>product,ag,bg,xg;GlobalTensor<uint32_t>control,wg;LocalTensor<float>mem,mat,xx,yy,rr,delta,temp,reduce,scratch,bb,ib,diag,buf;LocalTensor<int32_t>rev,ix,di;LocalTensor<uint32_t>packet;
};
// END Lanen64 cached original-A Jacobi20 with uniform fallback.

// END Lanen64 N200 split-LU GJ16 full solve.

template <typename DT_A, uint32_t MODE>
__global__ __aicore__ void linalg_solve(GM_ADDR A, GM_ADDR B, GM_ADDR X, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LinalgSolveTilingData);
    if constexpr(MODE==12){RootN50_ColumnMajor50 op;op.Init(A,B,X);op.Process();return;}
    if constexpr(MODE==15){
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveTilingData, Lanen64_splitData, tiling);
        {Lanen64_CachedJacobi20 fast;fast.Init(A,B,X,workspace,Lanen64_splitData);if(fast.Process())return;}
        {Lanen64_SplitGJ16Factor factor;factor.Init(A,B,workspace,Lanen64_splitData);factor.Process();}
        if ASCEND_IS_AIV{if(Lanen64_SplitNeedsFallback(workspace)){if((GetBlockIdx()%4)==0){auto fallbackData=Lanen64_splitData;fallbackData.blocked=false;fallbackData.paired=false;fallbackData.rhsSplit=false;fallbackData.blocks=fallbackData.batches;Solver<true> fallback;fallback.Init(A,B,X,workspace,fallbackData);fallback.Process(GetBlockIdx()/4);Lanen64_MarkFallbackIR(workspace);}}else{Lanen64_SplitGJSolver<true,true,true> solver;solver.Init(A,B,X,workspace,Lanen64_splitData);solver.Process();}}
        return;
    }
    if constexpr(MODE==14){
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveTilingData, laneData, tiling);
        if((GetBlockIdx()%4)==0){Lanen200_SharedFactor factor;factor.Init(A,B,workspace);factor.Process();}
        SyncAll<true>();
        Solver<true> solver;solver.Init(A,B,X,workspace,laneData);solver.Process();return;
    }
    if constexpr(MODE==13){Lanen64_JointNarrow64 op;op.Init(A,B,X);op.Process();return;}
    if constexpr(MODE==11){JointNarrowSolve<50>op;op.Init(A,B,X);op.Process();return;}
    if constexpr(MODE==10){Lanen64_Tiny16 op;op.Init(A,B,X);op.Process();return;}
    if constexpr(MODE==5){TinyExact<16>op;op.Init(A,B,X);op.Process();return;}
    else if constexpr(MODE==6){Lanen15_NarrowSolve op;op.Init(A,B,X);op.Process();return;}
    else if constexpr(MODE==7){bool ok;{SmallExact<64>op;op.Init(A,B,X);ok=op.Process();}if(!ok){SmallComplete<64>rescue;rescue.Init(A,B,X);rescue.Process();}return;}
    else if constexpr(MODE==8){bool ok;{SmallExact<50>op;op.Init(A,B,X);ok=op.Process();}if(!ok){SmallComplete<50>rescue;rescue.Init(A,B,X);rescue.Process();}return;}
    else{
    GET_TILING_DATA_WITH_STRUCT(LinalgSolveTilingData, data, tiling);
    if constexpr (MODE == 4) {
        // Lanen256_: complete local-transform Cube factor and cached block solver.
        if(Lanen256_TryCubeLUT16(A,B,X,workspace,data))return;
        {OriginalRowSchur factor;factor.Init(A,B,workspace,data);factor.Process();}
        if ASCEND_IS_AIV {
            if((GetBlockIdx()/2)%data.pairsPerMatrix<data.k){
                if(GetBlockIdx()&1){Solver<false>worker;worker.Init(A,B,X,workspace,data);worker.ResidualWorker(workspace);}
                else{Solver<false>solver;solver.Init(A,B,X,workspace,data);solver.Process();}
            }
        }
        return;
    } else if constexpr (MODE == 3) {
        if ASCEND_IS_AIC {PairBlockedFactor<false>factor;factor.Init(A,B,workspace,data);factor.Process();}
        else{
            if(GetBlockIdx()&1){PairBlockedFactor<true>factor;factor.Init(A,B,workspace,data);factor.Process();}
            else{PairBlockedFactor<false>factor;factor.Init(A,B,workspace,data);factor.Process();}
            if(GetBlockIdx()&1){if(data.k>=2){Solver<false>worker;worker.Init(A,B,X,workspace,data);worker.ResidualWorker(workspace);}}
            else{Solver<false>solver;solver.Init(A,B,X,workspace,data);solver.Process();}
        }
        return;
    } else if constexpr (MODE == 2) {
        { BlockedFactor factor; factor.Init(A,B,workspace,data);factor.Process(); }
        if ASCEND_IS_AIV { Solver<true> solver;solver.Init(A,B,X,workspace,data);solver.Process(); }
        return;
    } else {
    if constexpr (MODE == 1) {
        if (GetBlockIdx() & 1) { ColumnFactor<true> factor; factor.Init(A, B, workspace, data); factor.Process(); }
        else { ColumnFactor<false> factor; factor.Init(A, B, workspace, data); factor.Process(); }
        if (GetBlockIdx() & 1) {
            if (data.k >= 2) { Solver<false> worker;worker.Init(A,B,X,workspace,data);worker.ResidualWorker(workspace); }
            return;
        }
    }
    if (data.rhsSplit && data.n == 50) { SmallCrout<50> op; op.Init(A, B, X, data); op.Process(); return; }
    if (data.rhsSplit && data.n == 64) { SmallCrout<64> op; op.Init(A, B, X, data); op.Process(); return; }
    if (data.onChip && data.k == 1 && data.n == 15) {
        TinyInverse<15> op; op.Init(A, B, X, data); op.Process(); return;
    }
    if (data.onChip && data.k == 1 && data.n == 16) {
        TinyInverse<16> op; op.Init(A, B, X, data); op.Process(); return;
    }
    if (data.onChip) {
        Solver<true> solver; solver.Init(A, B, X, workspace, data); solver.Process();
    } else {
        Solver<false> solver; solver.Init(A, B, X, workspace, data); solver.Process();
    }
    }
    }
}
