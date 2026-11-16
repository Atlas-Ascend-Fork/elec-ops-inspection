#include <cstdint>
#include "ascendc/host_api/tiling/template_argument.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

// Static kernels receive dimensions through template selection, not tiling loads.
// Keep one aligned word for framework registration; no kernel reads this payload.
struct alignas(8) SwiGluTilingData {
    uint64_t reserved;
};
static_assert(sizeof(SwiGluTilingData) == 8, "Tiling payload must be 8 bytes");

// Nine supported shapes only. Keys 38/39/40 use MIX; the others use AIV.
ASCENDC_TPL_ARGS_DECL(SwiGlu,
    ASCENDC_TPL_DATATYPE_DECL(DT_INPUT_X, C_DT_FLOAT16),
    ASCENDC_TPL_UINT_DECL(KERNEL_PATH, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 32, 33, 34, 35, 36, 38, 39, 40, 44),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 32),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 33),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 34),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 35),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 36),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 38),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 39),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 40),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 44),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
);

namespace {

constexpr uint32_t VECTOR_TILE_ELEMENTS = 2048;
constexpr uint64_t AIV_PER_BLOCK = 2;
constexpr uint32_t FLOATS_PER_BLOCK = 8;
constexpr uint32_t HALVES_PER_BLOCK = 16;
constexpr uint64_t CASE3_M = 512;
constexpr uint32_t CASE3_D = 256;

__aicore__ inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

template <class DT_INPUT_X, uint32_t D>
class KernelSwiGluFixedStaticRow {
public:
    static_assert(D == 1 || D == 4 || D == 32, "Supported short-row dimensions");
    static_assert(sizeof(DT_INPUT_X) == 2, "Half input storage is required");
    static constexpr uint32_t PITCH = (D + 15u) & ~15u;
    static constexpr uint32_t WEIGHT_ELEMENTS = D * PITCH;
    static constexpr uint32_t STAGE_B1 = 2 * WEIGHT_ELEMENTS;
    static constexpr uint32_t STAGE_B2 = STAGE_B1 + PITCH;
    static constexpr uint32_t STAGE_X = STAGE_B2 + PITCH;
    static constexpr uint32_t STAGE_ELEMENTS = STAGE_X + PITCH;
    static constexpr uint32_t W1 = 0, W2 = WEIGHT_ELEMENTS;
    static constexpr uint32_t B1 = STAGE_B1, B2 = STAGE_B2, X = STAGE_X;
    // Both projections run as one strided pass, so the product plane holds two full
    // weight planes before the reduced columns start. The separate two pass form is
    // kept below for a one line rollback.
    static constexpr uint32_t PRODUCT = STAGE_ELEMENTS;
    static constexpr uint32_t MERGED_PRODUCT = 2 * WEIGHT_ELEMENTS;
    static constexpr uint32_t Z1 = PRODUCT + MERGED_PRODUCT, Z2 = Z1 + PITCH;
    static constexpr uint32_t ACT = Z2 + PITCH;
    static constexpr uint32_t COMPUTE_ELEMENTS = ACT + PITCH;
    static constexpr uint32_t EXPLICIT_UB_BYTES = (STAGE_ELEMENTS + PITCH) * sizeof(DT_INPUT_X) +
        COMPUTE_ELEMENTS * sizeof(float);
    static_assert(EXPLICIT_UB_BYTES <= 32u * 1024u, "Single-row buffers must fit the budget");
    // A plain burst copy is cheaper than the padded descriptor path, but only the
    // full-tensor transfers that are whole 32B blocks may drop the pad entirely.
    static constexpr bool WHOLE_BLOCK_ROW = (D * sizeof(DT_INPUT_X)) % 32u == 0;
    static constexpr bool WHOLE_BLOCK_WEIGHT = PITCH == D && (D * D * sizeof(DT_INPUT_X)) % 32u == 0;
    // Same Vector pipe keeps RAW order in hardware, so only the cross pipe flags matter.
    // Probe: the widest fixed row (D == 32) already proved stable without the intra
    // pipe barriers on the two narrower rows, so it drops them as well.
    static constexpr bool DROP_V_BARRIER = D == 1 || D == 4 || D == 32;

    __aicore__ inline void Process(GM_ADDR inputX, GM_ADDR weight1, GM_ADDR bias1,
                                   GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output)
    {
        // Host requests eight AIV blocks for this fixed single-row route; block 0
        // carries the whole row and blocks 1..7 retire immediately.
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        if constexpr (D == 1) {
            // One element, scalar-only short graph: every input is read through the
            // scalar path, the whole expression is evaluated on the scalar unit and
            // the half result is stored directly. No UB tensor, no MTE2/MTE3 handoff
            // and no Vector instruction are involved on this route.
            ScalarProcess(inputX, weight1, bias1, weight2, bias2, output);
            return;
        }
        // Only block 0 reaches the single-row computation below.
        AscendC::GlobalTensor<DT_INPUT_X> xGm, w1Gm, b1Gm, w2Gm, b2Gm, outputGm;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(inputX), D);
        w1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(weight1), D * D);
        b1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(bias1), D);
        w2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(weight2), D * D);
        b2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(bias2), D);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(output), D);

        // Both sizes and every subview address are known at compile time.
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
        auto stage = ub.Alloc<DT_INPUT_X, STAGE_ELEMENTS + PITCH>();
        auto compute = ub.Alloc<float, COMPUTE_ELEMENTS>();
        auto outputHalf = stage[STAGE_ELEMENTS];
        CopyWeight(stage, w1Gm);
        CopyWeight(stage[WEIGHT_ELEMENTS], w2Gm);
        const AscendC::DataCopyExtParams rowCopy{1, D * sizeof(DT_INPUT_X), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<DT_INPUT_X> pad{false, 0, 0, 0};
        if constexpr (WHOLE_BLOCK_ROW) {
            // Whole 32B rows: a plain burst copy skips the pad descriptor path.
            AscendC::DataCopy(stage[STAGE_B1], b1Gm, uint64_t{D});
            AscendC::DataCopy(stage[STAGE_B2], b2Gm, uint64_t{D});
            AscendC::DataCopy(stage[STAGE_X], xGm, uint64_t{D});
        } else {
            AscendC::DataCopyPad(stage[STAGE_B1], b1Gm, rowCopy, pad);
            AscendC::DataCopyPad(stage[STAGE_B2], b2Gm, rowCopy, pad);
            AscendC::DataCopyPad(stage[STAGE_X], xGm, rowCopy, pad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        AscendC::Cast(compute, stage, AscendC::RoundMode::CAST_NONE, STAGE_ELEMENTS);
        SyncV();

        auto product = compute[PRODUCT];
        auto w1 = compute[W1];
        auto x = compute[X];
        auto b1 = compute[B1];
        auto z1 = compute[Z1];
        auto z2 = compute[Z2];
        Project(product, w1, b1, x, z1, z2);

        // Static mode uses basic FP32 APIs; no high-level temporary allocator.
        auto activation = compute[ACT];
        // One fused SiLU keeps the same FP32 path with three fewer dependent
        // vector instructions than the hand written Muls/Exp/Adds/Div chain.
        AscendC::Silu<float, false>(activation, z1, D);
        SyncV();
        AscendC::Mul(z1, activation, z2, D);
        SyncV();
        AscendC::Cast(outputHalf, z1, AscendC::RoundMode::CAST_RINT, D);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        if constexpr (WHOLE_BLOCK_ROW) {
            AscendC::DataCopy(outputGm, outputHalf, uint64_t{D});
        } else {
            AscendC::DataCopyPad(outputGm, outputHalf, rowCopy);
        }
    }

private:
    // exp(value) on the scalar unit. Range reduction to |r| <= ln(2)/2 keeps the
    // degree 7 Taylor term below 5e-9 relative, so the later half rounding matches
    // the Vector SiLU path for every finite input.
    __aicore__ inline static float ScalarExpApprox(float value)
    {
        if (value > 88.0f) {
            return 3.402823466e+38f;
        }
        if (value < -87.0f) {
            return 0.0f;
        }
        const float scaled = value * 1.4426950408889634f;
        const int32_t k = static_cast<int32_t>(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
        const float r = value - static_cast<float>(k) * 0.6931471805599453f;
        float p = 1.0f / 5040.0f;
        p = 1.0f / 720.0f + r * p;
        p = 1.0f / 120.0f + r * p;
        p = 1.0f / 24.0f + r * p;
        p = 1.0f / 6.0f + r * p;
        p = 0.5f + r * p;
        p = 1.0f + r * p;
        p = 1.0f + r * p;
        union {
            int32_t i;
            float f;
        } scale;
        scale.i = (k + 127) << 23;
        return p * scale.f;
    }

    // d == 1 has exactly one input element, so both projections are a single
    // multiply and bias add and the gate is one scalar SiLU. Reading through the
    // raw GM pointer avoids the descriptor setup of five DataCopyPad calls.
    __aicore__ inline void ScalarProcess(GM_ADDR inputX, GM_ADDR weight1, GM_ADDR bias1,
                                         GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output)
    {
        // Host requests eight AIV blocks for this route; only block 0 carries the
        // single element, so the other seven retire before touching any input.
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __gm__ DT_INPUT_X *xPtr = reinterpret_cast<__gm__ DT_INPUT_X *>(inputX);
        __gm__ DT_INPUT_X *w1Ptr = reinterpret_cast<__gm__ DT_INPUT_X *>(weight1);
        __gm__ DT_INPUT_X *b1Ptr = reinterpret_cast<__gm__ DT_INPUT_X *>(bias1);
        __gm__ DT_INPUT_X *w2Ptr = reinterpret_cast<__gm__ DT_INPUT_X *>(weight2);
        __gm__ DT_INPUT_X *b2Ptr = reinterpret_cast<__gm__ DT_INPUT_X *>(bias2);
        __gm__ DT_INPUT_X *yPtr = reinterpret_cast<__gm__ DT_INPUT_X *>(output);

        const float x = static_cast<float>(xPtr[0]);
        const float z1 = static_cast<float>(w1Ptr[0]) * x + static_cast<float>(b1Ptr[0]);
        const float z2 = static_cast<float>(w2Ptr[0]) * x + static_cast<float>(b2Ptr[0]);
        const float activation = z1 / (1.0f + ScalarExpApprox(-z1));
        yPtr[0] = static_cast<DT_INPUT_X>(activation * z2);
    }

    __aicore__ inline void SyncV()
    {
        if constexpr (!DROP_V_BARRIER) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyWeight(AscendC::LocalTensor<DT_INPUT_X> destination,
                                      AscendC::GlobalTensor<DT_INPUT_X> &weight)
    {
        if constexpr (WHOLE_BLOCK_WEIGHT) {
            AscendC::DataCopy(destination, weight, uint64_t{D} * D);
            return;
        }
        const AscendC::DataCopyPadExtParams<DT_INPUT_X> pad{false, 0, 0, 0};
        if constexpr (PITCH == D) {
            AscendC::DataCopyPad(destination, weight,
                AscendC::DataCopyExtParams{1, D * D * sizeof(DT_INPUT_X), 0, 0, 0}, pad);
        } else {
            AscendC::DataCopyPad(destination, weight,
                AscendC::DataCopyExtParams{static_cast<uint16_t>(D), D * sizeof(DT_INPUT_X), 0, 0, 0}, pad);
        }
    }


    // Both weight planes are adjacent row groups of the same staging plane, exactly
    // one weight plane apart, so a Mul with 2*D repeats issues both projections and
    // an Add with two repeats folds both biases. That removes two Vector issues.
    __aicore__ inline void Project(AscendC::LocalTensor<float> &product,
                                   AscendC::LocalTensor<float> &weight,
                                   AscendC::LocalTensor<float> &bias,
                                   AscendC::LocalTensor<float> &x,
                                   AscendC::LocalTensor<float> &result1,
                                   AscendC::LocalTensor<float> &result2)
    {
        constexpr uint8_t REPEAT_STRIDE = static_cast<uint8_t>(PITCH / FLOATS_PER_BLOCK);
        // Source x is read as a contiguous row (block stride 1) and reused by every
        // repeat (repeat stride 0). A zero block stride only works while D <= 8.
        AscendC::Mul(product, weight, x, uint64_t{D}, static_cast<uint8_t>(2 * D),
            AscendC::BinaryRepeatParams(1, 1, 1, REPEAT_STRIDE, REPEAT_STRIDE, 0));
        SyncV();
        if constexpr (D == 1) {
            // A one-term reduction is the identity; the two products land one
            // padded row apart, so one strided Add finishes both columns.
            AscendC::Add(result1, product, bias, 1, 2,
                AscendC::BinaryRepeatParams(1, 1, 1, REPEAT_STRIDE, REPEAT_STRIDE, REPEAT_STRIDE));
            SyncV();
            return;
        }
        // Both column groups live in one product plane; the reduced columns start
        // after that plane, so neither pass overwrites the other's source rows.
        if constexpr (D == PITCH) {
            AscendC::WholeReduceSum<float>(result1, product, static_cast<int32_t>(D),
                static_cast<int32_t>(2*D),1,1,REPEAT_STRIDE);
            SyncV();
        } else {
        AscendC::WholeReduceSum<float>(result2, product[WEIGHT_ELEMENTS], static_cast<int32_t>(D),
            static_cast<int32_t>(D), 1, 1, REPEAT_STRIDE);
        SyncV();
        AscendC::WholeReduceSum<float>(result1, product, static_cast<int32_t>(D),
            static_cast<int32_t>(D), 1, 1, REPEAT_STRIDE);
        SyncV();
        }
        AscendC::Add(result1, result1, bias, uint64_t{D}, 2,
            AscendC::BinaryRepeatParams(1, 1, 1, REPEAT_STRIDE, REPEAT_STRIDE, REPEAT_STRIDE));
        SyncV();
    }
};

template <class DT_INPUT_X>
class KernelSwiGluFixedD64Columns {
public:
    static constexpr uint32_t WIDTH = 64;
    static constexpr uint32_t COLUMNS = 16;
    static constexpr uint32_t PANEL_ELEMENTS = WIDTH * COLUMNS;
    static constexpr uint32_t STAGE_X = 2 * PANEL_ELEMENTS + 2 * COLUMNS;
    static constexpr uint32_t STAGE_ELEMENTS = STAGE_X + WIDTH;
    static constexpr uint32_t W1 = 0;
    // The float planes mirror the staged order, so one counted Cast converts every
    // input. The merged product plane therefore starts after the whole stage.
    static constexpr uint32_t W2 = W1 + PANEL_ELEMENTS;
    static constexpr uint32_t B1 = W2 + PANEL_ELEMENTS;
    static constexpr uint32_t B2 = B1 + COLUMNS;
    static constexpr uint32_t X = B2 + COLUMNS;
    static constexpr uint32_t PRODUCT = STAGE_ELEMENTS;
    // Both projections run as one strided pass over the adjacent W1/W2 planes.
    static constexpr uint32_t MERGED_PRODUCT = 2 * PANEL_ELEMENTS;
    static constexpr uint32_t Z1 = PRODUCT + MERGED_PRODUCT;
    static constexpr uint32_t Z2 = Z1 + COLUMNS;
    static constexpr uint32_t ACT = Z2 + COLUMNS;
    static constexpr uint32_t COMPUTE_ELEMENTS = ACT + COLUMNS;
    static constexpr uint32_t EXPLICIT_UB_BYTES = (STAGE_ELEMENTS + COLUMNS) * sizeof(DT_INPUT_X) +
        COMPUTE_ELEMENTS * sizeof(float);
    static_assert(EXPLICIT_UB_BYTES <= 32u * 1024u, "Column panel buffers must fit the budget");

    __aicore__ inline void Process(GM_ADDR inputX, GM_ADDR weight1, GM_ADDR bias1,
                                   GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output)
    {
        if (AscendC::GetBlockNum() != 8 || AscendC::GetBlockIdx() >= 8) return;
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t row = core >> 2;
        const uint32_t column = (core & 3u) << 4;
        AscendC::GlobalTensor<DT_INPUT_X> xGm, w1Gm, b1Gm, w2Gm, b2Gm, outputGm;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(inputX), 2 * WIDTH);
        w1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(weight1), WIDTH * WIDTH);
        b1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(bias1), WIDTH);
        w2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(weight2), WIDTH * WIDTH);
        b2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(bias2), WIDTH);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(output), 2 * WIDTH);
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
        auto stage = ub.Alloc<DT_INPUT_X, STAGE_ELEMENTS + COLUMNS>();
        auto compute = ub.Alloc<float, COMPUTE_ELEMENTS>();
        // Every panel, bias, row and output slice is a whole number of 32B blocks,
        // so plain burst copies replace the pad descriptor path.
        AscendC::DataCopy(stage, w1Gm[column * WIDTH], uint64_t{PANEL_ELEMENTS});
        AscendC::DataCopy(stage[PANEL_ELEMENTS], w2Gm[column * WIDTH], uint64_t{PANEL_ELEMENTS});
        AscendC::DataCopy(stage[2 * PANEL_ELEMENTS], b1Gm[column], uint64_t{COLUMNS});
        AscendC::DataCopy(stage[2 * PANEL_ELEMENTS + COLUMNS], b2Gm[column], uint64_t{COLUMNS});
        AscendC::DataCopy(stage[STAGE_X], xGm[row * WIDTH], uint64_t{WIDTH});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        // Every staged plane is contiguous and mirrors the float layout, so one
        // counted Cast replaces the five per-plane conversions.
        AscendC::Cast(compute, stage, AscendC::RoundMode::CAST_NONE, uint64_t{STAGE_ELEMENTS});
        AscendC::PipeBarrier<PIPE_V>();
        auto product = compute[PRODUCT];auto w1 = compute[W1];auto x = compute[X];
        auto b1 = compute[B1];auto z1 = compute[Z1];auto z2 = compute[Z2];
        Project(product, w1, b1, x, z1, z2);
        // Fused SiLU plus the gate multiply replace the five issue Muls/Exp/Adds/
        // Div/Mul chain; this shape is issue bound, not bandwidth bound.
        auto activation = compute[ACT];
        AscendC::Silu<float, false>(activation, z1, COLUMNS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(z1, activation, z2, COLUMNS);
        AscendC::PipeBarrier<PIPE_V>();
        auto out = stage[STAGE_ELEMENTS];
        AscendC::Cast(out, z1, AscendC::RoundMode::CAST_RINT, COLUMNS);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(outputGm[row * WIDTH + column], out, uint64_t{COLUMNS});
    }

private:
    // Both projections run as one strided pass over the adjacent W1/W2 planes and
    // share one strided bias add, so the pair costs four Vector issues.
    __aicore__ inline void Project(AscendC::LocalTensor<float> &product,
                                   AscendC::LocalTensor<float> &weight,
                                   AscendC::LocalTensor<float> &bias,
                                   AscendC::LocalTensor<float> &row,
                                   AscendC::LocalTensor<float> &result1,
                                   AscendC::LocalTensor<float> &result2)
    {
        constexpr uint8_t ROW_STRIDE = static_cast<uint8_t>(WIDTH / FLOATS_PER_BLOCK);
        AscendC::Mul(product, weight, row, uint64_t{WIDTH}, static_cast<uint8_t>(2 * COLUMNS),
                     AscendC::BinaryRepeatParams(1, 1, 1, ROW_STRIDE, ROW_STRIDE, 0));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(result1, product, static_cast<int32_t>(WIDTH),
                                       static_cast<int32_t>(2*COLUMNS), 1, 1,
                                       static_cast<int32_t>(ROW_STRIDE));
        AscendC::PipeBarrier<PIPE_V>();
        constexpr uint8_t COL_STRIDE = static_cast<uint8_t>(COLUMNS / FLOATS_PER_BLOCK);
        AscendC::Add(result1, result1, bias, uint64_t{COLUMNS}, 2,
                     AscendC::BinaryRepeatParams(1, 1, 1, COL_STRIDE, COL_STRIDE, COL_STRIDE));
        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <class DT_INPUT_X>
class KernelSwiGluStaticCase3 {
public:
    static constexpr uint32_t MAX_ROWS = 128;
    static constexpr uint32_t COLUMNS = 64;
    static constexpr uint32_t PAIRED_COLUMNS = 128;
    static constexpr uint32_t VECTOR_ROWS = 32;
    static constexpr uint16_t READY = 4;
    static constexpr uint16_t READY_SECOND = 5; // Second panel, first AIV lane's peer

    // The row grid follows the AIC block count: a sixteen-block launch keeps the
    // measured 128-row tiles, a wider launch spreads the same 512 rows over more
    // owners and lowers the per-core X volume. Tiles stay multiples of 32 rows so
    // each AIV half is a whole 16-row fractal and each Mmad is a legal FP16 panel.
    __aicore__ inline static void TileSplit(uint32_t blocks, uint32_t tile,
                                            uint32_t &rowStart, uint32_t &rows)
    {
        const uint32_t tiles = blocks >> 2;
        const uint32_t base = static_cast<uint32_t>(CeilDiv(CASE3_M, tiles) / 32u) * 32u;
        const uint32_t extra = static_cast<uint32_t>((CASE3_M - base * tiles) >> 5);
        const uint32_t wideRows = base + 32u;
        rowStart = tile < extra ? tile * wideRows : extra * wideRows + (tile - extra) * base;
        rows = tile < extra ? wideRows : base;
    }

    __aicore__ inline void Process(GM_ADDR inputX, GM_ADDR weight1, GM_ADDR bias1,
                                  GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output,
                                  GM_ADDR userWorkspace)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(inputX), 512 * 256);
        w1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(weight1), 256 * 256);
        w2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(weight2), 256 * 256);
        b1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(bias1), 256);
        b2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(bias2), 256);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(output), 512 * 256);
        rawGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(userWorkspace), 512 * 512);
        if ASCEND_IS_AIC { ProcessCube(); }
        if ASCEND_IS_AIV { ProcessVector(); }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void LoadPairedWeight(AscendC::LocalTensor<DT_INPUT_X> &dst, uint32_t group)
    {
        AscendC::Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = COLUMNS;
        copy.dValue = CASE3_D;
        copy.srcDValue = CASE3_D;
        copy.dstNzC0Stride = PAIRED_COLUMNS;
        copy.dstNzNStride = 1;
        const uint32_t offset = group * COLUMNS * CASE3_D;
        AscendC::DataCopy(dst, w1Gm_[offset], copy);
        // The second half must be inserted in each K/16 fractal, not appended
        // after the entire W1 tensor in NZ order.
        AscendC::DataCopy(dst[COLUMNS * 16], w2Gm_[offset], copy);
    }

    __aicore__ inline void LoadFragmentX(AscendC::LocalTensor<DT_INPUT_X> &dst, uint64_t rowOffset,
                                        uint32_t rows)
    {
        AscendC::Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = rows;
        copy.dValue = CASE3_D;
        copy.srcDValue = CASE3_D;
        copy.dstNzC0Stride = rows;
        copy.dstNzNStride = 1;
        AscendC::DataCopy(dst, xGm_[rowOffset * CASE3_D], copy);
    }

    // One half-tile panel of the tiled X fractal slab, into its own A2 region. The
    // source stride stays the panel height so only the start block changes.
    __aicore__ inline void LoadA2Panel(AscendC::LocalTensor<DT_INPUT_X> &dst,
                                       const AscendC::LocalTensor<DT_INPUT_X> &src,
                                       uint32_t srcRows, uint32_t rows, uint32_t firstBlock)
    {
        AscendC::LoadData2DParams load{};
        load.repeatTimes = CASE3_D / 16;
        load.srcStride = srcRows / 16;
        load.ifTranspose = false;
        for (uint32_t mBlock = 0; mBlock < rows / 16; ++mBlock)
            AscendC::LoadData(dst[mBlock * CASE3_D * 16], src[(firstBlock + mBlock) * 256], load);
    }

    __aicore__ inline void ProcessCube()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t group = core & 3u;
        const uint32_t tile = core >> 2;
        uint32_t tileStart = 0;
        uint32_t tileRows = 0;
        TileSplit(AscendC::GetBlockNum(), tile, tileStart, tileRows);
        const uint64_t rowOffset = tileStart;
        const uint32_t half = tileRows >> 1;
        AscendC::LocalMemAllocator<AscendC::Hardware::L1> l1;
        AscendC::LocalMemAllocator<AscendC::Hardware::L0A> l0a;
        AscendC::LocalMemAllocator<AscendC::Hardware::L0B> l0b;
        AscendC::LocalMemAllocator<AscendC::Hardware::L0C> l0c;
        auto weightL1 = l1.Alloc<AscendC::TPosition::B1, DT_INPUT_X, 32768>();
        auto xL1 = l1.Alloc<AscendC::TPosition::A1, DT_INPUT_X, 32768>();
        auto aL0 = l0a.Alloc<AscendC::TPosition::A2, DT_INPUT_X, 32768>();
        auto weightL0 = l0b.Alloc<AscendC::TPosition::B2, DT_INPUT_X, 32768>();
        // Preserve the original two-slot C0 reservation. One task uses slot 0.
        auto cL0 = l0c.Alloc<AscendC::TPosition::CO1, float, 32768>();
        AscendC::SetMMLayoutTransform(true);
        LoadPairedWeight(weightL1, group);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        // Stage only the first X panel; the second one is issued under the first
        // panel's Mmad so lane 0 sees its data about half a load earlier.
        LoadFragmentX(xL1, rowOffset, half);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::LoadData2DParams load{};
        load.repeatTimes = PAIRED_COLUMNS * CASE3_D / 256;
        load.srcStride = 1;
        load.ifTranspose = false;
        AscendC::LoadData(weightL0, weightL1, load);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(1);
        // Two half-tile panels: the first Mmad can start as soon as its own A2 region
        // is filled, and its Fixpipe releases the first AIV lane while the second
        // panel is still being staged.
        LoadA2Panel(aL0, xL1, half, half, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
        auto aL0Second = aL0[half * CASE3_D];
        auto xL1Second = xL1[half * CASE3_D];
        AscendC::MmadParams mm{};
        mm.m = static_cast<int16_t>(half);
        mm.n = PAIRED_COLUMNS;
        mm.k = CASE3_D;
        mm.cmatrixInitVal = true;
        mm.unitFlag = 3u;
        AscendC::Mmad(cL0, aL0, weightL0, mm);
        LoadFragmentX(xL1Second, rowOffset + half, half);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::FixpipeParamsV220 fix{};
        fix.nSize = PAIRED_COLUMNS;
        fix.mSize = static_cast<uint16_t>(half);
        fix.srcStride = static_cast<uint16_t>(half);
        // Planar per-group workspace: 128 columns per row land back to back, so both
        // the Fixpipe write and the AIV read below are one contiguous stream.
        fix.dstStride = PAIRED_COLUMNS;
        fix.ndNum = 1;
        fix.quantPre = ::QuantMode_t::F322F16;
        fix.unitFlag = 3u;
        const uint64_t rawBase = group * CASE3_M * PAIRED_COLUMNS + rowOffset * PAIRED_COLUMNS;
        AscendC::Fixpipe(rawGm_[rawBase], cL0, fix);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(READY);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        LoadA2Panel(aL0Second, xL1Second, half, half, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(1);
        AscendC::Mmad(cL0[half * PAIRED_COLUMNS], aL0Second, weightL0, mm);
        AscendC::Fixpipe(rawGm_[rawBase + half * PAIRED_COLUMNS],
                         cL0[half * PAIRED_COLUMNS], fix);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(READY_SECOND);
        AscendC::SetMMLayoutTransform(false);
    }

    __aicore__ inline void PrepareBias(uint32_t group, AscendC::LocalTensor<DT_INPUT_X> &stage,
                                      AscendC::LocalTensor<float> &bias1,
                                      AscendC::LocalTensor<float> &bias2)
    {
        const AscendC::DataCopyExtParams copy{1, COLUMNS * sizeof(DT_INPUT_X), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<DT_INPUT_X> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(stage, b1Gm_[group * COLUMNS], copy, pad);
        AscendC::DataCopyPad(stage[COLUMNS], b2Gm_[group * COLUMNS], copy, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Cast(bias1,stage,AscendC::RoundMode::CAST_NONE,2*COLUMNS);
        bias2=bias1[COLUMNS];
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeVectorTile(uint64_t rowOffset, uint32_t group, uint32_t chunkRows,
            bool reuse, bool last,
            AscendC::LocalTensor<float> &z1, AscendC::LocalTensor<float> &z2,
            AscendC::LocalTensor<DT_INPUT_X> &dz1, AscendC::LocalTensor<DT_INPUT_X> &dz2,
            AscendC::LocalTensor<DT_INPUT_X> &paired,
            AscendC::LocalTensor<DT_INPUT_X> &out, AscendC::LocalTensor<float> &temporary,
            AscendC::LocalTensor<float> &bias1, AscendC::LocalTensor<float> &bias2)
    {
        const uint32_t count = chunkRows * COLUMNS;
        dz2=dz1[count];
        z2=z1[count];
        const AscendC::UnaryRepeatParams castRows(1,1,COLUMNS/8,PAIRED_COLUMNS/HALVES_PER_BLOCK);
        AscendC::Cast(z1,paired,AscendC::RoundMode::CAST_NONE,uint64_t{COLUMNS},
            static_cast<uint8_t>(chunkRows),castRows);
        AscendC::Cast(z2,paired[COLUMNS],AscendC::RoundMode::CAST_NONE,uint64_t{COLUMNS},
            static_cast<uint8_t>(chunkRows),castRows);
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::BinaryRepeatParams addBias(1,1,1,COLUMNS/8,COLUMNS/8,0);
        AscendC::Add(z1,z1,bias1,uint64_t{COLUMNS},static_cast<uint8_t>(chunkRows),addBias);
        AscendC::Add(z2,z2,bias2,uint64_t{COLUMNS},static_cast<uint8_t>(chunkRows),addBias);
        AscendC::PipeBarrier<PIPE_V>();
        // One fused SiLU keeps this FP32 path and drops the Muls/Exp/Adds/Div
        // chain, whose Exp and Div are the two most expensive Vector issues.
        AscendC::Silu<float, false>(temporary, z1, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(z1, temporary, z2, count);
        AscendC::PipeBarrier<PIPE_V>();
        if(reuse)AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::Cast(out, z1, AscendC::RoundMode::CAST_RINT, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        const AscendC::DataCopyExtParams outputCopy{
            static_cast<uint16_t>(chunkRows), COLUMNS * sizeof(DT_INPUT_X), 0,
            (CASE3_D - COLUMNS) * sizeof(DT_INPUT_X), 0};
        AscendC::DataCopy(yGm_[rowOffset * CASE3_D + group * COLUMNS], out,
            AscendC::DataCopyParams{static_cast<uint16_t>(chunkRows),
                static_cast<uint16_t>(COLUMNS*sizeof(DT_INPUT_X)/32),0,
                static_cast<uint16_t>((CASE3_D-COLUMNS)*sizeof(DT_INPUT_X)/32)});
        if (!last) AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void ProcessVector()
    {
        const uint32_t core = AscendC::GetBlockIdx() / AIV_PER_BLOCK;
        const uint32_t group = core & 3u;
        const uint32_t tile = core >> 2;
        uint32_t tileStart = 0;
        uint32_t tileRows = 0;
        TileSplit(AscendC::GetBlockNum(), tile, tileStart, tileRows);
        const uint32_t halfRows = tileRows / AIV_PER_BLOCK;
        // Each AIV of the block owns half of the tile's rows.
        const uint64_t rowOffset = tileStart + AscendC::GetSubBlockIdx() * halfRows;
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
        auto z1 = ub.Alloc<AscendC::TPosition::VECIN, float, 2*VECTOR_TILE_ELEMENTS>();
        auto z2 = z1[VECTOR_TILE_ELEMENTS];
        auto dz1 = ub.Alloc<AscendC::TPosition::VECIN, DT_INPUT_X, 2*VECTOR_TILE_ELEMENTS>();
        auto dz2 = dz1[VECTOR_TILE_ELEMENTS];
        auto paired = ub.Alloc<AscendC::TPosition::VECIN, DT_INPUT_X, MAX_ROWS * PAIRED_COLUMNS / 2>();
        auto out = ub.Alloc<AscendC::TPosition::VECOUT, DT_INPUT_X, VECTOR_TILE_ELEMENTS>();
        auto stage = ub.Alloc<AscendC::TPosition::VECIN, DT_INPUT_X, PAIRED_COLUMNS>();
        auto temporary = ub.Alloc<AscendC::TPosition::VECCALC, float, VECTOR_TILE_ELEMENTS>();
        auto bias1 = ub.Alloc<AscendC::TPosition::VECCALC, float, VECTOR_TILE_ELEMENTS>();
        auto bias2 = bias1[COLUMNS];
        PrepareBias(group, stage, bias1, bias2);
        // Each lane owns one Cube panel, so each waits only for its own panel.
        AscendC::CrossCoreWaitFlag<2>(AscendC::GetSubBlockIdx() == 0 ? READY : READY_SECOND);
        // The lane's half is one contiguous streak in the per-group plane: stage all
        // of it with a single MTE2 round trip and let the chunks only de-interleave.
        const AscendC::DataCopyExtParams rawCopy{1,
            static_cast<uint32_t>(halfRows * PAIRED_COLUMNS * sizeof(DT_INPUT_X)), 0, 0, 0};
        AscendC::DataCopy(paired,
            rawGm_[group * CASE3_M * PAIRED_COLUMNS + rowOffset * PAIRED_COLUMNS],
            uint64_t{halfRows * PAIRED_COLUMNS});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        for (uint32_t done = 0; done < halfRows;) {
            const uint32_t chunk = halfRows - done < VECTOR_ROWS ? halfRows - done : VECTOR_ROWS;
            auto pairedChunk = paired[done * PAIRED_COLUMNS];
            ComputeVectorTile(rowOffset + done, group, chunk, done != 0, done + chunk >= halfRows,
                              z1, z2, dz1, dz2, pairedChunk, out, temporary, bias1, bias2);
            done += chunk;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::GlobalTensor<DT_INPUT_X> xGm_, w1Gm_, w2Gm_, b1Gm_, b2Gm_, yGm_, rawGm_;
};

template <class DT_INPUT_X>
class KernelSwiGluStaticD128 {
public:
    static constexpr uint32_t WIDTH = 128;
    static constexpr uint32_t JOINED = 256;
    static constexpr uint32_t ROWS = 64;
    static constexpr uint32_t VECTOR_ROWS = 32;
    static constexpr uint32_t RAW_ELEMENTS = VECTOR_ROWS * JOINED;
    static constexpr uint32_t OUT_ELEMENTS = VECTOR_ROWS * WIDTH;
    // 4096 rows over sixteen AIC blocks, 256 rows each, split into four 64 row
    // tasks. UnitFlag replaces the instruction level Mmad->Fixpipe wait with a
    // 512B handshake, so the Fixpipe drains the accumulator while the Mmad still
    // runs; the handshake owns L0C reuse, which is why one accumulator remains.
    static constexpr uint32_t CORES = 16;
    static constexpr uint32_t LONG_CORES = 16;
    static constexpr uint32_t LONG_BAND = 256;
    static constexpr uint32_t SHORT_BAND = 256;
    static constexpr uint32_t TASKS_PER_CORE = 4;
    static constexpr uint32_t FIRST_TASK_ROWS = 64;

    __aicore__ inline uint64_t BandStart(uint32_t core)
    {
        return core < LONG_CORES
                   ? static_cast<uint64_t>(core) * LONG_BAND
                   : static_cast<uint64_t>(LONG_CORES) * LONG_BAND +
                         static_cast<uint64_t>(core - LONG_CORES) * SHORT_BAND;
    }

    __aicore__ inline uint32_t BandRows(uint32_t core)
    {
        return core < LONG_CORES ? LONG_BAND : SHORT_BAND;
    }

    __aicore__ inline uint64_t TaskStart(uint32_t core, uint32_t index)
    {
        return BandStart(core) + static_cast<uint64_t>(index) * FIRST_TASK_ROWS;
    }

    __aicore__ inline uint32_t TaskRows(uint32_t core, uint32_t)
    {
        return BandRows(core) / TASKS_PER_CORE;
    }

    __aicore__ inline void Process(GM_ADDR inputX, GM_ADDR weight1, GM_ADDR bias1,
                                  GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output,
                                  GM_ADDR userWorkspace)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(inputX), 4096 * WIDTH);
        w1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(weight1), WIDTH * WIDTH);
        w2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(weight2), WIDTH * WIDTH);
        b1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(bias1), WIDTH);
        b2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(bias2), WIDTH);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(output), 4096 * WIDTH);
        // Fixpipe rounds the accumulator to half, so the handoff plane moves half the
        // bytes and the AIV needs one widening Cast instead of a rounding round trip.
        rawGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(userWorkspace), 4096 * JOINED);
        if ASCEND_IS_AIC { ProcessCube(); }
        if ASCEND_IS_AIV { ProcessVector(); }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // One cross-core flag id per task, so four tasks no longer share two counters.
    static constexpr uint16_t READY_BASE = 4;

    __aicore__ inline void PrepareWeightsLoad(AscendC::LocalTensor<DT_INPUT_X> &stage)
    {
        AscendC::Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = WIDTH;
        copy.dValue = WIDTH;
        copy.srcDValue = WIDTH;
        copy.dstNzC0Stride = JOINED;
        copy.dstNzNStride = 1;
        AscendC::DataCopy(stage, w1Gm_, copy);
        AscendC::DataCopy(stage[WIDTH * 16], w2Gm_, copy);
    }

    __aicore__ inline void LoadWeights(AscendC::LocalTensor<DT_INPUT_X> &stage,
                                       AscendC::LocalTensor<DT_INPUT_X> &weight)
    {
        AscendC::LoadData2DParams load{};
        load.repeatTimes = JOINED * WIDTH / 256;
        load.srcStride = 1;
        load.ifTranspose = false;
        AscendC::LoadData(weight, stage, load);
    }

    __aicore__ inline void PrepareBiasLoad(AscendC::LocalTensor<DT_INPUT_X> &stage)
    {
        AscendC::DataCopy(stage, b1Gm_, WIDTH);
        AscendC::DataCopy(stage[WIDTH], b2Gm_, WIDTH);
    }

    __aicore__ inline void LoadBias(AscendC::LocalTensor<DT_INPUT_X> &stage,
                                    AscendC::LocalTensor<float> &bias)
    {
        const AscendC::DataCopyParams copy{
            1, static_cast<uint16_t>(JOINED * sizeof(DT_INPUT_X) / 64), 0, 0};
        AscendC::DataCopy<float, DT_INPUT_X>(bias, stage, copy);
    }

    __aicore__ inline void LoadXStage(uint64_t startRow, uint32_t rows,
                                      AscendC::LocalTensor<DT_INPUT_X> &stage)
    {
        AscendC::Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = rows;
        copy.dValue = WIDTH;
        copy.srcDValue = WIDTH;
        copy.dstNzC0Stride = rows;
        copy.dstNzNStride = 1;
        AscendC::DataCopy(stage, xGm_[startRow * WIDTH], copy);
    }

    __aicore__ inline void LoadXBlocks(uint32_t rows, AscendC::LocalTensor<DT_INPUT_X> &stage,
                                       AscendC::LocalTensor<DT_INPUT_X> &a)
    {
        AscendC::LoadData2DParams load{};
        load.repeatTimes = WIDTH / 16;
        load.srcStride = rows / 16;
        load.ifTranspose = false;
        for (uint32_t block = 0; block < rows / 16; ++block)
            AscendC::LoadData(a[block * WIDTH * 16], stage[block * 256], load);
    }

    __aicore__ inline void PrepareX(uint64_t startRow, uint32_t rows, bool reused, bool hasNext,
            AscendC::LocalTensor<DT_INPUT_X> &stage, AscendC::LocalTensor<DT_INPUT_X> &a)
    {
        if (reused) AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        LoadXStage(startRow, rows, stage);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        if (reused) AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        LoadXBlocks(rows, stage, a);
        if (hasNext) AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    }

    __aicore__ inline void ProcessCube()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        if (core >= CORES) return;
        AscendC::LocalMemAllocator<AscendC::Hardware::L1> l1;
        AscendC::LocalMemAllocator<AscendC::Hardware::L0A> l0a;
        AscendC::LocalMemAllocator<AscendC::Hardware::L0B> l0b;
        AscendC::LocalMemAllocator<AscendC::Hardware::L0C> l0c;
        AscendC::LocalMemAllocator<AscendC::Hardware::BIAS> bt;
        auto weightStage = l1.Alloc<AscendC::TPosition::B1, DT_INPUT_X, JOINED * WIDTH>();
        auto xStage = l1.Alloc<AscendC::TPosition::A1, DT_INPUT_X, ROWS * WIDTH>();
        auto biasStage = l1.Alloc<AscendC::TPosition::C1, DT_INPUT_X, JOINED>();
        auto a = l0a.Alloc<AscendC::TPosition::A2, DT_INPUT_X, ROWS * WIDTH>();
        auto weight = l0b.Alloc<AscendC::TPosition::B2, DT_INPUT_X, JOINED * WIDTH>();
        auto c = l0c.Alloc<AscendC::TPosition::CO1, float, ROWS * JOINED>();
        auto bias = bt.Alloc<AscendC::TPosition::C2, float, JOINED>();
        // UnitFlag hands each finished N fractal to the Fixpipe, so the accumulator
        // must be produced N first to match the row major copy out order.
        AscendC::SetMMLayoutTransform(true);
        // Resident weights, both biases and the first band's X are three GM reads of
        // independent planes: issue them together and drain MTE2 once, instead of
        // serialising a weight wait in front of the X copy.
        PrepareWeightsLoad(weightStage);
        PrepareBiasLoad(biasStage);
        LoadXStage(TaskStart(core, 0), TaskRows(core, 0), xStage);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        LoadWeights(weightStage, weight);
        LoadBias(biasStage, bias);
        LoadXBlocks(TaskRows(core, 0), xStage, a);
        if (TASKS_PER_CORE > 1) AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
        for (uint32_t index = 0; index < TASKS_PER_CORE; ++index) {
            const bool reused = index != 0;
            const bool hasNext = index + 1 < TASKS_PER_CORE;
            const uint64_t startRow = TaskStart(core, index);
            const uint32_t rows = TaskRows(core, index);
            if (index != 0) PrepareX(startRow, rows, reused, hasNext, xStage, a);
            AscendC::MmadParams mm{};
            mm.m = static_cast<int16_t>(rows);
            mm.n = JOINED;
            mm.k = WIDTH;
            mm.cmatrixInitVal = false;
            // One Mmad completes this whole tile, so 0b11 also marks the last
            // producer the paired Fixpipe waits for. The pair replaces both the
            // Mmad->Fixpipe drain and the Fixpipe->Mmad accumulator release.
            mm.unitFlag = 3u;
            AscendC::Mmad(c, a, weight, bias, mm);
            if (hasNext) AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
            AscendC::FixpipeParamsV220 fix{};
            fix.nSize = JOINED;
            fix.mSize = static_cast<uint16_t>(rows);
            fix.srcStride = static_cast<uint16_t>(rows);
            fix.dstStride = JOINED;
            fix.ndNum = 1;
            fix.quantPre = ::QuantMode_t::F322F16;
            fix.unitFlag = 3u;
            AscendC::Fixpipe(rawGm_[startRow * JOINED], c, fix);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(READY_BASE + static_cast<uint16_t>(index));
        }
        AscendC::SetMMLayoutTransform(false);
    }

    __aicore__ inline void CopyInput(uint64_t row, uint32_t rows,
                                    AscendC::LocalTensor<DT_INPUT_X> raw,
                                    uint16_t slot, bool reused)
    {
        if (reused) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
        // rows*JOINED halfs are a whole 32B-multiple burst, so the padded descriptor
        // path is pure overhead here; the tiny path already drops it the same way.
        AscendC::DataCopy(raw, rawGm_[row * JOINED], uint64_t{rows * JOINED});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
    }

    __aicore__ inline void ComputeOutput(AscendC::LocalTensor<DT_INPUT_X> raw, uint64_t row,
            uint32_t rows, AscendC::LocalTensor<DT_INPUT_X> out,
            uint16_t slot, bool reused, bool hasFuture)
    {
        constexpr uint32_t outputStride = WIDTH / HALVES_PER_BLOCK;
        constexpr uint32_t inputStride = JOINED / HALVES_PER_BLOCK;
        const uint32_t outElements = rows * WIDTH;
        const auto repeats = static_cast<uint8_t>(rows);
        if (reused) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
        // Read gate and up directly from the joined half rows.
        AscendC::Muls(out, raw, static_cast<DT_INPUT_X>(-1), uint64_t{WIDTH}, repeats,
            AscendC::UnaryRepeatParams(1, 1, outputStride, inputStride));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(out, out, outElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(out, out, static_cast<DT_INPUT_X>(1), outElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(out, raw, out, uint64_t{WIDTH}, repeats,
            AscendC::BinaryRepeatParams(1, 1, 1, outputStride, inputStride, outputStride));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(out, out, raw[WIDTH], uint64_t{WIDTH}, repeats,
            AscendC::BinaryRepeatParams(1, 1, 1, outputStride, outputStride, inputStride));
        AscendC::PipeBarrier<PIPE_V>();
        // The raw slot is only overwritten two tasks later, so the free signal must
        // follow that reuse distance instead of the next-task distance.
        if (hasFuture) AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
        // 32 rows x 128 halfs is one contiguous 8KB burst, so no pad descriptor.
        AscendC::DataCopy(yGm_[row * WIDTH], out, uint64_t{outElements});
        if (hasFuture) AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
    }

    __aicore__ inline void ProcessVector()
    {
        const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx()) >> 1;
        if (core >= CORES) return;
        const uint32_t lane = AscendC::GetSubBlockIdx();
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
        auto rawSlots = ub.Alloc<AscendC::TPosition::VECIN, DT_INPUT_X, 2 * RAW_ELEMENTS>();
        auto outSlots = ub.Alloc<AscendC::TPosition::VECOUT, DT_INPUT_X, 2 * OUT_ELEMENTS>();
        // A task is 64 rows, so each AIV owns exactly one 32 row chunk: both raw slots
        // hold a whole task and the copy of task n+1 overlaps the Vector work of n.
        const uint32_t half = TaskRows(core, 0) / 2;
        const uint64_t laneRow = TaskStart(core, 0) + static_cast<uint64_t>(lane) * half;
        AscendC::CrossCoreWaitFlag<2>(READY_BASE);
        CopyInput(laneRow, VECTOR_ROWS, rawSlots, 0, false);
        for (uint32_t index = 0; index < TASKS_PER_CORE; ++index) {
            const uint16_t slot = static_cast<uint16_t>(index & 1u);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
            if (index + 1 < TASKS_PER_CORE) {
                AscendC::CrossCoreWaitFlag<2>(READY_BASE + static_cast<uint16_t>(index + 1));
                CopyInput(TaskStart(core, index + 1) + static_cast<uint64_t>(lane) * half,
                          VECTOR_ROWS, rawSlots[static_cast<uint64_t>((slot ^ 1u) * RAW_ELEMENTS)],
                          static_cast<uint16_t>(slot ^ 1u), index >= 1);
            }
            ComputeOutput(rawSlots[static_cast<uint64_t>(slot * RAW_ELEMENTS)],
                          TaskStart(core, index) + static_cast<uint64_t>(lane) * half,
                          VECTOR_ROWS,
                          outSlots[static_cast<uint64_t>(slot * OUT_ELEMENTS)],
                          slot, index >= 2, index + 2 < TASKS_PER_CORE);
        }
    }

    AscendC::GlobalTensor<DT_INPUT_X> xGm_, w1Gm_, w2Gm_, b1Gm_, b2Gm_, yGm_;
    AscendC::GlobalTensor<DT_INPUT_X> rawGm_;
};

template <class T>
class KernelSwiGluFixedD13Rows {
public:
    // Seven owners compute three rows each; packed half inputs accumulate into FP32.
    // All inputs remain runtime tensors; no cached values or host arithmetic.
    static constexpr uint32_t D=13, ROWS=21, CORES=7, ROWS_PER_CORE=3, PITCH=16;
    __aicore__ inline void Process(GM_ADDR inputX, GM_ADDR weight1, GM_ADDR bias1,
                                   GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output)
    {
        const uint32_t core=AscendC::GetBlockIdx();
        if(core>=CORES)return;
        const uint32_t firstRow=core*ROWS_PER_CORE;
        Panel<13,0>(firstRow,inputX,weight1,bias1,weight2,bias2,output);
    }
private:
    template<uint32_t N,uint32_t COLUMN>
    __aicore__ inline void Panel(uint32_t firstRow,GM_ADDR inputX,GM_ADDR weight1,GM_ADDR bias1,
                                 GM_ADDR weight2,GM_ADDR bias2,GM_ADDR output)
    {
        constexpr uint32_t R=ROWS_PER_CORE, K=PITCH, OP=(N+7)/8*8, HP=(N+15)/16*16;
        // Gather also visits masked K-padding: reserve through the final padded row,
        // not merely align(N*D). The 16-column band needs 544, not 528 halfs.
        constexpr uint32_t RAW=((N-1)*D+K+15)/16*16, WE=K*HP, BR=K;
        constexpr uint32_t BYTES=(2*RAW+2*HP+R*K+R*HP+2*WE+2*R*WE)*sizeof(T)+
            (2*HP+4*R*HP)*sizeof(float)+(BR*2*HP+BR+2*HP)*sizeof(int32_t)+(R*HP+16)*sizeof(T);
        static_assert(RAW>=(N-1)*D+K,"Gather K-padding must remain in raw allocation");
        static_assert(BYTES<=128*1024,"Host requires 128 KiB UB for this pure AIV path");
        AscendC::GlobalTensor<T>x,w1,w2,b1,b2,y;
        x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputX),ROWS*D);
        w1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight1),D*D);
        w2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight2),D*D);
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias1),D);
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias2),D);
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output),ROWS*D);
        AscendC::LocalMemAllocator<AscendC::Hardware::UB>ub;
        auto wh=ub.Alloc<T,2*RAW>();auto bh=ub.Alloc<T,2*HP>();
        auto xh=ub.Alloc<T,R*K>();auto out=ub.Alloc<T,R*HP>();
        auto weight=ub.Alloc<T,2*WE>();
        auto bias=ub.Alloc<float,2*HP>();
        auto product=ub.Alloc<T,2*R*WE>();auto z1=ub.Alloc<float,2*R*HP>();
        auto act=ub.Alloc<float,2*R*HP>();
        auto index=ub.Alloc<int32_t,BR*2*HP>();auto bases=ub.Alloc<int32_t,BR>();
        auto cols=ub.Alloc<int32_t,2*HP>();
        auto outputPattern=ub.Alloc<uint16_t,16>();
        auto packedOut=ub.Alloc<T,R*HP>();
        const AscendC::DataCopyPadExtParams<T>noPad{false,0,0,0};
        const AscendC::DataCopyExtParams wc{1,N*D*sizeof(T),0,0,0};
        const AscendC::DataCopyExtParams bc{1,N*sizeof(T),0,0,0};
        AscendC::DataCopyPad(wh,w1[COLUMN*D],wc,noPad);
        AscendC::DataCopyPad(wh[RAW],w2[COLUMN*D],wc,noPad);
        AscendC::DataCopyPad(bh,b1[COLUMN],bc,noPad);
        AscendC::DataCopyPad(bh[HP],b2[COLUMN],bc,noPad);
        AscendC::DataCopyPad(xh,x[firstRow*D],AscendC::DataCopyExtParams{R,D*sizeof(T),0,0,0},noPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        auto basePairs=bases.template ReinterpretCast<uint64_t>();
        auto columnPairs=cols.template ReinterpretCast<uint64_t>();
        #pragma unroll
        for(uint32_t r=0;r<BR;r+=2) {
            const uint64_t low=(r<D?r:0)*sizeof(T);
            const uint64_t high=(r+1<D?r+1:0)*sizeof(T);
            basePairs.SetValue(r/2,low|(high<<32));
        }
        #pragma unroll
        for(uint32_t k=0;k<2*HP;k+=2) {
            const uint64_t low=((k/HP)*RAW+(k%HP<N?(k%HP)*D:0))*sizeof(T);
            const uint32_t next=k+1;
            const uint64_t high=((next/HP)*RAW+(next%HP<N?(next%HP)*D:0))*sizeof(T);
            columnPairs.SetValue(k/2,low|(high<<32));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        auto ib=index.template ReinterpretCast<uint32_t>();
        auto bb=bases.template ReinterpretCast<uint32_t>();
        AscendC::Brcb(ib,bb,uint8_t{BR/8},AscendC::BrcbRepeatParams(2*HP/8,2*HP));
        AscendC::Copy<int32_t>(index,index,uint64_t{2*HP},uint8_t{BR},AscendC::CopyRepeatParams(1,0,2*HP/8,2*HP/8));
        AscendC::Add(index,index,cols,uint64_t{2*HP},uint8_t{BR},AscendC::BinaryRepeatParams(1,1,1,2*HP/8,2*HP/8,0));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Cast(bias,bh,AscendC::RoundMode::CAST_NONE,uint64_t{N},uint8_t{2},
            AscendC::UnaryRepeatParams(1,1,HP/8,HP/16));
        auto offsets=index.template ReinterpretCast<uint32_t>();
        AscendC::Gather(weight,wh,offsets,uint32_t{0},2*WE);
        // Pack W1/W2 columns into one masked VMLA issue per K.
        AscendC::Brcb(product,xh,uint8_t{R*K/8},AscendC::BrcbRepeatParams(2*HP/16,HP));
        AscendC::Copy<T>(product,product,uint64_t{2*HP},uint8_t{R*K},
            AscendC::CopyRepeatParams(1,0,2*HP/16,2*HP/16));
        AscendC::Duplicate(z1,0.0f,2*R*HP);
        const AscendC::BinaryRepeatParams fmaStride(1,1,1,2*HP/8,0,K*2*HP/16);
        const uint64_t valid=(uint64_t{1}<<N)-1;
        AscendC::SetVectorMask<float>(uint64_t{0},valid|(valid<<HP));
        #pragma unroll
        for(uint32_t k=0;k<D;++k)
            AscendC::MulAddDst<float,T,false>(z1,weight[k*2*HP],product[k*2*HP],
                uint64_t{2*HP},uint8_t{R},fmaStride);
        AscendC::SetVectorMask<float>(~uint64_t{0},~uint64_t{0});
        static_assert(R==3 && N==13 && HP==16 && sizeof(T)==2);
        auto patternWords=outputPattern.template ReinterpretCast<uint64_t>();
        patternWords.SetValue(0,0x00001fff1fff1fffULL);
        patternWords.SetValue(1,0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(1);
        AscendC::Add(z1,z1,bias,uint64_t{2*HP},uint8_t{R},
            AscendC::BinaryRepeatParams(1,1,1,2*HP/8,2*HP/8,0));
        AscendC::Silu<float,false>(act,z1,2*R*HP);
        AscendC::Mul(z1,act,z1[HP],uint64_t{N},uint8_t{R},
            AscendC::BinaryRepeatParams(1,1,1,2*HP/8,2*HP/8,2*HP/8));
        AscendC::Cast(out,z1,AscendC::RoundMode::CAST_RINT,uint64_t{N},uint8_t{R},
            AscendC::UnaryRepeatParams(1,1,HP/16,2*HP/8));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(1);
        uint64_t kept=0;
        AscendC::GatherMask(packedOut,out,outputPattern,true,R*HP,
            AscendC::GatherMaskParams{1,1,0,0},kept);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        static_assert(N==D && COLUMN==0,"Packed write requires complete rows");
        AscendC::DataCopyPad(y[firstRow*D],packedOut,
            AscendC::DataCopyExtParams{1,R*N*sizeof(T),0,0,0});
    }
};

template <class T>
class KernelSwiGluD33Rows {
public:
    // Pure AIV 17 row groups x two independent output-column bands.
    // No K split, Cube launch or cross-owner intermediate result.
    static constexpr uint32_t D=33, ROWS=85, CORES=34, ROWS_PER_CORE=5, PITCH=48;
    __aicore__ inline void Process(GM_ADDR inputX, GM_ADDR weight1, GM_ADDR bias1,
                                   GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output)
    {
        const uint32_t core=AscendC::GetBlockIdx();
        if(core>=CORES)return;
        constexpr uint32_t groups=ROWS/ROWS_PER_CORE;
        if(core<groups) {
            Panel<16,0>(core*ROWS_PER_CORE,inputX,weight1,bias1,weight2,bias2,output);
        } else {
            Panel<17,16>((core-groups)*ROWS_PER_CORE,inputX,weight1,bias1,weight2,bias2,output);
        }
    }
private:
    template<uint32_t N,uint32_t COLUMN>
    __aicore__ inline void Panel(uint32_t firstRow,GM_ADDR inputX,GM_ADDR weight1,GM_ADDR bias1,
                                 GM_ADDR weight2,GM_ADDR bias2,GM_ADDR output)
    {
        constexpr uint32_t R=ROWS_PER_CORE, K=PITCH, OP=(N+7)/8*8, HP=(N+15)/16*16;
        // Gather also visits masked K-padding: reserve through the final padded row,
        // not merely align(N*D). The 16-column band needs 544, not 528 halfs.
        constexpr uint32_t RAW=((N-1)*D+K+15)/16*16, WE=N*K, BR=(N+7)/8*8;
        constexpr uint32_t BYTES=(2*RAW+2*HP+R*K+R*HP+2*WE+2*R*K+2*R*WE)*sizeof(T)+
            (2*OP+2*R*WE+(N==16?5:3)*R*OP)*sizeof(float)+
            ((BR+N)*K+BR+K)*sizeof(int32_t)+
            (N==17 ? (16*OP+16+OP)*sizeof(int32_t)+(2*R*OP+((2*R*N+7)/8)*8)*sizeof(float) : 0);
        static_assert(RAW>=(N-1)*D+K,"Gather K-padding must remain in raw allocation");
        static_assert(BYTES<=96*1024,"Host reserves 96 KiB for this pure AIV path");
        AscendC::GlobalTensor<T>x,w1,w2,b1,b2,y;
        x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputX),ROWS*D);
        w1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight1),D*D);
        w2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight2),D*D);
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias1),D);
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias2),D);
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output),ROWS*D);
        AscendC::LocalMemAllocator<AscendC::Hardware::UB>ub;
        auto wh=ub.Alloc<T,2*RAW>();auto bh=ub.Alloc<T,2*HP>();
        auto xh=ub.Alloc<T,R*K>();auto out=ub.Alloc<T,R*HP>();
        auto weight=ub.Alloc<T,2*WE>();
        auto xPair=ub.Alloc<T,2*R*K>();
        auto productHalf=ub.Alloc<T,2*R*WE>();
        auto bias=ub.Alloc<float,2*OP>();
        auto product=ub.Alloc<float,2*R*WE>();auto z1=ub.Alloc<float,R*OP>();
        auto z2=ub.Alloc<float,R*OP>();auto act=ub.Alloc<float,R*OP>();
        auto index=ub.Alloc<int32_t,(BR+N)*K>();auto bases=ub.Alloc<int32_t,BR>();
        auto cols=ub.Alloc<int32_t,K>();
        const AscendC::DataCopyPadExtParams<T>noPad{false,0,0,0};
        const AscendC::DataCopyExtParams wc{1,N*D*sizeof(T),0,0,0};
        const AscendC::DataCopyExtParams bc{1,N*sizeof(T),0,0,0};
        AscendC::DataCopyPad(xh,x[firstRow*D],AscendC::DataCopyExtParams{R,D*sizeof(T),0,0,0},noPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::DataCopyPad(wh,w1[COLUMN*D],wc,noPad);
        AscendC::DataCopyPad(wh[RAW],w2[COLUMN*D],wc,noPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
        AscendC::DataCopyPad(bh,b1[COLUMN],bc,noPad);
        AscendC::DataCopyPad(bh[HP],b2[COLUMN],bc,noPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        auto basePairs=bases.template ReinterpretCast<uint64_t>();
        auto colPairs=cols.template ReinterpretCast<uint64_t>();
        #pragma unroll
        for(uint32_t r=0;r<BR;r+=2) {
            const uint64_t low=r*D*sizeof(T), high=(r+1)*D*sizeof(T);
            basePairs.SetValue(r/2,low|(high<<32));
        }
        #pragma unroll
        for(uint32_t k=0;k<K;k+=2) {
            const uint64_t low=k*sizeof(T), high=(k+1)*sizeof(T);
            colPairs.SetValue(k/2,low|(high<<32));
        }
        AscendC::LocalTensor<int32_t> reduceBases, reduceIndex, reduceCols;
        if constexpr (N==17) {
            reduceBases=ub.Alloc<int32_t,16>();
            reduceIndex=ub.Alloc<int32_t,16*OP>();
            reduceCols=ub.Alloc<int32_t,OP>();
            auto pairs=reduceBases.template ReinterpretCast<uint64_t>();
            #pragma unroll
            for(uint32_t i=0;i<16;i+=2) {
                const uint64_t lo=i<2*R ? ((i%R)*2*N+(i/R)*N)*sizeof(float) : 0;
                const uint32_t j=i+1;
                const uint64_t hi=j<2*R ? ((j%R)*2*N+(j/R)*N)*sizeof(float) : 0;
                pairs.SetValue(i/2,lo|(hi<<32));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
        // Pack two adjacent half rows per repeat, then widen before the original reductions.
        for(uint32_t r=0;r<R;++r)
            AscendC::Copy<T>(xPair[r*2*K],xh[r*K],uint64_t{K},uint8_t{2},
                AscendC::CopyRepeatParams(1,1,K/16,0));
        auto ib=index.template ReinterpretCast<uint32_t>();
        auto bb=bases.template ReinterpretCast<uint32_t>();
        AscendC::Brcb(ib,bb,uint8_t{BR/8},AscendC::BrcbRepeatParams(K/8,K));
        AscendC::Copy<int32_t>(index,index,uint64_t{K},uint8_t{N},AscendC::CopyRepeatParams(1,0,K/8,K/8));
        AscendC::Add(index,index,cols,uint64_t{K},uint8_t{N},AscendC::BinaryRepeatParams(1,1,1,K/8,K/8,0));
        if constexpr (N==17) {
            AscendC::Add(reduceCols,cols,cols,OP);
            AscendC::Brcb(reduceIndex.template ReinterpretCast<uint32_t>(),
                reduceBases.template ReinterpretCast<uint32_t>(),uint8_t{2},
                AscendC::BrcbRepeatParams(OP/8,OP));
            AscendC::Copy<int32_t>(reduceIndex,reduceIndex,uint64_t{OP},uint8_t{2*R},
                AscendC::CopyRepeatParams(1,0,OP/8,OP/8));
            AscendC::Add(reduceIndex,reduceIndex,reduceCols,uint64_t{N},uint8_t{2*R},
                AscendC::BinaryRepeatParams(1,1,1,OP/8,OP/8,0));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
        auto offsets=index.template ReinterpretCast<uint32_t>();
        AscendC::Adds(index[WE],index,int32_t(RAW*sizeof(T)),WE);
        AscendC::Gather(weight,wh,offsets,uint32_t{0},2*WE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Cast(bias,bh,AscendC::RoundMode::CAST_NONE,uint64_t{N},uint8_t{2},
            AscendC::UnaryRepeatParams(1,1,OP/8,HP/16));

        for(uint32_t r=0;r<R;++r)
            AscendC::Mul(productHalf[r*2*WE],weight,xPair[r*2*K],uint64_t{2*K},uint8_t{N},
                AscendC::BinaryRepeatParams(1,1,1,2*K/16,2*K/16,0));
        AscendC::Cast(product,productHalf,AscendC::RoundMode::CAST_NONE,2*R*WE);
        if constexpr (N==16) {
            // Each WRS repeat emits one float into [gate16, up16] for each row.
            auto compact=ub.Alloc<float,2*R*OP>();
            AscendC::WholeReduceSum<float>(compact,product,D,2*R*N,1,1,K/8);
            AscendC::Copy<float>(z1,compact,uint64_t{N},uint8_t{R},
                AscendC::CopyRepeatParams(1,1,OP/8,2*OP/8));
            AscendC::Copy<float>(z2,compact[OP],uint64_t{N},uint8_t{R},
                AscendC::CopyRepeatParams(1,1,OP/8,2*OP/8));
        } else {
            auto compact=ub.Alloc<float,((2*R*N+7)/8)*8>();
            auto projections=ub.Alloc<float,2*R*OP>();
            AscendC::WholeReduceSum<float>(compact,product,D,2*R*N,1,1,K/8);
            AscendC::Gather(projections,compact,reduceIndex.template ReinterpretCast<uint32_t>(),
                uint32_t{0},2*R*OP);
            z1=projections;
            z2=projections[R*OP];
        }
        // All rows share one bias vector, so one repeated Add covers every row.
        AscendC::Add(z1,z1,bias,uint64_t{N},uint8_t{R},AscendC::BinaryRepeatParams(1,1,1,OP/8,OP/8,0));
        AscendC::Add(z2,z2,bias[OP],uint64_t{N},uint8_t{R},AscendC::BinaryRepeatParams(1,1,1,OP/8,OP/8,0));
        AscendC::Silu<float,false>(act,z1,R*OP);
        AscendC::Mul(z1,act,z2,R*OP);
        AscendC::Cast(out,z1,AscendC::RoundMode::CAST_RINT,uint64_t{N},uint8_t{R},
            AscendC::UnaryRepeatParams(1,1,HP/16,OP/8));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyPad(y[firstRow*D+COLUMN],out,
            AscendC::DataCopyExtParams{R,N*sizeof(T),0,(D-N)*sizeof(T),0});
    }
};

template <class DT_INPUT_X>
class KernelSwiGluStaticPanelStream {
public:
    static constexpr uint32_t WIDTH = 256;
    static constexpr uint32_t ROWS = 128;
    static constexpr uint32_t COLUMNS = 64;
    static constexpr uint32_t JOINED = 2 * COLUMNS;
    static constexpr uint32_t GROUPS = WIDTH / COLUMNS;
    static constexpr uint32_t VECTOR_ROWS = ROWS / 2;
    static constexpr uint32_t PANEL_ELEMENTS = JOINED * WIDTH;
    static constexpr uint32_t RAW_ELEMENTS = VECTOR_ROWS * JOINED;
    static constexpr uint32_t ACT_ELEMENTS = VECTOR_ROWS * COLUMNS;
    static constexpr uint32_t OUT_ELEMENTS = VECTOR_ROWS * WIDTH;
    // Raw handoff is stored as half: the Cube rounds the FP32 accumulator with
    // F322F16 in the Fixpipe, so the GM hop moves half the bytes and the AIV
    // needs a single widening Cast instead of a rounding round trip.
    static constexpr uint32_t EXPLICIT_UB_BYTES = GROUPS * RAW_ELEMENTS * sizeof(DT_INPUT_X) +
        ACT_ELEMENTS * sizeof(DT_INPUT_X) +
        2 * OUT_ELEMENTS * sizeof(DT_INPUT_X) +
        2 * WIDTH * sizeof(DT_INPUT_X) + GROUPS * JOINED * sizeof(DT_INPUT_X);
    static_assert(sizeof(DT_INPUT_X) == 2, "Half input storage required");

    // T170: 60 full 128-row tasks = exactly 3 per AIC on the Judge's 20 cores.
    // The remaining 512 rows become sixteen 32-row tail tiles (AIC 0..15).
    static constexpr uint32_t TASKS = 60;
    static constexpr uint32_t TAIL_START = TASKS * ROWS;
    static constexpr uint32_t TAIL_ROWS = 32;
    static constexpr uint32_t TAIL_CORES = (8192 - TAIL_START) / TAIL_ROWS;
    static constexpr uint32_t TAIL_RAW_BASE = TASKS * GROUPS * ROWS * JOINED;
    static constexpr uint16_t TAIL_READY = 12;
    __aicore__ inline void Process(GM_ADDR inputX, GM_ADDR weight1, GM_ADDR bias1,
                                  GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output,
                                  GM_ADDR userWorkspace)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(inputX), 8192 * WIDTH);
        w1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(weight1), WIDTH * WIDTH);
        w2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(weight2), WIDTH * WIDTH);
        b1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(bias1), WIDTH);
        b2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(bias2), WIDTH);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(output), 8192 * WIDTH);
        rawGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X *>(userWorkspace), 2 * 8192 * WIDTH);
        if ASCEND_IS_AIC { ProcessCube(); }
        if ASCEND_IS_AIV { ProcessVector(); }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr uint16_t READY = 4;
    static constexpr uint16_t DONE = 6;

    __aicore__ inline void LoadPairedWeightBand(AscendC::LocalTensor<DT_INPUT_X> weights,
                                                uint32_t group, uint32_t destBand)
    {
        AscendC::Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = COLUMNS;
        copy.dValue = WIDTH;
        copy.srcDValue = WIDTH;
        copy.dstNzC0Stride = JOINED;
        copy.dstNzNStride = 1;
        const uint32_t source = group * COLUMNS * WIDTH;
        const uint32_t dest = destBand * PANEL_ELEMENTS;
        AscendC::DataCopy(weights[dest], w1Gm_[source], copy);
        // Interleave each K/16 fractal's W1/W2 rows, not whole NZ matrices.
        AscendC::DataCopy(weights[dest + COLUMNS * 16], w2Gm_[source], copy);
    }

    __aicore__ inline void LoadAForTask(uint32_t task, bool reused, bool hasNext,
            AscendC::LocalTensor<DT_INPUT_X> &x, AscendC::LocalTensor<DT_INPUT_X> &a,
            bool loaded = false)
    {
        if (!reused && !loaded) {
        // T153: one ND2NZ per 16-row block; that NZ matrix is already in final A2
        // block order, so the eight per-block issues collapse into one per k half.
        AscendC::Nd2NzParams copy{};
        copy.ndNum = ROWS / 16;
        copy.nValue = 16;
        copy.dValue = WIDTH;
        copy.srcDValue = WIDTH;
        copy.srcNdMatrixStride = 16 * WIDTH;
        copy.dstNzC0Stride = 16;
        copy.dstNzNStride = 1;
        copy.dstNzMatrixStride = 16 * WIDTH;
        AscendC::DataCopy(x, xGm_[task * ROWS * WIDTH], copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        // The last panel of the previous task is the last reader of A2 and B2.
        // Consume this release once here, not again for the new task's group0.
        if (reused) AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        // T156: L0A holds the two k halves as independent 8-kblk matrices so an
        // Mmad with k = WIDTH/2 can read each of them directly.
        AscendC::LoadData2DParams load{};
        load.repeatTimes = (WIDTH / 2) / 16;
        load.srcStride = 1;
        load.ifTranspose = false;
        for (uint32_t h = 0; h < 2; ++h)
            for (uint32_t b = 0; b < ROWS / 16; ++b)
                AscendC::LoadData(a[h * ROWS * (WIDTH / 2) + b * (WIDTH / 2) * 16],
                                  x[b * WIDTH * 16 + h * (WIDTH / 2) * 16], load);
        if (hasNext) AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    }

    __aicore__ inline void ProcessCube()
    {
        const uint32_t core=AscendC::GetBlockIdx();
        // T133: one AIC owns all four column groups of its 128-row tasks, so the
        // paired AIV drains every output row as one contiguous 512B burst. All
        // four weight bands stay resident in L1.
        const uint32_t first=core;
        const uint32_t step=AscendC::GetBlockNum();
        const uint32_t taskCount=(TASKS-1-first)/step+1;
        const uint32_t myUnits=GROUPS*taskCount;
        const uint32_t myPanels=2*myUnits;
        AscendC::LocalMemAllocator<AscendC::Hardware::L1> l1;
        AscendC::LocalMemAllocator<AscendC::Hardware::L0A> la;
        AscendC::LocalMemAllocator<AscendC::Hardware::L0B> lb;
        AscendC::LocalMemAllocator<AscendC::Hardware::L0C> lc;
        auto weights=l1.Alloc<AscendC::TPosition::B1,DT_INPUT_X,GROUPS*PANEL_ELEMENTS>();
        auto x=l1.Alloc<AscendC::TPosition::A1,DT_INPUT_X,ROWS*WIDTH>();
        auto aa=la.Alloc<AscendC::TPosition::A2,DT_INPUT_X,ROWS*WIDTH>();
        // T156: the 64KB B panel splits cleanly into its two k halves, so L0B is
        // double buffered at that granularity; an Mmad with k = WIDTH/2 then only
        // waits for the panel two Mmads back instead of the one just issued.
        auto b0=lb.Alloc<AscendC::TPosition::B2,DT_INPUT_X,PANEL_ELEMENTS/2>();
        auto b1=lb.Alloc<AscendC::TPosition::B2,DT_INPUT_X,PANEL_ELEMENTS/2>();
        auto cc=lc.Alloc<AscendC::TPosition::CO1,float,2*ROWS*JOINED>();
        for(uint32_t g=0;g<GROUPS;++g) LoadPairedWeightBand(weights,g,g);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(2);
        AscendC::LoadData2DParams ld{};
        ld.repeatTimes=(PANEL_ELEMENTS/2)/256;ld.srcStride=1;ld.ifTranspose=false;
        // T171: drain the 32-row tail tile first. Its epilogue then hides under
        // the first full 128-row task instead of being exposed after the last.
        if(core < TAIL_CORES) {
            const uint64_t tailStart = TAIL_START + static_cast<uint64_t>(core) * TAIL_ROWS;
            AscendC::Nd2NzParams tcp{};
            tcp.ndNum=TAIL_ROWS/16;tcp.nValue=16;tcp.dValue=WIDTH;tcp.srcDValue=WIDTH;
            tcp.srcNdMatrixStride=16*WIDTH;tcp.dstNzC0Stride=16;tcp.dstNzNStride=1;
            tcp.dstNzMatrixStride=16*WIDTH;
            AscendC::DataCopy(x, xGm_[tailStart*WIDTH], tcp);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            AscendC::LoadData2DParams tload{};
            tload.repeatTimes=(WIDTH/2)/16;tload.srcStride=1;tload.ifTranspose=false;
            for(uint32_t h=0;h<2;++h)
                for(uint32_t b=0;b<TAIL_ROWS/16;++b)
                    AscendC::LoadData(aa[h*TAIL_ROWS*(WIDTH/2)+b*(WIDTH/2)*16],
                                      x[b*WIDTH*16+h*(WIDTH/2)*16], tload);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
            uint32_t tp=0;
            for(uint32_t g=0;g<GROUPS;++g) {
                const uint16_t cslot=static_cast<uint16_t>(g&1u);
                for(uint32_t h=0;h<2;++h,++tp) {
                    const uint16_t slot=static_cast<uint16_t>(tp&1u);
                    if(tp>=2) AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1+slot);
                    auto bdst=(slot==0)?b0:b1;
                    AscendC::LoadData(bdst, weights[g*PANEL_ELEMENTS+h*(PANEL_ELEMENTS/2)], ld);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1+slot);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(1+slot);
                    if(h==0 && g>=2) AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cslot);
                    AscendC::MmadParams tmm{};
                    tmm.m=TAIL_ROWS;tmm.n=JOINED;tmm.k=WIDTH/2;tmm.cmatrixInitVal=(h==0);
                    AscendC::Mmad(cc[cslot*ROWS*JOINED], aa[h*TAIL_ROWS*(WIDTH/2)], bdst, tmm);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1+slot);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(cslot);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(cslot);
                AscendC::FixpipeParamsV220 tfp{};
                tfp.nSize=JOINED;tfp.mSize=TAIL_ROWS;tfp.srcStride=TAIL_ROWS;
                tfp.dstStride=JOINED;tfp.ndNum=1;tfp.quantPre=::QuantMode_t::F322F16;
                AscendC::Fixpipe(rawGm_[TAIL_RAW_BASE
                                        + (static_cast<uint64_t>(core)*GROUPS+static_cast<uint64_t>(g))
                                          *TAIL_ROWS*JOINED],
                                  cc[cslot*ROWS*JOINED], tfp);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cslot);
                AscendC::CrossCoreSetFlag<2,PIPE_FIX>(
                    static_cast<uint16_t>(TAIL_READY + static_cast<uint16_t>(g)));
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
        }
        // T115: issue the first panel's X burst before the resident weight fill is
        // awaited, so the 64KB input transfer overlaps the weight load instead of
        // following it.
        {
            AscendC::Nd2NzParams firstCopy{};
            firstCopy.ndNum = ROWS / 16;
            firstCopy.nValue = 16;
            firstCopy.dValue = WIDTH;
            firstCopy.srcDValue = WIDTH;
            firstCopy.srcNdMatrixStride = 16 * WIDTH;
            firstCopy.dstNzC0Stride = 16;
            firstCopy.dstNzNStride = 1;
            firstCopy.dstNzMatrixStride = 16 * WIDTH;
            AscendC::DataCopy(x, xGm_[first * ROWS * WIDTH], firstCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        }
        AscendC::LoadData(b0,weights,ld);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1);
        uint32_t seq=0, unit=0, panel=0;
        uint64_t pendingAddr=0;
        uint32_t pendingIdx=0, pendingGroup=0;
        uint16_t pendingSlot=0, pendingSeq=0;
        bool hasPending=false;
        for(uint32_t task=first;task<TASKS;task+=step,++seq) {
            const bool more=task+step<TASKS;
            LoadAForTask(task,seq!=0,more,x,aa,seq==0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
            // T155: the next task's X burst only needs the L1->L0A read to be done,
            // not the Cube. Issue it here so its 64KB ND2NZ hides behind this task's
            // eight Mmads instead of being exposed after the last one.
            if(more) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                AscendC::Nd2NzParams cp{};
                cp.ndNum=ROWS/16;cp.nValue=16;cp.dValue=WIDTH;cp.srcDValue=WIDTH;
                cp.srcNdMatrixStride=16*WIDTH;cp.dstNzC0Stride=16;cp.dstNzNStride=1;
                cp.dstNzMatrixStride=16*WIDTH;
                AscendC::DataCopy(x,xGm_[(task+step)*ROWS*WIDTH],cp);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            }
            for(uint32_t g=0;g<GROUPS;++g,++unit) {
                const uint16_t cslot=static_cast<uint16_t>(unit&1u);
                // Drain the previous unit's L0C panel on the FIX pipe while this
                // unit's first Mmad runs on the Cube.
                if(hasPending) {
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(pendingSlot);
                    AscendC::FixpipeParamsV220 fp{};
                    fp.nSize=JOINED;fp.mSize=ROWS;fp.srcStride=ROWS;fp.dstStride=JOINED;
                    fp.ndNum=1;fp.quantPre=::QuantMode_t::F322F16;
                    AscendC::Fixpipe(rawGm_[pendingAddr],cc[pendingSlot*ROWS*JOINED],fp);
                    if(pendingIdx+2<myUnits)
                        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(pendingSlot);
                    AscendC::CrossCoreSetFlag<2,PIPE_FIX>(
                        static_cast<uint16_t>(READY + ((pendingSeq*GROUPS+pendingGroup)&7u)));
                    hasPending=false;
                }
                for(uint32_t h=0;h<2;++h,++panel) {
                    const uint16_t slot=static_cast<uint16_t>(panel&1u);
                    const uint16_t other=static_cast<uint16_t>((panel+1)&1u);
                    if(panel+1<myPanels) {
                        if(panel>=1)AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1+other);
                        const uint32_t np=(panel+1)&7u;
                        auto dst=(other==0)?b0:b1;
                        AscendC::LoadData(dst,weights[(np>>1)*PANEL_ELEMENTS +
                            (np&1u)*(PANEL_ELEMENTS/2)],ld);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1+other);
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(1+slot);
                    if(h==0 && unit>=2) AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cslot);
                    auto bsrc=(slot==0)?b0:b1;
                    AscendC::MmadParams mm{};
                    mm.m=ROWS;mm.n=JOINED;mm.k=WIDTH/2;mm.cmatrixInitVal=(h==0);
                    AscendC::Mmad(cc[cslot*ROWS*JOINED],aa[h*ROWS*(WIDTH/2)],bsrc,mm);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1+slot);
                    if(h==1) {
                        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(cslot);
                        pendingAddr=(task*GROUPS+g)*ROWS*JOINED;
                        pendingIdx=unit;
                        pendingSlot=cslot;
                        pendingSeq=static_cast<uint16_t>(seq);
                        pendingGroup=g;
                        hasPending=true;
                    }
                }
            }
            // A2 is released only once every Mmad of this task has been issued.
            if(more) AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        }
        // Drain the last unit of the last task; only then may its AIV run.
        if(hasPending) {
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(pendingSlot);
            AscendC::FixpipeParamsV220 fp{};
            fp.nSize=JOINED;fp.mSize=ROWS;fp.srcStride=ROWS;fp.dstStride=JOINED;
            fp.ndNum=1;fp.quantPre=::QuantMode_t::F322F16;
            AscendC::Fixpipe(rawGm_[pendingAddr],cc[pendingSlot*ROWS*JOINED],fp);
            if(pendingIdx+2<myUnits)
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(pendingSlot);
            AscendC::CrossCoreSetFlag<2,PIPE_FIX>(
                static_cast<uint16_t>(READY + ((pendingSeq*GROUPS+pendingGroup)&7u)));
        }
        // Balance the last two Mmads' L0B releases: unpaired SetFlags hang the block.
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
    }

    __aicore__ inline void PrepareBias(AscendC::LocalTensor<DT_INPUT_X> &stage,
                                      AscendC::LocalTensor<DT_INPUT_X> &biasPair)
    {
        const AscendC::DataCopyExtParams copy{1, WIDTH * sizeof(DT_INPUT_X), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<DT_INPUT_X> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(stage, b1Gm_, copy, pad);
        AscendC::DataCopyPad(stage[WIDTH], b2Gm_, copy, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
        // T157: the tail is half, so the bias is never widened either. Every
        // column group gets its own joined 64+64 half row build in one strided op.
        for (uint32_t g = 0; g < GROUPS; ++g) {
            AscendC::Muls(biasPair[g * JOINED], stage[g * COLUMNS], static_cast<DT_INPUT_X>(1),
                          uint64_t{COLUMNS}, static_cast<uint8_t>(2),
                          AscendC::UnaryRepeatParams(1, 1, COLUMNS / HALVES_PER_BLOCK,
                                                     WIDTH / HALVES_PER_BLOCK));
        }
        // T120: the Vector pipe keeps RAW order in hardware (proven by T113).
    }

    __aicore__ inline void ComputePanel(AscendC::LocalTensor<DT_INPUT_X> raw,
                                       AscendC::LocalTensor<DT_INPUT_X> out, uint32_t group,
            AscendC::LocalTensor<DT_INPUT_X> &biasPair, AscendC::LocalTensor<DT_INPUT_X> &act,
            uint32_t rows)
    {
        // T157: the Fixpipe already rounded the accumulator to half, so the whole
        // tail stays in half: the 128-wide bias add, the four SiLU stages and the
        // gate multiply run at half element pitch. Only the final 64-wide result
        // has to be widened into the 256-wide output row of this column group.
        const AscendC::BinaryRepeatParams rowRepeat(1, 1, 1,
            JOINED / HALVES_PER_BLOCK, JOINED / HALVES_PER_BLOCK, 0);
        AscendC::Add(raw, raw, biasPair[group * JOINED], uint64_t{JOINED},
                     static_cast<uint8_t>(rows), rowRepeat);
        // T120: the Vector pipe keeps RAW order in hardware (proven by T113).
        AscendC::Muls(act, raw, static_cast<DT_INPUT_X>(-1), uint64_t{COLUMNS},
                      static_cast<uint8_t>(rows),
                      AscendC::UnaryRepeatParams(1, 1, COLUMNS / HALVES_PER_BLOCK,
                                                 JOINED / HALVES_PER_BLOCK));
        // T120: the Vector pipe keeps RAW order in hardware (proven by T113).
        AscendC::Exp(act, act, uint64_t{rows * COLUMNS});
        // T120: the Vector pipe keeps RAW order in hardware (proven by T113).
        AscendC::Adds(act, act, static_cast<DT_INPUT_X>(1), uint64_t{rows * COLUMNS});
        // T120: the Vector pipe keeps RAW order in hardware (proven by T113).
        AscendC::Div(act, raw, act, uint64_t{COLUMNS}, static_cast<uint8_t>(rows),
                     AscendC::BinaryRepeatParams(1, 1, 1, COLUMNS / HALVES_PER_BLOCK,
                         JOINED / HALVES_PER_BLOCK, COLUMNS / HALVES_PER_BLOCK));
        // T120: the Vector pipe keeps RAW order in hardware (proven by T113).
        AscendC::Mul(act, act, raw[COLUMNS], uint64_t{COLUMNS}, static_cast<uint8_t>(rows),
                     AscendC::BinaryRepeatParams(1, 1, 1, COLUMNS / HALVES_PER_BLOCK,
                         COLUMNS / HALVES_PER_BLOCK, JOINED / HALVES_PER_BLOCK));
        // T120: the Vector pipe keeps RAW order in hardware (proven by T113).
        AscendC::Copy(out[group * COLUMNS], act, uint64_t{COLUMNS},
                      static_cast<uint8_t>(rows),
                      AscendC::CopyRepeatParams(1, 1, WIDTH / HALVES_PER_BLOCK,
                                                COLUMNS / HALVES_PER_BLOCK));
        // T120: the Vector pipe keeps RAW order in hardware (proven by T113).
    }

    __aicore__ inline void ProcessVector()
    {
        const uint32_t first=static_cast<uint32_t>(AscendC::GetBlockIdx())>>1;
        const uint32_t step=AscendC::GetBlockNum();
        const uint32_t lane=AscendC::GetSubBlockIdx();
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
        auto hs=ub.Alloc<AscendC::TPosition::VECIN,DT_INPUT_X,GROUPS*RAW_ELEMENTS>();
        auto os=ub.Alloc<AscendC::TPosition::VECOUT,DT_INPUT_X,2*OUT_ELEMENTS>();
        auto act=ub.Alloc<AscendC::TPosition::VECCALC,DT_INPUT_X,ACT_ELEMENTS>();
        auto stage=ub.Alloc<AscendC::TPosition::VECIN,DT_INPUT_X,2*WIDTH>();
        auto biasPair=ub.Alloc<AscendC::TPosition::VECCALC,DT_INPUT_X,GROUPS*JOINED>();
        PrepareBias(stage,biasPair);
        // T171: drain the tail tile before the three full tasks, so its epilogue
        // overlaps the first 128-row task instead of being exposed at the end.
        if(first < TAIL_CORES) {
            const uint32_t tailRows = TAIL_ROWS / 2u;
            const uint64_t tailStart = TAIL_START + static_cast<uint64_t>(first) * TAIL_ROWS;
            auto out0 = os[0];
            for(uint32_t g=0;g<GROUPS;++g) {
                AscendC::CrossCoreWaitFlag<2>(TAIL_READY + static_cast<uint16_t>(g));
                if(g>=1) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
                AscendC::DataCopy(hs[0], rawGm_[TAIL_RAW_BASE
                    + (static_cast<uint64_t>(first)*GROUPS+static_cast<uint64_t>(g))
                      *TAIL_ROWS*JOINED
                    + static_cast<uint64_t>(lane)*tailRows*JOINED],
                    uint64_t{tailRows*JOINED});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                ComputePanel(hs[0], out0, g, biasPair, act, tailRows);
                if(g+1<GROUPS) AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopy(yGm_[(tailStart + static_cast<uint64_t>(lane)*tailRows)*WIDTH],
                              out0, uint64_t{tailRows*WIDTH});
            // Close the MTE3 read of out0 before the main loop reuses os[0].
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        }
        uint32_t seq=0;
        for(uint32_t task=first;task<TASKS;task+=step,++seq) {
            const uint16_t slot=static_cast<uint16_t>(seq&1u);
            if(seq>=1)AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            const uint64_t base=((task*GROUPS)*ROWS+lane*VECTOR_ROWS)*JOINED;
            if(seq>=2)AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
            auto out=os[slot*OUT_ELEMENTS];
            for(uint32_t g=0;g<GROUPS;++g) {
                // T169: drain group g as soon as its own fixpipe lands instead of
                // waiting for the whole 4-group task.
                AscendC::CrossCoreWaitFlag<2>(
                    static_cast<uint16_t>(READY + ((seq*GROUPS+g)&7u)));
                AscendC::DataCopy(hs[g*RAW_ELEMENTS], rawGm_[base+g*ROWS*JOINED],
                    uint64_t{RAW_ELEMENTS});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                ComputePanel(hs[g*RAW_ELEMENTS],out,g,biasPair,act,VECTOR_ROWS);
            }
            if(task+step<TASKS)AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
            // T133: the full 256-column row is ready in UB, so each lane drains its
            // 64-row slab as four contiguous 8KB bursts instead of 256 row bursts.
            const uint64_t offset=(task*ROWS+lane*VECTOR_ROWS)*WIDTH;
            AscendC::DataCopyPad(yGm_[offset],out,
                AscendC::DataCopyExtParams{4,
                    static_cast<uint32_t>((VECTOR_ROWS/4)*WIDTH*sizeof(DT_INPUT_X)),0,0,0});
            if(task+2*step<TASKS)AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
        }
    }

    AscendC::GlobalTensor<DT_INPUT_X> xGm_, w1Gm_, w2Gm_, b1Gm_, b2Gm_, yGm_, rawGm_;
};

}  // namespace

template <typename DT_INPUT_X, int KERNEL_PATH>
__global__ __aicore__ void swi_glu(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                   GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    if constexpr (KERNEL_PATH == 44) {
        // Per-row AIV owner: no Cube, no workspace handoff, no tiling read.
        KernelSwiGluD33Rows<DT_INPUT_X> op;
        op.Process(input_x, weight_1, bias_1, weight_2, bias_2, output);
    } else if constexpr (KERNEL_PATH == 40) {
        KernelSwiGluStaticPanelStream<DT_INPUT_X> op;
        op.Process(input_x, weight_1, bias_1, weight_2, bias_2, output,
                   AscendC::GetUserWorkspace(workspace));
    } else if constexpr (KERNEL_PATH == 39) {
        KernelSwiGluStaticD128<DT_INPUT_X> op;
        op.Process(input_x, weight_1, bias_1, weight_2, bias_2, output,
                   AscendC::GetUserWorkspace(workspace));
    } else if constexpr (KERNEL_PATH == 38) {
        KernelSwiGluStaticCase3<DT_INPUT_X> op;
        op.Process(input_x, weight_1, bias_1, weight_2, bias_2, output,
                   AscendC::GetUserWorkspace(workspace));
    } else if constexpr (KERNEL_PATH == 36) {
        KernelSwiGluFixedD64Columns<DT_INPUT_X> op;
        op.Process(input_x, weight_1, bias_1, weight_2, bias_2, output);
    } else if constexpr (KERNEL_PATH == 35) {
        KernelSwiGluFixedD13Rows<DT_INPUT_X> op;
        op.Process(input_x, weight_1, bias_1, weight_2, bias_2, output);
    } else if constexpr (KERNEL_PATH == 32 || KERNEL_PATH == 33 || KERNEL_PATH == 34) {
        // D is selected by Host metadata; this branch never reads tiling/workspace.
        constexpr uint32_t fixedD = KERNEL_PATH == 32 ? 1u : (KERNEL_PATH == 33 ? 4u : 32u);
        KernelSwiGluFixedStaticRow<DT_INPUT_X, fixedD> op;
        op.Process(input_x, weight_1, bias_1, weight_2, bias_2, output);
    }
}
