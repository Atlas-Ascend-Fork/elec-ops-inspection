#include "kernel_operator.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "tiling_key_fused_disconnector_keypoint_soft_argmax.h"

namespace {
using namespace AscendC;

constexpr uint32_t INPUT_CHANNELS = 32;
constexpr uint32_t OUTPUT_CHANNELS = FUSED_KEYPOINT_CHANNELS;
constexpr uint32_t WEIGHTS_PER_CHANNEL = INPUT_CHANNELS * 9;
constexpr uint32_t RECORD_FLOATS = FUSED_KEYPOINT_RECORD_FLOATS;
constexpr uint32_t TILE_RECORD_FLOATS = OUTPUT_CHANNELS * RECORD_FLOATS;
constexpr uint32_t VEC_FLOATS = 64;
constexpr uint32_t FLOATS_PER_BLOCK = 8;
constexpr uint32_t SIGN_BIT = 0x80000000U;
constexpr uint32_t FLOAT_MASK = 0x007fffffU;
constexpr uint32_t LEADING_BIT = 0x00800000U;
constexpr uint32_t ONE_BITS = 0x3f800000U;
constexpr int32_t EXPONENT_BIAS = 127;
constexpr int32_t EXPONENT_LIMIT = EXPONENT_BIAS + 7; // 128.0f

union FloatBits {
    float value;
    uint32_t bits;
};

__aicore__ inline uint32_t Min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint32_t FloatOrder(uint32_t bits)
{
    return (bits & SIGN_BIT) != 0 ? ~bits : bits ^ SIGN_BIT;
}

__aicore__ inline uint32_t NormalSignificand(uint32_t bits, int32_t& exponent)
{
    exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - EXPONENT_BIAS;
    uint32_t significand = bits & FLOAT_MASK;
    if (exponent == -127) {
        exponent = -126;
        while (significand != 0 && significand < LEADING_BIT) {
            significand <<= 1;
            --exponent;
        }
    } else {
        significand |= LEADING_BIT;
    }
    return significand;
}

// Write (max - value) as a normalized significand and a binary exponent.
__aicore__ inline uint32_t PositiveGap(uint32_t maxBits, uint32_t valueBits, int32_t& exponent)
{
    int32_t maxExponent, valueExponent;
    const uint32_t maxSig = NormalSignificand(maxBits, maxExponent);
    const uint32_t valueSig = NormalSignificand(valueBits, valueExponent);
    if (maxSig == 0 || valueSig == 0) {
        exponent = maxSig == 0 ? valueExponent : maxExponent;
        return maxSig | valueSig;
    }
    exponent = maxExponent > valueExponent ? maxExponent : valueExponent;
    const uint32_t maxShift = exponent - maxExponent;
    const uint32_t valueShift = exponent - valueExponent;
    if (maxShift > 32 || valueShift > 32) {
        return maxShift == 0 ? maxSig : valueSig;
    }
    const int64_t maxTerm = static_cast<int64_t>(static_cast<uint64_t>(maxSig) << (32 - maxShift));
    const int64_t valueTerm = static_cast<int64_t>(static_cast<uint64_t>(valueSig) << (32 - valueShift));
    const int64_t difference = ((maxBits & SIGN_BIT) != 0 ? -maxTerm : maxTerm) -
        ((valueBits & SIGN_BIT) != 0 ? -valueTerm : valueTerm);
    if (difference <= 0) {
        return 0;
    }
    uint64_t normalized = static_cast<uint64_t>(difference);
    while (normalized < (uint64_t{LEADING_BIT} << 32)) {
        normalized <<= 1;
        --exponent;
    }
    if (normalized >= (uint64_t{LEADING_BIT} << 33)) {
        normalized = (normalized >> 1) | (normalized & 1);
        ++exponent;
    }
    uint32_t significand = static_cast<uint32_t>(normalized >> 32);
    const uint32_t remainder = static_cast<uint32_t>(normalized);
    if (remainder > SIGN_BIT || (remainder == SIGN_BIT && (significand & 1))) {
        ++significand;
    }
    if (significand == (LEADING_BIT << 1)) {
        significand >>= 1;
        ++exponent;
    }
    return significand;
}

template <uint32_t ShapeId, bool UnitTau>
class FusedKeypointKernel {
    static_assert(ShapeId >= 1 && ShapeId <= 3, "Unsupported spatial shape");
    static constexpr uint32_t CHANNEL_GROUP = 8;
    static constexpr uint32_t HALO_SLOTS = 2;
    static constexpr uint32_t height_ = ShapeId == 1 ? 48 : (ShapeId == 2 ? 64 : 72);
    static constexpr uint32_t width_ = ShapeId == 3 ? 96 : 64;
    static constexpr uint32_t tileHeight_ = ShapeId == 3 ? 6 : 4;
    static constexpr uint32_t tileWidth_ = 32;
    static constexpr uint32_t tilesPerBatch_ = (height_ / tileHeight_) * (width_ / tileWidth_);
    static constexpr uint32_t fullTileSize_ = tileHeight_ * tileWidth_;
    static constexpr uint32_t shiftCapacity_ = (tileHeight_ + 2) * tileWidth_;
    static constexpr uint32_t shiftChannelPitch_ = 3 * shiftCapacity_ + FLOATS_PER_BLOCK;
    static constexpr uint32_t haloPitch_ = 40;
    static constexpr uint32_t haloPlane_ = (tileHeight_ + 2) * haloPitch_;
    static constexpr uint32_t GROUP_TAPS = CHANNEL_GROUP * 9;

public:
    __aicore__ inline void Init(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints,
                               GM_ADDR scores, GM_ADDR workspaceBase,
                               const FusedDisconnectorKeypointSoftArgmaxTilingData& tiling)
    {
        batch_ = tiling.batch;
        blocks_ = tiling.blocks;
        stride_ = tiling.stride;
        tau_ = tiling.tau;
        featGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(feat), batch_ * INPUT_CHANNELS * height_ * width_);
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight), OUTPUT_CHANNELS * WEIGHTS_PER_CHANNEL);
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias), OUTPUT_CHANNELS);
        keypointsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keypoints), batch_ * OUTPUT_CHANNELS * 2);
        scoresGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scores), batch_ * OUTPUT_CHANNELS);
        const auto base = reinterpret_cast<uint64_t>(workspaceBase);
        auto* records = reinterpret_cast<__gm__ float*>((base + 63) & ~uint64_t{63});
        recordsGm_.SetGlobalBuffer(records, batch_ * tilesPerBatch_ * TILE_RECORD_FLOATS);

        pipe_.InitBuffer(haloQueue_, HALO_SLOTS, CHANNEL_GROUP * haloPlane_ * sizeof(float));
        pipe_.InitBuffer(weights_, OUTPUT_CHANNELS * WEIGHTS_PER_CHANNEL * sizeof(float));
        pipe_.InitBuffer(vectorWeights_, OUTPUT_CHANNELS * GROUP_TAPS * FLOATS_PER_BLOCK * sizeof(float));
        pipe_.InitBuffer(bias_, FLOATS_PER_BLOCK * sizeof(float));
        pipe_.InitBuffer(offsets_, shiftCapacity_ * sizeof(uint32_t));
        pipe_.InitBuffer(shifts_, CHANNEL_GROUP * shiftChannelPitch_ * sizeof(float));
        pipe_.InitBuffer(accumulators_, OUTPUT_CHANNELS * CHANNEL_GROUP * fullTileSize_ * sizeof(float));
        pipe_.InitBuffer(xCoords_, fullTileSize_ * sizeof(float));
        pipe_.InitBuffer(yCoords_, fullTileSize_ * sizeof(float));
        pipe_.InitBuffer(exponents_, fullTileSize_ * sizeof(float));
        pipe_.InitBuffer(products_, fullTileSize_ * sizeof(float));
        pipe_.InitBuffer(reduce_, 1024);
        pipe_.InitBuffer(records_, tilesPerBatch_ * TILE_RECORD_FLOATS * sizeof(float));
        pipe_.InitBuffer(published_, TILE_RECORD_FLOATS * sizeof(float));
        pipe_.InitBuffer(keypoints_, 4 * OUTPUT_CHANNELS * 2 * sizeof(float));
        pipe_.InitBuffer(scores_, 4 * OUTPUT_CHANNELS * sizeof(float));
        pipe_.InitBuffer(merge_, 6 * VEC_FLOATS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyParameters();
        const uint32_t block = GetBlockIdx();
        if (WeightsAreZero()) {
            if (block == 0) {
                WriteUniformResults();
                WriteOutputs();
            }
            return;
        }
        if constexpr (!UnitTau) {
            PrepareTemperature();
        }
        for (uint32_t task = block; task < batch_ * tilesPerBatch_; task += blocks_) {
            ComputeTile(task / tilesPerBatch_, task % tilesPerBatch_, published_.Get<float>());
            Wait<HardEvent::S_MTE3>();
            DataCopy(recordsGm_[task * TILE_RECORD_FLOATS], published_.Get<float>(), TILE_RECORD_FLOATS);
            Wait<HardEvent::MTE3_S>();
        }
        SyncAll<true>();
        if (block != 0) {
            return;
        }
        auto records = records_.Get<float>();
        for (uint32_t b = 0; b < batch_; ++b) {
            const uint32_t floats = tilesPerBatch_ * TILE_RECORD_FLOATS;
            DataCopy(records, recordsGm_[b * floats], floats);
            Wait<HardEvent::MTE2_S>();
            MergeBatch(b, records);
        }
        WriteOutputs();
    }

private:
    __aicore__ inline void WriteOutputs()
    {
        Wait<HardEvent::S_MTE3>();
        DataCopy(keypointsGm_, keypoints_.Get<float>(), batch_ * OUTPUT_CHANNELS * 2);
        const auto scoreBytes = static_cast<uint32_t>(batch_ * OUTPUT_CHANNELS * sizeof(float));
        DataCopyPad(scoresGm_, scores_.Get<float>(), {1, scoreBytes, 0, 0, 0});
        Wait<HardEvent::MTE3_S>();
    }

    template <HardEvent Event>
    __aicore__ inline void Wait()
    {
        const auto eventId = pipe_.FetchEventID(Event);
        SetFlag<Event>(eventId);
        WaitFlag<Event>(eventId);
    }

    __aicore__ inline void CopyParameters()
    {
        DataCopy(weights_.Get<float>(), weightGm_, OUTPUT_CHANNELS * WEIGHTS_PER_CHANNEL);
        DataCopyPad(bias_.Get<float>(), biasGm_, {1, OUTPUT_CHANNELS * sizeof(float), 0, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 0, 0.0f});
        Wait<HardEvent::MTE2_S>();
    }

    __aicore__ inline bool WeightsAreZero()
    {
        constexpr uint32_t halfWords = OUTPUT_CHANNELS * WEIGHTS_PER_CHANNEL * sizeof(float) / sizeof(uint16_t);
        constexpr uint32_t chunk = 1024;
        constexpr uint32_t halfWordsPerBlock = 32 / sizeof(uint16_t);
        static_assert(2 * chunk < halfWords && halfWords <= 3 * chunk);
        auto words = weights_.Get<uint16_t>();
        auto reduced = vectorWeights_.Get<uint16_t>();
        Or(reduced, words, words[chunk], chunk);
        PipeBarrier<PIPE_V>();
        Or(reduced, reduced, words[2 * chunk], halfWords - 2 * chunk);
        for (uint32_t n = chunk / 2; n >= halfWordsPerBlock; n /= 2) {
            PipeBarrier<PIPE_V>();
            Or(reduced, reduced, reduced[n], n);
        }
        Wait<HardEvent::V_S>();
        auto bits = reduced.ReinterpretCast<uint32_t>();
        uint32_t nonzero = 0;
        for (uint32_t i = 0; i < FLOATS_PER_BLOCK; ++i) {
            nonzero |= bits.GetValue(i);
        }
        // Every producer checks the same weights. Ignore the sign of exact zero.
        return (nonzero & ~SIGN_BIT) == 0;
    }

    __aicore__ inline void WriteUniformResults()
    {
        auto points = keypoints_.Get<float>();
        auto scores = scores_.Get<float>();
        auto bias = bias_.Get<float>();
        const float x = static_cast<float>(static_cast<int32_t>(width_) * stride_) * 0.5f;
        const float y = static_cast<float>(static_cast<int32_t>(height_) * stride_) * 0.5f;
        for (uint32_t b = 0; b < batch_; ++b) {
            for (uint32_t k = 0; k < OUTPUT_CHANNELS; ++k) {
                const uint32_t index = b * OUTPUT_CHANNELS + k;
                points.SetValue(2 * index, x);
                points.SetValue(2 * index + 1, y);
                scores.SetValue(index, bias.GetValue(k));
            }
        }
    }

    __aicore__ inline void PrepareTemperature()
    {
        // The common temperature range has a finite, normal float reciprocal.
        regularTau_ = tau_ >= 0x1p-32f && tau_ <= 0x1p32f;
        if (regularTau_) {
            auto work = reduce_.Get<float>();
            auto numerator = work;
            auto denominator = work[FLOATS_PER_BLOCK];
            auto reciprocal = work[2 * FLOATS_PER_BLOCK];
            Duplicate(numerator, 1.0f, FLOATS_PER_BLOCK);
            Duplicate(denominator, tau_, FLOATS_PER_BLOCK);
            PipeBarrier<PIPE_V>();
            Div(reciprocal, numerator, denominator, FLOATS_PER_BLOCK);
            Wait<HardEvent::V_S>();
            reciprocalTau_ = reciprocal.GetValue(0);
        } else {
            const uint32_t significand = NormalSignificand(FloatBits{tau_}.bits, tauExponent_);
            tauMantissa_ = ONE_BITS | (significand & FLOAT_MASK);
        }
    }

    __aicore__ inline void CopyHalo(uint32_t batch, uint32_t firstChannel, uint32_t v, uint32_t u,
                                   uint32_t rows, uint32_t cols)
    {
        const uint32_t top = v == 0;
        const uint32_t left = u == 0;
        const uint32_t bottom = v + rows == height_;
        const uint32_t right = u + cols == width_;
        const uint32_t copyRows = rows + 2 - top - bottom;
        const uint32_t copyCols = cols + 2 - left - right;
        const uint32_t paddedCols = (cols + 2 + FLOATS_PER_BLOCK - 1) / FLOATS_PER_BLOCK * FLOATS_PER_BLOCK;
        const DataCopyExtParams copy{static_cast<uint16_t>(CHANNEL_GROUP), static_cast<uint32_t>(copyCols * sizeof(float)),
                                    static_cast<uint32_t>((height_ * width_ - copyCols) * sizeof(float)),
                                    (haloPlane_ - paddedCols) / FLOATS_PER_BLOCK, 0};
        const DataCopyPadExtParams<float> pad{true, static_cast<uint8_t>(left), static_cast<uint8_t>(right), 0.0f};
        auto halo = haloQueue_.template AllocTensor<float>();
        const uint32_t source = ((batch * INPUT_CHANNELS + firstChannel) * height_ + v + top - 1) *
            width_ + u + left - 1;
        for (uint32_t row = 0; row < copyRows; ++row) {
            DataCopyPad(halo[(top + row) * haloPitch_], featGm_[source + row * width_], copy, pad);
        }
        haloQueue_.EnQue(halo);
    }

    __aicore__ inline void Convolve(uint32_t batch, uint32_t v, uint32_t u, uint32_t rows, uint32_t cols)
    {
        const uint32_t count = rows * cols;
        const uint32_t shiftedCount = (rows + 2) * cols;
        auto offsets = offsets_.Get<uint32_t>();
        for (uint32_t row = 0; row < rows + 2; ++row) {
            for (uint32_t col = 0; col < cols; ++col) {
                offsets.SetValue(row * cols + col, (row * haloPitch_ + col) * sizeof(float));
            }
        }
        CopyHalo(batch, 0, v, u, rows, cols);
        Wait<HardEvent::S_V>();
        auto acc = accumulators_.Get<float>();
        Duplicate(acc, 0.0f, OUTPUT_CHANNELS * CHANNEL_GROUP * fullTileSize_);
        auto shifts = shifts_.Get<float>();
        auto weights = weights_.Get<float>();
        auto vectorWeights = vectorWeights_.Get<float>();
        const bool top = v == 0;
        const bool bottom = v + rows == height_;
        for (uint32_t base = 0; base < INPUT_CHANNELS; base += CHANNEL_GROUP) {
            if (base + CHANNEL_GROUP < INPUT_CHANNELS) {
                CopyHalo(batch, base + CHANNEL_GROUP, v, u, rows, cols);
            }
            for (uint32_t k = 0; k < OUTPUT_CHANNELS; ++k) {
                Brcb(vectorWeights[k * GROUP_TAPS * FLOATS_PER_BLOCK],
                     weights[k * WEIGHTS_PER_CHANNEL + base * 9], GROUP_TAPS / FLOATS_PER_BLOCK, {1, 8});
            }
            auto halo = haloQueue_.template DeQue<float>();
            for (uint32_t c = 0; c < CHANNEL_GROUP; ++c) {
                auto plane = halo[c * haloPlane_];
                if (top) {
                    Duplicate(plane, 0.0f, haloPitch_);
                }
                if (bottom) {
                    Duplicate(plane[(rows + 1) * haloPitch_], 0.0f, haloPitch_);
                }
                if (top || bottom) {
                    PipeBarrier<PIPE_V>();
                }
                for (uint32_t j = 0; j < 3; ++j) {
                    Gather(shifts[c * shiftChannelPitch_ + j * shiftCapacity_], plane, offsets,
                           j * sizeof(float), shiftedCount);
                }
            }
            PipeBarrier<PIPE_V>();
            // Each block of eight lanes carries one input channel, not one output channel.
            SetVectorMask<float, MaskMode::NORMAL>(VEC_FLOATS);
            const BinaryRepeatParams channels{
                fullTileSize_ / FLOATS_PER_BLOCK, shiftChannelPitch_ / FLOATS_PER_BLOCK, 9, 1, 1, 0};
            for (uint32_t i = 0; i < 3; ++i) {
                for (uint32_t j = 0; j < 3; ++j) {
                    auto x = shifts[j * shiftCapacity_ + i * cols];
                    for (uint32_t k = 0; k < OUTPUT_CHANNELS; ++k) {
                        auto coefficients = vectorWeights[(k * GROUP_TAPS + i * 3 + j) * FLOATS_PER_BLOCK];
                        MulAddDst<float, float, false>(acc[k * CHANNEL_GROUP * fullTileSize_], x, coefficients,
                            MASK_PLACEHOLDER, static_cast<uint8_t>(count / FLOATS_PER_BLOCK), channels);
                    }
                    PipeBarrier<PIPE_V>();
                }
            }
            ResetMask();
            haloQueue_.FreeTensor(halo);
        }
        // Deterministic eight-way tree; all four output channels remain independent.
        for (uint32_t step = 1; step < CHANNEL_GROUP; step *= 2) {
            for (uint32_t k = 0; k < OUTPUT_CHANNELS; ++k) {
                for (uint32_t r = 0; r < CHANNEL_GROUP; r += 2 * step) {
                    auto dst = acc[(k * CHANNEL_GROUP + r) * fullTileSize_];
                    Add(dst, dst, acc[(k * CHANNEL_GROUP + r + step) * fullTileSize_], count);
                }
            }
            PipeBarrier<PIPE_V>();
        }
        // Compact into dead k=0 partial planes; destinations cannot clobber live k>0 sums.
        for (uint32_t k = 1; k < OUTPUT_CHANNELS; ++k) {
            Copy(acc[k * fullTileSize_], acc[k * CHANNEL_GROUP * fullTileSize_], VEC_FLOATS,
                 count / VEC_FLOATS, {1, 1, FLOATS_PER_BLOCK, FLOATS_PER_BLOCK});
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t k = 0; k < OUTPUT_CHANNELS; ++k) {
            auto logit = acc[k * fullTileSize_];
            Adds(logit, logit, bias_.Get<float>().GetValue(k), count);
        }
    }

    __aicore__ inline float ReduceSum(LocalTensor<float> source, uint32_t count)
    {
        auto sums = reduce_.Get<float>();
        const uint32_t repeats = count / VEC_FLOATS;
        WholeReduceSum(sums, source, VEC_FLOATS, repeats, 1, 1, FLOATS_PER_BLOCK);
        Wait<HardEvent::V_S>();
        float partial[16];
        for (uint32_t i = 0; i < repeats; ++i) {
            partial[i] = sums.GetValue(i);
        }
        for (uint32_t size = repeats; size > 1; size = (size + 1) / 2) {
            for (uint32_t i = 0; i < size / 2; ++i) {
                partial[i] = partial[2 * i] + partial[2 * i + 1];
            }
            if ((size & 1) != 0) {
                partial[size / 2] = partial[size - 1];
            }
        }
        Wait<HardEvent::S_V>();
        return partial[0];
    }

    __aicore__ inline float ReduceMax(LocalTensor<float> source, uint32_t count)
    {
        auto maxes = reduce_.Get<float>();
        if (!UnitTau && !regularTau_) {
            Wait<HardEvent::V_S>();
            auto bitValues = source.ReinterpretCast<uint32_t>();
            uint32_t maxBits = bitValues.GetValue(0);
            for (uint32_t i = 1; i < count; ++i) {
                const uint32_t value = bitValues.GetValue(i);
                maxBits = FloatOrder(value) > FloatOrder(maxBits) ? value : maxBits;
            }
            Wait<HardEvent::S_V>();
            FloatBits result;
            result.bits = maxBits;
            return result.value;
        }
        const uint32_t repeats = count / VEC_FLOATS;
        WholeReduceMax(maxes, source, VEC_FLOATS, repeats, 1, 1, FLOATS_PER_BLOCK);
        Wait<HardEvent::V_S>();
        float rawMax = maxes.GetValue(0);
        for (uint32_t j = 1; j < repeats; ++j) {
            float value = maxes.GetValue(j * 2);
            rawMax = value > rawMax ? value : rawMax;
        }
        Wait<HardEvent::S_V>();
        return rawMax;
    }

    __aicore__ inline void FillTileCoordinates(uint32_t v, uint32_t u, uint32_t rows, uint32_t cols)
    {
        auto x = xCoords_.Get<float>();
        auto y = yCoords_.Get<float>();
        for (uint32_t row = 0; row < rows; ++row) {
            CreateVecIndex(x[row * cols], static_cast<float>(static_cast<int32_t>(u)), cols);
            Duplicate(y[row * cols], static_cast<float>(static_cast<int32_t>(v + row)), cols);
        }
    }

    __aicore__ inline void ComputeTile(uint32_t batch, uint32_t tile, LocalTensor<float> records)
    {
        const uint32_t tilesWide = (width_ + tileWidth_ - 1) / tileWidth_;
        const uint32_t v = tile / tilesWide * tileHeight_;
        const uint32_t u = tile % tilesWide * tileWidth_;
        const uint32_t rows = Min(height_ - v, tileHeight_);
        const uint32_t cols = Min(width_ - u, tileWidth_);
        const uint32_t count = rows * cols;
        Convolve(batch, v, u, rows, cols);
        FillTileCoordinates(v, u, rows, cols);
        PipeBarrier<PIPE_V>();
        auto exp = exponents_.Get<float>();
        auto product = products_.Get<float>();
        for (uint32_t k = 0; k < OUTPUT_CHANNELS; ++k) {
            auto logits = accumulators_.Get<float>()[k * fullTileSize_];
            const float rawMax = ReduceMax(logits, count);
            FormExponent(exp, logits, product, rawMax, count);
            Exp(exp, exp, count);
            PipeBarrier<PIPE_V>();
            const float total = ReduceSum(exp, count);
            Mul(product, exp, xCoords_.Get<float>(), count);
            PipeBarrier<PIPE_V>();
            const float x = ReduceSum(product, count);
            Mul(product, exp, yCoords_.Get<float>(), count);
            PipeBarrier<PIPE_V>();
            const float y = ReduceSum(product, count);
            const uint32_t base = k * RECORD_FLOATS;
            records.SetValue(base, rawMax);
            records.SetValue(base + 1, total);
            records.SetValue(base + 2, x);
            records.SetValue(base + 3, y);
            for (uint32_t j = 4; j < RECORD_FLOATS; ++j) {
                records.SetValue(base + j, 0.0f);
            }
        }
    }

    __aicore__ inline void FormExponent(LocalTensor<float> dst, LocalTensor<float> logits,
                                       LocalTensor<float> denom, float rawMax, uint32_t count)
    {
        if (!UnitTau && !regularTau_) {
            auto raw = logits.ReinterpretCast<uint32_t>();
            auto mantissas = dst.ReinterpretCast<uint32_t>();
            auto shifts = denom.ReinterpretCast<int32_t>();
            const uint32_t maxBits = FloatBits{rawMax}.bits;
            for (uint32_t i = 0; i < count; ++i) {
                int32_t power = 0;
                const uint32_t significand = PositiveGap(maxBits, raw.GetValue(i), power);
                mantissas.SetValue(i, significand == 0 ? 0 : ONE_BITS | (significand & FLOAT_MASK));
                shifts.SetValue(i, power - tauExponent_);
            }
            // Reuse the raw logits after every gap has been formed.
            FloatBits tauMantissa;
            tauMantissa.bits = tauMantissa_;
            Wait<HardEvent::S_V>();
            Duplicate(logits, tauMantissa.value, count);
            PipeBarrier<PIPE_V>();
            auto quotient = shifts_.Get<float>(); // The convolution no longer needs the shift planes.
            Div(quotient, dst, logits, count);
            Wait<HardEvent::V_S>();
            auto quotientBits = quotient.ReinterpretCast<uint32_t>();
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t bits = quotientBits.GetValue(i);
                const int32_t adjusted = static_cast<int32_t>(bits >> 23) + shifts.GetValue(i);
                if (bits == 0 || adjusted <= 0) {
                    mantissas.SetValue(i, 0);
                } else if (adjusted >= EXPONENT_LIMIT) {
                    mantissas.SetValue(i, SIGN_BIT | (static_cast<uint32_t>(EXPONENT_LIMIT) << 23));
                } else {
                    mantissas.SetValue(i, SIGN_BIT | (static_cast<uint32_t>(adjusted) << 23) | (bits & FLOAT_MASK));
                }
            }
            Wait<HardEvent::S_V>();
            return;
        }
        Adds(dst, logits, -rawMax, count);
        PipeBarrier<PIPE_V>();
        Maxs(dst, dst, UnitTau ? -128.0f : -128.0f * tau_, count);
        PipeBarrier<PIPE_V>();
        if constexpr (!UnitTau) {
            Muls(dst, dst, reciprocalTau_, count);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void MergeBatch(uint32_t b, LocalTensor<float> records)
    {
        auto recordOffsets = offsets_.Get<uint32_t>();
        for (uint32_t t = 0; t < tilesPerBatch_; ++t) {
            recordOffsets.SetValue(t, t * TILE_RECORD_FLOATS * sizeof(float));
        }
        Wait<HardEvent::S_V>();
        auto work = merge_.Get<float>();
        auto maxes = work;
        auto exp = work[VEC_FLOATS];
        auto denom = work[5 * VEC_FLOATS];
        for (uint32_t k = 0; k < OUTPUT_CHANNELS; ++k) {
            const uint32_t base = k * RECORD_FLOATS;
            Duplicate(maxes, -0x1.fffffep127f, VEC_FLOATS);
            for (uint32_t s = 2; s < 5; ++s) {
                Duplicate(work[s * VEC_FLOATS], 0.0f, VEC_FLOATS);
            }
            PipeBarrier<PIPE_V>();
            Gather(maxes, records, recordOffsets, base * sizeof(float), tilesPerBatch_);
            for (uint32_t s = 1; s <= 3; ++s) {
                Gather(work[(s + 1) * VEC_FLOATS], records, recordOffsets, (base + s) * sizeof(float),
                       tilesPerBatch_);
            }
            PipeBarrier<PIPE_V>();
            const float rawMax = ReduceMax(maxes, VEC_FLOATS);
            FormExponent(exp, maxes, denom, rawMax, VEC_FLOATS);
            Exp(exp, exp, VEC_FLOATS);
            PipeBarrier<PIPE_V>();
            for (uint32_t s = 2; s < 5; ++s) {
                auto values = work[s * VEC_FLOATS];
                Mul(values, values, exp, VEC_FLOATS);
            }
            PipeBarrier<PIPE_V>();
            const float total = ReduceSum(work[2 * VEC_FLOATS], VEC_FLOATS);
            const float x = ReduceSum(work[3 * VEC_FLOATS], VEC_FLOATS);
            const float y = ReduceSum(work[4 * VEC_FLOATS], VEC_FLOATS);
            auto numerator = work;
            auto denominator = work[FLOATS_PER_BLOCK];
            auto result = work[2 * FLOATS_PER_BLOCK];
            numerator.SetValue(0, x);
            numerator.SetValue(1, y);
            for (uint32_t j = 2; j < FLOATS_PER_BLOCK; ++j) {
                numerator.SetValue(j, 0.0f);
            }
            Wait<HardEvent::S_V>();
            Duplicate(denominator, total, FLOATS_PER_BLOCK);
            PipeBarrier<PIPE_V>();
            Div(result, numerator, denominator, FLOATS_PER_BLOCK);
            Wait<HardEvent::V_S>();
            const uint32_t out = b * OUTPUT_CHANNELS + k;
            keypoints_.Get<float>().SetValue(2 * out, (result.GetValue(0) + 0.5f) * static_cast<float>(stride_));
            keypoints_.Get<float>().SetValue(2 * out + 1, (result.GetValue(1) + 0.5f) * static_cast<float>(stride_));
            scores_.Get<float>().SetValue(out, rawMax);
            Wait<HardEvent::S_V>();
        }
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, HALO_SLOTS> haloQueue_;
    TBuf<TPosition::VECIN> weights_, bias_;
    TBuf<TPosition::VECCALC> vectorWeights_;
    TBuf<TPosition::VECCALC> offsets_, shifts_, accumulators_, xCoords_, yCoords_, exponents_, products_, reduce_, merge_;
    TBuf<TPosition::VECIN> records_;
    TBuf<TPosition::VECOUT> published_, keypoints_, scores_;
    GlobalTensor<float> featGm_, weightGm_, biasGm_, keypointsGm_, scoresGm_, recordsGm_;
    uint32_t batch_, blocks_;
    int32_t stride_;
    float tau_;
    float reciprocalTau_ = 0.0f;
    uint32_t tauMantissa_ = 0;
    int32_t tauExponent_ = 0;
    bool regularTau_ = true;
};
}  // namespace

template <typename DT_FEAT, uint32_t SHAPE_ID, uint32_t UNIT_TAU>
__global__ __aicore__ void fused_disconnector_keypoint_soft_argmax(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints,
    GM_ADDR scores, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FusedDisconnectorKeypointSoftArgmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedDisconnectorKeypointSoftArgmaxTilingData, tilingData, tiling);
    FusedKeypointKernel<SHAPE_ID, UNIT_TAU != 0> kernel;
    // The ACLNN wrapper offsets workspace past the system region. Pure AIV hard
    // sync needs no system region, so use the original allocation for records.
    (void)workspace;
    kernel.Init(feat, weight, bias, keypoints, scores, GetSysWorkSpacePtr(), tilingData);
    kernel.Process();
}
