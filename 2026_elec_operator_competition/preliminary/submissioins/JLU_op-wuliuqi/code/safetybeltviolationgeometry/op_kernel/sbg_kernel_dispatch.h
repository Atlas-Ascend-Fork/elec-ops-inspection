// Copyright (c) 2026 JLU_op-wuliuqi.
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

template <typename T>
struct SbgKernelFamily;

template <>
struct SbgKernelFamily<float> {
    using Small = SbgFp32Range1To31;
    template <bool U, uint32_t R> using Point32 = SbgFp32B32<U, R>;
    template <uint32_t B, bool U, uint32_t R> using Middle = SbgFp32Range33To255<B, U, R>;
    template <bool U, uint32_t R> using Point256 = SbgFp32B256<U, R>;
    template <uint32_t B, bool U, uint32_t R, bool TAIL> using Large = SbgFp32Range257To4095<B, U, R, TAIL>;
    template <bool U, uint32_t R> using Point4096 = SbgFp32B4096<U, R>;
};

template <>
struct SbgKernelFamily<half> {
    using Small = SbgFp16Range1To31;
    template <bool U, uint32_t R> using Point32 = SbgFp16B32<U, R>;
    template <uint32_t B, bool U, uint32_t R> using Middle = SbgFp16Range33To255<B, U, R>;
    template <bool U, uint32_t R> using Point256 = SbgFp16B256<U, R>;
    template <uint32_t B, bool U, uint32_t R, bool TAIL> using Large = SbgFp16Range257To4095<B, U, R, TAIL>;
    template <bool U, uint32_t R> using Point4096 = SbgFp16B4096<U, R>;
};

struct SbgKernelArguments {
    GM_ADDR boxes;
    GM_ADDR anchors;
    GM_ADDR threshold;
    GM_ADDR output;
    GM_ADDR tiling;

    template <typename Kernel>
    __aicore__ inline void RunPoint() const {
        Kernel point;
        point.Run(boxes, anchors, threshold, output, threshold != nullptr ? 1U : 0U, tiling);
    }

    template <typename Kernel, uint32_t FIXED, uint32_t PER>
    __aicore__ inline void RunRange() const {
        __gm__ uint32_t *data = reinterpret_cast<__gm__ uint32_t*>(tiling);
        Kernel point;
        point.Run(boxes, anchors, threshold, output, FIXED ? FIXED : data[0],
            FIXED ? PER : data[1], FIXED ? (threshold != nullptr ? 1U : 0U) : data[2], tiling);
    }
};

template <typename T, uint32_t BASE, bool UNIT_RATIO, uint32_t RATIO_KIND, bool TAIL>
__aicore__ inline void SbgRunLargePlan(const SbgKernelArguments &args) {
    constexpr uint32_t fixed = BASE == 257U || BASE == 4095U ? BASE : 0U;
    constexpr uint32_t largePer = sizeof(T) == 4U ? 576U : 640U;
    constexpr uint32_t per = fixed == 4095U ? largePer : 272U;
    using Kernel = typename SbgKernelFamily<T>::template Large<fixed, UNIT_RATIO, RATIO_KIND, TAIL>;
    args.template RunRange<Kernel, fixed, per>();
}

template <typename T, uint32_t BASE, bool UNIT_RATIO, uint32_t RATIO_KIND>
__aicore__ inline void SbgRunLarge(const SbgKernelArguments &args) {
    if constexpr (sizeof(T) == 4U && BASE == 4095U) {
        if (AscendC::GetBlockIdx() == 7U) {
            SbgRunLargePlan<T, BASE, UNIT_RATIO, RATIO_KIND, true>(args);
        } else {
            SbgRunLargePlan<T, BASE, UNIT_RATIO, RATIO_KIND, false>(args);
        }
    } else {
        SbgRunLargePlan<T, BASE, UNIT_RATIO, RATIO_KIND, false>(args);
    }
}

template <typename T, uint32_t BATCH, uint32_t BASE, bool UNIT_RATIO, uint32_t RATIO_KIND>
__aicore__ inline void SbgRunCompatibility(const SbgKernelArguments &args) {
    if constexpr (BATCH == 0U) {
        __gm__ uint32_t *data = reinterpret_cast<__gm__ uint32_t*>(args.tiling);
        SbgIntervalFallback<T>(args.boxes, args.anchors, args.threshold,
            args.output, data[0], data[1], data[2], args.tiling);
    } else {
        constexpr uint32_t active = SbgShapePolicy(BASE, sizeof(T) == 2U).working;
        constexpr uint32_t per = ((BASE + active - 1U) / active + 15U) / 16U * 16U;
        SbgIntervalFallback<T, BASE, per, UNIT_RATIO, RATIO_KIND>(args.boxes, args.anchors,
            args.threshold, args.output, BASE, per, args.threshold != nullptr ? 1U : 0U, args.tiling);
    }
}

template <typename T, uint32_t BATCH>
struct SbgKernelDispatch {
    static constexpr bool unitRatio = BATCH == 8224U;
    static constexpr bool fixedRatio = BATCH >= 16384U;
    static constexpr uint32_t base = fixedRatio ? BATCH - 16384U : unitRatio ? 32U : BATCH;
    static constexpr uint32_t ratioKind = !fixedRatio ? 0U : base == 256U ? 2U : base == 257U ? 3U : 1U;

    __aicore__ static inline void Run(const SbgKernelArguments &args) {
        using Family = SbgKernelFamily<T>;
        if constexpr (base == 8193U) {
            args.template RunRange<typename Family::Small, 0U, 0U>();
        } else if constexpr (base == 32U) {
            args.template RunPoint<typename Family::template Point32<unitRatio, ratioKind>>();
        } else if constexpr (base == 8194U || base == 33U) {
            constexpr uint32_t fixed = base == 33U ? base : 0U;
            args.template RunRange<typename Family::template Middle<fixed, unitRatio, ratioKind>, fixed, 48U>();
        } else if constexpr (base == 256U) {
            args.template RunPoint<typename Family::template Point256<unitRatio, ratioKind>>();
        } else if constexpr (base == 8195U || base == 8196U || base == 257U || base == 4095U) {
            SbgRunLarge<T, base, unitRatio, ratioKind>(args);
        } else if constexpr (base == 4096U) {
            args.template RunPoint<typename Family::template Point4096<unitRatio, ratioKind>>();
        } else {
            SbgRunCompatibility<T, BATCH, base, unitRatio, ratioKind>(args);
        }
    }
};
