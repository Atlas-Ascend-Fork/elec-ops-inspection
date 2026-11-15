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

// Column-major LU for the 15- and 16-row scalar fallback. Template indices
// retain the original unrolled order and constant local-memory offsets.
template<Zq2 Size>
struct TinyScalarTriangular {
    static_assert(Size == 15u || Size == 16u, "Unsupported tiny shape");
    static constexpr Zq2 pitch = 16u;

    template<Zq2 Row, Zq2 Column>
    __aicore__ static inline void ForwardRow(Zq7<float> factors, float (&rhs)[Size])
    {
        if constexpr (Column < Row) {
            rhs[Row] = square_math::RoundSub(
                rhs[Row], square_math::RoundMul(factors.GetValue(Column * pitch + Row), rhs[Column]));
            ForwardRow<Row, Column + 1u>(factors, rhs);
        }
    }

    template<Zq2 Row>
    __aicore__ static inline void Forward(Zq7<float> factors, float (&rhs)[Size])
    {
        if constexpr (Row < Size) {
            ForwardRow<Row, 0u>(factors, rhs);
            Forward<Row + 1u>(factors, rhs);
        }
    }

    template<Zq2 Row, Zq2 Column>
    __aicore__ static inline void BackwardRow(Zq7<float> factors, float (&rhs)[Size])
    {
        if constexpr (Column > Row) {
            rhs[Row] = square_math::RoundSub(
                rhs[Row], square_math::RoundMul(factors.GetValue(Column * pitch + Row), rhs[Column]));
            BackwardRow<Row, Column - 1u>(factors, rhs);
        }
    }

    template<Zq2 Row>
    __aicore__ static inline void Backward(Zq7<float> factors, const __gm__ float* inverse, float (&rhs)[Size])
    {
        BackwardRow<Row, Size - 1u>(factors, rhs);
        const float diagonal = factors.GetValue(Row * pitch + Row);
        const float magnitude = diagonal < 0.0f ? -diagonal : diagonal;
        const bool reciprocalSafe = magnitude >= 1.0e-30f && magnitude <= 1.0e30f;
        rhs[Row] = diagonal == 0.0f ? 0.0f
                                   : (reciprocalSafe ? square_math::RoundMul(rhs[Row], inverse[Row])
                                                     : square_math::Divide(rhs[Row], diagonal));
        if constexpr (Row > 0u) {
            Backward<Row - 1u>(factors, inverse, rhs);
        }
    }

    template<Zq2 Row>
    __aicore__ static inline void Load(Zq7<float> source, float (&rhs)[Size])
    {
        if constexpr (Row < Size) {
            rhs[Row] = source.GetValue(Row);
            Load<Row + 1u>(source, rhs);
        }
    }

    template<Zq2 Row>
    __aicore__ static inline void Store(const float (&rhs)[Size], __gm__ float* output, Zq2 stride)
    {
        if constexpr (Row < Size) {
            output[Zq3(Row) * stride] = rhs[Row];
            Store<Row + 1u>(rhs, output, stride);
        }
    }

    __aicore__ static inline void Solve(Zq7<float> factors, Zq7<float> source, const __gm__ float* inverse,
                                        __gm__ float* output, Zq2 stride)
    {
        float rhs[Size];
        Load<0u>(source, rhs);
        Forward<0u>(factors, rhs);
        Backward<Size - 1u>(factors, inverse, rhs);
        Store<0u>(rhs, output, stride);
    }
};
