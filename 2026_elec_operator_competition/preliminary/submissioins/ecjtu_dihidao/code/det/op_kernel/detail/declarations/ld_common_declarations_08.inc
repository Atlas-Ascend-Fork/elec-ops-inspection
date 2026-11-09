/*
 * Copyright (c) 2026 dihidao.
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
// Intentionally included multiple times in class scope.
// Complete declarations only; template names and member order are preserved.
static constexpr bool GUARDED = ClassPlan46::GROUP == 8U || ClassPlan46::GROUP == 4U || ClassPlan46::HETERO6;
    static constexpr bool PREFIX_STREAM = ClassPlan46::HETERO6;
    GM_ADDR prefixBase;
    static_assert(!ClassPlan46::PIVOT8_AHEAD || B <= 32U, "first retired TMP tile covers all reduction destinations");
    static constexpr uint32_t GUARD_ROWS = GUARDED ? 8U : 0U, GUARD_COEFF = GUARDED ? 1U : 0U;
    static constexpr uint32_t MAT = 0, PANEL = ClassPlan46::THIN_H6 ? MAT : MAT + NMAX * S * 4U,
                              FAC = PANEL + P * (NMAX + GUARD_ROWS) * 4;
    static_assert(ClassPlan46::THIN_H6 || PANEL >= MAT + NMAX * S * sizeof(float), "cyclic matrix overlaps panel");
    static constexpr uint32_t META_WORDS = 8U + 2U * B;
    static constexpr uint32_t BRC = FAC + (B * NMAX + META_WORDS) * 4, COL = BRC + (NMAX + GUARD_COEFF) * 8 * 4;
    static constexpr uint32_t ABS = COL + NMAX * 4, ROW = ABS + NMAX * 4, PROW = ROW + S * 4;
    static constexpr uint32_t TMP = PROW + B * 4, TAB = TMP + 1024, LANE = TAB + NMAX * 4;
    static constexpr uint32_t META = FAC + B * NMAX * 4U, END = LANE + 8 * NMAX * 4U;
    static constexpr uint32_t GM_WORDS = B * NMAX + META_WORDS;
    static_assert(END <= 196352, "cyclic row UB budget");
    static_assert(ClassPlan46::DEFER_BOOKKEEP && ClassPlan46::UB_FINISH64 && NMAX == 256U && B == 32U,
                  "normalGuard prefix-only scope");
    LocalTensor<float> mat, panel, fac, brc, col, ab, row, prow, tmp, meta;
    LocalTensor<int32_t> tab, lanes;
