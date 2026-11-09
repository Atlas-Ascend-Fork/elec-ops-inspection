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
static constexpr uint32_t GUARD_ROWS =
                                  (ClassPlan46::DIM == 192U || ClassPlan46::DIM == 200U || ClassPlan46::DIM == 208U)
                                      ? 8U
                                      : 0U,
                              GUARD_COEFF =
                                  (ClassPlan46::DIM == 192U || ClassPlan46::DIM == 200U || ClassPlan46::DIM == 208U)
                                      ? 1U
                                      : 0U;
    static constexpr uint32_t MAT = 0,
                              PANEL = MAT +
                                      (NMAX + ((ClassPlan46::INPLACE_PANEL || (BALANCED200 && ClassPlan46::LOCAL_TAIL))
                                                   ? GUARD_ROWS
                                                   : 0U)) *
                                          S * 4U,
                              FAC = PANEL + ClassPlan46::PANEL_PITCH * (NMAX + GUARD_ROWS) * 4;
    static_assert(PANEL >= MAT + NMAX * S * sizeof(float), "cyclic matrix overlaps panel");
    static constexpr uint32_t BRC = FAC + B * NMAX * 4, COL = BRC + (NMAX + GUARD_COEFF) * 8 * 4;
    static constexpr uint32_t ABS = COL + NMAX * 4, ROW = ABS + NMAX * 4, PROW = ROW + S * 4;
    static constexpr uint32_t TMP_BYTES = NATIVE208 ? 768U : 1024U;
    static constexpr uint32_t TMP = PROW + B * 4, TAB = TMP + TMP_BYTES, LANE = TAB + NMAX * 4;
    static constexpr uint32_t META = LANE + 8 * NMAX * 4, META_WORDS = 8U + 2U * B, END = META + META_WORDS * 4U;
