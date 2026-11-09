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
static constexpr uint32_t META_WORDS = 8U + 2U * B;
    static constexpr uint32_t BRC = FAC + (B * NMAX + META_WORDS) * 4, COL = BRC + (NMAX + GUARD_COEFF) * 8 * 4;
    static constexpr uint32_t ABS = COL + NMAX * 4, ROW = ABS + NMAX * 4, PROW = ROW + S * 4;
    static constexpr uint32_t TMP = PROW + B * 4, TAB = TMP + 1024, LANE = TAB + NMAX * 4;
    static constexpr uint32_t META = FAC + B * NMAX * 4U, END = LANE + 8 * NMAX * 4U;
    static constexpr uint32_t GM_WORDS = B * NMAX + META_WORDS;
    static_assert(END <= 196352, "cyclic row UB budget");
    LocalTensor<float> mat, panel, fac, brc, col, ab, row, prow, tmp, meta;
    LocalTensor<int32_t> tab, lanes;
