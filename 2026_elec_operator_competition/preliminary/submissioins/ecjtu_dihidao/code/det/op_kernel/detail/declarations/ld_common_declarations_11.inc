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
GM_ADDR prefixFlag;
    static_assert(END <= 196352, "cyclic row UB budget");
    LocalTensor<float> mat, panel, fac, brc, col, ab, row, prow, tmp, meta;
    LocalTensor<int32_t> tab, lanes;
    LocalTensor<uint32_t> mu;
    GlobalTensor<float> in, ws, out;
    GlobalTensor<uint32_t> wu;
    uint32_t n, rank, matrix, blocks, owned, inputDim;
    float sign, prod;
    int32_t ex;
    uint32_t singular;
