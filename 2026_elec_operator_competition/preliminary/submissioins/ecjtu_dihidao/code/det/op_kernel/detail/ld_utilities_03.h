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
#pragma once
// Private device implementation; included by ld_device.h.
#if (defined(__DAV_C220_VEC__))
__aicore__ inline __inout_pipe__(V) void LdetVcmax128(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                      uint32_t cnt) {
    vcmax(reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(src.GetPhyAddr()), 1,
          1, 1, 8, Order_t::VALUE_INDEX);
}
#endif

#if (defined(__DAV_C220_VEC__))
__aicore__ inline __inout_pipe__(V) void LdetPanelAbs(const LocalTensor<float> &dst, const LocalTensor<float> &src) {
    vabs(reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(src.GetPhyAddr()), 1,
         1, 1, 8, 8);
}
#endif

#if (defined(__DAV_C220_VEC__))
__aicore__ inline __inout_pipe__(V) void LdetPanelMul(const LocalTensor<float> &dst, const LocalTensor<float> &a,
                                                      const LocalTensor<float> &b) {
    vmul(reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(a.GetPhyAddr()),
         reinterpret_cast<__ubuf__ float *>(b.GetPhyAddr()), 1, 1, 1, 1, 8, 8, 8);
}
#endif

#if (defined(__DAV_C220_VEC__))
__aicore__ inline __inout_pipe__(V) void LdetPanelMuls(const LocalTensor<float> &dst, float scale) {
    vmuls(reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr()),
          scale, 1, 1, 1, 8, 8);
}
#endif

#if (defined(__DAV_C220_VEC__))
__aicore__ inline __inout_pipe__(V) void LdetPanelMla(const LocalTensor<float> &dst, const LocalTensor<float> &factors,
                                                      const LocalTensor<float> &column, uint8_t repeat,
                                                      uint8_t dstRepeatStride = 64U) {
    vmla(reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(factors.GetPhyAddr()),
         reinterpret_cast<__ubuf__ float *>(column.GetPhyAddr()), repeat, 1, 0, 1, dstRepeatStride, 1, 0);
}
#endif

#if (defined(__DAV_C220_VEC__))
__aicore__ inline __inout_pipe__(V) void LdetTinyMulRows(__ubuf__ float *d, __ubuf__ float *a, __ubuf__ float *b,
                                                         uint8_t repeat) {
    vmul(d, a, b, repeat, 1, 0, 1, 1, 1, 0);
}
#endif

#if (defined(__DAV_C220_VEC__))
__aicore__ inline __inout_pipe__(V) void LdetTinyAddRows(__ubuf__ float *d, __ubuf__ float *a, uint8_t repeat) {
    vadd(d, d, a, repeat, 1, 1, 1, 1, 1, 1);
}
#endif
