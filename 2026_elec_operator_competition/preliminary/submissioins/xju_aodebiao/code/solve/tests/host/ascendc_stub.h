/*
 * Copyright (c) 2026 aodebiao.
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

// Host-only arithmetic/copy test double, not an NPU synchronization simulator.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
using GM_ADDR = unsigned char *;

namespace AscendC {
enum class HardEvent {
  S_MTE2,
  V_MTE2,
  MTE2_S,
  MTE2_V,
  S_MTE3,
  V_MTE3,
  MTE3_S,
  MTE3_V,
  MTE3_MTE2,
  S_V,
  V_S
};
enum class TPosition { VECCALC };

struct SingularSystem : std::runtime_error {
  SingularSystem() : std::runtime_error("singular system") {}
};
[[noreturn]] inline void Trap() { throw SingularSystem(); }
inline uint32_t GetBlockIdx() { return 0; }
template <HardEvent E> inline void SetFlag(int) {}
template <HardEvent E> inline void WaitFlag(int) {}

template <class T> struct LocalTensor {
  T *data;
  LocalTensor operator[](uint64_t offset) const { return {data + offset}; }
  template <class U> LocalTensor<U> ReinterpretCast() const {
    return {reinterpret_cast<U *>(data)};
  }
  T GetValue(uint64_t offset) const { return data[offset]; }
  void SetValue(uint64_t offset, T value) const { data[offset] = value; }
};
template <class T> struct GlobalTensor {
  T *data = nullptr;
  void SetGlobalBuffer(T *value) { data = value; }
  T GetValue(uint64_t offset) const { return data[offset]; }
  GlobalTensor operator[](uint64_t offset) const { return {data + offset}; }
};
template <TPosition P> struct TBuf {
  std::vector<uint64_t> storage;
  template <class T> LocalTensor<T> Get() {
    return {reinterpret_cast<T *>(storage.data())};
  }
};
struct TPipe {
  template <TPosition P> void InitBuffer(TBuf<P> &buffer, uint32_t bytes) {
    buffer.storage.resize((bytes + 7) / 8);
  }
  template <HardEvent E> int AllocEventID() { return 0; }
  template <HardEvent E> void ReleaseEventID(int) {}
};
struct DataCopyExtParams {
  uint16_t blockCount;
  uint32_t blockLen, srcStride, dstStride, rsv;
};
template <class T> struct DataCopyPadExtParams {
  bool pad;
  uint8_t left, right;
  T value;
};
inline void DataCopyPad(LocalTensor<float> dst, GlobalTensor<float> src,
                        DataCopyExtParams cp, DataCopyPadExtParams<float>) {
  auto *output = reinterpret_cast<unsigned char *>(dst.data);
  const auto *input = reinterpret_cast<const unsigned char *>(src.data);
  const uint32_t padded = (cp.blockLen + 31) / 32 * 32;
  for (uint32_t block = 0; block < cp.blockCount; ++block) {
    std::copy_n(input + block * (cp.blockLen + cp.srcStride), cp.blockLen,
                output + block * (padded + cp.dstStride * 32));
  }
}
inline void DataCopyPad(GlobalTensor<float> dst, LocalTensor<float> src,
                        DataCopyExtParams cp) {
  if (cp.blockCount != 1)
    throw std::runtime_error("unsupported host store");
  std::copy_n(src.data, cp.blockLen / sizeof(float), dst.data);
}
inline void Duplicate(LocalTensor<float> dst, float value, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i)
    dst.data[i] = value;
}
inline void Muls(LocalTensor<float> dst, LocalTensor<float> src, float value,
                 uint32_t count) {
  for (uint32_t i = 0; i < count; ++i)
    dst.data[i] = src.data[i] * value;
}
inline void Sub(LocalTensor<float> dst, LocalTensor<float> lhs,
                LocalTensor<float> rhs, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i)
    dst.data[i] = lhs.data[i] - rhs.data[i];
}
inline void Add(LocalTensor<float> dst, LocalTensor<float> lhs,
                LocalTensor<float> rhs, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i)
    dst.data[i] = lhs.data[i] + rhs.data[i];
}
} // namespace AscendC
