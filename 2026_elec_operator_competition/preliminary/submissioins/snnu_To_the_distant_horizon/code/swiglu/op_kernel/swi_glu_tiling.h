/*
 * Copyright (c) 2026 zhiyuan (致远队).
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

// Tiling结构体定义的头文件
// 【v128.2 终极防呆】零外部类型依赖: host(g++ 链) 与 kernel(ccec 链) 无论 include 根下有无 lib/
//   子目录, 本文件都必然编译通过 (不引用 TCubeTiling 等任何框架类型, 仅 cstdint)。
// Cube 使能协议 (双侧 __has_include 探测 + 运行期信号, 任一 miss 自动回退 v127):
//   - host 侧: __has_include 探测 lib/matmul_tiling.h ⇒ 命中且 MatmulApiTiling.GetTiling 成功
//     ⇒ 写 mmBaseM=tc.baseM>0 并将 TCubeTiling 原始字节 memcpy 进 mmRaw; 否则 mmBaseM 保持 0。
//   - kernel 侧: __has_include 探测 lib/matmul_intf.h ⇒ 命中才编译 Cube 家族(别名/成员/REGIST/
//     GEMM 调用), 并 reinterpret_cast 取用 mmRaw; miss ⇒ 编译期剪除 + t9Cube_ 恒 false。
//   - 双侧全命中 ⇒ 真 Cube 路径; 任一 miss ⇒ T9 自动回退 v127 二维分割 (线上已验证, 成绩无损)。
//   - v127 通用路径仅读前三个字段, 布局完全兼容。
#pragma once

#include <cstdint>

struct SwiGluTilingData {
    uint32_t length;      // 赛题原字段: x 总元素数 (= mTotal * d)
    uint32_t mTotal;      // 【v128 落定】行数 M (host 由 shape 解析)
    uint32_t d;           // 【v128 落定】特征维 (host 由 weight_1 shape 解析)
    uint32_t mmBaseM;     // 【v128.2】Cube 使能信号: 0=回退 v127; >0=真 TCubeTiling 已写入 mmRaw
    uint64_t mmRaw[1024]; // 【v128.2】TCubeTiling 裸字节缓冲 (8KB; uint64 保证 8B 对齐)
};
