/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FUSED_DISCONNECTOR_KEYPOINT_SOFT_ARGMAX_TILING_H
#define FUSED_DISCONNECTOR_KEYPOINT_SOFT_ARGMAX_TILING_H

#include <cstdint>

struct FusedDisconnectorKeypointSoftArgmaxTilingData {
    uint32_t batch;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t height;
    uint32_t width;
    uint32_t wPad;
    uint32_t activeCores;
    uint32_t coresPerBatch;
    uint32_t maxStripRows;
    uint32_t flags;
    float invTau;
    float strideF;
    float halfStrideF;
    uint32_t tempMode;
};

#endif
