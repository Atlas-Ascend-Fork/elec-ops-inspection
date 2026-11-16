#pragma once
#include <cstdint>
struct FusedDisconnectorKeypointSoftArgmaxTilingHeader {
uint32_t B;
uint32_t C;
uint32_t H;
uint32_t W;
uint32_t stripesPerBatch;
uint32_t usedCores;
uint32_t exactInvTau;
float tau;
float invTau;
float strideFloat;
};
struct FusedDisconnectorKeypointSoftArgmaxTilingData {
uint32_t B;
uint32_t C;
uint32_t H;
uint32_t W;
uint32_t stripesPerBatch;
uint32_t usedCores;
uint32_t exactInvTau;
float tau;
float invTau;
float strideFloat;
uint32_t pad[6];
float syncSlots[28 * 16];
};
