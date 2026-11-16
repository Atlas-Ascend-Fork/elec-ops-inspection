#pragma once
#include <cstdint>

struct LinalgSolveTilingData {
    uint32_t n, k, nPad, kPad;
    uint32_t rowPitch, batches, blocks, onChip;
    uint32_t batchRank;
    uint32_t dotSize;
    uint32_t paired;
    uint32_t rhsSplit;
    uint32_t blocked;
    uint32_t batchDims[8];
    uint32_t aStrides[8];
    uint32_t bStrides[8];
    uint64_t systemWorkspace;
    uint32_t originalSchur, pairsPerMatrix;
    uint64_t schurOffset;
};

constexpr uint32_t kOS_S=0,kOS_L=kOS_S+256*256,kOS_U=kOS_L+256*256;
constexpr uint32_t kOS_LBlock=kOS_U+256*256,kOS_L11=kOS_LBlock+256*16;
constexpr uint32_t kOS_Perm=kOS_L11+256,kOS_Identity=kOS_Perm+256;
constexpr uint32_t kOS_Private=kOS_Identity+128*128,kOS_Total=kOS_Private+4*4096;
