
#pragma once

#include <cstdint>

struct SplineConvTilingData {
    uint32_t length;
    uint32_t numNodes;
    uint32_t numEdges;
    uint32_t numCin;
    uint32_t numCout;
    uint32_t dim;
    uint32_t numBasis;
    uint32_t numKernel;
    uint32_t ks0;
    uint32_t ks1;
    uint32_t ks2;
    uint32_t open0;
    uint32_t open1;
    uint32_t open2;
    uint32_t degree;
    uint32_t aggrMean;
    uint32_t tileNodes;
    uint32_t edgeTile;
    uint32_t blockNum;
};
