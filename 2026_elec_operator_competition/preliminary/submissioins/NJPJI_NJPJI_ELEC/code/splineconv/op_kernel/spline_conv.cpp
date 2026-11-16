



















#include "kernel_operator.h"

#include "spline_conv_tiling.h"
#include "tiling_key_spline_conv.h"

using namespace AscendC;

namespace {
constexpr int64_t MAX_C = 128;
constexpr int64_t EDGE_TILE = 512;
constexpr int32_t SPLINE_MAX_S = 64;
}

template <class DT_X>
class KernelSplineConv {
public:
    __aicore__ inline KernelSplineConv() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edge_index, GM_ADDR edge_attr, GM_ADDR weight,
                                GM_ADDR out, const SplineConvTilingData& t) {
        numNodes_ = (int64_t)t.numNodes;
        numEdges_ = (int64_t)t.numEdges;
        cIn_ = (int64_t)t.numCin;
        cOut_ = (int64_t)t.numCout;
        dim_ = (int64_t)t.dim;
        numBasis_ = (int64_t)t.numBasis;
        numKernel_ = (int64_t)t.numKernel;
        degree_ = (int64_t)t.degree;
        aggrMean_ = (t.aggrMean != 0u);
        tileNodes_ = (int64_t)t.tileNodes;
        edgeTile_ = (int64_t)t.edgeTile;
        ks_[0] = (int64_t)t.ks0;
        ks_[1] = (int64_t)t.ks1;
        ks_[2] = (int64_t)t.ks2;
        open_[0] = (t.open0 != 0u);
        open_[1] = (t.open1 != 0u);
        open_[2] = (t.open2 != 0u);

        if (tileNodes_ > MAX_C) tileNodes_ = MAX_C;
        if (tileNodes_ < 1) tileNodes_ = 1;
        if (edgeTile_ > EDGE_TILE) edgeTile_ = EDGE_TILE;
        if (edgeTile_ < 1) edgeTile_ = 1;

        xGm_.SetGlobalBuffer((__gm__ Tf*)x, numNodes_ * cIn_);
        eiGm_.SetGlobalBuffer((__gm__ int64_t*)edge_index, 2 * numEdges_);
        attrGm_.SetGlobalBuffer((__gm__ Tf*)edge_attr, numEdges_ * dim_);
        wGm_.SetGlobalBuffer((__gm__ Tf*)weight, numKernel_ * cIn_ * cOut_);
        outGm_.SetGlobalBuffer((__gm__ Tf*)out, numNodes_ * cOut_);


        const int64_t wElems = cIn_ * cOut_;
        const int64_t vecLen = (cIn_ > cOut_) ? cIn_ : cOut_;
        int64_t attrBytes = edgeTile_ * dim_ * (int64_t)sizeof(Tf);
        if (attrBytes < 32) attrBytes = 32;
        pipe_.InitBuffer(dstBuf_, edgeTile_ * (int64_t)sizeof(int64_t));
        pipe_.InitBuffer(srcBuf_, edgeTile_ * (int64_t)sizeof(int64_t));
        pipe_.InitBuffer(attrBuf_, attrBytes);
        pipe_.InitBuffer(accBuf_, tileNodes_ * cOut_ * (int64_t)sizeof(float));
        pipe_.InitBuffer(degBuf_, tileNodes_ * (int64_t)sizeof(float));
        pipe_.InitBuffer(wEffBuf_, wElems * (int64_t)sizeof(float));
        pipe_.InitBuffer(wTmpBuf_, wElems * (int64_t)sizeof(float));
        // 权重行缓存：按 basis 下标缓存，命中则跳过 wElems 大小的 DMA（≤64KB 才启用）
        useCache_ = (numBasis_ * wElems * (int64_t)sizeof(float) <= 64 * 1024);
        if (useCache_) {
            pipe_.InitBuffer(wCacheBuf_, numBasis_ * wElems * (int64_t)sizeof(float));
            wCache_ = wCacheBuf_.Get<float>();
            for (int32_t s = 0; s < (int32_t)SPLINE_MAX_S; ++s) cachedW_[s] = -1;
        }
        pipe_.InitBuffer(xRowBuf_, vecLen * (int64_t)sizeof(float));
        pipe_.InitBuffer(tvBuf_, vecLen * (int64_t)sizeof(float));
        pipe_.InitBuffer(constBuf_, (cOut_ > 32 ? cOut_ : 32) * (int64_t)sizeof(float));

        dTile_ = dstBuf_.Get<int64_t>();
        sTile_ = srcBuf_.Get<int64_t>();
        aTile_ = attrBuf_.Get<Tf>();
        acc_ = accBuf_.Get<float>();
        deg_ = degBuf_.Get<float>();
        wEff_ = wEffBuf_.Get<float>();
        wTmp_ = wTmpBuf_.Get<float>();
        xRow_ = xRowBuf_.Get<float>();
        tv_ = tvBuf_.Get<float>();
        consts_ = constBuf_.Get<float>();
    }

    __aicore__ inline void Process() {
        if (numNodes_ <= 0 || cIn_ <= 0 || cOut_ <= 0) {
            return;
        }
        if (cIn_ > MAX_C || cOut_ > MAX_C) {
            return;
        }
        const int64_t blockIdx = (int64_t)GetBlockIdx();
        const int64_t lo = blockIdx * tileNodes_;
        if (lo >= numNodes_) {
            return;
        }
        const int64_t hi = (lo + tileNodes_ <= numNodes_) ? (lo + tileNodes_) : numNodes_;
        const int64_t rows = hi - lo;

        Duplicate<float>(acc_, 0.0f, (int32_t)(rows * cOut_));
        PipeBarrier<PIPE_V>();
        Duplicate<float>(deg_, 0.0f, (int32_t)rows);
        PipeBarrier<PIPE_ALL>();

        const DataCopyPadExtParams<int64_t> padI64{false, 0, 0, 0};
        const DataCopyPadExtParams<Tf> padTf{false, 0, 0, static_cast<Tf>(0)};


        for (int64_t e0 = 0; e0 < numEdges_; e0 += edgeTile_) {
            const int64_t nt = (e0 + edgeTile_ <= numEdges_) ? edgeTile_ : (numEdges_ - e0);
            DataCopyPad(dTile_, eiGm_[numEdges_ + e0],
                        DataCopyExtParams{1, (uint32_t)(nt * (int64_t)sizeof(int64_t)), 0, 0, 0}, padI64);
            DataCopyPad(sTile_, eiGm_[e0],
                        DataCopyExtParams{1, (uint32_t)(nt * (int64_t)sizeof(int64_t)), 0, 0, 0}, padI64);
            DataCopyPad(aTile_, attrGm_[e0 * dim_],
                        DataCopyExtParams{1, (uint32_t)(nt * dim_ * (int64_t)sizeof(Tf)), 0, 0, 0}, padTf);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

            for (int64_t j = 0; j < nt; ++j) {
                const int64_t dstNode = (int64_t)dTile_.GetValue((int32_t)j);
                if (dstNode < lo || dstNode >= hi) {
                    continue;
                }
                const int64_t srcNode = (int64_t)sTile_.GetValue((int32_t)j);


                float pseudo[3];
                for (int64_t d = 0; d < dim_; ++d) {
                    pseudo[d] = static_cast<float>(aTile_.GetValue((int32_t)(j * dim_ + d)));
                }
                float basis[SPLINE_MAX_S];
                int32_t widx[SPLINE_MAX_S];
                CalcBasis(pseudo, basis, widx);


                const int64_t wElems = cIn_ * cOut_;
                for (int64_t s = 0; s < numBasis_; ++s) {
                    LocalTensor<float> wSrc = useCache_ ? wCache_[s * wElems] : wTmp_;
                    if (!useCache_ || cachedW_[s] != widx[s]) {
                        DataCopyPad(wSrc, wGm_[(int64_t)widx[s] * wElems],
                                    DataCopyExtParams{1, (uint32_t)(wElems * (int64_t)sizeof(Tf)), 0, 0, 0}, padTf);
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                        PipeBarrier<PIPE_ALL>();
                        if (useCache_) cachedW_[s] = widx[s];
                    }
                    if (s == 0) {
                        Muls(wEff_, wSrc, basis[0], (int32_t)wElems);
                    } else {
                        Axpy(wEff_, wSrc, basis[s], (int32_t)wElems);
                    }
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
                }


                DataCopyPad(xRow_, xGm_[srcNode * cIn_],
                            DataCopyExtParams{1, (uint32_t)(cIn_ * (int64_t)sizeof(Tf)), 0, 0, 0}, padTf);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
                PipeBarrier<PIPE_ALL>();
                float xv[MAX_C];
                for (int64_t ci = 0; ci < cIn_; ++ci) {
                    xv[ci] = static_cast<float>(xRow_.GetValue((int32_t)ci));
                }
                PipeBarrier<PIPE_ALL>();
                LocalTensor<float> accRow = acc_[(dstNode - lo) * cOut_];
                for (int64_t ci = 0; ci < cIn_; ++ci) {
                    Axpy(accRow, wEff_[ci * cOut_], xv[ci], (int32_t)cOut_);
                    PipeBarrier<PIPE_V>();
                }
                PipeBarrier<PIPE_ALL>();
                const float prevDeg = deg_.GetValue((int32_t)(dstNode - lo));
                deg_.SetValue((int32_t)(dstNode - lo), prevDeg + 1.0f);
                PipeBarrier<PIPE_ALL>();
                SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
            }
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        }


        LocalTensor<float> degVec = consts_;
        for (int64_t i = lo; i < hi; ++i) {
            LocalTensor<float> accRow = acc_[(i - lo) * cOut_];
            if (aggrMean_) {
                const float d = deg_.GetValue((int32_t)(i - lo));
                if (d > 0.0f) {
                    Duplicate<float>(degVec, d, (int32_t)cOut_);
                    PipeBarrier<PIPE_V>();
                    Div(accRow, accRow, degVec, (int32_t)cOut_);
                    PipeBarrier<PIPE_V>();
                }
            }

        }

        // 整块一次写出：原来每条边一次 DataCopyPad（边多时 DMA 次数是主开销）
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (hi > lo) {
            DataCopyPad(outGm_[lo * cOut_], acc_,
                        DataCopyExtParams{1, (uint32_t)((hi - lo) * cOut_ * (int64_t)sizeof(Tf)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    using Tf = DT_X;


    __aicore__ static inline float BSpline(float v, int32_t kMod, int64_t deg) {
        if (deg == 1) {
            return (kMod == 0) ? (1.0f - v) : v;
        }
        if (deg == 2) {
            const float v2 = v * v;
            if (kMod == 0) return 0.5f * v2 - v + 0.5f;
            if (kMod == 1) return -v2 + v + 0.5f;
            return 0.5f * v2;
        }
        const float v2 = v * v;
        const float v3 = v2 * v;
        if (kMod == 0) return (1.0f - v) * (1.0f - v) * (1.0f - v) / 6.0f;
        if (kMod == 1) return (3.0f * v3 - 6.0f * v2 + 4.0f) / 6.0f;
        if (kMod == 2) return (-3.0f * v3 + 3.0f * v2 + 3.0f * v + 1.0f) / 6.0f;
        return v3 / 6.0f;
    }


    __aicore__ inline void CalcBasis(const float* pseudo, float* basis, int32_t* widx) const {
        const int32_t dq = (int32_t)(degree_ + 1);
        for (int64_t s = 0; s < numBasis_; ++s) {
            int32_t k = (int32_t)s;
            int32_t wi = 0;
            int32_t off = 1;
            float b = 1.0f;
            for (int64_t d = 0; d < dim_; ++d) {
                const int32_t kMod = k % dq;
                k /= dq;
                const int64_t ksD = ks_[d];
                const float scale = static_cast<float>(open_[d] ? (ksD - degree_) : ksD);
                const float v = pseudo[d] * scale;
                int32_t fl = (int32_t)v;
                if (static_cast<float>(fl) > v) {
                    fl -= 1;
                }
                int32_t idx = (fl + kMod) % (int32_t)ksD;
                if (idx < 0) {
                    idx += (int32_t)ksD;
                }
                wi += idx * off;
                off *= (int32_t)ksD;
                const float frac = v - static_cast<float>(fl);
                b *= BSpline(frac, kMod, degree_);
            }
            basis[s] = b;
            widx[s] = wi;
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> dstBuf_, srcBuf_, attrBuf_, accBuf_, degBuf_;
    TBuf<TPosition::VECCALC> wEffBuf_, wTmpBuf_, xRowBuf_, tvBuf_, constBuf_;
    TBuf<TPosition::VECCALC> wCacheBuf_;
    LocalTensor<float> wCache_;
    bool useCache_ = false;
    int32_t cachedW_[SPLINE_MAX_S];

    LocalTensor<int64_t> dTile_, sTile_;
    LocalTensor<Tf> aTile_;
    LocalTensor<float> acc_, deg_, wEff_, wTmp_, xRow_, tv_, consts_;

    GlobalTensor<Tf> xGm_, attrGm_, wGm_, outGm_;
    GlobalTensor<int64_t> eiGm_;

    int64_t numNodes_ = 0;
    int64_t numEdges_ = 0;
    int64_t cIn_ = 0;
    int64_t cOut_ = 0;
    int64_t dim_ = 2;
    int64_t numBasis_ = 1;
    int64_t numKernel_ = 1;
    int64_t degree_ = 1;
    int64_t ks_[3] = {2, 2, 2};
    bool open_[3] = {true, true, true};
    bool aggrMean_ = true;
    int64_t tileNodes_ = 64;
    int64_t edgeTile_ = EDGE_TILE;
};

template <typename DT_X>
 __global__ __aicore__ void spline_conv(GM_ADDR x, GM_ADDR edge_index, GM_ADDR edge_attr, GM_ADDR weight, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SplineConvTilingData);
    GET_TILING_DATA_WITH_STRUCT(SplineConvTilingData, tiling_data, tiling);
    KernelSplineConv<DT_X> op;
    op.Init(x, edge_index, edge_attr, weight, out, tiling_data);
    op.Process();
}
