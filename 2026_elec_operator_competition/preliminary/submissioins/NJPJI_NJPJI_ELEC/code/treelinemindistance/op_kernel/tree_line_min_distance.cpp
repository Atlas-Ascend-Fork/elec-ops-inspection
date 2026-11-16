// Kernel侧核函数实现
// 第 2 题：最小安全距离计算算子 TreeLineMinDistance
// 对每个树障点求到全部导线线段的最小欧氏距离与 argmin（距离相同取最小索引）；
// 零长度线段退化为到起点距离。
//
// 实现要点（真机 910B 验证）：
// 1) 向量化轴 = 树障点：单 tile 装 TILE_PT 个点，对线段顺序循环，
//    片上维护 curMin(距离平方) 与 curIdx(float32)，不落地 (B,N,M) 距离矩阵。
// 2) argmin 平局语义：线段升序 + 严格小于 => 保留最小索引；argmin d ≡ argmin d²。
// 3) 零长度线段走标量分支，绝不除以 0。
// 4) float16 输入在标量抽取时提升到 float32，输出前才降精度。
// 5) 真机约束：每个中间量独立 TBuf；链式向量算子间 PipeBarrier<PIPE_V>()；
//    标量/向量切换 PipeBarrier<PIPE_ALL>()；不用 Compare/Select 造布尔。
#include "kernel_operator.h"

#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"

using namespace AscendC;

namespace {
constexpr int64_t TILE_PT = 1024;    // 单 tile 树障点数上限
constexpr int64_t SEG_TILE = 512;    // 单次搬入 UB 的线段数
constexpr float BIG_INIT = 3.0e38f;  // 初始最小距离平方
constexpr float BIG_MUL = 1.0e20f;   // 谓词放大系数
}  // namespace

template <class DT_TREE_POINTS>
class KernelTreeLineMinDistance {
public:
    __aicore__ inline KernelTreeLineMinDistance() {}
    __aicore__ inline void Init(GM_ADDR tree_points, GM_ADDR line_segments, GM_ADDR min_distance,
                                GM_ADDR nearest_segment_idx, const TreeLineMinDistanceTilingData& t) {
        batch_ = (int64_t)t.batch;
        numPt_ = (int64_t)t.numPoints;
        numSeg_ = (int64_t)t.numSeg;
        tilePt_ = (int64_t)t.tilePoint;
        tileSeg_ = (int64_t)t.tileSeg;
        if (tilePt_ <= 0 || tilePt_ > TILE_PT) {
            tilePt_ = TILE_PT;
        }
        if (tileSeg_ <= 0 || tileSeg_ > SEG_TILE) {
            tileSeg_ = SEG_TILE;
        }

        ptGm_.SetGlobalBuffer((__gm__ Tf*)tree_points, batch_ * numPt_ * 3);
        segGm_.SetGlobalBuffer((__gm__ Tf*)line_segments, batch_ * numSeg_ * 6);
        distGm_.SetGlobalBuffer((__gm__ Tf*)min_distance, batch_ * numPt_);
        idxGm_.SetGlobalBuffer((__gm__ int32_t*)nearest_segment_idx, batch_ * numPt_);

        pipe_.InitBuffer(rawPtBuf_, TILE_PT * 3 * (int64_t)sizeof(Tf));
        pipe_.InitBuffer(rawSegBuf_, SEG_TILE * 6 * (int64_t)sizeof(Tf));
        pipe_.InitBuffer(pxBuf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(pyBuf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(pzBuf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(t1Buf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(t2Buf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(t3Buf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(w1Buf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(w2Buf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(minBuf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(idxBuf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(jBuf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(distBuf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(distTfBuf_, TILE_PT * (int64_t)sizeof(Tf));
        pipe_.InitBuffer(idxOutBuf_, TILE_PT * (int64_t)sizeof(int32_t));
        pipe_.InitBuffer(sABuf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(sBBuf_, TILE_PT * (int64_t)sizeof(float));
        pipe_.InitBuffer(sCBuf_, TILE_PT * (int64_t)sizeof(float));

        rawPt_ = rawPtBuf_.Get<Tf>();
        rawSeg_ = rawSegBuf_.Get<Tf>();
        px_ = pxBuf_.Get<float>();
        py_ = pyBuf_.Get<float>();
        pz_ = pzBuf_.Get<float>();
        t1_ = t1Buf_.Get<float>();
        t2_ = t2Buf_.Get<float>();
        t3_ = t3Buf_.Get<float>();
        w1_ = w1Buf_.Get<float>();
        w2_ = w2Buf_.Get<float>();
        curMin_ = minBuf_.Get<float>();
        curIdx_ = idxBuf_.Get<float>();
        jVec_ = jBuf_.Get<float>();
        dist_ = distBuf_.Get<float>();
        distTf_ = distTfBuf_.Get<Tf>();
        idxOut_ = idxOutBuf_.Get<int32_t>();
        sA_ = sABuf_.Get<float>();
        sB_ = sBBuf_.Get<float>();
        sC_ = sCBuf_.Get<float>();
    }

    __aicore__ inline void Process() {
        if (batch_ <= 0 || numPt_ <= 0 || numSeg_ <= 0) {
            return;
        }
        const int64_t blockIdx = (int64_t)GetBlockIdx();
        const int64_t blockNum = (int64_t)GetBlockNum();
        const int64_t tilesPerBatch = (numPt_ + tilePt_ - 1) / tilePt_;
        const int64_t totalTiles = batch_ * tilesPerBatch;
        // 连续切分：同一核尽量连续处理同 batch 的 tile，线段分块可在核内复用
        const int64_t perCore = (totalTiles + blockNum - 1) / blockNum;
        int64_t begin = blockIdx * perCore;
        int64_t end = begin + perCore;
        if (begin > totalTiles) begin = totalTiles;
        if (end > totalTiles) end = totalTiles;

        const DataCopyPadExtParams<Tf> padTf{false, 0, 0, static_cast<Tf>(0)};
        int64_t cachedBatch = -1;
        int64_t cachedSeg = -1;

        for (int64_t tIdx = begin; tIdx < end; ++tIdx) {
            const int64_t b = tIdx / tilesPerBatch;
            const int64_t tp = tIdx % tilesPerBatch;
            const int64_t ptOff = tp * tilePt_;
            const int64_t n = (ptOff + tilePt_ <= numPt_) ? tilePt_ : (numPt_ - ptOff);

            // 点 tile 搬入 + 标量抽取 3 个坐标 plane
            DataCopyPad(rawPt_, ptGm_[(b * numPt_ + ptOff) * 3],
                        DataCopyExtParams{1, (uint32_t)(n * 3 * (int64_t)sizeof(Tf)), 0, 0, 0}, padTf);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            for (int64_t i = 0; i < n; ++i) {
                const int64_t f3 = i * 3;
                px_.SetValue((int32_t)i, static_cast<float>(rawPt_.GetValue((int32_t)(f3 + 0))));
                py_.SetValue((int32_t)i, static_cast<float>(rawPt_.GetValue((int32_t)(f3 + 1))));
                pz_.SetValue((int32_t)i, static_cast<float>(rawPt_.GetValue((int32_t)(f3 + 2))));
            }
            PipeBarrier<PIPE_ALL>();

            const int32_t cnt = (int32_t)n;
            Duplicate<float>(curMin_, BIG_INIT, (int32_t)TILE_PT);
            PipeBarrier<PIPE_V>();
            Duplicate<float>(curIdx_, 0.0f, (int32_t)TILE_PT);
            PipeBarrier<PIPE_V>();

            for (int64_t sOff = 0; sOff < numSeg_; sOff += tileSeg_) {
                const int64_t nSeg = (sOff + tileSeg_ <= numSeg_) ? tileSeg_ : (numSeg_ - sOff);
                if (cachedBatch != b || cachedSeg != sOff) {
                    DataCopyPad(rawSeg_, segGm_[(b * numSeg_ + sOff) * 6],
                                DataCopyExtParams{1, (uint32_t)(nSeg * 6 * (int64_t)sizeof(Tf)), 0, 0, 0},
                                padTf);
                    SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
                    WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
                    cachedBatch = b;
                    cachedSeg = sOff;
                }

                for (int64_t j = 0; j < nSeg; ++j) {
                    const int64_t f6 = j * 6;
                    // line_segments 布局 [b,m,0,:]=起点 A、[b,m,1,:]=终点 C => AC = C - A
                    const float axv = static_cast<float>(rawSeg_.GetValue((int32_t)(f6 + 0)));
                    const float ayv = static_cast<float>(rawSeg_.GetValue((int32_t)(f6 + 1)));
                    const float azv = static_cast<float>(rawSeg_.GetValue((int32_t)(f6 + 2)));
                    const float cxv = static_cast<float>(rawSeg_.GetValue((int32_t)(f6 + 3)));
                    const float cyv = static_cast<float>(rawSeg_.GetValue((int32_t)(f6 + 4)));
                    const float czv = static_cast<float>(rawSeg_.GetValue((int32_t)(f6 + 5)));
                    const float acxv = cxv - axv;
                    const float acyv = cyv - ayv;
                    const float aczv = czv - azv;
                    const float len2 = acxv * acxv + acyv * acyv + aczv * aczv;

                    // AP = P - A
                    Adds(t1_, px_, -axv, cnt);
                    Adds(t2_, py_, -ayv, cnt);
                    Adds(t3_, pz_, -azv, cnt);

                    if (len2 > 0.0f) {
                        const float invLen2 = 1.0f / len2;
                        Muls(w1_, t1_, acxv, cnt);
                        Axpy(w1_, t2_, acyv, cnt);
                        Axpy(w1_, t3_, aczv, cnt);          // dot
                        Muls(w1_, w1_, invLen2, cnt);     // t
                        // 三段掩码：w2_=mA(t<0)、jVec_=mB(t>1)、dist_=mI(内部)
                        Muls(w2_, w1_, -1.0e30f, cnt);
                        Maxs(w2_, w2_, 0.0f, cnt);
                        Mins(w2_, w2_, 1.0f, cnt);
                        Adds(jVec_, w1_, -1.0f, cnt);
                        Muls(jVec_, jVec_, 1.0e30f, cnt);
                        Maxs(jVec_, jVec_, 0.0f, cnt);
                        Mins(jVec_, jVec_, 1.0f, cnt);
                        Duplicate(dist_, 1.0f, cnt);
                        Sub(dist_, dist_, w2_, cnt);
                        Sub(dist_, dist_, jVec_, cnt);
                        // dA² = |AP|²（t<0 段；t1..t3 保持 AP）
                        // 注意：px_/py_/pz_ 是 P 坐标，跨线段迭代复用，绝不能当草稿缓冲！
                        Mul(sA_, t1_, t1_, cnt);
                        Mul(sB_, t2_, t2_, cnt);
                        Add(sA_, sA_, sB_, cnt);
                        Mul(sB_, t3_, t3_, cnt);
                        Add(sA_, sA_, sB_, cnt);
                        Mul(sA_, sA_, w2_, cnt);          // mAdA²（w2_ 的 mA 用毕，可复用）
                        // dB² = |PC|²，PC = AP − AC（t>1 段）；草稿用 sB_/sC_/w2_
                        Adds(sB_, t1_, -acxv, cnt);
                        Adds(sC_, t2_, -acyv, cnt);
                        Adds(w2_, t3_, -aczv, cnt);
                        Mul(sB_, sB_, sB_, cnt);
                        Mul(sC_, sC_, sC_, cnt);
                        Add(sB_, sB_, sC_, cnt);
                        Mul(w2_, w2_, w2_, cnt);
                        Add(sB_, sB_, w2_, cnt);
                        Mul(sB_, sB_, jVec_, cnt);        // mBdB²（jVec_ 的 mB 用毕，可复用）
                        Add(sA_, sA_, sB_, cnt);
                        // dI² = |AP×AC|²/L（内部段；与 t 的舍入无关）
                        // 叉积分量：c1 = APy*ACz − APz*ACy → sC_；c2 = APz*ACx − APx*ACz → sB_；
                        //           c3 = APx*ACy − APy*ACx → jVec_
                        Muls(sB_, t2_, aczv, cnt);
                        Muls(sC_, t3_, acyv, cnt);
                        Sub(sC_, sB_, sC_, cnt);
                        Muls(sB_, t3_, acxv, cnt);
                        Muls(w2_, t1_, aczv, cnt);
                        Sub(sB_, sB_, w2_, cnt);
                        Muls(w2_, t1_, acyv, cnt);
                        Muls(jVec_, t2_, acxv, cnt);
                        Sub(jVec_, w2_, jVec_, cnt);
                        Mul(sC_, sC_, sC_, cnt);
                        Mul(sB_, sB_, sB_, cnt);
                        Mul(jVec_, jVec_, jVec_, cnt);
                        Add(sC_, sC_, sB_, cnt);
                        Add(sC_, sC_, jVec_, cnt);
                        Muls(sC_, sC_, invLen2, cnt);     // dI²
                        Mul(sC_, sC_, dist_, cnt);        // mIdI²
                        Add(sA_, sA_, sC_, cnt);          // d² = mAdA² + mBdB² + mIdI²
                        Adds(w1_, sA_, 0.0f, cnt);        // w1_ = |d|²
                    } else {
                        // len2 == 0：t1..t3 已是 P - A
                        Mul(w1_, t1_, t1_, cnt);
                        Mul(w2_, t2_, t2_, cnt);
                        Add(w1_, w1_, w2_, cnt);
                        Mul(w2_, t3_, t3_, cnt);
                        Add(w1_, w1_, w2_, cnt);
                    }

                    const bool isFirstSeg = (sOff == 0 && j == 0);
                    if (isFirstSeg) {
                        Adds(curMin_, w1_, 0.0f, cnt);
                        PipeBarrier<PIPE_V>();
                        Duplicate<float>(curIdx_, 0.0f, (int32_t)TILE_PT);
                        PipeBarrier<PIPE_V>();
                    } else {
                        // 谓词 p：d2 < curMin − 1e-7·curMin 时为 1。
                        // 容差内的差异视为平局，保留最小索引（对冲 fp32 噪声破坏对称平局）。
                        Sub(w2_, curMin_, w1_, cnt);
                        PipeBarrier<PIPE_V>();
                        Muls(t3_, curMin_, 1.0e-7f, cnt);   // 仅约 1 ulp：容差取大会引入 δd≈ε·d/2 的系统误差
                        PipeBarrier<PIPE_V>();
                        Sub(w2_, w2_, t3_, cnt);
                        PipeBarrier<PIPE_V>();
                        Muls(w2_, w2_, 1.0e30f, cnt);
                        PipeBarrier<PIPE_V>();
                        Maxs(w2_, w2_, 0.0f, cnt);
                        PipeBarrier<PIPE_V>();
                        Mins(w2_, w2_, 1.0f, cnt);
                        PipeBarrier<PIPE_V>();
                        // curIdx += p * (j - curIdx)
                        Duplicate<float>(jVec_, (float)(sOff + j), (int32_t)TILE_PT);
                        PipeBarrier<PIPE_V>();
                        Sub(t1_, jVec_, curIdx_, cnt);
                        PipeBarrier<PIPE_V>();
                        Mul(t1_, t1_, w2_, cnt);
                        PipeBarrier<PIPE_V>();
                        Add(curIdx_, curIdx_, t1_, cnt);
                        PipeBarrier<PIPE_V>();
                        // curMin 用向量 Min 精确取小：
                        // 不能用 curMin += p*(d2−curMin)，当首条线段很远时 ulp(curMin)≈0.5
                        // 会把 d² 误差直接放大到 δd≈0.5/(2d)（实测 1.1e-3）。
                        Min(curMin_, curMin_, w1_, cnt);
                        PipeBarrier<PIPE_V>();
                    }
                }
                SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
            }

            // 输出：真实欧氏距离（平方根只做一次）+ 最近线段索引
            Sqrt(dist_, curMin_, cnt);
            if constexpr (sizeof(Tf) == sizeof(float)) {
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyPad(distGm_[b * numPt_ + ptOff], dist_,
                            DataCopyExtParams{1, (uint32_t)(n * (int64_t)sizeof(Tf)), 0, 0, 0});
            } else {
                PipeBarrier<PIPE_V>();
                Cast(distTf_, dist_, RoundMode::CAST_RINT, (uint32_t)cnt);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyPad(distGm_[b * numPt_ + ptOff], distTf_,
                            DataCopyExtParams{1, (uint32_t)(n * (int64_t)sizeof(Tf)), 0, 0, 0});
            }
            Cast(idxOut_, curIdx_, RoundMode::CAST_RINT, (uint32_t)cnt);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            DataCopyPad(idxGm_[b * numPt_ + ptOff], idxOut_,
                        DataCopyExtParams{1, (uint32_t)(n * (int64_t)sizeof(int32_t)), 0, 0, 0});

            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

private:
    using Tf = DT_TREE_POINTS;

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawPtBuf_;
    TBuf<TPosition::VECCALC> rawSegBuf_;
    TBuf<TPosition::VECCALC> pxBuf_, pyBuf_, pzBuf_;
    TBuf<TPosition::VECCALC> t1Buf_, t2Buf_, t3Buf_;
    TBuf<TPosition::VECCALC> w1Buf_, w2Buf_;
    TBuf<TPosition::VECCALC> minBuf_, idxBuf_, jBuf_;
    TBuf<TPosition::VECCALC> distBuf_, distTfBuf_, idxOutBuf_;
    TBuf<TPosition::VECCALC> sABuf_, sBBuf_, sCBuf_;

    LocalTensor<Tf> rawPt_, rawSeg_;
    LocalTensor<float> px_, py_, pz_;
    LocalTensor<float> t1_, t2_, t3_;
    LocalTensor<float> w1_, w2_;
    LocalTensor<float> curMin_, curIdx_, jVec_;
    LocalTensor<float> dist_;
    LocalTensor<Tf> distTf_;
    LocalTensor<int32_t> idxOut_;
    LocalTensor<float> sA_, sB_, sC_;

    GlobalTensor<Tf> ptGm_, segGm_, distGm_;
    GlobalTensor<int32_t> idxGm_;

    int64_t batch_ = 0;
    int64_t numPt_ = 0;
    int64_t numSeg_ = 0;
    int64_t tilePt_ = TILE_PT;
    int64_t tileSeg_ = SEG_TILE;
};

template <typename DT_TREE_POINTS>
 __global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points, GM_ADDR line_segments, GM_ADDR min_distance, GM_ADDR nearest_segment_idx, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, tiling_data, tiling);
    KernelTreeLineMinDistance<DT_TREE_POINTS> op;
    op.Init(tree_points, line_segments, min_distance, nearest_segment_idx, tiling_data);
    op.Process();
}
