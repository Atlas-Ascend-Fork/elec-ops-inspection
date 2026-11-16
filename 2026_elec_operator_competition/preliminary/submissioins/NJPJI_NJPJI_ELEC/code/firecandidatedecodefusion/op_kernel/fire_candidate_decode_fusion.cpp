// Kernel侧核函数实现
// 第 3 题：目标候选框解码与置信度融合算子 FireCandidateDecodeFusion
//
// s  = sigmoid(tx/ty/tw/th/obj/fire)
// x  = (2*sx - 0.5 + gx) * stride
// y  = (2*sy - 0.5 + gy) * stride
// w  = (2*sw)^2 * anchor_w * stride
// h  = (2*sh)^2 * anchor_h * stride
// fire_score = sigmoid(obj) * sigmoid(fire)
//
// 实现要点（真机 910B 验证）：
// 1) raw_pred[B,N,6]、priors[N,5] 行宽非 32B 整数倍，整块连续 DMA 后标量抽取到 float32 plane；
// 2) 全程 float32 中间精度，float16 输入在抽取时提升，输出最后一步才降精度；
// 3) sigmoid 用 Exp + 精确除法 Div，不用 Reciprocal（近似算子误差约 1e-3）；
// 4) 真机约束：独立 TBuf；向量链之间 PipeBarrier<PIPE_V>()；标量/向量切换 PipeBarrier<PIPE_ALL>()。
#include "kernel_operator.h"

#include "fire_candidate_decode_fusion_tiling.h"
#include "tiling_key_fire_candidate_decode_fusion.h"

using namespace AscendC;

namespace {
constexpr int64_t TILE = 1024;     // 单 tile 候选数上限
constexpr int64_t PLANE_IN = 6;    // tx,ty,tw,th,obj,fire
constexpr int64_t PLANE_PR = 5;    // gx,gy,stride,anchor_w,anchor_h
}  // namespace

template <class DT_RAW_PRED>
class KernelFireCandidateDecodeFusion {
public:
    __aicore__ inline KernelFireCandidateDecodeFusion() {}
    __aicore__ inline void Init(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded,
                                const FireCandidateDecodeFusionTilingData& t) {
        batch_ = (int64_t)t.batch;
        numCand_ = (int64_t)t.numCand;
        tile_ = (int64_t)t.tileLen;
        if (tile_ <= 0 || tile_ > TILE) {
            tile_ = TILE;
        }

        predGm_.SetGlobalBuffer((__gm__ Tf*)raw_pred, batch_ * numCand_ * PLANE_IN);
        // 题面规定 priors 固定 float32
        priorGm_.SetGlobalBuffer((__gm__ float*)priors, numCand_ * PLANE_PR);
        outGm_.SetGlobalBuffer((__gm__ Tf*)decoded, batch_ * numCand_ * PLANE_PR);

        pipe_.InitBuffer(predBuf_, TILE * PLANE_IN * (int64_t)sizeof(Tf));
        pipe_.InitBuffer(priorBuf_, TILE * PLANE_PR * (int64_t)sizeof(float));
        pipe_.InitBuffer(txBuf_, TILE * (int64_t)sizeof(float));
        pipe_.InitBuffer(tyBuf_, TILE * (int64_t)sizeof(float));
        pipe_.InitBuffer(twBuf_, TILE * (int64_t)sizeof(float));
        pipe_.InitBuffer(thBuf_, TILE * (int64_t)sizeof(float));
        pipe_.InitBuffer(objBuf_, TILE * (int64_t)sizeof(float));
        pipe_.InitBuffer(fireBuf_, TILE * (int64_t)sizeof(float));
        pipe_.InitBuffer(gxBuf_, TILE * (int64_t)sizeof(float));
        pipe_.InitBuffer(gyBuf_, TILE * (int64_t)sizeof(float));
        pipe_.InitBuffer(strideBuf_, TILE * (int64_t)sizeof(float));
        pipe_.InitBuffer(awBuf_, TILE * (int64_t)sizeof(float));
        pipe_.InitBuffer(ahBuf_, TILE * (int64_t)sizeof(float));
        pipe_.InitBuffer(oneBuf_, TILE * (int64_t)sizeof(float));
        pipe_.InitBuffer(outBuf_, TILE * PLANE_PR * (int64_t)sizeof(Tf));
        pipe_.InitBuffer(idxBuf_, TILE * (int64_t)sizeof(int32_t));   // Gather 偏移表（单位：字节）
        pipe_.InitBuffer(rawTfBuf_, TILE * (int64_t)sizeof(Tf));   // half: Gather 暂存

        rawPred_ = predBuf_.Get<Tf>();
        rawPrior_ = priorBuf_.Get<float>();
        tx_ = txBuf_.Get<float>();
        ty_ = tyBuf_.Get<float>();
        tw_ = twBuf_.Get<float>();
        th_ = thBuf_.Get<float>();
        obj_ = objBuf_.Get<float>();
        fire_ = fireBuf_.Get<float>();
        gx_ = gxBuf_.Get<float>();
        gy_ = gyBuf_.Get<float>();
        strideP_ = strideBuf_.Get<float>();
        aw_ = awBuf_.Get<float>();
        ah_ = ahBuf_.Get<float>();
        one_ = oneBuf_.Get<float>();
        outRaw_ = outBuf_.Get<Tf>();
        rawTf_ = rawTfBuf_.Get<Tf>();

        Duplicate<float>(one_, 1.0f, (int32_t)TILE);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process() {
        if (batch_ <= 0 || numCand_ <= 0) {
            return;
        }
        const int64_t blockIdx = (int64_t)GetBlockIdx();
        const int64_t blockNum = (int64_t)GetBlockNum();
        const int64_t numNTiles = (numCand_ + tile_ - 1) / tile_;
        const int64_t totalTiles = batch_ * numNTiles;
        const DataCopyPadExtParams<Tf> padTf{false, 0, 0, static_cast<Tf>(0)};
        const DataCopyPadExtParams<float> padF32{false, 0, 0, 0.0f};

        for (int64_t t = blockIdx; t < totalTiles; t += blockNum) {
            const int64_t b = t / numNTiles;
            const int64_t tn = t % numNTiles;
            const int64_t candOff = tn * tile_;
            const int64_t n = (candOff + tile_ <= numCand_) ? tile_ : (numCand_ - candOff);

            // 搬入（整块连续）
            DataCopyPad(rawPred_, predGm_[(b * numCand_ + candOff) * PLANE_IN],
                        DataCopyExtParams{1, (uint32_t)(n * PLANE_IN * (int64_t)sizeof(Tf)), 0, 0, 0}, padTf);
            DataCopyPad(rawPrior_, priorGm_[candOff * PLANE_PR],
                        DataCopyExtParams{1, (uint32_t)(n * PLANE_PR * (int64_t)sizeof(float)), 0, 0, 0}, padF32);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

            {
                // Gather 列抽取：srcOffset 单位是**字节**（文档语义），按 sizeof(Tf)/sizeof(float) 缩放
                LocalTensor<int32_t> idxI = idxBuf_.Get<int32_t>();
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                for (int32_t k = 0; k < (int32_t)PLANE_IN; ++k) {
                    ArithProgression(idxI, (int32_t)0, (int32_t)(PLANE_IN * (int64_t)sizeof(Tf)), (int32_t)n);
                    PipeBarrier<PIPE_V>();
                    Adds(idxI, idxI, (int32_t)(k * (int32_t)sizeof(Tf)), (int32_t)n);
                    PipeBarrier<PIPE_V>();
                    LocalTensor<uint32_t> offU = idxI.ReinterpretCast<uint32_t>();
                    if constexpr (IsSameType<Tf, float>::value) {
                        if (k == 0) Gather(tx_, rawPred_, offU, 0, (uint32_t)n);
                        else if (k == 1) Gather(ty_, rawPred_, offU, 0, (uint32_t)n);
                        else if (k == 2) Gather(tw_, rawPred_, offU, 0, (uint32_t)n);
                        else if (k == 3) Gather(th_, rawPred_, offU, 0, (uint32_t)n);
                        else if (k == 4) Gather(obj_, rawPred_, offU, 0, (uint32_t)n);
                        else Gather(fire_, rawPred_, offU, 0, (uint32_t)n);
                    } else {
                        Gather(rawTf_, rawPred_, offU, 0, (uint32_t)n);
                        PipeBarrier<PIPE_V>();
                        if (k == 0) Cast(tx_, rawTf_, RoundMode::CAST_NONE, (int32_t)n);
                        else if (k == 1) Cast(ty_, rawTf_, RoundMode::CAST_NONE, (int32_t)n);
                        else if (k == 2) Cast(tw_, rawTf_, RoundMode::CAST_NONE, (int32_t)n);
                        else if (k == 3) Cast(th_, rawTf_, RoundMode::CAST_NONE, (int32_t)n);
                        else if (k == 4) Cast(obj_, rawTf_, RoundMode::CAST_NONE, (int32_t)n);
                        else Cast(fire_, rawTf_, RoundMode::CAST_NONE, (int32_t)n);
                    }
                    PipeBarrier<PIPE_V>();
                }
                for (int32_t k = 0; k < (int32_t)PLANE_PR; ++k) {
                    ArithProgression(idxI, (int32_t)0, (int32_t)(PLANE_PR * (int64_t)sizeof(float)), (int32_t)n);
                    PipeBarrier<PIPE_V>();
                    Adds(idxI, idxI, (int32_t)(k * (int32_t)sizeof(float)), (int32_t)n);
                    PipeBarrier<PIPE_V>();
                    LocalTensor<uint32_t> offU = idxI.ReinterpretCast<uint32_t>();
                    if (k == 0) Gather(gx_, rawPrior_, offU, 0, (uint32_t)n);
                    else if (k == 1) Gather(gy_, rawPrior_, offU, 0, (uint32_t)n);
                    else if (k == 2) Gather(strideP_, rawPrior_, offU, 0, (uint32_t)n);
                    else if (k == 3) Gather(aw_, rawPrior_, offU, 0, (uint32_t)n);
                    else Gather(ah_, rawPrior_, offU, 0, (uint32_t)n);
                    PipeBarrier<PIPE_V>();
                }
            }
            PipeBarrier<PIPE_ALL>();

            const int32_t cnt = (int32_t)n;
            // sigmoid：1/(1+exp(-x))，用精确除法
            SigmoidInplace(tx_, cnt);
            SigmoidInplace(ty_, cnt);
            SigmoidInplace(tw_, cnt);
            SigmoidInplace(th_, cnt);
            SigmoidInplace(obj_, cnt);
            SigmoidInplace(fire_, cnt);

            // 中心坐标解码：x = (2*s - 0.5 + g) * stride
            CenterDecode(tx_, gx_, strideP_, cnt);
            CenterDecode(ty_, gy_, strideP_, cnt);
            // 宽高解码：w = (2*s)^2 * anchor * stride
            SizeDecode(tw_, aw_, strideP_, cnt);
            SizeDecode(th_, ah_, strideP_, cnt);
            // fire_score = sigmoid(obj) * sigmoid(fire)
            Mul(obj_, obj_, fire_, cnt);
            // 向量 -> 标量：必须全量屏障
            PipeBarrier<PIPE_ALL>();

            // 交错写回 [n,5]：x,y,w,h,fire_score
            for (int64_t i = 0; i < n; ++i) {
                const int64_t b5 = i * PLANE_PR;
                outRaw_.SetValue((int32_t)(b5 + 0), static_cast<Tf>(tx_.GetValue((int32_t)i)));
                outRaw_.SetValue((int32_t)(b5 + 1), static_cast<Tf>(ty_.GetValue((int32_t)i)));
                outRaw_.SetValue((int32_t)(b5 + 2), static_cast<Tf>(tw_.GetValue((int32_t)i)));
                outRaw_.SetValue((int32_t)(b5 + 3), static_cast<Tf>(th_.GetValue((int32_t)i)));
                outRaw_.SetValue((int32_t)(b5 + 4), static_cast<Tf>(obj_.GetValue((int32_t)i)));
            }
            PipeBarrier<PIPE_ALL>();

            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(outGm_[(b * numCand_ + candOff) * PLANE_PR], outRaw_,
                        DataCopyExtParams{1, (uint32_t)(n * PLANE_PR * (int64_t)sizeof(Tf)), 0, 0, 0});

            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

private:
    using Tf = DT_RAW_PRED;

    __aicore__ inline void SigmoidInplace(const LocalTensor<float>& v, int32_t n) {
        Muls(v, v, -1.0f, n);
        PipeBarrier<PIPE_V>();
        Exp(v, v, n);
        PipeBarrier<PIPE_V>();
        Adds(v, v, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Div(v, one_, v, n);   // 精确除法，不用 Reciprocal
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CenterDecode(const LocalTensor<float>& s, const LocalTensor<float>& g,
                                        const LocalTensor<float>& strideV, int32_t n) {
        Muls(s, s, 2.0f, n);
        PipeBarrier<PIPE_V>();
        Add(s, s, g, n);
        PipeBarrier<PIPE_V>();
        Adds(s, s, -0.5f, n);
        PipeBarrier<PIPE_V>();
        Mul(s, s, strideV, n);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SizeDecode(const LocalTensor<float>& s, const LocalTensor<float>& anchor,
                                      const LocalTensor<float>& strideV, int32_t n) {
        Muls(s, s, 2.0f, n);
        PipeBarrier<PIPE_V>();
        Mul(s, s, s, n);
        PipeBarrier<PIPE_V>();
        Mul(s, s, anchor, n);
        PipeBarrier<PIPE_V>();
        Mul(s, s, strideV, n);
        PipeBarrier<PIPE_V>();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> predBuf_, priorBuf_;
    TBuf<TPosition::VECCALC> txBuf_, tyBuf_, twBuf_, thBuf_, objBuf_, fireBuf_;
    TBuf<TPosition::VECCALC> gxBuf_, gyBuf_, strideBuf_, awBuf_, ahBuf_, oneBuf_;
    TBuf<TPosition::VECCALC> outBuf_, idxBuf_, rawTfBuf_;

    LocalTensor<Tf> rawPred_, rawTf_;
    LocalTensor<float> rawPrior_;
    LocalTensor<float> tx_, ty_, tw_, th_, obj_, fire_;
    LocalTensor<float> gx_, gy_, strideP_, aw_, ah_, one_;
    LocalTensor<Tf> outRaw_;

    GlobalTensor<Tf> predGm_, outGm_;
    GlobalTensor<float> priorGm_;

    int64_t batch_ = 0;
    int64_t numCand_ = 0;
    int64_t tile_ = TILE;
};

template <typename DT_RAW_PRED>
 __global__ __aicore__ void fire_candidate_decode_fusion(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FireCandidateDecodeFusionTilingData);
    GET_TILING_DATA_WITH_STRUCT(FireCandidateDecodeFusionTilingData, tiling_data, tiling);
    KernelFireCandidateDecodeFusion<DT_RAW_PRED> op;
    op.Init(raw_pred, priors, decoded, tiling_data);
    op.Process();
}
