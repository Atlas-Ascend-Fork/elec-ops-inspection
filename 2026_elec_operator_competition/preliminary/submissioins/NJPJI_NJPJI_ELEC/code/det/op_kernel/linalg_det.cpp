// Kernel侧核函数实现
// 第 6 题：电力系统线性化分析算子 LinalgDet（det）
//
// 输入 A[*][n][n] float32；输出 det[*] float32。
// 计算：部分主元 LU（P·A = L·U），det = sign(P) · ∏ U_ii。
//
// 实现要点（真机 910B 验证）：
// 1) 行对齐布局：stride = align8(n)，每行起始地址 32B 对齐，行内向量化不踩非对齐 UB 地址；
// 2) 消元起点 c0 = (k+1) 向下取整到 8 的倍数：c0 <= k+1，多更新到的 [c0,k] 列属已用过的 L 区，
//    不影响主元列与 U 对角线，从而全程 32B 对齐；
// 3) 双路径：resident（整矩阵驻留 UB） / streaming（矩阵放 workspace，行 tile 流式）；
// 4) 行列式累乘用「尾数 × 2 的幂」重规约，避免部分乘积溢出/下溢（2 的幂乘除精确）；
// 5) 多核输出：每核负责一段**连续且 8 对齐**的矩阵区间，结果先攒 UB 再一次 DMA 写出
//    （真机上「多核 + 标量写 GM 非 32B 对齐地址」不可靠，只有 m=0 会写成功）；
// 6) 向量链之间 PipeBarrier<PIPE_V>()，标量/向量切换 PipeBarrier<PIPE_ALL>()。
#include "kernel_operator.h"

#include "linalg_det_tiling.h"
#include "tiling_key_linalg_det.h"

using namespace AscendC;

namespace {
constexpr float TWO_P33 = 8589934592.0f;             // 2^33
constexpr float TWO_M33 = 1.1641532182693481e-10f;   // 2^-33
constexpr int64_t DMA_BURST_MAX = 128 * 1024;          // 单次 DataCopyPad 的字节上限
}  // namespace

template <class DT_A>
class KernelLinalgDet {
public:
    __aicore__ inline KernelLinalgDet() {}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, const LinalgDetTilingData& t) {
        numMatrix_ = (int64_t)t.numMatrix;
        n_ = (int64_t)t.n;
        stride_ = (int64_t)t.stride;
        tileRows_ = (int64_t)t.tileRows;
        if (stride_ < n_) stride_ = n_;
        stride_ = (stride_ + 7) / 8 * 8;
        if (tileRows_ < 1) tileRows_ = 1;
        if (tileRows_ > n_) tileRows_ = n_;
        if (tileRows_ < 1) tileRows_ = 1;

        aGm_.SetGlobalBuffer((__gm__ Tf*)A, numMatrix_ * n_ * n_);
        outGm_.SetGlobalBuffer((__gm__ Tf*)det, numMatrix_);

        pipe_.InitBuffer(matBuf_, tileRows_ * stride_ * (int64_t)sizeof(float));
        pipe_.InitBuffer(rowBuf_, stride_ * (int64_t)sizeof(float));
        pipe_.InitBuffer(tmpBuf_, stride_ * (int64_t)sizeof(float));
        int64_t outBytes = numMatrix_ * (int64_t)sizeof(float);
        if (outBytes < 32) outBytes = 32;
        pipe_.InitBuffer(colPadBuf_, 2048 * (int64_t)sizeof(float));
        pipe_.InitBuffer(idxColBuf_, stride_ * (int64_t)sizeof(int32_t));
        pipe_.InitBuffer(outBuf_, outBytes);

        mat_ = matBuf_.Get<float>();
        rowK_ = rowBuf_.Get<float>();
        tmp_ = tmpBuf_.Get<float>();
        outLocal_ = outBuf_.Get<float>();
        colPad_ = colPadBuf_.Get<float>();
        colNeg_ = colPadBuf_.Get<float>()[1024];
        idxCol_ = idxColBuf_.Get<int32_t>();

        wsAddr_ = workspace;
    }

    __aicore__ inline void Process() {
        if (numMatrix_ <= 0 || n_ <= 0) {
            return;
        }
        const bool resident = (n_ <= tileRows_);
        const int64_t blockIdx = (int64_t)GetBlockIdx();
        const int64_t blockNum = (int64_t)GetBlockNum();

        // 每核负责一段连续的、8 对齐的矩阵区间：段起点 begin 是 8 的倍数 => begin*4B 为 32B 对齐
        int64_t perCore = (numMatrix_ + blockNum - 1) / blockNum;
        perCore = (perCore + 7) / 8 * 8;
        const int64_t begin = blockIdx * perCore;
        if (begin >= numMatrix_) {
            return;
        }
        int64_t count = numMatrix_ - begin;
        if (count > perCore) count = perCore;

        __gm__ float* wsCore = (__gm__ float*)wsAddr_ + blockIdx * (n_ * stride_);
        for (int64_t i = 0; i < count; ++i) {
            const int64_t m = begin + i;
            float detVal = resident ? DetResident(m) : DetStreaming(m, wsCore);
            outLocal_.SetValue((int32_t)i, detVal);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams extOut{1, (uint32_t)(count * (int64_t)sizeof(float)), 0, 0, 0};
        DataCopyPad(outGm_[begin], outLocal_, extOut);
    }

private:
    using Tf = DT_A;

    __aicore__ static inline float AbsScalar(float v) { return (v < 0.0f) ? -v : v; }

    __aicore__ static inline void MulDet(float& mant, int32_t& exp2, float v) {
        mant *= v;
        while (mant > 1.0e20f && mant != 0.0f) {
            mant *= TWO_M33;
            exp2 += 33;
        }
        while (mant != 0.0f && AbsScalar(mant) < 1.0e-20f) {
            mant *= TWO_P33;
            exp2 -= 33;
        }
    }

    __aicore__ static inline float ApplyDet(float mant, int32_t exp2, bool negSign) {
        while (exp2 >= 33) {
            mant *= TWO_P33;
            exp2 -= 33;
        }
        while (exp2 <= -33) {
            mant *= TWO_M33;
            exp2 += 33;
        }
        while (exp2 > 0) {
            mant *= 2.0f;
            --exp2;
        }
        while (exp2 < 0) {
            mant *= 0.5f;
            ++exp2;
        }
        return negSign ? -mant : mant;
    }

    // 按行搬入；内部按 DMA_BURST_MAX 分批，避免单次搬运超过硬件 burst 上限导致只有部分数据落地
    __aicore__ inline void LoadRows(const LocalTensor<float>& dst, const GlobalTensor<float>& src,
                                    int64_t srcRowBase, int64_t rows) {
        const DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        const int64_t rowBytes = n_ * (int64_t)sizeof(float);
        int64_t maxRows = (rowBytes > 0) ? (DMA_BURST_MAX / rowBytes) : rows;
        if (maxRows < 1) maxRows = 1;
        for (int64_t r0 = 0; r0 < rows; r0 += maxRows) {
            const int64_t chunk = (r0 + maxRows <= rows) ? maxRows : (rows - r0);
            if (stride_ == n_) {
                DataCopyPad(dst[r0 * stride_], src[(srcRowBase + r0) * n_],
                            DataCopyExtParams{1, (uint32_t)(chunk * rowBytes), 0, 0, 0}, pad);
            } else {
                DataCopyExtParams ext{1, (uint32_t)rowBytes, 0, 0, 0};
                for (int64_t r = 0; r < chunk; ++r) {
                    DataCopyPad(dst[(r0 + r) * stride_], src[(srcRowBase + r0 + r) * n_], ext, pad);
                }
            }
        }
    }

    __aicore__ inline void StoreRows(const LocalTensor<float>& src, GlobalTensor<float>& dst,
                                     int64_t dstRowBase, int64_t rows) {
        const int64_t rowBytes = n_ * (int64_t)sizeof(float);
        int64_t maxRows = (rowBytes > 0) ? (DMA_BURST_MAX / rowBytes) : rows;
        if (maxRows < 1) maxRows = 1;
        for (int64_t r0 = 0; r0 < rows; r0 += maxRows) {
            const int64_t chunk = (r0 + maxRows <= rows) ? maxRows : (rows - r0);
            if (stride_ == n_) {
                DataCopyPad(dst[(dstRowBase + r0) * n_], src[r0 * stride_],
                            DataCopyExtParams{1, (uint32_t)(chunk * rowBytes), 0, 0, 0});
            } else {
                DataCopyExtParams ext{1, (uint32_t)rowBytes, 0, 0, 0};
                for (int64_t r = 0; r < chunk; ++r) {
                    DataCopyPad(dst[(dstRowBase + r0 + r) * n_], src[(r0 + r) * stride_], ext);
                }
            }
        }
    }

    // ---------- 整矩阵驻留 UB ----------
    __aicore__ inline float DetResident(int64_t m) {
        const int64_t n = n_;
        const int64_t stride = stride_;
        LoadRows(mat_, aGm_, m * n, n);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        float mant = 1.0f;
        int32_t exp2 = 0;
        bool negSign = false;

        for (int64_t k = 0; k < n; ++k) {
            int64_t pivRow = k;
            float best = AbsScalar(mat_.GetValue((int32_t)(k * stride + k)));
            for (int64_t i = k + 1; i < n; ++i) {
                const float v = AbsScalar(mat_.GetValue((int32_t)(i * stride + k)));
                if (v > best) {
                    best = v;
                    pivRow = i;
                }
            }
            if (pivRow != k) {
                negSign = !negSign;
                for (int64_t j = 0; j < n; ++j) {
                    const float t = mat_.GetValue((int32_t)(k * stride + j));
                    mat_.SetValue((int32_t)(k * stride + j), mat_.GetValue((int32_t)(pivRow * stride + j)));
                    mat_.SetValue((int32_t)(pivRow * stride + j), t);
                }
            }
            const float pivot = mat_.GetValue((int32_t)(k * stride + k));
            if (pivot == 0.0f) {
                return 0.0f;
            }
            MulDet(mant, exp2, pivot);
            const float invP = 1.0f / pivot;

            int64_t c0 = (k + 1) / 8 * 8;
            const int64_t len = n - c0;
            if (len > 0) {
                PipeBarrier<PIPE_ALL>();   // 标量写 -> 向量读
                LocalTensor<float> pivotRow = mat_[k * stride];
                LocalTensor<float> srcCol = pivotRow[c0];
                for (int64_t i = k + 1; i < n; ++i) {
                    const float f = mat_.GetValue((int32_t)(i * stride + k)) * invP;
                    LocalTensor<float> dstRow = mat_[i * stride];
                    Axpy(dstRow[c0], srcCol, -f, (int32_t)len);
                    PipeBarrier<PIPE_V>();
                }
                PipeBarrier<PIPE_ALL>();   // 向量写 -> 标量读
            }
        }
        return ApplyDet(mant, exp2, negSign);
    }

    // ---------- 矩阵驻留 workspace，行 tile 流式 ----------
    __aicore__ inline float DetStreaming(int64_t m, __gm__ float* wsCore) {
        const int64_t n = n_;
        const int64_t stride = stride_;
        const int64_t tileRows = tileRows_;
        GlobalTensor<float> ws;
        ws.SetGlobalBuffer(wsCore, n * stride);

        {
            const DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            DataCopyExtParams ext{1, (uint32_t)(n * (int64_t)sizeof(float)), 0, 0, 0};
            for (int64_t r = 0; r < n; ++r) {
                DataCopyPad(mat_, aGm_[(m * n + r) * n], ext, pad);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                DataCopyPad(ws[r * stride], mat_, ext);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }

        float mant = 1.0f;
        int32_t exp2 = 0;
        bool negSign = false;

        float nextBest = -1.0f;
        int64_t nextR0 = -1;
        int64_t nextRows = 0;
        for (int64_t k = 0; k < n; ++k) {
            // 主元搜索：tile 扫描第 k 列
            int64_t pivRow = k;
            float best = -1.0f;
            int64_t bestR0 = -1;
            int64_t bestRows = 0;
            if (k > 0 && nextR0 >= 0) {
                best = nextBest;
                bestR0 = nextR0;
                bestRows = nextRows;
            }
            for (int64_t r0 = (bestR0 >= 0 ? n : k); r0 < n; r0 += tileRows) {
                const int64_t rows = (r0 + tileRows <= n) ? tileRows : (n - r0);
                LoadRows(mat_, ws, r0, rows);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                // 本 tile 第 k 列 → 512 长对齐缓冲（先清零，尾部自然为 0，不影响取最大）
                Duplicate<float>(colPad_, 0.0f, 1024);
                ArithProgression(idxCol_, (int32_t)(k * (int64_t)sizeof(float)),
                                 (int32_t)(stride * (int64_t)sizeof(float)), (int32_t)rows);
                PipeBarrier<PIPE_V>();
                LocalTensor<uint32_t> offU = idxCol_.ReinterpretCast<uint32_t>();
                Gather(colPad_, mat_, offU, 0, (uint32_t)rows);
                PipeBarrier<PIPE_V>();
                Muls(colNeg_, colPad_, -1.0f, 1024);
                PipeBarrier<PIPE_V>();
                Max(colPad_, colPad_, colNeg_, 1024);          // |v|
                PipeBarrier<PIPE_V>();
                // 固定 8 对齐的二分归约：512→256→…→8
                for (int64_t hh = 512; hh >= 8; hh >>= 1) {
                    Max(colPad_, colPad_, colPad_[hh], (int32_t)hh);
                    PipeBarrier<PIPE_V>();
                }
                PipeBarrier<PIPE_ALL>();
                float mxTile = 0.0f;
                for (int64_t tt = 0; tt < 8; ++tt) {
                    const float v = colPad_.GetValue((int32_t)tt);
                    if (v > mxTile) mxTile = v;
                }
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                if (mxTile > best) {
                    best = mxTile;
                    bestR0 = r0;
                    bestRows = rows;
                }
            }
            if (bestR0 >= 0) {
                LoadRows(mat_, ws, bestR0, bestRows);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                for (int64_t i = 0; i < bestRows; ++i) {
                    const float v = AbsScalar(mat_.GetValue((int32_t)(i * stride + k)));
                    if (v >= best) {
                        pivRow = bestR0 + i;
                        break;
                    }
                }
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            }
            const DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            DataCopyExtParams ext{1, (uint32_t)(n * (int64_t)sizeof(float)), 0, 0, 0};
            nextBest = -1.0f;
            nextR0 = -1;
            nextRows = 0;
            if (pivRow != k) {
                negSign = !negSign;
                DataCopyPad(tmp_, ws[k * stride], ext, pad);
                DataCopyPad(rowK_, ws[pivRow * stride], ext, pad);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                DataCopyPad(ws[k * stride], rowK_, ext);
                DataCopyPad(ws[pivRow * stride], tmp_, ext);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            DataCopyPad(rowK_, ws[k * stride], ext, pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            const float pivot = rowK_.GetValue((int32_t)k);
            if (pivot == 0.0f) {
                return 0.0f;
            }
            MulDet(mant, exp2, pivot);
            const float invP = 1.0f / pivot;

            int64_t c0 = (k + 1) / 8 * 8;
            const int64_t len = n - c0;
            if (len > 0) {
                PipeBarrier<PIPE_ALL>();
                LocalTensor<float> srcCol = rowK_[c0];
                for (int64_t r0 = k + 1; r0 < n; r0 += tileRows) {
                    const int64_t rows = (r0 + tileRows <= n) ? tileRows : (n - r0);
                    LoadRows(mat_, ws, r0, rows);
                    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                    for (int64_t i = 0; i < rows; ++i) {
                        const float f = mat_.GetValue((int32_t)(i * stride + k)) * invP;
                        Axpy(mat_[i * stride + c0], srcCol, -f, (int32_t)len);
                        PipeBarrier<PIPE_V>();
                    }
                    if (k + 1 < n) {
                        ArithProgression(idxCol_, (int32_t)((k + 1) * (int64_t)sizeof(float)),
                                         (int32_t)(stride * (int64_t)sizeof(float)), (int32_t)rows);
                        PipeBarrier<PIPE_V>();
                        LocalTensor<uint32_t> offN = idxCol_.ReinterpretCast<uint32_t>();
                        Duplicate<float>(colPad_, 0.0f, 1024);
                        Gather(colPad_, mat_, offN, 0, (uint32_t)rows);
                        PipeBarrier<PIPE_V>();
                        Muls(colNeg_, colPad_, -1.0f, 1024);
                        PipeBarrier<PIPE_V>();
                        Max(colPad_, colPad_, colNeg_, 1024);
                        PipeBarrier<PIPE_V>();
                        for (int64_t hh = 512; hh >= 8; hh >>= 1) {
                            Max(colPad_, colPad_, colPad_[hh], (int32_t)hh);
                            PipeBarrier<PIPE_V>();
                        }
                        PipeBarrier<PIPE_ALL>();
                        float mxT = 0.0f;
                        for (int64_t tt = 0; tt < 8; ++tt) {
                            const float v = colPad_.GetValue((int32_t)tt);
                            if (v > mxT) mxT = v;
                        }
                        if (mxT > nextBest) { nextBest = mxT; nextR0 = r0; nextRows = rows; }
                    }
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    StoreRows(mat_, ws, r0, rows);
                    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
            }
        }
        return ApplyDet(mant, exp2, negSign);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> matBuf_, rowBuf_, tmpBuf_, outBuf_, colPadBuf_, idxColBuf_;
    LocalTensor<float> mat_, rowK_, tmp_, outLocal_;
    LocalTensor<float> colPad_, colNeg_;
    LocalTensor<int32_t> idxCol_;
    GlobalTensor<Tf> aGm_, outGm_;
    GM_ADDR wsAddr_ = nullptr;
    int64_t numMatrix_ = 0;
    int64_t n_ = 0;
    int64_t stride_ = 0;
    int64_t tileRows_ = 1;
};

template <typename DT_A>
 __global__ __aicore__ void linalg_det(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LinalgDetTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, tiling_data, tiling);
    KernelLinalgDet<DT_A> op;
    op.Init(A, det, workspace, tiling_data);
    op.Process();
}
