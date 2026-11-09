/*
 * Copyright (c) 2026 dihidao.
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
// Private device implementation; included by ld_device.h.
#if (defined(__DAV_C220_VEC__))
template <class DT_A, uint32_t LD_BATCH, bool FAST_MC = false, bool PAD128 = false, bool PAD_RAGGED_FAMILY = false,
          bool RAGGED_HANDOFF = false, bool DUAL_LU = false, uint32_t PAIR_SCHEDULE = 0U, bool CUBE_LU = false,
          bool SUPER32 = false, bool BATCH_U01 = false, bool UB_TAIL = false, bool FORCE_FAST = false,
          bool WIDE512 = false>
class KernelLinalgDet : private LdetCubeState<CUBE_LU> {
    static constexpr uint32_t kPanel = WIDE512 ? 32U : ::kPanel;
    static constexpr uint32_t kFactorPitch = WIDE512 ? 512U : 256U;
    static constexpr uint32_t kLpPanel = WIDE512 ? 0U : ::kLpPanel;
    static constexpr uint32_t kLpLbuf = WIDE512 ? 65536U : ::kLpLbuf;
    static constexpr uint32_t kLpUbuf = WIDE512 ? 0U : ::kLpUbuf;
    static constexpr uint32_t kLpTile = WIDE512 ? 131072U : ::kLpTile;
    static constexpr uint32_t kLpTmp = WIDE512 ? 163840U : ::kLpTmp;
    static constexpr uint32_t kLpCp = WIDE512 ? 164864U : ::kLpCp;
    static constexpr uint32_t kLpRes = WIDE512 ? 173056U : ::kLpRes;
    static constexpr uint32_t kLpTab = WIDE512 ? 174080U : ::kLpTab;
    static constexpr uint32_t kLpColoff = WIDE512 ? 176128U : ::kLpColoff;
    static constexpr uint32_t kLpMeta = WIDE512 ? 178176U : ::kLpMeta;
    static constexpr uint32_t kIds = WIDE512 ? 179200U : (kLpPanel + 32768U);
    static constexpr uint32_t kRows = WIDE512 ? 181248U : (kLpPanel + 34816U);
    static constexpr uint32_t kAlive = WIDE512 ? 182272U : (kLpPanel + 4096U);
    static constexpr uint32_t kInvPos = WIDE512 ? 184320U : (kLpPanel + 2048U);
    static constexpr uint32_t kPos = WIDE512 ? 186368U : (kLpPanel);
    static constexpr uint32_t kPoll = WIDE512 ? 188416U : (kLpPanel + 3072U);
    static constexpr uint32_t kBar = WIDE512 ? 189440U : (kLpCp);
    static constexpr uint32_t kScratch = WIDE512 ? 190464U : ((kLpMeta + 48U * sizeof(uint32_t) + 31U) & ~31U);
    static_assert(!WIDE512 || kScratch + 576U <= 196608U, "Wide LU UB exceeds 192 KiB");
    bool firstCopyPending_ = false;

  public:
#include "ld_kernel_linalg_det_m01.h"

#include "ld_kernel_linalg_det_m02.h"

#include "ld_kernel_linalg_det_m03.h"

  private:
#include "ld_kernel_linalg_det_m04.h"

#include "ld_kernel_linalg_det_m05.h"

#include "ld_kernel_linalg_det_m06.h"

#include "ld_kernel_linalg_det_m07.h"

#include "ld_kernel_linalg_det_m08.h"

    struct SmallLUState {
        __ubuf__ float *matrix, *column, *absolute, *row, *reduction, *factors;
        uint32_t k = 0U;
        bool done = false;
        bool pending = false;
        bool singular = false;
        float sign = 1.0f;
        float product = 1.0f;
        int32_t exponent = 0;
        bool fast = true;
    };
#include "ld_kernel_linalg_det_m09.h"

#include "ld_kernel_linalg_det_m10.h"

#include "ld_kernel_linalg_det_m11.h"

#include "ld_kernel_linalg_det_m12.h"

#include "ld_kernel_linalg_det_m13.h"

#include "ld_kernel_linalg_det_m14.h"

#include "ld_kernel_linalg_det_m15.h"

#include "ld_kernel_linalg_det_m16.h"

#include "ld_kernel_linalg_det_m17.h"

#include "ld_kernel_linalg_det_m18.h"

#include "ld_kernel_linalg_det_m19.h"

  private:
    LocalTensor<float> mat_;
    LocalTensor<float> panel_;
    LocalTensor<float> lbuf_;
    LocalTensor<float> panelCM_;
    LocalTensor<float> ubuf_;
    LocalTensor<float> tile_;
    LocalTensor<float> tmp_;
    LocalTensor<float> cpbuf_;
    LocalTensor<float> res_;
    LocalTensor<float> colbuf_;
    LocalTensor<float> absbuf_;
    LocalTensor<float> rowkbuf_;
    LocalTensor<int32_t> tabI_;
    LocalTensor<int32_t> tabRI_;
    LocalTensor<uint32_t> tabRU_;
    LocalTensor<int32_t> coloffI_;
    LocalTensor<uint32_t> coloffU_;
    LocalTensor<uint32_t> tabU_;
    LocalTensor<uint32_t> metaU_;
    LocalTensor<uint32_t> barU_;
    LocalTensor<uint32_t> posU_;
    LocalTensor<uint32_t> dstU_;
    LocalTensor<uint32_t> srcU_;
    LocalTensor<uint32_t> pollU_;
    GlobalTensor<DT_A> aGm_;
    GlobalTensor<float> detGm_;
    GlobalTensor<float> wsGm_;
    GlobalTensor<uint32_t> wsGmU_;
    uint32_t n_ = 0;
    uint32_t s_ = 0;
    uint32_t batch_ = 0;
    uint32_t matPerCore_ = 0;
    uint32_t isLarge_ = 0;
    uint32_t start_ = 0;
    uint32_t cnt_ = 0;
    uint32_t groupSize_ = 1;
    uint32_t rank_ = 0;
    uint32_t matIdx_ = 0;
    uint64_t wsSpan_ = 0;
    uint64_t nn_ = 0;
    uint64_t nnS_ = 0;

    struct SmallRawViews51 {
        __ubuf__ float *column, *absolute, *row, *reduction, *matrix;
        __ubuf__ uint32_t *offsets;
    };
#include "ld_kernel_linalg_det_m20.h"

    struct TinyProductState60 {
        float sign, prod;
        int32_t ex;
        bool fastProd;
        uint32_t singular;
    };
#include "ld_kernel_linalg_det_m21.h"

    struct PanelStepGeometry61 {
        uint32_t tA, j0, cnt, segN, nCh;
    };
#include "ld_kernel_linalg_det_m22.h"

};
#endif
