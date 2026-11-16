/*
 * Copyright (c) 2026 ceprei_Tnx.
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

// v126 branch manifest; host/kernel use literal keys, not this header.
// 1: AIV-only small-output implementation (n<=256, outputs<=512).
//    v122: FP16 boundary repair pass removed from the rounding hand-off;
//    hardware Cast (CAST_NONE) already rounds +/-overflow to inf and keeps
//    NaN, matching numpy semantics (verified on 910B2/CANN 9.0).
//    v124: no more whole-struct tiling copy on this path; the eight needed
//    scalars are read directly from the tiling buffer (the copy embedded
//    the full TCubeTiling and dominated the scalar pipe at these sizes).
//    v126: batch bias (<=16 columns, rows aligned) arrives via one tiny DMA
//    per bias plus a vector cast instead of the per-element scalar UB copy;
//    the V->S handshake is skipped on that path.
// 2: MIX_AIC_1_1. AIC owns MatmulImpl; AIV owns FP32 bias/SiLU/gating.
//    Two ready/release notification slots. cvTaskM/N are the task grid;
//    Each route uses synchronous IterateAll instead of an extra PIPE_ALL.
//    cubeTilingData retains the SDK resource and accumulation configuration.
//    113a: paired groups with one rectangle run one fused FinishTile.
//    113b: bias cache keyed by (nStart, nUse).
//    113c: bias UB sized by singleCoreN instead of full N.
// 3: AIV-only d=1 implementation; 113d: runtime chunk (<=256) driven by
//    vecChunkElements, so small totals spread over many AIV cores.
//    v121: weight/bias scalars read directly from GM (no DMA round trips).
//    v122: FP16 boundary repair pass removed (see key 1).
//    v124: tiling read trimmed (see key 1); the four scalar GM reads are
//    issued while the first x DMA is in flight, hiding their latency; the
//    dead param queue / V-S events were dropped from the prologue.
// 4: MIX_AIC_1_1 basic-API cube engine (v120). AIC drives Nd2Nz / LoadData /
//    Mmad / Fixpipe by hand. L1 K-slabs (128 wide -> 256B MTE2 rows) are
//    double buffered and consumed in 64-wide L0 sub-blocks with a 2-stage
//    pipeline; A/W L1 slots plus L0A/L0B slots are ping-ponged. Every pipe
//    pair owns a distinct EVENT_ID range (MTE2_MTE1 0/1, MTE1_MTE2 2/3,
//    MTE1_M 4/5, M_MTE1 6/7) -- sharing IDs across pairs mis-pairs hardware
//    events in this pipeline depth and triggers aicore MPU faults.
//    Bit-identical rounding: FP32 L0C accumulation over K ascending, one
//    F322F16 Fixpipe rounding per output.
//    v127 (host): for MAC >= 200M shapes the host refines cvTaskM/N (split N
//    then M into 16-aligned halves, tails >= 16) until every core holds at
//    least two task rounds -- with one task per core the AIC and its paired
//    AIV ran strictly one after the other (measured -8~12% once split).
//    v128: L1 slab width 256 (512B Nd2Nz rows) for k>=512, falling back to
//    128 for k<512 so short-K shapes keep two slabs (measured -10% 1024^3,
//    -8% 512x768, no regressions on k<512).
//    v129: L1-resident bulk path for tasks with m,n <= 128 and k <= 256 --
//    the whole A/W1/W2 panels stay in L1 for the task (3x128KB, same budget),
//    W is fetched once per task instead of once per m-block group (-9%
//    4096x256). The host routes (m>=4096 && k==256) shapes back to key 4 and
//    clamps their task extent to 128 for bulk eligibility. Extents > 128
//    (Nd2Nz layout not proven there) fall back to the slab path.
//    v130: the bulk panels are double buffered (2x3x64KB = the same 384KB)
//    and the next task's panels are prefetched while the current task is
//    consumed, so the once-serial prologue disappears. With this the bulk
//    engine beats the library on the whole huge-M k<=256 class (16384x256
//    56.6 vs 61.1, 8192x256 36.5 vs 42.1, 32768x128 48.4 vs 52.2).
//    v131: half-K units extend the L1-resident engine to k<=512 (panels stay
//    128x256x2B = 64KB; k>256 tasks prefetch their next K-half within the
//    task, cross-task prefetch stays enabled for single-half shapes). The
//    k=512 huge-M class measures ~-30% vs MatmulImpl (4096x512: 46 vs 67,
//    8192x512: 75 vs 105).
//    v132: the refined slab engine also wins on the k=1024 wide-K class
//    (4096x1024: 170 vs 250 local, 2048x1024: 88 vs 96), so only k>1024
//    coarse-M shapes (and k>=2048) stay excluded.
//    v134: two fixes -- (1) bigM_k now leaves a k=1024 hole so 4096x1024
//    actually reaches key 4 (the v132 catch-all still excluded it);
//    (2) half-K bulk is limited to k<=256 or (k<=512 and m>=4096), because
//    mid-M k=512 shapes (512^3, 1024x512) run ~12-23% faster on the slab.
//    Selected automatically except (m>=4096 && k>512 && k!=1024),
//    (m>=2048 && k>1024) and k>=2048, where MatmulImpl MDL still wins.
// 6/8: old opt-in local experiments, disabled in the default submission.
// 12-15: generic <=32 output entry for 1-4 work blocks; keys carry the worker
// count, so idle blocks read no metadata and workers need one 32-bit load.

// 9: key9 Mmad/Fixpipe width aligned to 16; padded columns are workspace slack.

// 9: key9-only 16-aligned workspace pitch; other keys keep 64.

// 9: whole-plane contiguous read; padded columns are computed and discarded.

// 16/17/18/19: non-flat K>16 W-slice variants of 7/8/13/15; same ABI.

// 66/67/68: grouped single-projection weight reuse (1:1 / 1:2 / dual planes).
