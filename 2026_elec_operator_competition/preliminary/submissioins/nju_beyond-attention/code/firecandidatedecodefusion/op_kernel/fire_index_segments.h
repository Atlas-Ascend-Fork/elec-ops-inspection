/*
 * Copyright (c) 2026 Beyond Attention. All rights reserved.
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

// Intentionally re-included after each table header undefines these constant-data macros.
// Channels cache 32 candidates; packet priors rotate through a 33-candidate row.
#define FIRE_INDEX_IDENTITY(v) v
#define FIRE_INDEX_HALF(v) ((v) / 2U)
#define FIRE_INDEX_CHANNEL32(BASE, STEP, SHIFT)                                                              \
	((BASE) + (((SHIFT) + 0U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 1U) % 33U) * (STEP)),                  \
		((BASE) + (((SHIFT) + 2U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 3U) % 33U) * (STEP)),              \
		((BASE) + (((SHIFT) + 4U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 5U) % 33U) * (STEP)),              \
		((BASE) + (((SHIFT) + 6U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 7U) % 33U) * (STEP)),              \
		((BASE) + (((SHIFT) + 8U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 9U) % 33U) * (STEP)),              \
		((BASE) + (((SHIFT) + 10U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 11U) % 33U) * (STEP)),            \
		((BASE) + (((SHIFT) + 12U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 13U) % 33U) * (STEP)),            \
		((BASE) + (((SHIFT) + 14U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 15U) % 33U) * (STEP)),            \
		((BASE) + (((SHIFT) + 16U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 17U) % 33U) * (STEP)),            \
		((BASE) + (((SHIFT) + 18U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 19U) % 33U) * (STEP)),            \
		((BASE) + (((SHIFT) + 20U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 21U) % 33U) * (STEP)),            \
		((BASE) + (((SHIFT) + 22U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 23U) % 33U) * (STEP)),            \
		((BASE) + (((SHIFT) + 24U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 25U) % 33U) * (STEP)),            \
		((BASE) + (((SHIFT) + 26U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 27U) % 33U) * (STEP)),            \
		((BASE) + (((SHIFT) + 28U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 29U) % 33U) * (STEP)),            \
		((BASE) + (((SHIFT) + 30U) % 33U) * (STEP)), ((BASE) + (((SHIFT) + 31U) % 33U) * (STEP)),
#define FIRE_INDEX_RAW32                                                                                     \
	FIRE_INDEX_CHANNEL32(0U, 24U, 0U)                                                                        \
	FIRE_INDEX_CHANNEL32(4U, 24U, 0U)                                                                        \
	FIRE_INDEX_CHANNEL32(8U, 24U, 0U)                                                                        \
	FIRE_INDEX_CHANNEL32(12U, 24U, 0U)                                                                       \
	FIRE_INDEX_CHANNEL32(16U, 24U, 0U)                                                                       \
	FIRE_INDEX_CHANNEL32(20U, 24U, 0U)
#define FIRE_INDEX_PRIOR32                                                                                   \
	FIRE_INDEX_CHANNEL32(0U, 20U, 0U)                                                                        \
	FIRE_INDEX_CHANNEL32(4U, 20U, 0U)                                                                        \
	FIRE_INDEX_CHANNEL32(8U, 20U, 0U)                                                                        \
	FIRE_INDEX_CHANNEL32(12U, 20U, 0U)                                                                       \
	FIRE_INDEX_CHANNEL32(16U, 20U, 0U)
#define FIRE_INDEX_OUTPUT32(INDEX)                                                                           \
	INDEX(0U), INDEX(128U), INDEX(256U), INDEX(384U), INDEX(512U), INDEX(4U), INDEX(132U), INDEX(260U),      \
		INDEX(388U), INDEX(516U), INDEX(8U), INDEX(136U), INDEX(264U), INDEX(392U), INDEX(520U), INDEX(12U), \
		INDEX(140U), INDEX(268U), INDEX(396U), INDEX(524U), INDEX(16U), INDEX(144U), INDEX(272U),            \
		INDEX(400U), INDEX(528U), INDEX(20U), INDEX(148U), INDEX(276U), INDEX(404U), INDEX(532U),            \
		INDEX(24U), INDEX(152U), INDEX(280U), INDEX(408U), INDEX(536U), INDEX(28U), INDEX(156U),             \
		INDEX(284U), INDEX(412U), INDEX(540U), INDEX(32U), INDEX(160U), INDEX(288U), INDEX(416U),            \
		INDEX(544U), INDEX(36U), INDEX(164U), INDEX(292U), INDEX(420U), INDEX(548U), INDEX(40U),             \
		INDEX(168U), INDEX(296U), INDEX(424U), INDEX(552U), INDEX(44U), INDEX(172U), INDEX(300U),            \
		INDEX(428U), INDEX(556U), INDEX(48U), INDEX(176U), INDEX(304U), INDEX(432U), INDEX(560U),            \
		INDEX(52U), INDEX(180U), INDEX(308U), INDEX(436U), INDEX(564U), INDEX(56U), INDEX(184U),             \
		INDEX(312U), INDEX(440U), INDEX(568U), INDEX(60U), INDEX(188U), INDEX(316U), INDEX(444U),            \
		INDEX(572U), INDEX(64U), INDEX(192U), INDEX(320U), INDEX(448U), INDEX(576U), INDEX(68U),             \
		INDEX(196U), INDEX(324U), INDEX(452U), INDEX(580U), INDEX(72U), INDEX(200U), INDEX(328U),            \
		INDEX(456U), INDEX(584U), INDEX(76U), INDEX(204U), INDEX(332U), INDEX(460U), INDEX(588U),            \
		INDEX(80U), INDEX(208U), INDEX(336U), INDEX(464U), INDEX(592U), INDEX(84U), INDEX(212U),             \
		INDEX(340U), INDEX(468U), INDEX(596U), INDEX(88U), INDEX(216U), INDEX(344U), INDEX(472U),            \
		INDEX(600U), INDEX(92U), INDEX(220U), INDEX(348U), INDEX(476U), INDEX(604U), INDEX(96U),             \
		INDEX(224U), INDEX(352U), INDEX(480U), INDEX(608U), INDEX(100U), INDEX(228U), INDEX(356U),           \
		INDEX(484U), INDEX(612U), INDEX(104U), INDEX(232U), INDEX(360U), INDEX(488U), INDEX(616U),           \
		INDEX(108U), INDEX(236U), INDEX(364U), INDEX(492U), INDEX(620U), INDEX(112U), INDEX(240U),           \
		INDEX(368U), INDEX(496U), INDEX(624U), INDEX(116U), INDEX(244U), INDEX(372U), INDEX(500U),           \
		INDEX(628U), INDEX(120U), INDEX(248U), INDEX(376U), INDEX(504U), INDEX(632U), INDEX(124U),           \
		INDEX(252U), INDEX(380U), INDEX(508U), INDEX(636U),
