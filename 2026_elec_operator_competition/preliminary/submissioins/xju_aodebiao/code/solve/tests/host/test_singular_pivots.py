# Copyright (c) 2026 aodebiao.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compile the real LU methods against host test doubles; no NPU is needed."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("SOLVE_TEST_SOURCE", ROOT / "op_kernel/linalg_solve.cpp"))
HARNESS = r"""
#include <cmath>
#include <cstdio>
#include <string>
#include "ascendc_stub.h"
#include "linalg_solve_tiling.h"
#define private public
#include "kernel_body.inc"
#undef private

static bool Near(float actual, float expected) {
    return std::isfinite(actual) && std::fabs(actual - expected) <= 1e-4F;
}

int main(int argc, char **argv) {
    if (argc != 2) return 9;
    const std::string mode = argv[1];
    LinalgSolveTilingData t{2, 1, 1, 8, 8, 1, 0};
    std::vector<float> a{3, 1, 1, 2}, b{9, 8}, expected{2, 3};
    const bool singular = mode.find("singular") == 0;
    if (mode == "pivot") { a = {0, 1, 2, 3}; b = {2, 8}; expected = {1, 2}; }
    if (mode == "multiple") {
        t.k = 2; a = {2, 0, 0, 4}; b = {2, 4, 12, 16}; expected = {1, 2, 3, 4};
    }
    if (mode == "tiny") { a = {1e-10F, 0, 0, 2}; b = {1e-10F, 6}; expected = {1, 3}; }
    if (mode == "right") {
        t.rightSide = 1; a = {2, 1, 0, 3}; b = {2, 7}; expected = {1, 2};
    }
    if (mode == "right_multiple") {
        t.rightSide = 1; t.k = 2; a = {2, 1, 0, 3}; b = {2, 7, 6, 15}; expected = {1, 2, 3, 4};
    }
    if (mode == "broadcast_rhs") {
        t.batch = 2; t.batchRank = 1; t.batchDims[0] = 2; t.matrixBatchStrides[0] = 4;
        a = {2, 0, 0, 3, 1, 0, 0, 2}; b = {2, 6}; expected = {1, 2, 2, 3};
    }
    if (mode == "broadcast_matrix") {
        t.batch = 2; t.batchRank = 1; t.batchDims[0] = 2; t.rhsBatchStrides[0] = 2;
        a = {2, 0, 0, 4}; b = {2, 8, 6, 16}; expected = {1, 2, 3, 4};
    }
    if (mode == "empty_rhs") { t.k = 0; t.q = 0; b.clear(); expected.clear(); }
    if (mode == "singular_first") a = {0, 1, 0, 2};
    if (mode == "singular_later") { a = {1, 2, 2, 4}; b = {3, 6}; }
    std::vector<float> output(expected.size(), -999.0F), workspace(256, 0.0F);
    LuKernel kernel;
    kernel.Init(reinterpret_cast<GM_ADDR>(a.data()), reinterpret_cast<GM_ADDR>(b.data()),
                reinterpret_cast<GM_ADDR>(output.data()), reinterpret_cast<GM_ADDR>(workspace.data()), t);
    bool trapped = false;
    try {
        if (mode == "singular_back_positive" || mode == "singular_back_negative") {
            kernel.ao = 0; kernel.bo = t.n * t.p;
            workspace[0] = 1.0F;
            workspace[t.p + 1] = mode == "singular_back_negative" ? -0.0F : 0.0F;
            workspace[kernel.bo] = 1.0F; workspace[kernel.bo + t.q] = 2.0F;
            kernel.Back();
        } else {
            kernel.Process();
        }
    } catch (const AscendC::SingularSystem&) { trapped = true; }
    if (singular) {
        if (!trapped) return 2;
        for (float value : output) if (value != -999.0F) return 3;
    } else {
        if (trapped) return 4;
        for (size_t i = 0; i < output.size(); ++i) if (!Near(output[i], expected[i])) return 5;
    }
    std::puts("PASS");
    return 0;
}
"""


class SingularPivotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = os.environ.get("SOLVE_TEST_CXX", "")
        cls.command = (
            [part.strip('"') for part in shlex.split(compiler, posix=os.name != "nt")]
            if compiler
            else [shutil.which("c++") or shutil.which("g++") or ""]
        )
        if not cls.command[0]:
            raise unittest.SkipTest("set SOLVE_TEST_CXX to a host C++ compiler command")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        work = Path(cls.temporary.name)
        source = SOURCE.read_text(encoding="utf-8")
        # Only replace device entry/SDK includes; compile the production LU class verbatim.
        source = source.split("template<typename DT_A>", 1)[0]
        source = "\n".join(
            line for line in source.splitlines() if not line.startswith("#include")
        )
        (work / "kernel_body.inc").write_text(source, encoding="utf-8")
        (work / "harness.cpp").write_text(HARNESS, encoding="utf-8")
        cls.binary = work / ("solve-test.exe" if os.name == "nt" else "solve-test")
        args = cls.command + [
            "-std=c++17",
            "-O0",
            "-fno-fast-math",
            "-I",
            str(Path(__file__).parent),
            "-I",
            str(ROOT / "op_kernel"),
            str(work / "harness.cpp"),
            "-o",
            str(cls.binary),
        ]
        result = subprocess.run(args, capture_output=True, text=True, timeout=240)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def run_case(self, mode):
        result = subprocess.run(
            [str(self.binary), mode], capture_output=True, text=True, timeout=15
        )
        self.assertEqual(
            result.returncode, 0, f"{mode}: {result.stdout}{result.stderr}"
        )
        self.assertIn("PASS", result.stdout)

    def test_regular_solve(self):
        self.run_case("regular")

    def test_row_pivoting(self):
        self.run_case("pivot")

    def test_multiple_right_hand_sides(self):
        self.run_case("multiple")

    def test_nonzero_small_pivot_is_not_clamped(self):
        self.run_case("tiny")

    def test_right_side_solve(self):
        self.run_case("right")

    def test_right_side_multiple_rows(self):
        self.run_case("right_multiple")

    def test_broadcast_rhs(self):
        self.run_case("broadcast_rhs")

    def test_broadcast_matrix(self):
        self.run_case("broadcast_matrix")

    def test_empty_rhs_has_no_output_writes(self):
        self.run_case("empty_rhs")

    def test_zero_first_pivot_stops_before_division(self):
        self.run_case("singular_first")

    def test_zero_later_pivot_stops_before_division(self):
        self.run_case("singular_later")

    def test_back_substitution_rejects_positive_zero(self):
        self.run_case("singular_back_positive")

    def test_back_substitution_rejects_negative_zero(self):
        self.run_case("singular_back_negative")


if __name__ == "__main__":
    unittest.main()
