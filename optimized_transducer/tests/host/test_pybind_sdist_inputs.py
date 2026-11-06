# Copyright 2026 Electrical Engineering SIG - CANN Community
#
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

"""Build real source distributions and inspect actual extension build inputs."""

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PYBIND = ROOT / "optimized_transducer/pybind"


@pytest.fixture(params=("dist", "dist with spaces"))
def source_distribution(request, tmp_path):
    source = tmp_path / "source"
    shutil.copytree(
        PYBIND,
        source,
        ignore=shutil.ignore_patterns("*.egg-info", "build", "dist", "__pycache__"),
    )
    output = tmp_path / request.param
    run = subprocess.run(
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(output)],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    archives = list(output.glob("*.tar.gz"))
    assert len(archives) == 1
    return archives[0]


def test_real_source_distribution_contains_complete_build_inputs(source_distribution):
    with tarfile.open(source_distribution) as archive:
        root = archive.getnames()[0].split("/", 1)[0]
        for filename in ("CMakeLists.txt", "pybind_wrapper.cpp"):
            member = root + "/" + filename
            assert member in archive.getnames(), "actual sdist is missing " + filename
            assert (
                archive.extractfile(member).read() == (PYBIND / filename).read_bytes()
            )


def test_existing_metadata_and_setup_preserved(source_distribution):
    with tarfile.open(source_distribution) as archive:
        root = archive.getnames()[0].split("/", 1)[0]
        assert (
            archive.extractfile(root + "/setup.py").read()
            == (PYBIND / "setup.py").read_bytes()
        )
        metadata = archive.extractfile(root + "/PKG-INFO").read()
        assert b"Version: 0.0.1" in metadata
        assert b"torch==2.8.0" in metadata
        assert b"torch_npu==2.8.0" in metadata
