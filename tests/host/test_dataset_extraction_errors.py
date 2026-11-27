# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest

DOWNLOAD_SCRIPT = Path(__file__).resolve().parents[2] / "examples/datasets/download_dataset.py"
PAYLOAD = b"dataset contents\n"


@pytest.fixture(scope="module")
def downloader():
    spec = importlib.util.spec_from_file_location("gauss_dataset_downloader_test", DOWNLOAD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_archive(path, suffix):
    if suffix == ".zip":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("scene/frame.txt", PAYLOAD)
    else:
        with tarfile.open(path, "w:gz") as archive:
            item = tarfile.TarInfo("scene/frame.txt")
            item.size = len(PAYLOAD)
            archive.addfile(item, io.BytesIO(PAYLOAD))
    return path


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
def test_extraction_failure_propagates_and_preserves_download(downloader, tmp_path, suffix):
    source = tmp_path / ("broken" + suffix)
    source.write_bytes(b"not an archive")
    destination = tmp_path / "download with spaces" / source.name
    output = tmp_path / "extract with spaces"
    with pytest.raises(RuntimeError, match="Error extracting file") as error:
        downloader.download_and_extract(source.as_uri(), destination, output)
    assert isinstance(error.value.__cause__, subprocess.CalledProcessError)
    assert error.value.__cause__.returncode != 0
    assert destination.read_bytes() == source.read_bytes()
    assert not (output / "scene/frame.txt").exists()


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
def test_extraction_failure_returns_nonzero_to_calling_process(tmp_path, suffix):
    source = tmp_path / ("broken" + suffix)
    source.write_bytes(b"not an archive")
    destination = tmp_path / "download" / source.name
    output = tmp_path / "extract"
    command = (
        "import importlib.util, pathlib, sys; "
        "spec = importlib.util.spec_from_file_location('dataset_download_process', sys.argv[1]); "
        "module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; "
        "spec.loader.exec_module(module); "
        "module.download_and_extract(sys.argv[2], pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command, str(DOWNLOAD_SCRIPT), source.as_uri(),
         str(destination), str(output)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "RuntimeError: Error extracting file" in result.stderr
    assert destination.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
def test_valid_archive_extracts_and_removes_download(downloader, tmp_path, suffix):
    source = make_archive(tmp_path / ("valid" + suffix), suffix)
    destination = tmp_path / "download with spaces" / source.name
    output = tmp_path / "extract with spaces"
    assert downloader.download_and_extract(source.as_uri(), destination, output) is None
    assert (output / "scene/frame.txt").read_bytes() == PAYLOAD
    assert not destination.exists()
    assert source.exists()


def test_download_failure_still_propagates(downloader, tmp_path):
    source = tmp_path / "missing.zip"
    with pytest.raises(RuntimeError, match="Error downloading file") as error:
        downloader.download_and_extract(source.as_uri(), tmp_path / "download.zip", tmp_path / "extract")
    assert isinstance(error.value.__cause__, subprocess.CalledProcessError)


def test_multi_archive_dataset_stops_at_first_extraction_failure(downloader, tmp_path, monkeypatch):
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not an archive")
    later = make_archive(tmp_path / "later.zip", ".zip")
    monkeypatch.setitem(downloader.urls, "zipnerf", [broken.as_uri(), later.as_uri()])
    output = tmp_path / "dataset"
    task = downloader.DownloadData(dataset="zipnerf", save_dir=output)
    with pytest.raises(RuntimeError, match="Error extracting file"):
        task.main()
    assert (output / "zipnerf/broken.zip").read_bytes() == broken.read_bytes()
    assert not (output / "zipnerf/later.zip").exists()
    assert not (output / "zipnerf/scene/frame.txt").exists()
