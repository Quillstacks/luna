"""Path resolution and remote-scrape guard tests (no network access)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from luna import paths
from luna.io.pds_index import PDSIndex

VOLUME = {
    "volume_id": "LROLRC_1001",
    "index_url": "https://example.invalid/LROLRC_1001/INDEX/INDEX.TAB",
    "record_bytes": 100,
    "row_count": 10,
    "first_row_index": 0,
    "first_pid": "M100000000RC",
    "last_pid": "M199999999RC",
}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for var in ("LUNA_DATA_DIR", "LUNA_CATALOG_DIR", "LUNA_PDS_CACHE_DIR"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_pdsindex_finds_cwd_data_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: Path
):
    proj = tmp_path / "proj"
    (proj / "data").mkdir(parents=True)
    (proj / "catalogs").mkdir()
    (proj / "data" / "pds_volume_index_cdr.json").write_text(json.dumps([VOLUME]))
    (proj / "data" / "pds_geometry_cache.json").write_text(
        json.dumps({"M100000001RC": {"file_spec_name": "LROLRC_1001/DATA/M100000001RC.IMG"}})
    )
    monkeypatch.chdir(proj)

    with patch("requests.Session.request", side_effect=AssertionError("network used")):
        idx = PDSIndex()
        vols = idx.volumes()
        url = idx.url_for("M100000001R")

    assert idx.cache_path == (proj / "data" / "pds_volume_index_cdr.json")
    assert idx.geom_cache_path == proj / "data" / "pds_geometry_cache.json"
    assert [v.volume_id for v in vols] == ["LROLRC_1001"]
    assert url.endswith("M100000001RC.IMG")
    for p in (idx.cache_path, idx.geom_cache_path):
        assert "site-packages" not in p.parts
        assert not str(p).startswith(str(paths.BUNDLED_DIR))


def test_env_var_beats_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: Path):
    env_dir = tmp_path / "env"
    cwd_dir = tmp_path / "cwd" / "data"
    env_dir.mkdir()
    cwd_dir.mkdir(parents=True)
    (env_dir / "x.json").write_text("{}")
    (cwd_dir / "x.json").write_text("{}")
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setenv("LUNA_DATA_DIR", str(env_dir))
    assert paths.find_file("x.json", "data") == env_dir / "x.json"


def test_bundled_catalogs_resolve_outside_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: Path
):
    monkeypatch.chdir(tmp_path)
    for name in ("lpa.csv", "pit_nacs.json"):
        found = paths.find_file(name, "catalogs")
        assert found is not None and found.parent == paths.BUNDLED_DIR / "catalogs"


def test_missing_volume_index_raises_without_scraping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: Path
):
    monkeypatch.chdir(tmp_path)
    idx = PDSIndex()
    with patch.object(PDSIndex, "_characterize_volume") as char, patch.object(
        PDSIndex, "_list_volumes"
    ) as lister:
        with pytest.raises(RuntimeError, match="Searched"):
            idx.volumes()
    char.assert_not_called()
    lister.assert_not_called()


def test_writable_path_never_in_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: Path
):
    monkeypatch.chdir(tmp_path)
    w = paths.writable_path("pds_geometry_cache.json", "data")
    assert w == clean_env / ".cache" / "luna" / "pds_geometry_cache.json"
