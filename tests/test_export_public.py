"""公開版匯出：要改嘅地方改到，唔應該公開嘅嘢會擋住。"""
import importlib.util
import shutil
from pathlib import Path

import pytest

import config

ROOT = Path(config.PROJECT_DIR)
spec = importlib.util.spec_from_file_location("export_public", ROOT / "deploy" / "export_public.py")
export_public = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_public)


@pytest.fixture
def tree(tmp_path):
    for rel in ("README.md", "README.zh-HK.md", ".github/workflows/rebalance.yml", ".github/workflows/tests.yml"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    return tmp_path


def test_make_public_rewrites_readmes_and_workflow(tree):
    export_public.make_public(tree)
    readme = (tree / "README.md").read_text()
    assert "ai-trader-research/actions/workflows/tests.yml" in readme
    assert "HxCSYxC/ai-trader/actions" not in readme
    assert "This public copy has no keys and runs no scheduled jobs." in readme
    assert "呢個係展示版" in (tree / "README.zh-HK.md").read_text()
    wf = (tree / ".github/workflows/rebalance.yml").read_text()
    assert "workflow_dispatch:" in wf
    assert "\n  schedule:" not in wf
    export_public.check(tree)  # 改完之後應該過到檢查


@pytest.mark.parametrize("rel, text", [
    (".env", "ALPACA_API_KEY=x"),
    ("db/trading.sqlite", ""),
    ("data/cache/sp500.csv", ""),
    ("notes.txt", "key " + "PK" + "ABCDEFGHIJKLMNOPQR"),  # 拆開寫，唔好令呢個檔本身俾檢查擋住
])
def test_check_blocks_private_files_and_keys(tree, rel, text):
    export_public.make_public(tree)
    (tree / rel).parent.mkdir(parents=True, exist_ok=True)
    (tree / rel).write_text(text)
    with pytest.raises(SystemExit):
        export_public.check(tree)


def test_check_blocks_active_schedule(tree):
    with pytest.raises(SystemExit):  # 私人版 workflow 有 schedule，未改就唔可以公開
        export_public.check(tree)
