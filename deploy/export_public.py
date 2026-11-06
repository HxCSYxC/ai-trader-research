"""將私人 repo 嘅 HEAD 同步去公開展示版（HxCSYxC/ai-trader-research）。

    python deploy/export_public.py -m "說明今次改咗咩"            # 準備好 commit，唔會 push
    python deploy/export_public.py -m "…" --push                  # 檢查全部過咗先 push

公開版同私人版只差幾樣嘢（全部喺下面寫死）：
  - README 嘅 badge 指向公開 repo；講明公開版冇 key、冇定時運行
  - 廣東話 README 開頭加「展示版」說明
  - rebalance.yml 冇 schedule，只可以手動觸發
推之前會檢查：冇 .env / SQLite / log / cache、冇私人 email、冇似 Alpaca key 嘅字串、workflow 冇開定時。
Commit 作者一定用 GitHub 隱藏 email。
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_REPO = "HxCSYxC/ai-trader-research"
PRIVATE_REPO = "HxCSYxC/ai-trader"
AUTHOR = ("HxCSYxC", "230942076+HxCSYxC@users.noreply.github.com")

ZH_INTRO = """自己用嘅美股 AI 交易研究系統，用 Alpaca **模擬盤**（paper trading）驗證，唔涉及真錢。

> **呢個係展示版。** 每月自動換股喺另一個私人 repo 運行；呢度嘅 GitHub Actions 冇開定時，亦冇任何 key。
> 研究結論：用公開價量數據，喺大型美股搵唔到扣成本後穩定賺錢嘅 edge（日線 ML、S&P 500 AI 選股、即日形態全部測試過），詳情睇下面「進度」同 `intraday/PREREGISTRATION.md`。
>
> 唔係投資建議。

"""
MANUAL_ON = """on:
  # 展示版冇開定時觸發。自己用嘅話：喺 Settings → Secrets 加入 Alpaca Paper key，
  # 再加返 schedule，例如每個交易日開市時間每個鐘一次：
  #   schedule:
  #     - cron: "47 13-20 * * 1-5"
"""
FORBIDDEN_PATHS = re.compile(r"(^|/)(\.env|KILL_SWITCH)$|\.sqlite$|^logs/|^data/cache/|^db/")
ALPACA_KEY = re.compile(r"\b(PK|AK)[A-Z0-9]{16,}\b")


def run(*cmd, cwd=None, capture=True) -> str:
    out = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=capture)
    return out.stdout.strip() if capture else ""


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        sys.exit(f"❌ {path.name}：搵唔到要改嘅字（私人版改過？更新 export_public.py）：{old[:60]}")
    path.write_text(text.replace(old, new))


def replace_between(path: Path, start: str, end: str, new: str) -> None:
    text = path.read_text()
    a, b = text.find(start), text.find(end)
    if a < 0 or b < a:
        sys.exit(f"❌ {path.name}：搵唔到 {start!r} … {end!r}")
    path.write_text(text[:a] + new + text[b:])


def make_public(tree: Path) -> None:
    for name in ("README.md", "README.zh-HK.md"):
        p = tree / name
        p.write_text(p.read_text().replace(f"{PRIVATE_REPO}/actions", f"{PUBLIC_REPO}/actions"))
    replace_once(tree / "README.md", "(`paper=True` is hard-coded).",
                 "(`paper=True` is hard-coded). This public copy has no keys and runs no scheduled jobs.")
    replace_once(tree / "README.md", "Monthly rebalance job (weekdays after the US open; skips unless it is a rebalance day)",
                 "Monthly rebalance job (manual trigger only in this public copy)")
    replace_between(tree / "README.zh-HK.md", "自己用嘅美股 AI 交易系統", "## 開始", ZH_INTRO)
    replace_between(tree / ".github/workflows/rebalance.yml", "on:\n", "  workflow_dispatch:", MANUAL_ON)


def check(tree: Path) -> None:
    problems = []
    files = [p for p in tree.rglob("*") if p.is_file() and ".git" not in p.relative_to(tree).parts]
    # 冇設定（例如 CI）就當冇；唔用 run()，因為 git config 搵唔到會 exit 1
    private_email = subprocess.run(["git", "config", "user.email"], cwd=ROOT, text=True,
                                   capture_output=True).stdout.strip()
    for p in files:
        rel = p.relative_to(tree).as_posix()
        if FORBIDDEN_PATHS.search(rel):
            problems.append(f"唔應該公開嘅檔案：{rel}")
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        if private_email and private_email in text:
            problems.append(f"{rel} 有私人 email")
        if ALPACA_KEY.search(text):
            problems.append(f"{rel} 有似 Alpaca key 嘅字串")
    for wf in (tree / ".github/workflows").glob("*.yml"):
        if re.search(r"^\s*schedule:", wf.read_text(), re.M):
            problems.append(f"{wf.name} 開咗定時運行")
    if problems:
        sys.exit("❌ 檢查唔過，冇 commit：\n  " + "\n  ".join(problems))
    print(f"✅ 檢查通過（{len(files)} 個檔案）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--message", required=True, help="公開版 commit 訊息")
    parser.add_argument("--dir", type=Path, help="公開 repo 嘅 clone（唔填就 clone 去暫存資料夾）")
    parser.add_argument("--push", action="store_true", help="檢查通過之後 push 上 GitHub")
    args = parser.parse_args()

    if run("git", "status", "--porcelain", cwd=ROOT):
        sys.exit("❌ 私人 repo 有未 commit 嘅改動，先 commit 再同步")
    tree = args.dir or Path(tempfile.mkdtemp(prefix="ai-trader-public-")) / "repo"
    if not (tree / ".git").exists():
        run("gh", "repo", "clone", PUBLIC_REPO, str(tree), "--", "-q", capture=False)
    run("git", "pull", "-q", cwd=tree)

    # 成個 tree 換做私人版 HEAD（git archive 只有已 commit 嘅檔案，.gitignore 嘅嘢唔會入）
    tracked = run("git", "ls-files", "-z", cwd=tree).split("\0")
    for f in filter(None, tracked):
        (tree / f).unlink(missing_ok=True)
    archive = subprocess.run(["git", "archive", "HEAD"], cwd=ROOT, check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
    make_public(tree)
    check(tree)

    run("git", "add", "-A", cwd=tree)
    if not run("git", "status", "--porcelain", cwd=tree):
        print("公開版已經係最新，冇嘢要同步")
        return
    head = run("git", "rev-parse", "--short", "HEAD", cwd=ROOT)
    msg = f"{args.message}\n\nSynced from the private repo at {head}."
    run("git", "-c", f"user.name={AUTHOR[0]}", "-c", f"user.email={AUTHOR[1]}", "commit", "-q", "-m", msg, cwd=tree)
    print(run("git", "show", "--stat", "--format=%h %an <%ae>%n%s", "HEAD", cwd=tree))
    if args.push:
        run("git", "push", "-q", "origin", "HEAD", cwd=tree, capture=False)
        print(f"🚀 已經 push 去 https://github.com/{PUBLIC_REPO}")
    else:
        print(f"\n未 push。睇清楚之後：git -C {tree} push origin HEAD")


if __name__ == "__main__":
    main()
