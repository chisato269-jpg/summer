"""rlkit.localdata — OneDrive の同期の外に置くデータの場所（V7.1-S 以降）。

容量の大きい次の 2 種類を、OneDrive の同期フォルダの外（ローカルだけ）に置く。
  エミュレータの組の複製（Shizuku2_slot*）… 原本から作り直せる。走行のたびに CSV などを書き換えるため、同期との競合の原因にもなる
  診断の分単位の出力（runs/diag の slot_output・K10/source・R8_baseline_output。エミュレータの CSV の保全）
置き場所は環境変数 RLKIT_LOCAL_ROOT（無ければ ~/rl_local）。原本 Shizuku2/、報告書、指示書、コードは今までどおりプロジェクトの中。

V8.3-S の後に追加（通信量の対策）: 新しい走行の記録（updates・siblings・episodes の jsonl、行動列、教師、起動の記録など）は、走行中に
書き足され続け、OneDrive が変更のたびに送り直すため、OneDrive の外の RUNS_ROOT（~/rl_local/runs/<走行名>）に置く。
  新しい走行は、設定の run_dir に絶対パス（~/rl_local/runs/<走行名>）を書く（config.repo_path は絶対パスをそのまま返す）。
  新しい診断 script の出力（教師・結果・ログ）は work_dir(版, ...) の下に置く。報告書（.md）は今までどおり runs/diag/<版>/。
  以前の走行（runs/ の下）は触らない。状況表示は両方を探す（latest_run）。
"""
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_ROOT = os.environ.get("RLKIT_LOCAL_ROOT") or os.path.join(os.path.expanduser("~"), "rl_local")
EMULATOR_ROOT = os.path.join(LOCAL_ROOT, "emulators")
DIAG_CSV_ROOT = os.path.join(LOCAL_ROOT, "diag_csv")
RUNS_ROOT = os.path.join(LOCAL_ROOT, "runs")                 # 新しい走行の記録（OneDrive の外）
WORK_ROOT = os.path.join(LOCAL_ROOT, "work")                 # 新しい診断 script の出力（OneDrive の外）
OLD_RUNS_ROOT = os.path.join(_REPO_ROOT, "runs")             # 以前の走行（OneDrive の中）
DIAG_DIR = os.path.join(_REPO_ROOT, "runs", "diag")


def diag_csv_dir(path):
    """runs/diag の下のパス（相対でも絶対でも）を、同じ相対位置の DIAG_CSV_ROOT の下のパスに写す。
    runs/diag の外のパスは、そのまま返す（試験の一時フォルダなど）。"""
    full = os.path.abspath(path)
    try:
        rel = os.path.relpath(full, DIAG_DIR)
    except ValueError:                       # 別のドライブ
        return path
    if rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        return path
    return os.path.join(DIAG_CSV_ROOT, rel)


def run_dir_for(name):
    """新しい走行の記録の場所（OneDrive の外）。設定の run_dir に書く値（絶対パス）。"""
    return os.path.join(RUNS_ROOT, name)


def work_dir(version, *parts):
    """新しい診断 script の出力の置き場所（OneDrive の外）。version は "V84S" など。作成して返す。"""
    path = os.path.join(WORK_ROOT, version, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def latest_run(default_planned=600):
    """updates.jsonl が最も新しい走行を、OneDrive の外（RUNS_ROOT）と以前の走行（runs/）の両方から探す。
    Returns: (最終更新の時刻, 走行ディレクトリ, 計画更新数, 走行名) または None。計画更新数は最初の meta_start_*.json の設定から読み、無ければ
    以前の走行の名前による既定（v2s_trial08 は 1500）か default_planned。"""
    import glob
    import json
    best = None
    for root in (RUNS_ROOT, OLD_RUNS_ROOT):
        for upd in glob.glob(os.path.join(root, "*", "updates.jsonl")):
            d = os.path.dirname(upd)
            name = os.path.basename(d)
            m = os.path.getmtime(upd)
            if best is not None and m <= best[0]:
                continue
            planned = 1500 if name.startswith("v2s_") else default_planned
            metas = sorted(glob.glob(os.path.join(d, "meta_start_*.json")))
            if metas:
                try:
                    with open(metas[0], encoding="utf-8") as f:
                        planned = int(json.load(f)["config"]["planned_updates"])
                except Exception:
                    pass
            best = (m, d, planned, name)
    return best
