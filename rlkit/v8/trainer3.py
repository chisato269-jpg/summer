"""rlkit.v8.trainer3 — V8.3-S の学習の入口（指示書 V8.3-S 第 3〜7 章）。V82Trainer を継承し、L-6R の損失・数値、教師の保存、G-20 の基準だけを替える。

  C83-1 L-6R の損失・学習率・最大エポック: 設定の l6r_loss（k1・k3・k4）・l6r_lr・l6r_max_epochs（較正で選んだもの）。ほかの L-6R の決まりは同じ
  C83-2 有効な判断の定義: rlkit/v8/adopt.py の effective_decision_mask（V8.2-S から変更なし。承認された）
  C83-3 教師の保存: 試した兄弟の（観測・マスク・行動）を runs/<run>/teachers/u{更新}_s{組}.npz に圧縮して保存する。
        更新 1〜50 は試したもの全て、更新 51 以降は採用に失敗したものだけ（既存のファイルは上書きしない）
  C83-4 G-20 の実時間の基準: 450 秒
"""
import os

import numpy as np

from rlkit import config as config_mod
from rlkit import runlog
from rlkit.v8 import adopt
from rlkit.v8 import trainer2 as tr2

INSTRUCTION_VERSION = "V8.3-S"
TEACHER_ALL_UNTIL = 50                 # この更新まで、試した教師を全て保存する（以降は失敗したものだけ）
G20_SECONDS_MAX = 450.0


def should_save_teacher(update, adopted):
    """C83-3: 更新 1〜50 は全て、更新 51 以降は採用に失敗したものだけ保存する。"""
    return bool(update <= TEACHER_ALL_UNTIL or not adopted)


def save_teacher(path, teacher):
    """教師（観測・マスク・行動）を圧縮して保存する。観測は float32、マスクは bool、行動は int8。"""
    np.savez_compressed(path, obs=np.asarray(teacher["obs"], dtype=np.float32), masks=np.asarray(teacher["masks"], dtype=bool),
                        actions=np.asarray(teacher["actions"]).astype(np.int8))


def load_teacher(path):
    """save_teacher で保存した教師を、make_teacher と同じ形（eff を含む）で読み込む。"""
    z = np.load(path)
    return adopt.make_teacher(z["obs"], z["masks"], z["actions"].astype(np.int64))


class V83Trainer(tr2.V82Trainer):
    VERSION = INSTRUCTION_VERSION
    G20_SECONDS_MAX = G20_SECONDS_MAX

    def l6r_settings(self):
        loss = self.cfg.get("l6r_loss", "k1")
        if loss not in adopt.LOSS_KINDS:
            raise SystemExit(f"V8.3-S: l6r_loss が未知（{loss}）。{adopt.LOSS_KINDS} のいずれか")
        return {"loss": loss, "lr": float(self.cfg.get("l6r_lr", adopt.LEARNING_RATE)),
                "max_epochs": int(self.cfg.get("l6r_max_epochs", adopt.MAX_EPOCHS))}

    def init_from_weights(self):
        super().init_from_weights()
        self.init_meta["instruction_version"] = INSTRUCTION_VERSION
        self.init_meta["l6r"] = self.l6r_settings()                      # NOT-11: 走行の途中で変えない（設定から読むのは起動時の記録のため）

    def _teacher_path(self, update, sibling):
        return self._path(os.path.join("teachers", f"u{update}_s{int(self.slots[sibling])}.npz"))

    def try_adopt(self, sibs, adv):
        st = self.init_meta.get("l6r") or self.l6r_settings()            # 再開時は保存点に記録された値（起動時の設定）を使う
        u = self.update_index + 1

        def on_tried(s, teacher, adopted):
            if not should_save_teacher(u, adopted):
                return
            path = self._teacher_path(u, s)
            if os.path.exists(path):                                     # 既存のファイルは上書きしない
                self.log_event(event="teacher_exists_not_overwritten", path=path, update=u)
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_teacher(path, teacher)
        return adopt.try_adopt(self.policy, sibs, adv, lr=st["lr"], max_epochs=st["max_epochs"], loss=st["loss"], on_tried=on_tried)


def main(argv=None):
    import argparse
    import torch as th
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(repo_root)
    p = argparse.ArgumentParser(description="夏季の学習（指示書 V8.3-S）")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--updates", type=int, default=None, help="通算の更新数の上限")
    a = p.parse_args(argv)
    cfg = config_mod.load(a.config)
    missing = config_mod.unresolved_items(cfg)
    if missing:
        raise SystemExit("未確定項目が設定されていないため起動しない:\n" + "\n".join(f"  {k}（{i}）: {d}" for k, i, d in missing))
    runlog.check_thread_env()
    th.set_num_threads(1)
    run_dir = config_mod.repo_path(cfg["run_dir"])
    cfg["model_dir"] = config_mod.resolve_model_dir(cfg, run_dir, a.resume)
    print(f"[v8.3.train] モデルの置き場所: {cfg['model_dir']}", flush=True)
    trainer = V83Trainer(cfg, run_dir, max_updates=a.updates)
    print(f"[v8.3.train] 停止: {trainer.run(resume=a.resume)}", flush=True)


if __name__ == "__main__":
    main()
