"""rlkit.v9.switch — 段階 A から段階 B への切り替え（指示書 C9-5・C9-6）。

C9-5 切り替えの規則: 段階 A の更新 u の終わりに、B(u) を走行全体の最良の ECI として、
  u ≥ 600 かつ B(u) − B(u − 300) < 0.002 となった最初の u（停滞）、または u = 1,200 のどちらか早いほうで段階 A を終える。
C9-6 段階 B の始点の選び方（自動）:
  候補 1: 切り替えの時点の方策で、基準の軌跡（確率最大の選択）を 1 本走らせる
  候補 2: 段階 A の走行全体の最良の軌跡 H* の行動列を再生して観測とマスクを作り、H* を採取したときの重みに、K-3・学習率 3e-3・最大 3,000
          エポックで採用の学習を行う。基準を満たしたら基準の軌跡を 1 本走らせ、H* と E・DR が 16 進で一致するかを確かめる
  一致した候補のうち ECI の大きいほうを始点にする（同じなら候補 1）。候補 2 が基準を満たさない・一致しない・例外で終わる場合は候補 1。
  候補 1 の基準の軌跡が走らなかった場合は停止する（この関数は例外を投げる）。
"""
import os
import time
import traceback

import numpy as np
import torch as th

from rlkit import constants, eci as eci_mod
from rlkit.parallel.batched import SlotVecEnv
from rlkit.v8 import adopt
from rlkit.v8 import policy as pol8

STAGNATION_MIN_UPDATE = 600
STAGNATION_WINDOW = 300
STAGNATION_RISE = 0.002
SWITCH_CAP_UPDATE = 1200
CAND2_LR = 3e-3
CAND2_MAX_EPOCHS = 3000
CAND2_LOSS = "k3"


def switch_due(b_history, u, min_update=STAGNATION_MIN_UPDATE, window=STAGNATION_WINDOW, rise=STAGNATION_RISE,
               cap=SWITCH_CAP_UPDATE):
    """b_history[k − 1] = B(k)（更新 k の終わりの走行全体の最良の ECI）。更新 u の終わりに切り替えるかを返す。
    Returns: (切り替える, 理由 "stagnation" | "cap" | None, 判定の値)"""
    if len(b_history) < u:
        raise ValueError("b_history が更新 u まで無い")
    detail = {"update": u, "B_u": float(b_history[u - 1])}
    if u >= min_update and u - window >= 1:
        gain = float(b_history[u - 1]) - float(b_history[u - 1 - window])
        detail["B_u_minus_window"] = float(b_history[u - 1 - window])
        detail["gain_window"] = gain
        if gain < rise:
            return True, "stagnation", detail
    if u >= cap:
        return True, "cap", detail
    return False, None, detail


def select_start(cand1, cand2):
    """cand1: {"ok": bool, "ECI": float, ...}（基準の軌跡が走ったか）、cand2: None か {"ok": bool, "ECI": float, ...}（採用でき、基準の軌跡が
    H* と 16 進一致したか）。Returns: {"chosen": 1 | 2, "reason": 文字列}。候補 1 が走っていなければ例外。"""
    if not cand1 or not cand1.get("ok"):
        raise RuntimeError("切り替え: 候補 1 の基準の軌跡が走らなかった（停止規則）")
    if cand2 is None or not cand2.get("ok"):
        return {"chosen": 1, "reason": "候補 2 が基準を満たさない・一致しない・例外で終わったので候補 1"}
    if cand2["ECI"] > cand1["ECI"]:
        return {"chosen": 2, "reason": "候補 2 が H* と 16 進一致し、ECI が候補 1 より大きい"}
    if cand2["ECI"] == cand1["ECI"]:
        return {"chosen": 1, "reason": "ECI が同じなので候補 1"}
    return {"chosen": 1, "reason": "候補 1 の ECI が候補 2 より大きい"}


# ---- 環境を使う部分 ----

def _vec(cfg, factory):
    kw = {"stage": 1, "include_indoor_unit_power": cfg["include_indoor_unit_power"],
          "zone_step_energy_norm_gj": cfg["zone_step_energy_norm_gj"], "m9_enabled": cfg["m9_enabled"],
          "comm_rpm": bool(cfg.get("comm_rpm", False))}
    return SlotVecEnv([cfg["eval_slot"]], kw, constants.ZONE_STEPS_PER_WEEK, factory=factory)


def run_base_episode(cfg, policy, factory="real_v8"):
    """方策の確率最大の選択（マスク付き）で 1 エピソードを走らせる（評価用の組 1 本）。Returns: (終端の info, 行動列 (T, 80))"""
    vec = _vec(cfg, factory)
    acts = []
    try:
        obs = vec.reset()
        for _t in range(constants.ZONE_STEPS_PER_WEEK):
            masks = np.stack(vec.env_method("action_masks"))
            with th.no_grad():
                logits = policy(th.as_tensor(np.asarray(obs), dtype=th.float32)).numpy()
            act = pol8.masked_argmax(logits, masks)
            acts.append(act[0].copy())
            obs, _r, dones, infos = vec.step(act)
            if dones[0]:
                return infos[0], np.stack(acts)
    finally:
        vec.close()
    raise RuntimeError("1 週間で終端しなかった")


def replay_and_record(cfg, actions, factory="real_v8"):
    """行動列を再生し、各ステップの観測・マスクと終端の info を返す。"""
    vec = _vec(cfg, factory)
    obs_l, mask_l = [], []
    info = None
    try:
        obs = vec.reset()
        for t in range(actions.shape[0]):
            mask_l.append(np.stack(vec.env_method("action_masks"))[0].copy())
            obs_l.append(np.asarray(obs)[0].copy())
            obs, _r, dones, infos = vec.step(actions[t][None, :])
            if dones[0]:
                info = infos[0]
                break
    finally:
        vec.close()
    if info is None:
        raise RuntimeError("再生が終端しなかった")
    return np.stack(obs_l), np.stack(mask_l), info


def _hex(info):
    return float.hex(float(info["E_rep"])), float.hex(float(info["DR_rep"]))


def candidate1(cfg, policy, factory="real_v8"):
    """候補 1: 切り替えの時点の方策の基準の軌跡。"""
    try:
        info, actions = run_base_episode(cfg, policy, factory)
    except Exception:
        return {"ok": False, "error": traceback.format_exc()[-2000:]}
    e_hex, d_hex = _hex(info)
    return {"ok": True, "E": float(info["E_rep"]), "DR": float(info["DR_rep"]), "ECI": float(info["ECI"]),
            "E_hex": e_hex, "DR_hex": d_hex, "P_E": float(info["P_E"]), "P_DR": float(info["P_DR"]), "actions": actions}


def candidate2(cfg, harvest, actions_path, weights_path, factory="real_v8"):
    """候補 2: H*（段階 A の走行全体の最良の軌跡）を、H* を採取したときの重みに取り込む。
    Returns: ({"ok", ...記録}, 取り込んだ方策 または None)。例外は記録して ok = False にする（停止しない）。"""
    rec = {"ok": False, "H_update": harvest["update"], "H_role": harvest.get("role"), "H_E_hex": harvest["E_hex"],
           "H_DR_hex": harvest["DR_hex"], "H_ECI": harvest["ECI"], "loss": CAND2_LOSS, "lr": CAND2_LR, "max_epochs": CAND2_MAX_EPOCHS}
    try:
        actions = np.load(actions_path)["actions"].astype(np.int64)
        entry = th.load(weights_path, weights_only=False)
        policy = pol8.PolicyNetworkV8()
        policy.load_state_dict(entry["weights"])
        policy.eval()
        obs, masks, info = replay_and_record(cfg, actions, factory)
        rec["replay_E_hex"], rec["replay_DR_hex"] = _hex(info)
        rec["replay_matches_H"] = (rec["replay_E_hex"], rec["replay_DR_hex"]) == (harvest["E_hex"], harvest["DR_hex"])
        teacher = adopt.make_teacher(obs, masks, actions)
        t0 = time.time()
        fit = adopt.fit_to_teacher(policy, teacher, lr=CAND2_LR, max_epochs=CAND2_MAX_EPOCHS, loss=CAND2_LOSS)
        rec["fit"] = {"adopted": fit["adopted"], "epochs": fit["epochs"], "seconds": time.time() - t0, "broken_max": fit["broken_max"],
                      "n_effective": fit["before"]["n_effective"], "n_mismatch_before": fit["before"]["n_mismatch"],
                      "n_margin_fail_before": fit["before"]["n_margin_fail"], "min_margin_before": fit["before"]["min_margin"],
                      "n_mismatch_after": fit["after"]["n_mismatch"], "n_margin_fail_after": fit["after"]["n_margin_fail"],
                      "min_margin_after": fit["after"]["min_margin"]}
        if not fit["adopted"]:
            return rec, None
        base_info, _bacts = run_base_episode(cfg, policy, factory)
        rec["base_E_hex"], rec["base_DR_hex"] = _hex(base_info)
        rec["base_matches_H"] = (rec["base_E_hex"], rec["base_DR_hex"]) == (harvest["E_hex"], harvest["DR_hex"])
        rec["ECI"] = float(base_info["ECI"])
        rec["ok"] = bool(rec["base_matches_H"])
        return rec, (policy if rec["ok"] else None)
    except Exception:
        rec["error"] = traceback.format_exc()[-2000:]
        rec["ok"] = False
        return rec, None
