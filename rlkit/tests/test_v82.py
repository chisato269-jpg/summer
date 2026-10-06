"""指示書 V8.2-S の単体試験（UT-V8-11〜UT-V8-15）。エミュレータは使わない（模擬の環境）。"""
import json
import os

import numpy as np
import pytest
import torch

from rlkit import config, constants
from rlkit import policy as p7
from rlkit.v8 import action as a8
from rlkit.v8 import adopt
from rlkit.v8 import blocks as blk
from rlkit.v8 import observation as o8
from rlkit.v8 import policy as p8
from rlkit.v8 import trainer as tr1
from rlkit.v8 import trainer2 as tr2

T_SYN = 40


def _flat_mask():
    return np.ones(a8.ACTION_LOGITS, dtype=bool)


TRIAL08_BEST = os.path.join("runs", "v2s_trial08", "best", "best_official.pt")


def _policy(seed=1):
    """学習済みの方策（trial08 の best_official.pt を V8 に拡張したもの。無ければ無作為の初期値）と、その V7 の entry。"""
    if os.path.exists(TRIAL08_BEST):
        entry = torch.load(TRIAL08_BEST, weights_only=False)
    else:
        torch.manual_seed(seed)
        entry = {"stage": 1, "update": 0, "E": 0.3, "DR": 0.05, "eci_smooth": 0.3057, "E_hex": float.hex(0.3), "DR_hex": float.hex(0.05),
                 "weights": {"policy_net": p7.PolicyNetwork(1).state_dict(), "value_net": p7.ValueNetwork().state_dict()}}
    return p8.build_from_v7(entry)[0], entry


def _teacher(seed=0, T=T_SYN, conflict=False, n_flips=8, policy=None):
    """方策の確率最大の選択に、いくつかの判断だけを「2 番目に高い選択」に変えた軌跡（兄弟の軌跡と同じ形。変える判断は、1 位と 2 位の
    差が小さいものから選ぶ。差の大きい判断を変えるには、ロジットを大きく動かす必要があり、300 エポックで足りないことがある）。"""
    rng = np.random.default_rng(seed)
    obs = rng.uniform(0, 1, size=(T, o8.TOTAL_DIM)).astype(np.float32)
    masks = np.ones((T, a8.ACTION_LOGITS), dtype=bool)
    policy = policy or _policy(1)[0]
    with torch.no_grad():
        logits = policy(torch.as_tensor(obs)).numpy()
    actions = p8.masked_argmax(logits, masks)
    gaps, second = [], {}
    for d, (o, n) in enumerate(zip(a8.OFFSETS, a8.NVEC)):
        seg = logits[:, o:o + n].copy()
        best = actions[:, d]
        np.put_along_axis(seg, best[:, None], -np.inf, axis=1)
        sec = seg.argmax(axis=1)
        gap = logits[np.arange(T), o + best] - logits[np.arange(T), o + sec]
        for t in range(T):
            if 0.05 < gap[t] < 4.0 and n > 2 or (n == 2 and 0.05 < gap[t] < 4.0):
                gaps.append((float(gap[t]), t, d))
                second[(t, d)] = int(sec[t])
    gaps.sort()
    for _g, t, d in gaps[:n_flips]:
        actions[t, d] = second[(t, d)]
    if conflict:                                                # 同じ観測に、違う選択を要求する（学習では満たせない）
        obs[1] = obs[0]
        actions[1] = (actions[0] + 1) % a8.NVEC
    return obs, masks, actions


# ---- UT-V8-11: L-4R ----

def test_ut_v8_11_sparse_never_returns_a_noop_and_picks_a_non_argmax_choice(monkeypatch):
    monkeypatch.setattr(blk, "SPARSE_Q", 0.0)                   # 選び直しが 0 個になる乱数（確率 0）
    table = blk.block_steps_table()
    base = np.zeros(a8.ACTION_DIMS, dtype=np.int64)
    for b in (3, 40, 70):
        p = blk.Perturber(b, "sparse", blk.sibling_seed(1, 1, b), table[b])
        assert p.fallback and p.fallback["kind"] == "sparse_one_decision"
        for step in table[b]:
            out = p.apply(int(step), base, _flat_mask())
        assert len(p.changes) == 1                              # 1 個だけ選び直される
        st, d, old, new = p.changes[0]
        assert old == int(base[d]) and new != old and d in p.dims and st in set(int(s) for s in table[b])
        q = blk.Perturber(b, "sparse", blk.sibling_seed(1, 1, b), table[b])                     # 同じ種なら同じ
        for step in table[b]:
            q.apply(int(step), base, _flat_mask())
        assert q.changes == p.changes
    # 候補が 1 択にマスクされていれば飛ばし、次の候補へ進む（最初の候補の次元を 1 択にする）
    p = blk.Perturber(40, "sparse", blk.sibling_seed(5, 5, 5), table[40])
    first_step, first_dim = p.fallback["first_candidate"]
    m = _flat_mask()
    m[a8.OFFSETS[first_dim]:a8.OFFSETS[first_dim] + a8.NVEC[first_dim]] = False
    m[a8.OFFSETS[first_dim]] = True
    for step in table[40]:
        p.apply(int(step), base, m)
    assert len(p.changes) == 1 and p.changes[0][1] != first_dim or (first_step, first_dim) != tuple(p.changes[0][:2])
    # 通常の乱数（q = 0.03）では空振り防止は働かない場合がある。働かないときは L-4 と同じ
    monkeypatch.setattr(blk, "SPARSE_Q", 0.5)
    p = blk.Perturber(40, "sparse", blk.sibling_seed(2, 2, 2), table[40])
    assert p.fallback is None


def test_ut_v8_11_coherent_with_no_selected_unit_picks_one(monkeypatch):
    monkeypatch.setattr(blk, "COHERENT_P", 0.0)                 # 単位が 0 個になる確率
    table = blk.block_steps_table()
    for b in (5, 30, 60):
        p = blk.Perturber(b, "coherent", blk.sibling_seed(1, 1, b), table[b])
        assert len(p.coherent["units"]) == 1 and p.coherent["units"][0] in p.coherent["all_units"]
        assert p.fallback["kind"] == "coherent_unit"
    monkeypatch.setattr(blk, "COHERENT_P", 1.0)
    p = blk.Perturber(5, "coherent", blk.sibling_seed(1, 1, 5), table[5])
    assert p.fallback is None and len(p.coherent["units"]) == len(p.coherent["all_units"])


# ---- UT-V8-14: 有効な判断 ----

def test_ut_v8_14_effective_decisions():
    T = 6
    acts = np.zeros((T, a8.ACTION_DIMS), dtype=np.int64)
    acts[:, a8.dim_slice("zone_duty")] = a8.DUTY_FULL
    acts[2, a8.LAYOUT["zone_duty"][0] + 4] = a8.DUTY_OFF                   # ゾーン 4 は刻み 2 で割合 0
    masks = np.ones((T, a8.ACTION_LOGITS), dtype=bool)
    one = a8.LAYOUT["tenant_vent"][0]
    masks[:, a8.OFFSETS[one]:a8.OFFSETS[one] + 4] = False
    masks[:, a8.OFFSETS[one]] = True                                       # 南の換気は 1 択（M-9）
    z = a8.LAYOUT["zone_duty"][0]
    masks[3, a8.OFFSETS[z + 7]:a8.OFFSETS[z + 7] + 4] = False
    masks[3, a8.OFFSETS[z + 7] + a8.DUTY_OFF] = True                       # ゾーン 7 は刻み 3 で割合 0 だけ（M-5'）
    eff = adopt.effective_decision_mask(acts, masks)
    sp, fan = a8.LAYOUT["zone_setpoint"][0], a8.LAYOUT["zone_fan"][0]
    assert not eff[:, one].any()                                           # 1 択にマスクされた判断は除く
    assert not eff[3, z + 7]
    assert not eff[2, sp + 4]                                              # 割合 0 のゾーンの設定温度は環境にも観測にも入らない
    assert eff[2, fan + 4]                                                 # 割合 0 のゾーンの風量は観測（直前の風量）に入るので含める
    assert eff[1, sp + 4] and eff[1, fan + 4] and eff[2, z + 4]            # 書き込まれる判断は全て含まれる
    for g in ("zone_remote", "tenant_bypass", "system_evap"):              # リモコン・バイパス・蒸発温度は常に書き込まれる
        s = a8.dim_slice(g)
        assert eff[1, s].all()
    n_valid = a8.n_valid_options(masks)
    assert not (eff & (n_valid <= 1)).any()                                # 有効な判断は全て、選択肢が 2 つ以上


# ---- UT-V8-12: L-6R ----

def test_ut_v8_12_adopts_only_when_all_effective_decisions_match_with_margin():
    pol, _ = _policy()
    obs, masks, actions = _teacher()
    teacher = adopt.make_teacher(obs, masks, actions)
    before = adopt.check_match(pol, teacher)
    assert not before["ok"] and before["n_mismatch"] > 0
    r = adopt.fit_to_teacher(pol, teacher)
    assert r["adopted"] and 1 <= r["epochs"] <= adopt.MAX_EPOCHS
    after = adopt.check_match(pol, teacher)
    assert after["ok"] and after["n_mismatch"] == 0 and after["n_margin_fail"] == 0 and after["min_margin"] >= adopt.LOGIT_MARGIN
    # 採用した方策を決定論的に走らせると、有効な判断が全て教師と同じ
    with torch.no_grad():
        mine = p8.masked_argmax(pol(torch.as_tensor(obs)).numpy(), masks)
    assert ((mine == actions) | ~teacher["eff"]).all()


def test_ut_v8_12_restores_weights_when_not_satisfied_in_300_epochs():
    pol, _ = _policy(2)
    obs, masks, actions = _teacher(1, conflict=True)
    teacher = adopt.make_teacher(obs, masks, actions)
    before = {k: v.clone() for k, v in pol.state_dict().items()}
    r = adopt.fit_to_teacher(pol, teacher, max_epochs=40)                  # 同じ観測に違う選択: 満たせない（短く打ち切って確認）
    assert not r["adopted"] and r["epochs"] == 40
    assert all(torch.equal(before[k], v) for k, v in pol.state_dict().items())   # 学習前の重みに戻る


def test_ut_v8_12_tries_up_to_three_siblings_in_descending_A_and_adopts_one(monkeypatch):
    pol, _ = _policy()
    sibs = [{"obs": np.zeros((2, 3)), "masks": np.ones((2, 3), bool), "actions": np.zeros((2, 80), int)} for _ in range(12)]
    adv = np.array([0.0, 0.3, -0.1, 0.2, 0.5, 0.1, 0.0, 0.4, 0.05, 0.0, 0.0, 0.0])
    calls = []
    outcome = {4: False, 7: False, 1: True}

    def fake_fit(policy, teacher, lr, max_epochs, margin):
        s = next(i for i, sb in enumerate(sibs) if sb["actions"] is teacher["actions"] or np.array_equal(sb["actions"], teacher["actions"]))
        raise AssertionError("unused")
    # 試す順序を、make_teacher の呼び出しで記録する
    order = []

    def fake_make(obs, masks, actions):
        order.append(int(np.asarray(actions)[0, 0]))
        return {"idx": int(np.asarray(actions)[0, 0])}

    def fake_fit2(policy, teacher, lr, max_epochs, margin, loss="k1"):
        calls.append(teacher["idx"])
        ok = outcome[teacher["idx"]]
        info = {"ok": ok, "n_effective": 10, "n_mismatch": 0 if ok else 3, "n_margin_fail": 0, "min_margin": 1.0}
        return {"adopted": ok, "epochs": 5 if ok else 300, "before": info, "after": info}
    for i, sb in enumerate(sibs):
        sb["actions"] = np.full((2, 80), i, dtype=np.int64)
    monkeypatch.setattr(adopt, "make_teacher", fake_make)
    monkeypatch.setattr(adopt, "fit_to_teacher", fake_fit2)
    r = adopt.try_adopt(pol, sibs, adv)
    assert calls == [4, 7, 1]                                              # A_s の大きい順（0.5、0.4、0.3）、最大 3 本
    assert r["adopted"] == 1 and [t["sibling"] for t in r["tried"]] == [4, 7, 1] and [t["adopted"] for t in r["tried"]] == [False, False, True]
    calls.clear()
    outcome[4] = True                                                      # 1 本目で採用したら、そこで終わる（1 更新で 1 本だけ）
    r = adopt.try_adopt(pol, sibs, adv)
    assert calls == [4] and r["adopted"] == 4 and len(r["tried"]) == 1
    calls.clear()
    outcome.update({4: False, 7: False, 1: False, 3: True})                # 4 本目（3 番）は試さない
    r = adopt.try_adopt(pol, sibs, adv)
    assert calls == [4, 7, 1] and r["adopted"] is None and len(r["tried"]) == 3
    assert adopt.try_adopt(pol, sibs, np.zeros(12))["tried"] == []         # A_s > 0 の兄弟が無ければ何も試さない


# ---- UT-V8-13: 再現の論理 ----

def test_ut_v8_13_adopted_policy_reproduces_the_sibling_trajectory_deterministically():
    from rlkit.v8.mockenv import MockSummerEnvV8
    pol, _ = _policy(3)
    env = MockSummerEnvV8(seed=7)
    N_STEPS = 80                                               # 論理の確認なので、週の全部でなく最初の 80 ステップで行う

    def run(policy, flips=0):
        """flips > 0 なら、最初の flips 個の「1 位と 2 位の差が小さい判断」を 2 位の選択に変えた兄弟の軌跡を走らせる。"""
        obs, _ = env.reset()
        O, M, A = [], [], []
        left = flips
        for t in range(N_STEPS):
            mask = env.action_masks()
            with torch.no_grad():
                logits = policy(torch.as_tensor(obs[None, :])).numpy()[0]
            act = p8.masked_argmax(logits[None, :], mask[None, :])[0]
            if left > 0 and 50 <= t < 70:
                for d in range(a8.ACTION_DIMS):
                    o, n = int(a8.OFFSETS[d]), int(a8.NVEC[d])
                    seg = np.where(mask[o:o + n], logits[o:o + n], -np.inf)
                    order = np.argsort(-seg)
                    if n > 1 and np.isfinite(seg[order[1]]) and 0.05 < seg[order[0]] - seg[order[1]] < 1.0 and left > 0:
                        act[d] = int(order[1])
                        left -= 1
            O.append(obs.copy())
            M.append(mask.copy())
            A.append(act.copy())
            obs, _r, done, _t, info = env.step(act)
        return np.stack(O), np.stack(M), np.stack(A), info
    O, M, A, info_sib = run(pol, flips=4)
    O0, M0, A0, info_base0 = run(pol)
    assert (A != A0).sum() >= 4                                            # 兄弟は基準と違う（マスクの前回値の固定で、変えた判断が後のステップにも波及する）
    teacher = adopt.make_teacher(O, M, A)
    r = adopt.fit_to_teacher(pol, teacher)
    assert r["adopted"] and r["epochs"] >= 1
    O2, M2, A2, info_base = run(pol)                                       # 採用した方策の決定論的な走行
    assert ((A2 == A) | ~teacher["eff"]).all()                             # 有効な判断は完全に一致する
    assert np.array_equal(O2, O)                                           # 模擬の観測はステップ番号で決まる
    assert (A2 == A).mean() > 0.999                                        # 有効でない判断（割合 0 のゾーンの設定温度など）以外も、ほぼ一致


# ---- UT-V8-15: L-7R ----

def _trainer_cfg(tmp, planned=4):
    os.makedirs(str(tmp), exist_ok=True)
    cfg = config.load(os.path.join("rlkit", "config_summer_v82.json"))
    _pol, entry = _policy()
    entry_path = os.path.join(str(tmp), "init_entry.pt")
    torch.save(entry, entry_path)
    cfg.update({"run_dir": str(tmp / "run"), "model_dir": str(tmp / "models"), "planned_updates": planned,
                "init_weights": entry_path, "init_policy": None, "start_E_hex": None, "start_DR_hex": None})
    return cfg


def _mock_base(tmp):
    """模擬の環境で、初期の方策の基準の軌跡の (E, DR) を得る（始点の設定に使う）。"""
    cfg = _trainer_cfg(tmp / "probe", planned=1)
    first = json.loads(open(os.path.join(cfg["run_dir"], "updates.jsonl"), encoding="utf-8").readline()) \
        if os.path.exists(os.path.join(cfg["run_dir"], "updates.jsonl")) else None
    return cfg, first


def _run_mock(cfg, planned, update_policy=None):
    cfg = dict(cfg, planned_updates=planned)
    t = tr2.V82Trainer(cfg, cfg["run_dir"], factory="mock_v8")
    if update_policy is not None:
        t.update_policy = update_policy.__get__(t)
    t.run()
    return t


def _start_from_first_update(tmp):
    cfg = _trainer_cfg(tmp / "p", planned=1)
    cfg["start_E_hex"], cfg["start_DR_hex"] = float.hex(1.0), float.hex(0.1)           # 一致しない始点（更新 1 で止まる）で基準の (E, DR) を得る
    t = _run_mock(cfg, 1)
    row = json.loads(open(os.path.join(cfg["run_dir"], "updates.jsonl"), encoding="utf-8").readline())
    return float.hex(row["eval_E"]), float.hex(row["eval_DR"])


def test_ut_v8_15_l7r_stops_on_mismatch_and_continues_on_match(tmp_path):
    e_hex, d_hex = _start_from_first_update(tmp_path)
    # 始点が基準の軌跡と 16 進で一致し、採用が無い（方策が変わらない）更新が続けば、毎更新の基準が前の基準と一致して最後まで進む
    cfg = dict(_trainer_cfg(tmp_path / "ok", planned=4), start_E_hex=e_hex, start_DR_hex=d_hex)

    def never_adopt(self, base, sibs, adv):
        E, DR = self._cur["E"], self._cur["DR"]
        self._check_base = self.expected_base
        self._l7r = tr2.l7r_check(self._check_base, (float(E[tr1.BASE]), float(DR[tr1.BASE])))
        self.expected_base = (float(E[tr1.BASE]), float(DR[tr1.BASE]))
        return {"n_improved": 0, "updated": False, "epochs_applied": 0, "kl_cutoff": False, "kl_per_epoch": [], "loss_per_epoch": [],
                "adoption_rate": None, "n_decisions": 0, "weights": {}, "nonfinite": False,
                "extra": {"adopt": {"adopted_sibling": None, "tried": []}, "l7r": self._l7r}}
    t = _run_mock(cfg, 4, never_adopt)
    assert t.update_index == 4 and t.stop_reason.startswith("計画更新数")
    l7r = [json.loads(l) for l in open(os.path.join(cfg["run_dir"], "l7r.jsonl"), encoding="utf-8")]
    assert len(l7r) == 4 and all(x["passed"] for x in l7r)
    # 始点が基準と違えば、更新 1 で停止する
    bad = dict(_trainer_cfg(tmp_path / "bad", planned=4), start_E_hex=float.hex(float.fromhex(e_hex) + 1e-6), start_DR_hex=d_hex)
    t2 = _run_mock(bad, 4, never_adopt)
    assert t2.update_index == 1 and "L-7R" in t2.stop_reason


def test_ut_v8_15_next_base_must_equal_the_adopted_sibling(tmp_path):
    e_hex, d_hex = _start_from_first_update(tmp_path)
    cfg = dict(_trainer_cfg(tmp_path / "adopt", planned=4), start_E_hex=e_hex, start_DR_hex=d_hex)

    def fake_adopt_sibling0(self, base, sibs, adv):
        # 兄弟 0 を採用したことにするが、方策は変えない: 次の基準は兄弟 0 の (E, DR) と一致しない（兄弟 0 は基準と違う成績）
        E, DR = self._cur["E"], self._cur["DR"]
        self._check_base = self.expected_base
        self._l7r = tr2.l7r_check(self._check_base, (float(E[tr1.BASE]), float(DR[tr1.BASE])))
        self.expected_base = (float(E[0]), float(DR[0]))
        return {"n_improved": 1, "updated": True, "epochs_applied": 1, "kl_cutoff": False, "kl_per_epoch": [], "loss_per_epoch": [],
                "adoption_rate": None, "n_decisions": 1, "weights": {}, "nonfinite": False,
                "extra": {"adopt": {"adopted_sibling": 0, "tried": []}, "l7r": self._l7r}}
    t = _run_mock(cfg, 4, fake_adopt_sibling0)
    assert t.update_index == 2 and "L-7R" in t.stop_reason                  # 更新 2 の基準が兄弟 0 の E・DR と一致しないので停止
    rows = [json.loads(l) for l in open(os.path.join(cfg["run_dir"], "l7r.jsonl"), encoding="utf-8")]
    assert rows[0]["passed"] and not rows[1]["passed"]


def test_ut_v8_15_trainer_records_adoption_fields_per_update(tmp_path):
    """L-8R: 毎更新、試した兄弟の番号と A_s、採否、エポック数、学習の前に一致していなかった判断の数、最小のロジットの差を記録する。"""
    e_hex, d_hex = _start_from_first_update(tmp_path)
    cfg = dict(_trainer_cfg(tmp_path / "rec", planned=3), start_E_hex=e_hex, start_DR_hex=d_hex)
    t = tr2.V82Trainer(cfg, cfg["run_dir"], factory="mock_v8")
    t.run()
    ups = [json.loads(l) for l in open(os.path.join(cfg["run_dir"], "updates.jsonl"), encoding="utf-8")]
    assert len(ups) >= 1
    for u in ups:
        a = u["adopt"]
        assert {"adopted_sibling", "tried", "n_tried", "epochs_to_adopt", "n_mismatch_before", "min_margin_after"} <= set(a)
        for tr in a["tried"]:
            assert {"sibling", "A", "adopted", "epochs", "n_mismatch_before", "min_margin_before", "min_margin_after"} <= set(tr)
        assert a["n_tried"] == len(a["tried"]) <= adopt.MAX_SIBLINGS_TRIED
        assert "l7r" in u
