"""指示書 V8.3-S の単体試験（UT-V8-16〜UT-V8-19）。エミュレータは使わない（模擬の環境）。"""
import json
import os

import numpy as np
import pytest
import torch

from rlkit.v8 import action as a8
from rlkit.v8 import adopt
from rlkit.v8 import trainer3 as tr3

from test_v82 import _policy, _start_from_first_update, _teacher, _trainer_cfg


def _logits_with_margins(margins):
    """全判断の選択肢 0 を教師の選択とし、(T, 80) の m_d になるロジット (T, 372) を作る（他の選択肢のロジットは 0）。"""
    T = margins.shape[0]
    logits = torch.zeros(T, a8.ACTION_LOGITS)
    for d, o in enumerate(a8.OFFSETS.tolist()):
        logits[:, o] = torch.as_tensor(margins[:, d], dtype=torch.float32)
    return logits


# ---- UT-V8-16: K-3 ----

def test_ut_v8_16_margin_loss_is_a_sum_over_violators_only():
    T = 2
    margins = np.zeros((T, a8.ACTION_DIMS))
    margins[0, 0] = 0.2                       # 余裕を満たした（損失に入らない）
    margins[0, 1] = -0.3                      # 違反: 0.05 − (−0.3) = 0.35
    margins[1, 2] = 0.03                      # 違反: 0.05 − 0.03 = 0.02
    margins[1, 3] = -5.0                      # 有効でない判断（損失に入らない）
    eff = torch.ones(T, a8.ACTION_DIMS, dtype=torch.bool)
    eff[1, 3] = False
    masks = torch.ones(T, a8.ACTION_LOGITS, dtype=torch.bool)
    actions = torch.zeros(T, a8.ACTION_DIMS, dtype=torch.long)
    logits = _logits_with_margins(margins).requires_grad_(True)
    loss = adopt.training_loss("k3", logits, masks, actions, eff)
    # 有効な判断は 159 個。m = 0 の判断（違反。0.05 ずつ）が 159 − 3 = 156 個、0.35 と 0.02 が 1 個ずつ、0.2 は 0
    expected = 156 * 0.05 + 0.35 + 0.02
    assert float(loss) == pytest.approx(expected, abs=1e-4)
    assert float(loss) > 1.0                                  # 判断の総数で割っていない（割れば 0.05 程度）
    loss.backward()
    g = logits.grad.numpy()
    assert g[0, a8.OFFSETS[0]] == 0.0                         # 余裕を満たした判断は勾配が 0
    assert g[1, a8.OFFSETS[3]] == 0.0                         # 有効でない判断も 0
    assert g[0, a8.OFFSETS[1]] == pytest.approx(-1.0)         # 違反した判断は、選択肢のロジットを上げる向き（係数 1）
    # マスクで有効な他の選択肢だけを数える: 最大の他の選択肢を無効にすると、余裕は次に大きい有効な選択肢との差になる
    masks2 = masks.clone()
    o1, n1 = int(a8.OFFSETS[1]), int(a8.NVEC[1])
    logits2 = _logits_with_margins(margins)
    logits2[0, o1 + 1] = 10.0                                 # 無効にする選択肢に大きなロジットを置く
    masks2[0, o1 + 1] = False
    m = adopt.torch_margins(logits2, masks2, actions).numpy()
    assert m[0, 1] == pytest.approx(-0.3)                     # 無効な選択肢は他の最大に入らない
    one = torch.zeros_like(masks2)
    one[:, :] = masks2
    for k in range(1, n1):
        one[0, o1 + k] = False                                # 1 択になれば損失に入らない大きな正の値
    assert adopt.torch_margins(logits2, one, actions)[0, 1] > 1e6


def test_k3_and_other_losses_reach_adoption_on_a_small_teacher():
    pol, _ = _policy()
    for kind in adopt.LOSS_KINDS:
        p = type(pol)()
        p.load_state_dict(pol.state_dict())
        obs, masks, actions = _teacher()
        teacher = adopt.make_teacher(obs, masks, actions)
        r = adopt.fit_to_teacher(p, teacher, lr=3e-3, max_epochs=1000, loss=kind)
        assert r["adopted"], kind
        assert adopt.check_match(p, teacher)["ok"]
        assert r["broken_max"] >= 0 and r["seconds"] >= 0


# ---- UT-V8-17: K-4 ----

def test_ut_v8_17_weights_are_100_for_initially_unmet_decisions_and_fixed_during_training(monkeypatch):
    pol, _ = _policy()
    obs, masks, actions = _teacher()
    teacher = adopt.make_teacher(obs, masks, actions)
    ok0, mm, low, _ = adopt.decision_ok_matrix(pol, teacher)
    w = adopt.k4_weights(ok0, teacher["eff"])
    bad = teacher["eff"] & ~ok0
    assert bad.sum() > 0 and (w[bad] == 100.0).all() and (w[~bad] == 1.0).all()
    # 学習中に損失へ渡される重みは、全エポックで同じ
    seen = []
    real = adopt.training_loss

    def spy(kind, logits, masks_, actions_, eff_, weights=None):
        seen.append(None if weights is None else weights.clone())
        return real(kind, logits, masks_, actions_, eff_, weights)
    monkeypatch.setattr(adopt, "training_loss", spy)
    r = adopt.fit_to_teacher(pol, teacher, lr=3e-3, max_epochs=1000, loss="k4")
    assert r["epochs"] >= 2 and len(seen) == r["epochs"]
    assert all(torch.equal(seen[0], x) for x in seen)
    assert float((seen[0] == 100.0).sum()) == float(bad.sum())
    # 損失は重み付きの平均: Σ w·(−log p) / Σ w
    obs_t = torch.as_tensor(teacher["obs"])
    masks_t = torch.as_tensor(teacher["masks"])
    act_t = torch.as_tensor(teacher["actions"])
    eff_t = torch.as_tensor(teacher["eff"])
    logits = pol(obs_t)
    logp = adopt.pol8.chosen_log_prob(adopt.pol8.per_dim_log_softmax(logits, masks_t), act_t)
    wt = torch.as_tensor(w) * eff_t
    manual = float(-(logp * wt).sum() / wt.sum())
    assert float(real("k4", logits, masks_t, act_t, eff_t, torch.as_tensor(w))) == pytest.approx(manual, rel=1e-5)
    # K-1 は重みなし（平均）
    k1 = float(real("k1", logits, masks_t, act_t, eff_t))
    assert k1 == pytest.approx(float(-(logp * eff_t).sum() / eff_t.sum()), rel=1e-5)


# ---- UT-V8-18: C83-2 ----

def test_ut_v8_18_effective_decisions_include_next_observation_inputs():
    T = 4
    acts = np.zeros((T, a8.ACTION_DIMS), dtype=np.int64)
    acts[:, a8.dim_slice("zone_duty")] = a8.DUTY_FULL
    acts[1, a8.LAYOUT["zone_duty"][0] + 4] = a8.DUTY_OFF                   # ゾーン 4 は刻み 1 で割合 0
    masks = np.ones((T, a8.ACTION_LOGITS), dtype=bool)
    one = a8.LAYOUT["tenant_vent"][0]
    masks[:, a8.OFFSETS[one]:a8.OFFSETS[one] + 4] = False
    masks[:, a8.OFFSETS[one]] = True                                       # 1 択にマスクされた判断
    eff = adopt.effective_decision_mask(acts, masks)
    sp, fan = a8.LAYOUT["zone_setpoint"][0], a8.LAYOUT["zone_fan"][0]
    assert eff[1, fan + 4]                                                 # 割合 0 のゾーンの風量は次の観測（直前の風量）に入るので含まれる
    assert not eff[1, sp + 4]                                              # 設定温度は書き込まれず観測にも入らないので除かれる
    assert eff[0, sp + 4]                                                  # 割合が 0 でなければ含まれる
    assert not eff[:, one].any()                                           # 1 択の判断は除かれる


# ---- UT-V8-19: C83-3 ----

def test_ut_v8_19_teacher_saving_rule_and_roundtrip(tmp_path):
    assert tr3.should_save_teacher(1, True) and tr3.should_save_teacher(50, True) and tr3.should_save_teacher(50, False)
    assert not tr3.should_save_teacher(51, True)
    assert tr3.should_save_teacher(51, False) and tr3.should_save_teacher(600, False)
    obs, masks, actions = _teacher()
    teacher = adopt.make_teacher(obs, masks, actions)
    path = str(tmp_path / "t.npz")
    tr3.save_teacher(path, teacher)
    back = tr3.load_teacher(path)
    assert np.array_equal(back["obs"], teacher["obs"]) and back["obs"].dtype == np.float32
    assert np.array_equal(back["masks"], teacher["masks"]) and back["masks"].dtype == bool
    assert np.array_equal(back["actions"], teacher["actions"])
    assert np.array_equal(back["eff"], teacher["eff"])


def test_ut_v8_19_trainer_saves_tried_teachers_by_update_and_outcome(tmp_path, monkeypatch):
    e_hex, d_hex = _start_from_first_update(tmp_path)
    cfg = dict(_trainer_cfg(tmp_path / "t", planned=3), start_E_hex=e_hex, start_DR_hex=d_hex,
               l6r_loss="k3", l6r_lr=3e-3, l6r_max_epochs=3)             # 3 エポックでは満たせない設定（失敗の試みを作る）
    t = tr3.V83Trainer(cfg, cfg["run_dir"], factory="mock_v8")
    t.run()
    ups = [json.loads(l) for l in open(os.path.join(cfg["run_dir"], "updates.jsonl"), encoding="utf-8")]
    assert ups[0]["adopt"]["n_tried"] >= 0
    for u in ups:
        for tr in u["adopt"]["tried"]:
            f = os.path.join(cfg["run_dir"], "teachers", f"u{u['update']}_s{t.slots[tr['sibling']]}.npz")
            assert os.path.exists(f)                                       # 更新 50 までは試したものがすべて保存される
            z = np.load(f)
            assert set(z.files) == {"obs", "masks", "actions"}
    assert t.init_meta["l6r"] == {"loss": "k3", "lr": 3e-3, "max_epochs": 3}
    # 更新 51 以降の規則（試みを模擬して、採用した教師は保存せず、失敗した教師だけ保存する）
    obs, masks, actions = _teacher()
    teacher = adopt.make_teacher(obs, masks, actions)

    def fake_try(policy, sibs, adv, lr, max_epochs, loss, on_tried):
        on_tried(0, teacher, True)
        on_tried(1, teacher, False)
        return {"adopted": 0, "tried": []}
    monkeypatch.setattr(adopt, "try_adopt", fake_try)
    t.update_index = 50                                                    # 次は更新 51
    t.try_adopt([], np.zeros(12))
    d = os.path.join(cfg["run_dir"], "teachers")
    assert not os.path.exists(os.path.join(d, f"u51_s{t.slots[0]}.npz"))
    assert os.path.exists(os.path.join(d, f"u51_s{t.slots[1]}.npz"))
    t.update_index = 49                                                    # 次は更新 50: 採用した教師も保存する
    t.try_adopt([], np.zeros(12))
    assert os.path.exists(os.path.join(d, f"u50_s{t.slots[0]}.npz")) and os.path.exists(os.path.join(d, f"u50_s{t.slots[1]}.npz"))


def test_c83_4_g20_limit_is_450_and_defaults_are_unchanged():
    assert tr3.V83Trainer.G20_SECONDS_MAX == 450.0
    from rlkit.v8 import trainer2 as tr2
    from rlkit.v8 import trainer as tr1
    assert tr2.V82Trainer.G20_SECONDS_MAX == tr1.G20_SECONDS_MAX == 400.0     # V8.2-S の値は変えない
    assert (adopt.LEARNING_RATE, adopt.MAX_EPOCHS) == (3e-4, 300)              # 設定が無いときの L-6R は V8.2-S のまま
