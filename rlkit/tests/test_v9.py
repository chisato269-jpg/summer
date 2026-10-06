"""指示書 V9-S の単体試験（UT-V9-1〜UT-V9-7）。エミュレータは使わない（模擬の環境 mock_v8）。"""
import json
import os
import types

import numpy as np
import pytest
import torch

from rlkit import config, constants
from rlkit.learner import clipped_policy_loss
from rlkit.v8 import action as a8
from rlkit.v8 import observation as o8
from rlkit.v8 import policy as p8
from rlkit.v9 import credit, nets, phase_a, phase_b as pb, switch
from rlkit.v9 import trainer_single as ts9

FACTORY = "mock_v8"
CFG_PATH = os.path.join("rlkit", "config_summer_v9.json")


def _cfg(tmp, planned=4, force=2):
    cfg = config.load(CFG_PATH)
    cfg.update({"run_dir": str(tmp / "run"), "model_dir": str(tmp / "models"), "planned_updates": planned})
    cfg["v9"] = dict(cfg["v9"], force_switch_at=force)
    return cfg


# ---- UT-V9-1: C9-3 ----

def test_ut_v9_1_new_options_start_at_relative_probability_e_minus_4_and_masked_options_are_never_chosen():
    torch.manual_seed(0)
    net = nets.build_policy_net()
    rows = nets.new_option_rows()
    for head, idx in rows.items():
        b = getattr(net, head).bias.detach().numpy()
        assert (b[idx] == nets.NEW_OPTION_BIAS).all()
        assert (np.delete(b, idx) == 0.0).all()                              # それ以外の偏りは 0
    obs = torch.as_tensor(np.random.default_rng(1).uniform(0, 1, size=(64, o8.TOTAL_DIM)), dtype=torch.float32)
    with torch.no_grad():
        logits = net(obs)
    is_new = nets.new_option_logit_indices()
    masks = torch.ones(64, a8.ACTION_LOGITS, dtype=torch.bool)
    logp = p8.per_dim_log_softmax(logits, masks).numpy()
    for d, (o, n) in enumerate(zip(a8.OFFSETS, a8.NVEC)):
        new = is_new[o:o + n]
        if not new.any():
            continue
        lp = logp[:, o:o + n]
        old_mean = lp[:, ~new].mean(axis=1)
        for j in np.flatnonzero(new):
            ratio = np.exp(lp[:, j] - old_mean)
            assert 0.0183 * 0.7 < ratio.mean() < 0.0183 * 1.4, (d, j, ratio.mean())     # 約 0.018（頭部の重みは gain 0.01 で小さい）
        assert np.ptp(lp[:, ~new], axis=1).max() < 0.5                      # 新しい選択肢以外はほぼ等しい
    # マスクされた選択肢は選ばれない（確率最大・標本のどちらでも）
    masks2 = masks.clone()
    rng = np.random.default_rng(2)
    for d, (o, n) in enumerate(zip(a8.OFFSETS, a8.NVEC)):
        if n >= 3:
            drop = rng.integers(0, n, size=64)
            for b in range(64):
                masks2[b, o + drop[b]] = False
    am = p8.masked_argmax(logits.numpy(), masks2.numpy())
    lp2 = p8.per_dim_log_softmax(logits, masks2)
    probs = lp2.exp().numpy()
    assert (probs[~masks2.numpy()] == 0).all()
    for _ in range(20):
        sampled = np.stack([torch.multinomial(torch.as_tensor(probs[:, o:o + n]), 1)[:, 0].numpy()
                            for o, n in zip(a8.OFFSETS, a8.NVEC)], axis=1)
        chosen_ok = np.take_along_axis(masks2.numpy(), a8.OFFSETS[None, :] + sampled, axis=1)
        assert chosen_ok.all()
    assert np.take_along_axis(masks2.numpy(), a8.OFFSETS[None, :] + am, axis=1).all()
    v = nets.build_value_net()                                               # 価値網は V7 と同じ初期化（価値頭部 gain 1.0、偏り 0）
    assert float(v.zone_value_head.bias.abs().sum()) == 0.0 and float(v.zone_value_head.weight.norm()) > 0.5
    v.set_reward_context(0.4, 0.3)
    assert v.reward_context.tolist() == pytest.approx([0.4, 0.3])


# ---- UT-V9-2: C9-5 ----

def test_ut_v9_2_switch_rule():
    # 更新 600 未満では切り替わらない（B が全く伸びなくても）
    flat = [0.5] * 700
    for u in (1, 100, 301, 599):
        assert switch.switch_due(flat, u)[0] is False
    due, reason, _ = switch.switch_due(flat, 600)                            # u = 600 で B(600) − B(300) = 0 < 0.002
    assert due and reason == "stagnation"
    # 300 更新の伸びが 0.002 以上なら切り替わらず、0.002 未満になった最初の更新で切り替わる
    series = [0.1 + 0.00001 * k for k in range(1, 1300)]                   # 300 更新で +0.003
    for u in (600, 601, 900, 1199):
        assert switch.switch_due(series, u)[0] is False
    series2 = [0.1 + 0.00001 * min(k, 650) for k in range(1, 1300)]        # 更新 650 から伸びない
    firsts = [u for u in range(600, 1200) if switch.switch_due(series2, u)[0]]
    # B(u) − B(u−300) < 0.002 となる最初の u: 伸びは (min(u,650) − min(u−300,650)) × 1e-5
    exp_first = next(u for u in range(600, 1200) if (min(u, 650) - min(u - 300, 650)) * 1e-5 < 0.002)
    assert firsts[0] == exp_first
    # 更新 1,200 で必ず切り替わる（伸び続けていても）
    assert switch.switch_due(series, 1200) == (True, "cap", switch.switch_due(series, 1200)[2])
    assert switch.switch_due(series, 1200)[1] == "cap"
    # 停滞と上限が同時なら停滞を理由にする（早いほう。同じ更新）
    assert switch.switch_due([0.5] * 1300, 1200)[1] == "stagnation"


# ---- UT-V9-3: C9-6 ----

def test_ut_v9_3_start_selection_rule():
    c1 = {"ok": True, "ECI": 0.60}
    assert switch.select_start(c1, {"ok": True, "ECI": 0.65})["chosen"] == 2           # 候補 2 が成功して ECI が高い
    assert switch.select_start(c1, {"ok": True, "ECI": 0.55})["chosen"] == 1           # 候補 1 のほうが高い
    assert switch.select_start(c1, {"ok": True, "ECI": 0.60})["chosen"] == 1           # 同じなら候補 1
    for bad in (None, {"ok": False, "ECI": 0.9}, {"ok": False, "error": "x"}):          # 失敗・不一致・例外は候補 1
        r = switch.select_start(c1, bad)
        assert r["chosen"] == 1 and r["reason"]
    with pytest.raises(RuntimeError):
        switch.select_start({"ok": False}, {"ok": True, "ECI": 0.9})                  # 候補 1 が走らなければ停止


def _mock_cfg(tmp):
    return _cfg(tmp)


def test_ut_v9_3_candidate2_adoption_with_mock_env(tmp_path, monkeypatch):
    cfg = _mock_cfg(tmp_path)
    monkeypatch.setattr(switch, "CAND2_MAX_EPOCHS", 30)                                      # 試験の時間を短くする
    torch.manual_seed(3)
    net = nets.build_policy_net()
    with torch.no_grad():                                                            # 頭部の重みを大きくして、有効な判断の余裕（1e-3 以上）を確保する
        for head in (net.zone_head, net.system_head, net.tenant_head):
            head.weight.mul_(1.0e5)
    info, acts = switch.run_base_episode(cfg, net, FACTORY)                          # 方策 net の基準の軌跡を H* とする
    harvest = {"update": 5, "role": "eval", "E_hex": float.hex(float(info["E_rep"])), "DR_hex": float.hex(float(info["DR_rep"])),
               "ECI": float(info["ECI"])}
    apath, wpath = str(tmp_path / "h.npz"), str(tmp_path / "w.pt")
    np.savez_compressed(apath, actions=acts.astype(np.int8))
    torch.save({"weights": net.state_dict()}, wpath)
    rec, pol = switch.candidate2(cfg, harvest, apath, wpath, FACTORY)
    assert rec["ok"] and rec["base_matches_H"] and rec["replay_matches_H"] and pol is not None
    assert rec["fit"]["adopted"] and rec["fit"]["epochs"] == 0                        # 採取に使った重みは、そのまま H* を再現する
    # 重みを別の初期化にすると、取り込み（K-3）で H* を再現できるか、できなければ候補 2 は失敗（どちらでも ok と基準の一致は整合する）
    torch.manual_seed(4)
    other = nets.build_policy_net()
    torch.save({"weights": other.state_dict()}, wpath)
    rec2, pol2 = switch.candidate2(cfg, harvest, apath, wpath, FACTORY)
    assert rec2["ok"] == bool(rec2.get("base_matches_H") and rec2["fit"]["adopted"])
    assert (pol2 is not None) == rec2["ok"]
    if rec2["ok"]:
        info_b, _ = switch.run_base_episode(cfg, pol2, FACTORY)
        assert float.hex(float(info_b["E_rep"])) == harvest["E_hex"]
    # 例外（重みのファイルが無い）→ 失敗として記録する（停止しない）
    rec3, pol3 = switch.candidate2(cfg, harvest, apath, str(tmp_path / "none.pt"), FACTORY)
    assert not rec3["ok"] and pol3 is None and "error" in rec3
    # 基準の軌跡が H* と 16 進で一致しない → 失敗
    torch.save({"weights": net.state_dict()}, wpath)
    harvest_bad = dict(harvest, E_hex="0x1.0000000000000p+0")
    rec4, pol4 = switch.candidate2(cfg, harvest_bad, apath, wpath, FACTORY)
    assert not rec4["ok"] and pol4 is None and rec4["base_matches_H"] is False


# ---- UT-V9-4: C9-7 ----

def _fake_info(E, DR, ECI):
    return {"E_rep": E, "DR_rep": DR, "ECI": ECI, "P_E": 0.5, "P_DR": 0.5, "eci_official": ECI, "eci_smooth": ECI}


def test_ut_v9_4_best_files_are_saved_only_when_the_run_best_improves_and_never_overwritten(tmp_path):
    cfg = _cfg(tmp_path)
    pa = phase_a.PhaseA(cfg, cfg["run_dir"], factory=FACTORY)
    os.makedirs(cfg["run_dir"], exist_ok=True)
    n_steps = constants.ZONE_STEPS_PER_WEEK
    rollout = {"actions": np.zeros((12 * n_steps, a8.ACTION_DIMS), dtype=np.int64),
               "action_masks": np.ones((12 * n_steps, a8.ACTION_LOGITS), dtype=bool)}
    pol = types.SimpleNamespace(policy_network=nets.build_policy_net())

    def terminal(eci_list):
        return [(n_steps - 1, i, _fake_info(4.0 + i, 0.2, e)) for i, e in enumerate(eci_list)]
    eval_actions = np.ones((n_steps, a8.ACTION_DIMS), dtype=np.int64)
    pa.harvest_run_best(1, terminal([0.30] * 12), rollout, _fake_info(4.5, 0.2, 0.31), eval_actions, pol)      # 評価が最良
    assert pa.best_run["update"] == 1 and pa.best_run["role"] == "eval"
    pa.harvest_run_best(2, terminal([0.20] * 12), rollout, _fake_info(4.5, 0.2, 0.25), eval_actions, pol)      # 更新なし
    pa.harvest_run_best(3, terminal([0.40] + [0.1] * 11), rollout, _fake_info(4.5, 0.2, 0.35), eval_actions, pol)  # 訓練の組 0 が最良
    assert pa.best_run["update"] == 3 and pa.best_run["role"] == "train"
    h, b = os.path.join(cfg["run_dir"], "harvest"), os.path.join(cfg["model_dir"], "best")
    assert os.path.exists(os.path.join(h, "best_harvest_u1.npz")) and os.path.exists(os.path.join(h, "best_harvest_u3.npz"))
    assert not os.path.exists(os.path.join(h, "best_harvest_u2.npz"))                             # 更新されなかった更新は残さない
    assert os.path.exists(os.path.join(b, "best_harvest_weights_u1.pt")) and os.path.exists(os.path.join(b, "best_harvest_weights_u3.pt"))
    assert not os.path.exists(os.path.join(b, "best_harvest_weights_u2.pt"))
    assert np.load(os.path.join(h, "best_harvest_u1.npz"))["actions"].shape == (n_steps, a8.ACTION_DIMS)     # 評価の行動列
    assert pa.best_eci_history == [0.31, 0.31, 0.40]
    # 既存のファイルは上書きしない
    first_bytes = open(os.path.join(h, "best_harvest_u3.npz"), "rb").read()
    pa.best_run["ECI"] = 0.0                                                                     # 次の更新で再び改善する状況にする
    pa.harvest_run_best(3, terminal([0.50] * 12), rollout, _fake_info(4.5, 0.2, 0.2), eval_actions, pol)
    assert open(os.path.join(h, "best_harvest_u3.npz"), "rb").read() == first_bytes
    events = [json.loads(l)["event"] for l in open(os.path.join(cfg["run_dir"], "events.jsonl"), encoding="utf-8")]
    assert "harvest_alias_exists_not_overwritten" in events and "harvest_weights_exist_not_overwritten" in events


# ---- UT-V9-5: C9-4 ----

def test_ut_v9_5_credit_assignment_and_gradient_cut():
    rng = np.random.default_rng(0)
    T = 5
    zone_adv = rng.normal(size=(T, 18))
    sys_adv = rng.normal(size=T)
    acts = np.zeros((T, a8.ACTION_DIMS), dtype=np.int64)
    acts[:, a8.dim_slice("zone_duty")] = a8.DUTY_FULL
    acts[0, a8.LAYOUT["zone_duty"][0] + 0] = a8.DUTY_OFF
    out = credit.assign_per_dim_advantages(zone_adv, sys_adv, acts)
    building = zone_adv.sum(axis=1) + sys_adv
    for g in ("zone_duty", "zone_setpoint", "zone_fan", "zone_remote"):                    # 建物全体
        s = a8.dim_slice(g)
        assert np.allclose(out[:, s], building[:, None])
    t_of = np.asarray(a8.ZONE_TENANT)
    for g in ("tenant_vent", "tenant_bypass"):                                              # テナント内の和（バイパスも）
        s = a8.dim_slice(g)
        for t in range(2):
            assert np.allclose(out[:, s][:, t], zone_adv[:, t_of == t].sum(axis=1))
    es = a8.LAYOUT["system_evap"][0]
    sys_of = np.asarray(a8.ZONE_SYSTEM)
    on = a8.duty_indices(acts) > 0
    for s in range(4):                                                                       # 蒸発温度: Σ_{k∈s} A_k + A_sys × 運転ゾーンの割合
        share = on[:, sys_of == s].sum(axis=1) / on.sum(axis=1)
        assert np.allclose(out[:, es + s], zone_adv[:, sys_of == s].sum(axis=1) + sys_adv * share)
    lv = credit.dim_levels()
    assert set(lv) == {"building", "tenant", "system_evap"} and (lv[a8.dim_slice("zone_remote")] == "building").all()
    # 全ゾーン割合 0 のステップでは、蒸発温度への A_sys は a_s の比で配る（運転ゾーン数が 0）
    acts0 = np.zeros_like(acts)
    out0 = credit.assign_per_dim_advantages(zone_adv, sys_adv, acts0)
    a = np.asarray(constants.OUTDOOR_UNIT_A_KW)
    assert np.allclose(out0[:, es + 1], zone_adv[:, sys_of == 1].sum(axis=1) + sys_adv * a[1] / a.sum())
    # 作用しない次元の勾配の遮断: 割合 0 のゾーンの設定温度・風量と、1 択の判断は include が偽で、勾配が 0
    masks = np.ones((T, a8.ACTION_LOGITS), dtype=bool)
    o_vent = int(a8.OFFSETS[a8.LAYOUT["tenant_vent"][0]])
    masks[:, o_vent:o_vent + 4] = False
    masks[:, o_vent] = True                                                                  # 南の換気は 1 択
    gc1 = a8.gc1_excluded(acts)
    one = a8.n_valid_options(masks) < 2
    include = torch.as_tensor((~gc1) & (~one))
    assert not include[0, a8.LAYOUT["zone_setpoint"][0]] and not include[0, a8.LAYOUT["zone_fan"][0]]    # ゾーン 0 は割合 0
    assert include[1, a8.LAYOUT["zone_setpoint"][0]]
    assert not include[:, a8.LAYOUT["tenant_vent"][0]].any()
    logp = torch.randn(T, a8.ACTION_DIMS, requires_grad=True)
    old = torch.randn(T, a8.ACTION_DIMS)
    loss, _ = clipped_policy_loss(logp, old, torch.as_tensor(rng.normal(size=(T, a8.ACTION_DIMS))), include)
    loss.backward()
    g = logp.grad.numpy()
    assert (g[~include.numpy()] == 0).all() and (g[include.numpy()] != 0).any()


# ---- UT-V9-6 / UT-V9-7: 模擬の環境で段階 A → 切り替え → 段階 B を通しで走らせる ----

def _read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def test_ut_v9_6_and_7_end_to_end_with_mock_env(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, planned=4, force=2)
    run = ts9.V9Run(cfg, cfg["run_dir"], factory=FACTORY)
    stop_created = {"n": 0}
    orig = phase_a.PhaseA.one_update

    def one_update_with_stop_file(self):
        if stop_created["n"] == 0:                          # 走行中に STOP ファイルが置かれても、読まず、止まらない（UT-V9-7）
            open(os.path.join(self.run_dir, "STOP"), "w").write("x")
            stop_created["n"] = 1
        return orig(self)
    monkeypatch.setattr(phase_a.PhaseA, "one_update", one_update_with_stop_file)
    monkeypatch.setattr(switch, "CAND2_MAX_EPOCHS", 20)                                    # 試験の時間を短くする（取り込みは失敗して候補 1 になる）
    # 模擬の環境は、有効でない判断（割合 0 のゾーンの設定温度など）も成績に入れるため、採用の後の L-7R が一致しない（本物の環境では一致する。
    # V8.3-S の走行で確認済み）。通しの試験では採用を行わず、段階 B の記録と L-7R の経路だけを確かめる
    monkeypatch.setattr(pb.PhaseB, "try_adopt", lambda self, sibs, adv: {"adopted": None, "tried": []})
    reason = run.run()
    assert reason.startswith("計画更新数 4 に到達した"), reason
    ups = _read_jsonl(os.path.join(cfg["run_dir"], "updates.jsonl"))
    assert [u["update"] for u in ups] == [1, 2, 3, 4]
    assert [u["phase"] for u in ups] == ["A", "A", "B", "B"]                                  # 各更新の記録に段階の名前
    sw_rec = json.load(open(os.path.join(cfg["run_dir"], "switch.json"), encoding="utf-8"))
    assert sw_rec["update"] == 2 and sw_rec["choice"]["chosen"] in (1, 2) and sw_rec["candidate1"]["ok"]
    events = _read_jsonl(os.path.join(cfg["run_dir"], "events.jsonl"))
    names = [e["event"] for e in events]
    assert names.count("stop_file_ignored") == 1 and "phase_switch" in names and "resumed" not in names
    # UT-V9-6: 段階 B では PPO・価値網の更新が無く、比 ρ と走行全体の最良が引き継がれる
    b = run.B
    assert all(not p.requires_grad for p in b.value.parameters())
    for u in ups[2:]:
        assert "adopt" in u and u["adopt"] is not None and u["l7r"]["passed"] and "sibling_A" in u
        assert u["kl_per_epoch"] == [] and u["epochs_applied"] >= 0
    assert ups[2]["reference"]["ref_update"] == ups[1]["reference"]["ref_update"] or ups[2]["reference"]["ref_update"] >= 1
    assert ups[2]["harvest_best"]["ECI"] >= ups[1]["harvest_best"]["ECI"]                     # 走行全体の最良は引き継がれ、下がらない
    assert run.B.reference is run.A.reference                                               # 比 ρ の状態（R-7）が同じ
    assert ups[2]["rho"] == pytest.approx(ups[1]["reference_update"]["rho"])               # 段階 A の最後の更新で決めた ρ を使う
    # 終了: C-12 の再生と最終の方策の確認
    st = json.load(open(os.path.join(cfg["run_dir"], "status.json"), encoding="utf-8"))
    assert st["phase"] == "B" and st["harvest_replay"]["F-1_passed"] is True
    assert st["final_policy_check"]["match_expected_base"] is True


def test_ut_v9_7_refuses_resume_and_stop_file_at_start_without_touching_files(tmp_path):
    cfg = _cfg(tmp_path)
    cfg_path = str(tmp_path / "cfg.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in cfg.items() if k in config.DEFAULTS or k in config.OPTIONAL_KEYS}, f)
    with pytest.raises(SystemExit):
        ts9.main(["--config", cfg_path, "--resume"])
    assert not os.path.exists(cfg["run_dir"]) and not os.path.exists(cfg["model_dir"])              # 1 バイトも書かない
    os.makedirs(cfg["run_dir"])
    open(os.path.join(cfg["run_dir"], "STOP"), "w").write("x")
    before = sorted((n, os.path.getsize(os.path.join(cfg["run_dir"], n))) for n in os.listdir(cfg["run_dir"]))
    with pytest.raises(SystemExit):
        ts9.main(["--config", cfg_path])
    after = sorted((n, os.path.getsize(os.path.join(cfg["run_dir"], n))) for n in os.listdir(cfg["run_dir"]))
    assert before == after and not os.path.exists(cfg["model_dir"])
    os.remove(os.path.join(cfg["run_dir"], "STOP"))
    open(os.path.join(cfg["run_dir"], "updates.jsonl"), "w").write("{}\n")                       # 記録のある走行ディレクトリでも拒否（C-15）
    with pytest.raises(SystemExit):
        ts9.main(["--config", cfg_path])
    assert open(os.path.join(cfg["run_dir"], "updates.jsonl")).read() == "{}\n"
