"""rlkit.v8.adopt — 方策の更新 L-6R（指示書 V8.2-S 第 3 章）: 最良の兄弟の軌跡の正確な再現。

環境が決定論的なので、方策の確率最大の選択が、ある軌跡の全ての状態で、その軌跡の選択と一致すれば、その方策を決定論的に走らせると同じ軌跡が
再現される（1 ステップ目の状態は同じで、選択が同じなら次の状態も同じ。378 ステップ目まで帰納的に成り立つ）。
更新ごとに、A_s > 0 の兄弟を A_s の大きい順に最大 3 本まで次を試し、1 本採用したら終える（MUST-16）。
  兄弟 s の軌跡の全ステップの「有効な判断」を教師とし、方策の出力のマスク付きの交差エントロピーを最小にする
  （Adam・学習率 3e-4・ミニバッチは全体・最大 300 エポック。試みごとに Adam の状態は新しくする）。
  各エポックの後に、全ての有効な判断で、確率最大の選択が兄弟の選択と一致し、かつその選択のロジットが（有効な選択肢の中の）2 番目より
  1e-3 以上大きいかを判定する。満たした時点で学習を終えて採用する（MUST-15: 一致率が高いだけでは採用しない）。
  300 エポックで満たさなければ学習前の重みに戻して次の兄弟を試す。3 本とも満たさなければ方策を変えない。
有効な判断: 1 択にマスクされた判断（確率最大の選択が必ずその 1 択になる）と、環境と次の観測に影響しない判断を除いたもの。
  設定温度は割合 0 のゾーンで環境に書き込まれず、観測にも入らない（観測の設定温度の特徴は割合 0 のとき 0）ので除く。
  割合 0 のゾーンの風量は、環境には書き込まれないが、観測のゾーン +20（直前の風量）に入り次のステップの観測を変えるので、含める
  （指示書の「環境に書き込まれない判断を除く」への補足。含めないと軌跡の再現が保証されない。V82S-a。V8.3-S の C83-2 で承認）。
蒸発温度は常に書き込まれ、観測にも入る。

V8.3-S（C83-1）: 損失・学習率・最大エポックを設定で選べる（損失 K-1 平均の交差エントロピー、K-3 余裕の損失、K-4 重み付きの交差エントロピー）。
既定（設定が無いとき）は V8.2-S と同じ K-1・3e-4・300。他の決まり（A_s の大きい順に最大 3 本・採用は 1 本・満たさなければ学習前の重みに戻す）は同じ。
"""
import time

import numpy as np
import torch

from rlkit.v8 import action as act8
from rlkit.v8 import policy as pol8

LEARNING_RATE = 3e-4
MAX_EPOCHS = 300
LOGIT_MARGIN = 1e-3
MAX_SIBLINGS_TRIED = 3
LOSS_KINDS = ("k1", "k3", "k4")
K3_TARGET_MARGIN = 0.05                      # K-3: 余裕の目標（これより小さい判断だけが損失に入る）
K4_MISMATCH_WEIGHT = 100.0                   # K-4: 学習の開始の時点で基準を満たしていなかった判断の重み（ほかは 1）


def effective_decision_mask(actions, flat_masks):
    """軌跡の有効な判断 (T, 80)。選択肢が 1 つの判断と、割合 0 のゾーンの設定温度を除く。"""
    actions = np.asarray(actions)
    n_opt = act8.n_valid_options(np.asarray(flat_masks))
    eff = n_opt > 1
    on = act8.duty_indices(actions) > 0
    sp = act8.dim_slice("zone_setpoint")
    eff[:, sp] &= on                                         # 割合 0 のゾーンの設定温度は環境にも観測にも影響しない
    return eff


def decision_margins(logits, masks, actions):
    """各判断で、選択したロジットから、有効な他の選択肢の最大のロジットを引いた値 (T, 80)。有効な選択肢が 1 つなら +inf。"""
    logits = np.asarray(logits, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    T = logits.shape[0]
    out = np.full((T, act8.ACTION_DIMS), np.inf)
    for d, (o, n) in enumerate(zip(act8.OFFSETS, act8.NVEC)):
        seg = np.where(masks[:, o:o + n], logits[:, o:o + n], -np.inf)
        chosen = np.take_along_axis(seg, actions[:, d:d + 1], axis=1)[:, 0]
        seg2 = seg.copy()
        np.put_along_axis(seg2, actions[:, d:d + 1], -np.inf, axis=1)
        other = seg2.max(axis=1)
        out[:, d] = np.where(np.isfinite(other), chosen - other, np.inf)
    return out


def decision_ok_matrix(policy, teacher, margin=LOGIT_MARGIN):
    """各判断が採用の基準（確率最大が一致し、ロジットの差が margin 以上）を満たすか (T, 80) bool（有効でない判断は True）。
    Returns: (ok 行列, 不一致の行列, 余裕不足の行列, 有効な判断の最小のロジット差)"""
    with torch.no_grad():
        logits = policy(torch.as_tensor(teacher["obs"], dtype=torch.float32)).numpy()
    masks = np.asarray(teacher["masks"], dtype=bool)
    actions = np.asarray(teacher["actions"], dtype=np.int64)
    eff = teacher["eff"]
    mine = pol8.masked_argmax(logits, masks)
    mismatch = (mine != actions) & eff
    m = decision_margins(logits, masks, actions)
    low = (m < margin) & eff
    min_m = float(m[eff].min()) if int(eff.sum()) else float("inf")
    return ~(mismatch | low), mismatch, low, min_m


def _summary(mismatch, low, min_m, n_eff):
    return {"ok": bool(not mismatch.any() and not low.any()), "n_effective": n_eff, "n_mismatch": int(mismatch.sum()),
            "n_margin_fail": int(low.sum()), "min_margin": min_m}


def check_match(policy, teacher, margin=LOGIT_MARGIN):
    """教師の軌跡に対し、方策の確率最大の選択が全ての有効な判断で一致し、ロジットの差が margin 以上かを調べる。
    Returns: dict(ok, n_effective, n_mismatch, n_margin_fail, min_margin)"""
    _ok, mismatch, low, min_m = decision_ok_matrix(policy, teacher, margin)
    return _summary(mismatch, low, min_m, int(teacher["eff"].sum()))


def make_teacher(obs, masks, actions):
    actions = np.asarray(actions, dtype=np.int64)
    return {"obs": np.asarray(obs, dtype=np.float32), "masks": np.asarray(masks, dtype=bool), "actions": actions,
            "eff": effective_decision_mask(actions, masks)}


def torch_margins(logits, masks, actions):
    """各判断で、選んだロジットから、有効な他の選択肢の最大のロジットを引いた値 (B, 80)（torch。勾配が通る）。
    有効な選択肢が 1 つの判断は大きな正の値になる（K-3 の損失に入らない）。"""
    big = 1.0e9
    masked = torch.where(masks, logits, torch.full_like(logits, -big))
    cols = []
    for d, (o, n) in enumerate(zip(act8.OFFSETS.tolist(), act8.NVEC.tolist())):
        seg = masked[:, o:o + n]
        a = actions[:, d:d + 1]
        chosen = torch.gather(seg, 1, a)[:, 0]
        onehot = torch.zeros_like(seg, dtype=torch.bool).scatter_(1, a, True)
        other = seg.masked_fill(onehot, -big).max(dim=1).values
        cols.append(chosen - other)
    return torch.stack(cols, dim=1)


def training_loss(kind, logits, masks, actions, eff, weights=None):
    """学習の損失（eff は有効な判断の bool (B, 80)）。
    k1: 交差エントロピーの平均（有効な判断の総数で割る）。
    k3: 余裕の損失 Σ_d max(0, 0.05 − m_d)（和であり、判断の総数で割らない。m_d は torch_margins）。
    k4: 重み付きの交差エントロピーの平均（weights (B, 80) は学習の開始の時点で固定。重みの和で割る）。"""
    effm = eff.to(logits.dtype)
    if kind == "k3":
        m = torch_margins(logits, masks, actions)
        return (torch.clamp(K3_TARGET_MARGIN - m, min=0.0) * effm).sum()
    logp = pol8.chosen_log_prob(pol8.per_dim_log_softmax(logits, masks), actions)
    if kind == "k1":
        return -(logp * effm).sum() / max(int(eff.sum()), 1)
    if kind == "k4":
        w = weights * effm
        return -(logp * w).sum() / w.sum().clamp(min=1e-12)
    raise ValueError(f"未知の損失: {kind}")


def k4_weights(ok_matrix, eff):
    """K-4 の重み (T, 80) float32: 学習の開始の時点で採用の基準を満たしていなかった有効な判断が 100、ほかが 1。"""
    bad = np.asarray(eff) & ~np.asarray(ok_matrix)
    return np.where(bad, K4_MISMATCH_WEIGHT, 1.0).astype(np.float32)


def fit_to_teacher(policy, teacher, lr=LEARNING_RATE, max_epochs=MAX_EPOCHS, margin=LOGIT_MARGIN, loss="k1"):
    """方策を教師の軌跡に合わせる（L-6R の 1 本ぶん）。合う（check_match が ok）と学習を終えて採用。
    合わなければ学習前の重みに戻す。Returns: dict(adopted, epochs, before, after, broken_max, seconds)。
    broken_max: 学習の開始の時点で基準を満たしていた有効な判断のうち、あるエポックの後に基準を満たさなくなった数の、全エポックでの最大。"""
    if loss not in LOSS_KINDS:
        raise ValueError(f"未知の損失: {loss}")
    t0 = time.time()
    eff_np = np.asarray(teacher["eff"])
    ok0, mm0, low0, min0 = decision_ok_matrix(policy, teacher, margin)
    before = _summary(mm0, low0, min0, int(eff_np.sum()))
    saved = {k: v.clone() for k, v in policy.state_dict().items()}
    if before["ok"]:
        return {"adopted": True, "epochs": 0, "before": before, "after": before, "broken_max": 0, "seconds": time.time() - t0}
    obs = torch.as_tensor(teacher["obs"], dtype=torch.float32)
    masks = torch.as_tensor(teacher["masks"], dtype=torch.bool)
    actions = torch.as_tensor(teacher["actions"], dtype=torch.long)
    eff = torch.as_tensor(eff_np)
    weights = torch.as_tensor(k4_weights(ok0, eff_np)) if loss == "k4" else None
    good0 = ok0 & eff_np                                              # 開始の時点で基準を満たしていた有効な判断
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    after = before
    broken_max = 0
    for epoch in range(1, max_epochs + 1):
        out = training_loss(loss, policy(obs), masks, actions, eff, weights)
        opt.zero_grad()
        out.backward()
        opt.step()
        okm, mm, low, min_m = decision_ok_matrix(policy, teacher, margin)
        broken_max = max(broken_max, int((good0 & ~okm).sum()))
        after = _summary(mm, low, min_m, before["n_effective"])
        if after["ok"]:
            policy.eval()
            return {"adopted": True, "epochs": epoch, "before": before, "after": after, "broken_max": broken_max,
                    "seconds": time.time() - t0}
    policy.load_state_dict(saved)                                    # 満たさなければ学習前の重みに戻す
    policy.eval()
    return {"adopted": False, "epochs": max_epochs, "before": before, "after": after, "broken_max": broken_max,
            "seconds": time.time() - t0}


def try_adopt(policy, siblings, adv, max_tried=MAX_SIBLINGS_TRIED, lr=LEARNING_RATE, max_epochs=MAX_EPOCHS, margin=LOGIT_MARGIN,
              loss="k1", on_tried=None):
    """A_s > 0 の兄弟を A_s の大きい順に最大 max_tried 本まで試し、1 本採用したら終える。siblings は長さ 12 のリストで、各要素は
    dict(obs, masks, actions)。on_tried(s, teacher, adopted) は 1 本の試みが終わるたびに呼ばれる（C83-3 の教師の保存用）。
    Returns: dict(adopted (兄弟の番号か None)、tried: 試みの記録のリスト)"""
    adv = np.asarray(adv, dtype=np.float64)
    order = [int(s) for s in np.argsort(-adv) if adv[s] > 0.0][:max_tried]
    tried = []
    for s in order:
        teacher = make_teacher(siblings[s]["obs"], siblings[s]["masks"], siblings[s]["actions"])
        r = fit_to_teacher(policy, teacher, lr, max_epochs, margin, loss)
        tried.append({"sibling": s, "A": float(adv[s]), "adopted": r["adopted"], "epochs": r["epochs"],
                      "seconds": r.get("seconds"), "broken_max": r.get("broken_max"),
                      "n_effective": r["before"]["n_effective"], "n_mismatch_before": r["before"]["n_mismatch"],
                      "n_margin_fail_before": r["before"]["n_margin_fail"], "min_margin_before": r["before"]["min_margin"],
                      "n_mismatch_after": r["after"]["n_mismatch"], "n_margin_fail_after": r["after"]["n_margin_fail"],
                      "min_margin_after": r["after"]["min_margin"]})
        if on_tried is not None:
            on_tried(s, teacher, r["adopted"])
        if r["adopted"]:
            return {"adopted": s, "tried": tried}
    return {"adopted": None, "tried": tried}
