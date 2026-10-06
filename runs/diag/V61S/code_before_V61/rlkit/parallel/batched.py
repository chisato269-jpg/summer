"""rlkit.parallel.batched — 組ごとの子プロセスを束ねるベクトル環境（現行 batched.SlotVecEnv を移植）。

組番号の順に並んだ N 個の子プロセス（spawn）。終端での自動 reset は SB3 の SubprocVecEnv と同じだが、
reset() から数えて steps_per_collection 回目の step（1 回の採取の最後）だけは自動 reset しない
（次の採取で親が明示的に reset するため、使われないエミュレータを起動しない）。
"""
import multiprocessing as mp

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rlkit.parallel import slot_worker


class SlotVecEnvError(RuntimeError):
    pass


class SlotVecEnv(VecEnv):
    def __init__(self, slot_indices, env_kwargs, steps_per_collection, factory="real"):
        slot_indices = list(slot_indices)
        if slot_indices != sorted(set(slot_indices)):
            raise ValueError(f"組番号は重複なく昇順に並べる: {slot_indices}")
        self.slot_indices = slot_indices
        self.steps_per_collection = steps_per_collection
        self._steps_since_reset = 0
        self.waiting = False
        self.closed = False
        ctx = mp.get_context("spawn")
        self.remotes, self.processes = [], []
        for k in slot_indices:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=slot_worker.worker, args=(child, parent, factory, k, dict(env_kwargs)), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.processes.append(p)
        try:
            spaces_list = [self._recv(r) for r in self.remotes]
        except SlotVecEnvError:
            self.close()
            raise
        observation_space, action_space = spaces_list[0]
        super().__init__(len(slot_indices), observation_space, action_space)

    def _recv(self, remote):
        try:
            status, payload = remote.recv()
        except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as exc:
            raise SlotVecEnvError(f"組の子プロセスとの通信が切れた: {exc!r}")
        if status != "ok":
            raise SlotVecEnvError(f"組の子プロセスで失敗した: {payload}")
        return payload

    def reset(self):
        for i, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[i], self._options[i])))
        results = [self._recv(r) for r in self.remotes]
        obs, self.reset_infos = zip(*results)
        self._reset_seeds()
        self._reset_options()
        self._steps_since_reset = 0
        return np.stack(obs)

    def step_async(self, actions):
        self._steps_since_reset += 1
        last = self._steps_since_reset >= self.steps_per_collection
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", (action, last)))
        self.waiting = True

    def step_wait(self):
        results = [self._recv(r) for r in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), list(infos)

    def close(self):
        """全組へ終了を伝え、終わらなければ PID で終了させる。

        S-11（指示書 V5-S 第 2-5 節）: 親から終了させた分も組ごとの起動記録へ 1 行残し、
        開始・BACnet 応答・終了の対応が取れるようにする。
        """
        if self.closed:
            return
        from rlkit.parallel import launch
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for k, p in zip(self.slot_indices, self.processes):
            p.join(timeout=30)
            killed = p.is_alive()
            if killed:
                p.kill()
                p.join(timeout=10)
            launch.log_slot_event(k, "parent_closed_worker", worker_pid=p.pid, killed=killed,
                                  exitcode=p.exitcode)
        self.closed = True

    def _targets(self, indices):
        return [self.remotes[i] for i in self._get_indices(indices)]

    def get_attr(self, attr_name, indices=None):
        targets = self._targets(indices)
        for r in targets:
            r.send(("get_attr", attr_name))
        return [self._recv(r) for r in targets]

    def has_attr(self, attr_name):
        for r in self.remotes:
            r.send(("has_attr", attr_name))
        return all(self._recv(r) for r in self.remotes)

    def set_attr(self, attr_name, value, indices=None):
        targets = self._targets(indices)
        for r in targets:
            r.send(("set_attr", (attr_name, value)))
        for r in targets:
            self._recv(r)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        targets = self._targets(indices)
        for r in targets:
            r.send(("env_method", (method_name, method_args, method_kwargs)))
        return [self._recv(r) for r in targets]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
