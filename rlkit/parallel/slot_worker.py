"""rlkit.parallel.slot_worker — 組ごとの子プロセス（現行 slot_worker.py を移植）。

batched.SlotVecEnv が spawn で起動する。命令と応答は SB3 の SubprocVecEnv の worker と同じ意味で、次の 2 点が違う。
  応答を ("ok", 値) / ("error", 追跡情報) の形にし、子プロセスでの失敗を親へ伝える
  "step" に「1 回の採取の最後のステップか」を添え、最後のステップでは終端でも自動 reset しない
子プロセスを軽く保つため、このモジュールは torch・stable_baselines3 を import しない。
"""
import traceback


def make_env(factory, slot_index, env_kwargs):
    if factory == "real":
        from rlkit.env import SummerEnv
        from rlkit.parallel import slots
        slot = slots.make_slot(slot_index)
        return SummerEnv(slot=slot, **env_kwargs), slot
    if factory == "mock":
        from rlkit.mockenv import MockSummerEnv
        return MockSummerEnv(**env_kwargs), None
    if factory == "real_v8":                       # V8-S（rlkit/v8/env.py）
        from rlkit.parallel import slots
        from rlkit.v8.env import SummerEnvV8
        slot = slots.make_slot(slot_index)
        return SummerEnvV8(slot=slot, **env_kwargs), slot
    if factory == "mock_v8":                       # V8-S の試験用の模擬環境
        from rlkit.v8.mockenv import MockSummerEnvV8
        return MockSummerEnvV8(**env_kwargs), None
    raise ValueError(f"未知の環境の作り方: {factory}")


def worker(remote, parent_remote, factory, slot_index, env_kwargs):
    parent_remote.close()
    env = slot = None
    try:
        env, slot = make_env(factory, slot_index, env_kwargs)
        remote.send(("ok", (env.observation_space, env.action_space)))
    except Exception:
        remote.send(("error", traceback.format_exc()))
        remote.close()
        return
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                action, no_auto_reset = data
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                reset_info = {}
                if done:
                    info["terminal_observation"] = obs
                    if not no_auto_reset:
                        obs, reset_info = env.reset()
                remote.send(("ok", (obs, reward, done, info, reset_info)))
            elif cmd == "reset":
                seed, options = data
                kw = {"options": options} if options else {}
                obs, reset_info = env.reset(seed=seed, **kw)
                remote.send(("ok", (obs, reset_info)))
            elif cmd == "env_method":
                name, args, kwargs = data
                remote.send(("ok", getattr(env, name)(*args, **kwargs)))
            elif cmd == "get_attr":
                remote.send(("ok", getattr(env, data)))
            elif cmd == "has_attr":
                remote.send(("ok", hasattr(env, data)))
            elif cmd == "set_attr":
                setattr(env, data[0], data[1])
                remote.send(("ok", None))
            elif cmd == "rpm_stats":     # V7.1-S C-13: 一括読み取りの計数（使っていなければ空）
                try:
                    from rlkit import comm_rpm
                    remote.send(("ok", dict(comm_rpm.COMM_RPM_STATS)))
                except Exception:
                    remote.send(("ok", {}))
            elif cmd == "close":
                break
            else:
                remote.send(("error", f"未知の命令: {cmd}"))
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception:
        try:
            remote.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            env.close()  # 組のエミュレータを終了させる（EpisodeDriver.close）
        except Exception:
            pass
        if slot is not None:
            from rlkit.parallel import slots
            slots.runtime_for(slot).release()
        try:
            remote.close()
        except Exception:
            pass
