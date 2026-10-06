import gymnasium as gym
import numpy as np
import torch.nn as nn
import os
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import BaseCallback

OUTPUT_INTERVAL = 100 #データ書き出し間隔
NO_LEARNING = False

# カスタムコールバッククラス
class CustomSaveCallback(BaseCallback):

    def __init__(self, save_freq, save_path, verbose=0):
        super(CustomSaveCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.reward = 0
        self.n_epsd = 0        
        self.hist_rwds = [] # 学習過程の記録
        self.episode_actions = []

    def _on_step(self) -> bool:
        # 報酬を積算
        self.reward += self.locals["rewards"][0]
        if self.n_epsd % OUTPUT_INTERVAL == 0:
            self.episode_actions.append(np.array(self.locals["actions"]))

        # エピソード終了時
        if self.locals["dones"][0]:
            if self.n_epsd % OUTPUT_INTERVAL == 0:
                # モデルを保存
                model_file = os.path.join(self.save_path, f"model_{self.n_epsd}.zip")
                self.model.save(model_file)
                # 行動を保存
                actions_array = np.array(self.episode_actions)
                self.episode_actions = []
                action_file = os.path.join(self.save_path, f"actions_{self.n_epsd}.npy")
                np.save(action_file, actions_array)

            print(f"Episode finished with total reward: {self.reward:.2f}")
            # 報酬を保存
            self.hist_rwds.append(self.reward)
            np.save(os.path.join(self.save_path, f"rewards.npy"), self.hist_rwds)
            self.reward = 0

            self.n_epsd += 1 # episode数をインクリメント

        return True


# ここから学習処理
model_file_path = "./model.zip"

# 環境の作成
TARGET = "Shizuku2EnvironmentSample"
register(
    id=TARGET,
    entry_point=f"{TARGET}:{TARGET}",
)
env = gym.make(
    TARGET, 
    backup_interval=OUTPUT_INTERVAL,
    eval_mode=NO_LEARNING)

# モデルの定義
if os.path.exists(model_file_path):
    print("Loaded an existing model. Resuming training...")
    model = PPO.load(model_file_path, env=env)
else:
    print("No existing model found. Creating a new model and starting training...")
    policy_kwargs = dict(
        net_arch=[512, 256, 128, 64],
        activation_fn=nn.ReLU,
    )
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma=1.0)

print(model.policy)

# 学習
if not NO_LEARNING:
    print("Starting reinforcement learning...")
    custom_callback = CustomSaveCallback(save_freq=1, save_path="./saved_models/")
    model.learn(total_timesteps=48*7*10000, callback=custom_callback)
    print("Reinforcement learning completed.")
# 結果確認
else:
    print("Starting evaluation run...")
    obs, info = env.reset()

    terminated = False
    truncated = False
    total_reward = 0.0

    while not terminated and not truncated:
        # 決定論的な行動を選択
        action, _states = model.predict(obs, deterministic=True)
        # 環境を1ステップ進める
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Evaluation finished. Total Reward: {total_reward}")
            break

    print("Evaluation run completed.")