import gym
from stable_baselines3 import PPO

class AdversarialCyberEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)  # Defender actions
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6,))

    def step(self, blue_action):
        reward = 1 if blue_action == 2 else -1
        return self.observation_space.sample(), reward, False, {}


def train_battle_hardened_agent():
    env = AdversarialCyberEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)
    model.save("models/battle_hardened_axon")


if __name__ == "__main__":
    train_battle_hardened_agent()