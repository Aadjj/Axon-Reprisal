import os
import logging
from stable_baselines3 import DQN
from src.response_env import ThreatResponseEnv
import numpy as np


def train_agent():
    logging.basicConfig(level=logging.INFO)
    os.makedirs("models", exist_ok=True)

    env = ThreatResponseEnv()

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500
    )

    logging.info("Starting RL Agent training...")
    model.learn(total_timesteps=10000)

    model.save("models/dqn_threat_response")
    logging.info("RL Agent saved to models/dqn_threat_response.zip")


if __name__ == "__main__":
    train_agent()