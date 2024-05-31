import os
import sys
sys.path.append('../')

from carla_env import CarlaEnv
from stable_baselines3 import PPO

NUM_EPISODES = 5

models_dir = "../models/PPO/"
model_path = f"{models_dir}/PPO-Agent-500K.zip"

# Start the enviroment 
carla_env = CarlaEnv()
carla_env.reset()

# Load the specific model
model = PPO.load(model_path, env=carla_env)

for _ in range(NUM_EPISODES):
    obs = carla_env.reset()
    done = False

    while not done:
        action, _space = model.predict(obs)
        obs, reward, done, info = carla_env.step(action)

    print(reward)


