import os
import sys
sys.path.append('../')

from carla_env import CarlaEnv
from stable_baselines3 import A2C

NUM_EPISODES = 5

models_dir = "../models/A2C/"
model_path = f"{models_dir}/A2C-Agent-500K.zip"

# Start the enviroment 
carla_env = CarlaEnv()
carla_env.reset()

# Load the specific model
model = A2C.load(model_path, env=carla_env)

print("\nA2C AGENT")
print("_________")
for i in range(NUM_EPISODES):
    obs = carla_env.reset()
    done = False

    while not done:
        action, _space = model.predict(obs)
        obs, reward, done, info = carla_env.step(action)

    print(f"Episode {i}: {reward =}")

