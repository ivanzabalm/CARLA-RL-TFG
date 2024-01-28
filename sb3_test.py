import os
from carla_env import CarlaEnv
from stable_baselines3 import PPO

NUM_EPISODES = 4

models_dir = "models/PPO/"
model_path = f"{models_dir}/300000"

# Start the enviroment 
carla_env = CarlaEnv()
carla_env.reset()

# Load the specific model
model = PPO.load(model_path, env=carla_env)

for _ in range(NUM_EPISODES):
    obs = carla_env.reset()
    action, _space = model.predict(obs)
    obs, reward, done, info = carla_env.step(action)
    print(reward)