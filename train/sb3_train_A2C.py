import os
import sys
sys.path.append('../')

from carla_env import CarlaEnv
from stable_baselines3 import A2C

NUM_EPISODES = 10
NUM_TIMESTEPS = 100_000

logs_dir = "../logs"
models_dir = "../models/A2C/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logs_dir):
	os.makedirs(logs_dir)

# Start the enviroment 
carla_env = CarlaEnv()
carla_env.reset()

# Define RL Algorithm
model = A2C('MlpPolicy', carla_env, verbose=1, learning_rate=0.0005, tensorboard_log=logs_dir)

# Training
for i in range(1,NUM_EPISODES+1):
	model.learn(total_timesteps=NUM_TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
	model.save(f"{models_dir}/A2C-Agent-{round((NUM_TIMESTEPS*i)/1000)}K")
