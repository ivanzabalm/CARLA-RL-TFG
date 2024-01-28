import os
from carla_env import CarlaEnv
from stable_baselines3 import A2C, PPO

NUM_EPISODES = 10
NUM_TIMESTEPS = 100000 

logs_dir = "logs"
models_dir = "models/PPO/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logs_dir):
	os.makedirs(logs_dir)

# Start the enviroment 
carla_env = CarlaEnv()
carla_env.reset()

# Define RL Algorithm
model = PPO('MlpPolicy', carla_env, verbose=1, tensorboard_log=logs_dir)

# Training
for i in range(1,NUM_EPISODES+1):
	model.learn(total_timesteps=NUM_TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{NUM_TIMESTEPS*i}")