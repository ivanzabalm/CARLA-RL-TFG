import random
import time
import math 
import cv2
import carla
import gym
import numpy as np
from gym import spaces
from agents.navigation.global_route_planner import GlobalRoutePlanner

SHOW_CAMERA = True     # Watch car camera while training

class CarlaEnv(gym.Env):
    def __init__(self):
        self.route = None       # List of waypoint that car must follow
        self.pos = None         # Target waypoint index
        self.target_wp = None   # Target waypoint object
        self.lap_count = 0      # Circuit laps
        self.wp_count = 0       # Waypoints counter
        self.wp_ends = [19, 46, 33, 59] # Routes ends (specific waypoint)
        self.pos_wp_ends = 1    # End waypoints index
        self.change_route_count = 0 # Route change counter
        self.lane_inv_count = 0 # Lane invasion counter 
        self.params_dict = {}   # Parameters values dictionary for reward function
        self.actors_lst = []    # Actors list refrence to mange instances (vehicle, cameras, sensors)
        self.collision_hist = []    # Register if agent commit a collision event 
        
        # Observation space (Normalized from 0 to 1)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(240, 320, 3), dtype=np.float32)
		
        # Action space (11 for steering: [0-10], 5 for throttle: [0-4])
        self.action_space = spaces.MultiDiscrete([11,5])

        # Connect to CARLA and config World
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.world = self.client.load_world('Town02')
        self.world.set_weather(carla.WeatherParameters.Default)

		# Clean CARLA Simulator (To ensure an empty scene)
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()

		# Car blueprint
        self.vehicle_bp = self.world.get_blueprint_library().filter("model3")[0]
        self.vehicle_bp.set_attribute('color', '0,0,0')

		# Spawn points
        self.spawn_points = self.world.get_map().get_spawn_points()

    # Route maker from origin to destination
    def __route_maker(self, i):
        # Route Planner
        grp = GlobalRoutePlanner(self.world.get_map(), 2)

        # Origin: Car position
        origin = self.vehicle.get_transform().location       

        # Route of waypoints (list)
        route = grp.trace_route(origin, self.spawn_points[self.wp_ends[i]].location)

        # Create the route
        self.route = route
            
    # Calculate parameteres values
    def __get_params_values(self):
        # Distance from Car to Waypoint target
        self.params_dict["dist_target"] = self.__euclidean_distance(self.vehicle.get_transform().location.x, self.vehicle.get_transform().location.y, self.target_wp[0].transform.location.x, self.target_wp[0].transform.location.y)
                    
        # Param 2: Angle between Car vetor and Road vector
        self.params_dict["angle_cr"] = self.vehicle.get_transform().get_forward_vector().get_vector_angle(self.route[self.pos][0].transform.get_forward_vector())

        # Param 3: Speed
        self.params_dict["speed"] = 3.6 * np.sqrt(self.vehicle.get_velocity().x**2 + self.vehicle.get_velocity().y**2)

        # Param 4: Done (Fail)
        if len(self.collision_hist) > 0 or self.params_dict["dist_target"] > 6.5:
            self.params_dict["done"] = 1
        else:
            self.params_dict["done"] = 0

    # Define the reward points obtain from agent actions
    def __reward_func(self):
        reward = 0
        done = False

        # Big reward for done a route
        if self.change_route_count > 0:
            reward += 50
            self.change_route_count -= 1

        # Big reward to agent for doing a lap
        if self.lap_count > 0: 
            reward += 200

        # Reward agent for number obtain of waypoints
        if self.wp_count > 0:
            reward += 1
            self.wp_count -= 1

        # Reward agent for be more centered on the lane
        if self.params_dict["dist_target"] < 3:
            reward += 1
            
        # Reward agent for drive more straight 
        if self.params_dict["angle_cr"] < 0.2:
            reward += 1
        else:
            reward -= 4

        # Reward for appropriate speed (15-35 km/h)
        if self.params_dict["speed"] > 15 and self.params_dict["speed"] <= 30:
            reward += 1
        else:
            reward -= 4

        # Punish the agent for lane invasion
        if self.lane_inv_count > 0:
            reward -= 10
            self.lane_inv_count -= 1

        # Punish the agent if is out of lane
        if self.params_dict["done"] == 1:
                done = True
                reward -= 100

                # Reset
                self.__clean_scene()
                self.params_dict["done"] =  0

        return reward, done
    
    # Euclidean distance formula between two points 
    def __euclidean_distance(self, p1_x, p1_y, p2_x, p2_y):
        loc_p1 = np.array([p1_x, p1_y])
        loc_p2 = np.array([p2_x, p2_y])
        distance = np.linalg.norm(loc_p1 - loc_p2)

        return distance
    
    # Callback function to print the camera image
    def __process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((240, 320, 4))[:, :, :3]
        self.front_camera = i

    # Clean all elements from scene (vehicle, actors and cameras)
    def __clean_scene(self):
        for actor in self.actors_lst:
            actor.destroy()

        # Reset params
        self.actors_lst = []
        self.collision_hist = []

        if SHOW_CAMERA:
            cv2.destroyAllWindows()

    def __collision_data(self, event):
        self.collision_hist.append(event)

    def __on_lane_invasion(self, event):
        self.lane_inv_count += 1

    def reset(self):
        super().reset(seed=None)

        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        
        # Vehicle instance
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_points[19])
        self.actors_lst.append(self.vehicle)

        # Create initial route
        self.pos_wp_ends = 1
        self.__route_maker(self.pos_wp_ends)

        # Segmentation camera
        sem_cam_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        sem_cam_bp.set_attribute("image_size_x", "320")
        sem_cam_bp.set_attribute("image_size_y", "240")

        cam_transform = carla.Transform(carla.Location(x=1.0, y=0, z=1.40))
        self.sem_cam = self.world.spawn_actor(sem_cam_bp, cam_transform, attach_to=self.vehicle)
        self.sem_cam.listen(lambda data: self.__process_img(data))

        self.actors_lst.append(self.sem_cam)
        
        time.sleep(2)

        if SHOW_CAMERA:
            cv2.namedWindow('Sem Camera',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Sem Camera', self.front_camera)
            cv2.waitKey(1)

        # Collision sensor
        sensor_col = self.world.get_blueprint_library().find("sensor.other.collision")
        self.sensor_col = self.world.spawn_actor(sensor_col, carla.Transform(), attach_to=self.vehicle)
        self.sensor_col.listen(lambda event: self.__collision_data(event))
        
        self.actors_lst.append(self.sensor_col)

        # Lane invasion sensor
        lane_invasion_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor_lane_invasion = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)
        self.sensor_lane_invasion.listen(lambda event: self.__on_lane_invasion(event))

        # Reset class vars
        self.step_counter = 0
        self.route = None       
        self.pos = None         
        self.target_wp = None   
        self.params_dict = {}
        self.collision_hist = []
        self.lane_inv_count = 0
        self.wp_count = 0
        self.lap_count = 0
        self.pos_wp_ends = 1

        # Initial car state
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera/255.0
    
    def step(self, actions):
        self.step_counter +=1
            
        if self.step_counter == 1:
            # Initial route
            self.__route_maker(self.pos_wp_ends)

        # Actions selection (by agent)
        sel_steer = actions[0]
        sel_throttle = actions[1]

        # Steering actions list
        actions_steer = [-0.8, -0.4, -0.3, -0.2, 0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.8]
        steer = actions_steer[sel_steer]

        # Throttle actions list
        actions_throttle = [0.25, 0.5, 0.75, 1.0]

        if sel_throttle == 0:
            # Brake
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=steer, brake=actions_throttle[3]))
        else:
            # Throttle
            self.vehicle.apply_control(carla.VehicleControl(throttle=actions_throttle[sel_throttle-1], steer=steer, brake=0.0))

        cam = self.front_camera

        if SHOW_CAMERA:
            cv2.imshow('Sem Camera', cam)
            cv2.waitKey(1)

        while self.route == None:
            time.sleep(0.01)

        # Update target waypoint selector
        if self.step_counter == 1:
            self.pos = 0
            self.target_wp = self.route[self.pos]
        else:
            if self.params_dict["dist_target"] <= 2:
                # In case of reaching the end
                if self.pos+1 > len(self.route)-1:
                    if self.pos_wp_ends == 3:
                        self.pos_wp_ends = 0
                        self.lap_count += 1
                    else:
                        self.pos_wp_ends += 1

                    self.route = self.__route_maker(self.pos_wp_ends)
                    self.pos = 0
                    self.change_route_count += 1

                # Until not reaching it
                else:
                    self.pos += 1

                self.target_wp = self.route[self.pos]
                self.wp_count += 1                    

        # Calculate params values
        self.__get_params_values()

        # Calculate the reward 
        reward, done = self.__reward_func()
    
        # Stable Baseline 3 Format
        return cam/255.0, reward, done, {}