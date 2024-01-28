import sys
import pygame
import carla
import numpy as np
import random
import math
import cv2
from agents.navigation.global_route_planner import GlobalRoutePlanner

# CARLA Debug Draw (to see waypoints and vectors)
DEBUG_DRAW = True

# Carla path
sys.path.append('~/Documents/CARLA_0.9.14/PythonAPI/carla')

# Carla connection
client = carla.Client('localhost', 2000)

# Collision callback
def on_collision(event):
    global fail
    fail = 1

# Lane invsion callback
def on_lane_invasion(event):
    print(f"\nLane Invasion!\n")

# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj):
    # (Semantic segmentation) Encoded red channel convert to RGB to tag all elements in the scene
    # data.convert(carla.ColorConverter.CityScapesPalette)
    
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))

# Control object to manage vehicle controls
class ControlObject(object):
    def __init__(self, veh):
        # Control parameters to store the control state
        self._vehicle = veh
        self._throttle = 0
        self._brake = 0
        self._steer = 0
        self._steer_cache = 0
        # A carla.VehicleControl object is needed to alter the
        # vehicle's control state
        self._control = carla.VehicleControl()

    # Check for key press events in the PyGame window
    # and define the control state
    def parse_control(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self._vehicle.set_autopilot(False)
            if event.key == pygame.K_UP:
                self._throttle = 1
            if event.key == pygame.K_DOWN:
                self._brake = 1
            if event.key == pygame.K_RIGHT:
                self._steer = 1
            if event.key == pygame.K_LEFT:
                self._steer = -1
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                self._throttle = 0
            if event.key == pygame.K_DOWN:
                self._brake = 0
            if event.key == pygame.K_RIGHT or event.key == pygame.K_LEFT:
                self._steer = 0

    # Process the current control state, change the control parameter
    # if the key remains pressed
    def process_control(self):
        self._control.throttle = self._throttle
        self._control.brake = self._brake
        self._control.steer = self._steer

        # Apply the control parameters to the ego vehicle
        self._vehicle.apply_control(self._control)
    
    def get_speed_kmh(self):
        velocity = self._vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed

# Basic world configuration
def setup_carla_world():
    world = client.get_world()
    world = client.load_world('Town02')
    world.set_weather(carla.WeatherParameters.Default)

    return world

world = setup_carla_world()

# Spawn points
spawn_points = world.get_map().get_spawn_points()
start_point = spawn_points[19]

# Basic vehicle configuration
def setup_agent_vehicle(start_point):
    vehicle_bp = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    vehicle_bp.set_attribute('color', '0,0,0')
    vehicle = world.try_spawn_actor(vehicle_bp, start_point)

    return vehicle

vehicle = setup_agent_vehicle(start_point)

# Euclidian distance formula
def euclidean_distance(p1_x, p1_y, p2_x, p2_y):
    loc_p1 = np.array([p1_x, p1_y])
    loc_p2 = np.array([p2_x, p2_y])
    distance = np.linalg.norm(loc_p1 - loc_p2)

    return distance

# Collision detector
def setup_sensor_collision():
    collision_sensor_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=vehicle)

    return collision_sensor

collision_sensor = setup_sensor_collision()

# Listen to collision sensor callback
collision_sensor.listen(lambda event: on_collision(event))

# Lane invasion detector
def setup_sensor_lane_invasion():
    lane_invasion_bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
    lane_invasion_sensor = world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=vehicle)

    return lane_invasion_sensor

lane_invasion_sensor = setup_sensor_lane_invasion()

# Listen to lane invasion sensor
lane_invasion_sensor.listen(lambda event: on_lane_invasion(event))

# Target waypoint index
pos_wp_ends = 1

# List of wp targets
wp_ends = [19, 46, 33, 59]

# Route maker from car location to destination
def route_maker(i):
    # Route Planner
    grp = GlobalRoutePlanner(world.get_map(), 2)

    # Origin: Car position
    origin = vehicle.get_transform().location       

    # Route of waypoints (list)
    route = grp.trace_route(origin, spawn_points[wp_ends[i]].location)
    
    return route

route = route_maker(pos_wp_ends)

# Camera config

# Third person camera
camera_init_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# Start camera with PyGame callback
camera.listen(lambda image: pygame_callback(image, renderObject))

# Get camera dimensions
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

# Instantiate objects for rendering and vehicle control
renderObject = RenderObject(image_w, image_h)
controlObject = ControlObject(vehicle)

# Initialise the display
pygame.init()
gameDisplay = pygame.display.set_mode((image_w, image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)

# Draw black to the display
gameDisplay.fill((0, 0, 0))
gameDisplay.blit(renderObject.surface, (0, 0))
pygame.display.flip()

# Game loop
crashed = False
font = pygame.font.Font(None, 36)

# Target waypoint
pos = 0
target_wp = route[pos]
route_draw = route

# Waypoint counter
wp_count = -1

# Lap counter
lap_count = 0

# Case of collision or lose wp target
fail = 0

# Traffic signs
speed_limit = 30

while not crashed:
    # Advance the simulation time
    world.tick()        

    # Update the display
    gameDisplay.blit(renderObject.surface, (0, 0))

    # Calculate the distance between vehicle and target   
    distance_target = euclidean_distance(vehicle.get_transform().location.x, vehicle.get_transform().location.y, target_wp[0].transform.location.x, target_wp[0].transform.location.y)

    # The car lose the route
    if distance_target > 6.5:
        fail = 1

    # When the car come to the target choose a new one
    if distance_target <= 2:
        if pos+1 > len(route)-1:
            if pos_wp_ends == 3:
                pos_wp_ends = 0
                lap_count += 1
            else:
                pos_wp_ends += 1
            
            route = route_maker(pos_wp_ends)
            pos = 0
        else:
            pos += 1

        # Change next target
        target_wp = route[pos]

        # Cut route
        route_draw = route_draw[pos:]

        wp_count += 1

    # Car speed
    vehicle_speed = vehicle.get_velocity()
    velocidad_kmh = 3.6 * np.sqrt(vehicle_speed.x**2 + vehicle_speed.y**2)
    
    text = font.render(f"Speed: {velocidad_kmh:.2f} km/h", True, (255, 255, 255))
    gameDisplay.blit(text, (10, image_h - 40))

    # Vehicle location
    vehicle_location = vehicle.get_transform().location

    # Car vector
    forward_vector = vehicle.get_transform().get_forward_vector()
    scaled_forward_vector = forward_vector * 5.0
    vector_end = vehicle_location + carla.Location(x=scaled_forward_vector.x, y=scaled_forward_vector.y, z=0.0)

    # Angle between car orientation (yaw) and road tangent (cos(x))
    angle = forward_vector.get_vector_angle(route[pos][0].transform.get_forward_vector())

    # Env parameters
    print(f'{lap_count = }   {wp_count = }   distance_target = {distance_target:.3f}    angle(car,road) = {angle:.3f}    {fail = }   speed = {velocidad_kmh:.2f} kmh   {speed_limit = }     traffic light: {vehicle.is_at_traffic_light() } = {vehicle.get_traffic_light().state if vehicle.is_at_traffic_light() else 0}')

    # Traffic Signs
    if target_wp[0].get_landmarks(50.0, True):
        for sign in target_wp[0].get_landmarks(50.0, True):
            # Get specific car lane signs
            if target_wp[0].road_id == sign.road_id:
                if sign.name.split("_")[0] == "Speed":
                    speed_limit = sign.value
              
    # Graphic draws
    if DEBUG_DRAW:
        for i in range(pos+1,len(route)):
            world.debug.draw_point(carla.Location(x=route[i][0].transform.location.x, y=route[i][0].transform.location.y, z=0.8),
                            color=carla.Color(r=255, g=0, b=0),
                            life_time=0.05)

        world.debug.draw_point(carla.Location(x=vehicle_location.x, y=vehicle_location.y, z=0.8),
                            color=carla.Color(r=0, g=0, b=255),
                            life_time=0.05)
            
        world.debug.draw_arrow(carla.Location(x=vehicle_location.x, y=vehicle_location.y, z=0.8),
                            vector_end,
                            life_time=0.04)

        world.debug.draw_arrow(carla.Location(x=vehicle_location.x, y=vehicle_location.y, z=0.8),
                            carla.Location(x=target_wp[0].transform.location.x, y=target_wp[0].transform.location.y, z=0.8),
                            color=carla.Color(r=0, g=0, b=255),
                            life_time=0.04)

    pygame.display.flip()

    # Process the current control state
    controlObject.process_control()

    # Collect key press events
    for event in pygame.event.get():

        # If the window is closed, break the while loop
        if event.type == pygame.QUIT:
            crashed = True

        # Parse effect of key press event on control state
        controlObject.parse_control(event)
        if event.type == pygame.KEYUP:
            # TAB key switches vehicle
            if event.key == pygame.K_TAB:
                vehicle.set_autopilot(True)
                vehicle = vehicle
                # Ensure the vehicle is still alive (might have been destroyed)
                if vehicle.is_alive:
                    # Stop and remove the camera
                    camera.stop()
                    camera.destroy()

                    # Spawn a new camera and attach it to the new vehicle
                    controlObject = ControlObject(vehicle)
                    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
                    camera.listen(lambda image: pygame_callback(image, renderObject))

                    # Update the PyGame window
                    gameDisplay.fill((0, 0, 0))
                    gameDisplay.blit(renderObject.surface, (0, 0))
                    pygame.display.flip()

# Delete scene elements
for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()

# Stop the camera and quit PyGame after exiting the game loop
camera.stop()
pygame.quit()
