import sys
import pygame
import carla
import numpy as np
import random
import math
import cv2
from agents.navigation.global_route_planner import GlobalRoutePlanner

# CARLA Debug Draw (to see waypoints and vectors)
DEBUG_DRAW = False

fail = 0

# Carla path
sys.path.append('~/Documents/CARLA_0.9.14/PythonAPI/carla')

# Carla connection
client = carla.Client('localhost', 2000)

# Collision callback
def on_collision(event):
    global fail
    fail = 0

# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj):
    # Encoded red channel convert to RGB to tag all elements in the scene
    data.convert(carla.ColorConverter.CityScapesPalette)
    
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
start_point = spawn_points[21]

# Basic vehicle configuration
def setup_agent_vehicle(start_point):
    vehicle_bp = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    vehicle_bp.set_attribute('color', '0,0,0')
    vehicle = world.try_spawn_actor(vehicle_bp, start_point)

    return vehicle

vehicle = setup_agent_vehicle(start_point)

# Collision detector
def setup_sensor_collision():
    collision_sensor_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor_location = carla.Location(x=0.0, y=0.0, z=0.0)
    collision_sensor = world.spawn_actor(collision_sensor_bp, carla.Transform(collision_sensor_location), attach_to=vehicle)

    return collision_sensor

collision_sensor = setup_sensor_collision()

# Listen to sensor callback
collision_sensor.listen(lambda event: on_collision(event))

# Route maker from point A to point B
def route_maker():
    # Route Planner
    grp = GlobalRoutePlanner(world.get_map(), 2)
    point_a = carla.Location(x=start_point.location.x, y=start_point.location.y, z=start_point.location.z)
    point_b = carla.Location(x=spawn_points[2].location.x, y=spawn_points[2].location.y, z=spawn_points[2].location.z)
    route = grp.trace_route(point_a, point_b)

    return route

route = route_maker()

# Camera config
camera_init_trans = carla.Transform(carla.Location(x=1.0, y=0, z=1.40))
camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
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
pos = 2
target_wp = route[pos]
route_draw = route

def euclidean_distance(p1_x, p1_y, p2_x, p2_y):
    loc_p1 = np.array([p1_x, p1_y])
    loc_p2 = np.array([p2_x, p2_y])
    distance = np.linalg.norm(loc_p1 - loc_p2)

    return distance

while not crashed:
    # Advance the simulation time
    world.tick()        

    # Update the display
    gameDisplay.blit(renderObject.surface, (0, 0))
    
    # Closest waypoint (Avoid get lane change waypoints)
    if world.get_map().get_waypoint(vehicle.get_transform().location).transform.get_forward_vector().get_vector_angle(route[pos][0].transform.get_forward_vector()) < 1.5:
        closest_wp = world.get_map().get_waypoint(vehicle.get_transform().location)

    # Calculate the distance between vehicle and target   
    distance_target = euclidean_distance(vehicle.get_transform().location.x, vehicle.get_transform().location.y, target_wp[0].transform.location.x, target_wp[0].transform.location.y)

    # Calculate the distance between vehicle and lane center (car orientation)
    vehicle_yaw = math.radians(vehicle.get_transform().rotation.yaw)

    delta_x = closest_wp.transform.location.x - vehicle.get_transform().location.x
    delta_y = closest_wp.transform.location.y - vehicle.get_transform().location.y

    rotated_delta_x = delta_x * math.cos(vehicle_yaw) - delta_y * math.sin(vehicle_yaw)
    rotated_delta_y = delta_x * math.sin(vehicle_yaw) + delta_y * math.cos(vehicle_yaw)

    distance_center = math.sqrt(rotated_delta_x**2 + rotated_delta_y**2)

    # Distance to center (y) > 1.3 => Car is out of the lane
    if distance_center > 1.3:
        fail = 1

    # When the car come to the target choose a new one
    if distance_target <= 2:
        if pos+2 > len(route)-1:
            pos = len(route)-1
        else:
            pos = pos + 2

        # Change next target
        target_wp = route[pos]

        # Cut route
        route_draw = route_draw[pos:]

    # Car speed
    vehicle_speed = vehicle.get_velocity()
    velocidad_kmh = 3.6 * np.sqrt(vehicle_speed.x**2 + vehicle_speed.y**2)

    # Vehicle location
    vehicle_location = vehicle.get_transform().location

    # Car vector
    forward_vector = vehicle.get_transform().get_forward_vector()
    scaled_forward_vector = forward_vector * 5.0
    vector_end = vehicle_location + carla.Location(x=scaled_forward_vector.x, y=scaled_forward_vector.y, z=0.0)

    # Angle between car orientation (yaw) and road tangent (cos(x))
    angle = forward_vector.get_vector_angle(route[pos][0].transform.get_forward_vector())

    # Reward function parameters
    print(f'Dis target: {round(distance_target, 4)}    Dis center: {round(distance_center, 4)}    Angle(Car,Road): {round(angle, 4)}  Fail: {fail}   Speed: {velocidad_kmh:.2f} kmh')

    # Graphic draws
    if DEBUG_DRAW:
        if pos+10 > len(route)-1: 
            for i in range(pos+1,len(route)):
                world.debug.draw_point(carla.Location(x=route[i][0].transform.location.x, y=route[i][0].transform.location.y, z=0.8),
                                color=carla.Color(r=255, g=0, b=0),
                                life_time=0.05)
        else:
            for i in range(pos+1,pos+11):
                world.debug.draw_point(carla.Location(x=route[i][0].transform.location.x, y=route[i][0].transform.location.y, z=0.8),
                                color=carla.Color(r=255, g=0, b=0),
                                life_time=0.05)

        world.debug.draw_point(carla.Location(x=target_wp[0].transform.location.x, y=target_wp[0].transform.location.y, z=0.8),
                            color=carla.Color(r=0, g=0, b=255),
                            life_time=0.05)
            
        world.debug.draw_arrow(carla.Location(x=vehicle_location.x, y=vehicle_location.y, z=0.8),
                            vector_end,
                            life_time=0.04)

        world.debug.draw_arrow(carla.Location(x=closest_wp.transform.location.x, y=closest_wp.transform.location.y, z=0.8),
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
