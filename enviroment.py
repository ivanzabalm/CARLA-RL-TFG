# Delete support pygame message
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import sys
sys.path.append('~/Documents/CARLA_0.9.14/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner

import pygame
import carla
import numpy as np
import random

# Client and world connection
client = carla.Client('localhost', 2000)
world = client.get_world()

# Change world
world = client.load_world('Town02')

# Set clear weather
world.set_weather(carla.WeatherParameters.Default)

# Spawn points
spawn_points = world.get_map().get_spawn_points()
start_point = spawn_points[21]

# Vehicle slection and spawn
vehicle_bp = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
vehicle_bp.set_attribute('color', '0,0,0')
vehicle = world.try_spawn_actor(vehicle_bp, start_point)

# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))

# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

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

# Route Planner
grp = GlobalRoutePlanner(world.get_map(), 2)

point_a = carla.Location(x=start_point.location.x, y=start_point.location.y, z=start_point.location.z)
point_b = carla.Location(x=spawn_points[2].location.x, y=spawn_points[2].location.y, z=spawn_points[2].location.z)

route = grp.trace_route(point_a, point_b)

# Select a vehicle to follow with the camera
ego_vehicle = vehicle

# Initialise the camera floating behind the vehicle
camera_init_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# Start camera with PyGame callback
camera.listen(lambda image: pygame_callback(image, renderObject))

# Get camera dimensions
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

# Instantiate objects for rendering and vehicle control
renderObject = RenderObject(image_w, image_h)
controlObject = ControlObject(ego_vehicle)

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
pos = 5
target_wp = route[pos]

route_draw = route

if target_wp in route:
    print(route.index(target_wp))

while not crashed:
    # Advance the simulation time
    world.tick()        

    # Update the display
    gameDisplay.blit(renderObject.surface, (0, 0))
    
    # Closest waypoint
    closest_wp = world.get_map().get_waypoint(ego_vehicle.get_transform().location)

    # Calculate the distance between vehicle and waypoint 
    vehicle_location_np = np.array([ego_vehicle.get_transform().location.x, ego_vehicle.get_transform().location.y])
    waypoint_location_np = np.array([target_wp[0].transform.location.x, target_wp[0].transform.location.y])
    distance = np.linalg.norm(vehicle_location_np - waypoint_location_np)

    # When the car come to the target choose a new one
    if distance <= 2:
        pos = pos + 5

        # Change next target
        target_wp = route[pos]

        # Cut route
        route_draw = route_draw[pos:]

    # Car speed
    velocidad_vehiculo = ego_vehicle.get_velocity()
    velocidad_kmh = 3.6 * np.sqrt(velocidad_vehiculo.x**2 + velocidad_vehiculo.y**2)

    text = font.render(f"Speed: {velocidad_kmh:.2f} km/h", True, (255, 255, 255))
    gameDisplay.blit(text, (10, image_h - 40))

    # Vehicle location
    vehicle_location = ego_vehicle.get_transform().location

    # Car vector
    forward_vector = vehicle.get_transform().get_forward_vector()
    scaled_forward_vector = forward_vector * 7.0
    vector_end = vehicle_location + carla.Location(x=scaled_forward_vector.x, y=scaled_forward_vector.y, z=0.0)

    # Angle between car orientation (yaw) and road tangent (cos(x))
    angle = forward_vector.get_vector_angle(route[pos][0].transform.get_forward_vector())

    # Reward function parameters
    print(f'Distance to target: {distance}    Angle(Car,Road): {angle}    Colision: {0}')

    # Graphic draws
    for i in range(pos+1,(pos+21)):
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
                ego_vehicle.set_autopilot(True)
                ego_vehicle = vehicle
                # Ensure the vehicle is still alive (might have been destroyed)
                if ego_vehicle.is_alive:
                    # Stop and remove the camera
                    camera.stop()
                    camera.destroy()

                    # Spawn a new camera and attach it to the new vehicle
                    controlObject = ControlObject(ego_vehicle)
                    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
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
