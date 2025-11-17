import glob
import os
import sys
import numpy as np
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import pickle

from synch_mode import CarlaSyncMode
from controllers import PIDLongitudinalController
from utils import *

random.seed(78)

class SimEnv(object):
    def __init__(self, 
        visuals=True,
        target_speed = 30,
        max_iter = 4000,
        start_buffer = 10,
        train_freq = 1,
        save_freq = 200,
        start_ep = 0,
        max_dist_from_waypoint = 20
    ) -> None:
        self.visuals = visuals
        if self.visuals:
            self._initiate_visuals()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        # --- Use the SIMPLE, FAST map ---
        print("Loading Town02_Opt...")
        self.world = self.client.load_world('Town02_Opt')
        
        # Poll the server until the map is ready
        map_ready = False
        while not map_ready:
            try:
                self.world.get_map()
                map_ready = True
            except RuntimeError:
                print("Waiting for map to load...")
                time.sleep(0.5)
        print("Map loaded successfully.")

        # Unload map layers *after* map is ready
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)
        
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.blueprint_library = self.world.get_blueprint_library()
        
        # --- Fix the player car to a Tesla ---
        self.vehicle_blueprint = self.blueprint_library.find('vehicle.tesla.model3')

        # Environment parameters
        self.global_t = 0
        self.target_speed = target_speed
        self.max_iter = max_iter
        self.start_buffer = start_buffer
        self.train_freq = train_freq
        self.save_freq = save_freq
        self.start_ep = start_ep
        self.max_dist_from_waypoint = max_dist_from_waypoint
        self.start_train = self.start_ep + self.start_buffer
        
        # Reward/Actor lists
        self.total_rewards = 0 # This tracks reward for saving, not for episodes
        self.average_rewards_list = []
        self.player_actor_list = []
        self.npc_actor_list = []
        self.traffic_manager = None
    
    def _initiate_visuals(self):
        pygame.init()
        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()
    
    # -----------------------------------------------------------------
    # Creates traffic (runs only once)
    # -----------------------------------------------------------------
    def create_traffic(self):
        print("Setting up Traffic Manager...")
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(3.0) 
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_respawn_dormant_vehicles(True)

        npc_blueprints = self.blueprint_library.filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # --- Use SIMPLE traffic (30 cars) ---
        num_vehicles = 30
        print(f"Spawning {num_vehicles} NPC vehicles...")

        for i in range(num_vehicles):
            try:
                blueprint = random.choice(npc_blueprints)
                spawn_point = spawn_points.pop() 
                
                npc = self.world.spawn_actor(blueprint, spawn_point)
                npc.set_autopilot(True, self.traffic_manager.get_port())
                self.npc_actor_list.append(npc)
            except RuntimeError:
                pass
            except IndexError:
                break

        print(f"Successfully spawned {len(self.npc_actor_list)} NPC vehicles.")

    # -----------------------------------------------------------------
    # Creates the player (runs every episode)
    # -----------------------------------------------------------------
    def create_player_agent(self):
        # Clear any old player actors
        self.reset() 

        player_spawn_point = None
        spawn_attempts = 0
        
        while player_spawn_point is None:
            spawn_attempts += 1
            if spawn_attempts > 20: 
                print("CRITICAL ERROR: Could not find a free spawn point for the player after 20 attempts.")
                player_spawn_point = self.spawn_points[0]
            else:
                proposed_spawn_point = random.choice(self.spawn_points)
            
            try:
                self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, proposed_spawn_point)
                player_spawn_point = proposed_spawn_point
            except RuntimeError:
                if spawn_attempts % 5 == 0:
                    print(f"Spawn point {proposed_spawn_point.location} occupied, trying another...")
                time.sleep(0.01)

        self.player_actor_list.append(self.vehicle)

        # --- Create sensors ---
        self.camera_rgb = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.player_actor_list.append(self.camera_rgb)

        self.camera_rgb_vis = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.player_actor_list.append(self.camera_rgb_vis)

        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.player_actor_list.append(self.collision_sensor)

        self.speed_controller = PIDLongitudinalController(self.vehicle)
    
    # -----------------------------------------------------------------
    # Resets only the player
    # -----------------------------------------------------------------
    def reset(self):
        for actor in self.player_actor_list:
            if actor.is_alive:
                actor.destroy()
        self.player_actor_list = []
    
    def generate_episode(self, model, replay_buffer, ep, action_map=None, eval=True):
        # --- ADD A VARIABLE TO TRACK THIS EPISODE'S REWARD ---
        episode_reward = 0.0
        
        try:
            with CarlaSyncMode(self.world, self.camera_rgb, self.camera_rgb_vis, self.collision_sensor, fps=30) as sync_mode:
                counter = 0
                snapshot, image_rgb, image_rgb_vis, collision = sync_mode.tick(timeout=2.0)

                if snapshot is None or image_rgb is None:
                    print("No data, skipping episode")
                    return 0.0 # <-- Return 0

                image = process_img(image_rgb)
                next_state = image 

                while True:
                    spectator = self.world.get_spectator()
                    
                    if not self.vehicle or not self.vehicle.is_alive:
                        break 
                        
                    vehicle_transform = self.vehicle.get_transform()
                    spectator_transform = carla.Transform(
                        vehicle_transform.location + carla.Location(x=-10, z=5), 
                        vehicle_transform.rotation
                    )
                    spectator.set_transform(spectator_transform)
                    
                    if self.visuals:
                        if should_quit():
                            return 0.0 # <-- Return 0
                        self.clock.tick_busy_loop(30)

                    vehicle_location = self.vehicle.get_location()
                    waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True, 
                        lane_type=carla.LaneType.Driving)
                    
                    state = next_state
                    counter += 1
                    self.global_t += 1

                    action = model.select_action(state, eval=eval)
                    steer = action
                    if action_map is not None:
                        steer = action_map[action]

                    control = self.speed_controller.run_step(self.target_speed)
                    control.steer = steer
                    self.vehicle.apply_control(control)

                    fps = round(1.0 / snapshot.timestamp.delta_seconds)
                    snapshot, image_rgb, image_rgb_vis, collision = sync_mode.tick(timeout=2.0)
                    
                    if snapshot is None: 
                        print("Lost connection to server, ending episode.")
                        break

                    cos_yaw_diff, dist, collision_data = get_reward_comp(self.vehicle, waypoint, collision)
                    reward = reward_value(cos_yaw_diff, dist, collision_data)

                    if image_rgb is None:
                        print("No camera data, ending episode.")
                        break

                    image = process_img(image_rgb)
                    done = 1 if collision_data else 0
                    
                    # --- MODIFIED REWARD TRACKING ---
                    episode_reward += reward
                    
                    next_state = image
                    replay_buffer.add(state, action, next_state, reward, done)

                    if not eval:
                        if ep > self.start_train and (self.global_t % self.train_freq) == 0:
                            model.train(replay_buffer)

                    # --- ADDED BLOCK FOR REAL-TIME TERMINAL OUTPUT ---
                    if counter % 100 == 0: # Print status every 100 steps
                        print(f"  [Ep: {ep+1}, Step: {counter}/{self.max_iter}] "
                              f"Running Ep. Reward: {episode_reward:.2f}, "
                              f"Epsilon: {model.current_eps:.4f}")
                    # --- END OF ADDED BLOCK ---

                    if self.visuals:
                        draw_image(self.display, image_rgb_vis)
                        self.display.blit(
                            self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),
                            (8, 10))
                        self.display.blit(
                            self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                            (8, 28))
                        pygame.display.flip()

                    if collision_data == 1 or counter >= self.max_iter or dist > self.max_dist_from_waypoint:
                        print("Episode {} processed".format(ep), counter)
                        break
                
                # --- Add to the global total for saving ---
                self.total_rewards += episode_reward
                
                if not eval and ep % self.save_freq == 0 and ep > 0:
                    self.save(model, ep)
                    
        except Exception as e:
            print(f"Error in generate_episode (e.g., sync_mode failed): {e}")
            return 0.0 # <-- Return 0 if it failed

        # --- RETURN THE EPISODE'S REWARD ---
        return episode_reward

    def save(self, model, ep):
        if ep % self.save_freq == 0 and ep > self.start_ep:
            # Note: This avg_reward is a bit skewed as it's not reset,
            # but it matches the original logic.
            avg_reward = self.total_rewards / (ep - self.start_ep)
            self.average_rewards_list.append(avg_reward)
            
            model.save('weights/model_ep_{}'.format(ep))
            print("Saved model with average reward =", avg_reward)
    
    # -----------------------------------------------------------------
    # Cleans up everything at the end
    # -----------------------------------------------------------------
    def cleanup(self):
        print("Destroying all actors...")
        
        all_actors = self.player_actor_list + self.npc_actor_list
        for actor in all_actors:
            if actor.is_alive:
                actor.destroy()
        
        self.player_actor_list = []
        self.npc_actor_list = []
        
        print("Cleanup complete.")
        
        if self.visuals:
            pygame.quit()

# -----------------------------------------------------------------
# Global helper functions (must be outside the class)
# -----------------------------------------------------------------
def get_reward_comp(vehicle, waypoint, collision):
    vehicle_location = vehicle.get_location()
    x_wp = waypoint.transform.location.x
    y_wp = waypoint.transform.location.y

    x_vh = vehicle_location.x
    y_vh = vehicle_location.y

    wp_array = np.array([x_wp, y_wp])
    vh_array = np.array([x_vh, y_vh])

    dist = np.linalg.norm(wp_array - vh_array)

    vh_yaw = correct_yaw(vehicle.get_transform().rotation.yaw)
    wp_yaw = correct_yaw(waypoint.transform.rotation.yaw)
    cos_yaw_diff = np.cos((vh_yaw - wp_yaw)*np.pi/180.)

    collision_data = 0 if collision is None else 1
    
    return cos_yaw_diff, dist, collision_data

def reward_value(cos_yaw_diff, dist, collision, lambda_1=1, lambda_2=1, lambda_3=5):
    reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision)
    return reward