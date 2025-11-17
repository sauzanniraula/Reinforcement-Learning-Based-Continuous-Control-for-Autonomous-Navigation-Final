import os
import pygame
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Function for initial_setup.py ---
def create_folders(folders):
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)

# --- Functions for controllers.py ---
def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.
        :param vehicle: actor to get speed from
        :return: speed in Km/h
    """
    vel = vehicle.get_velocity()
    return 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def correct_yaw(yaw):
    """
    Corrects the yaw angle to be within the [-180, 180] range
    """
    return yaw if yaw <= 180.0 else yaw - 360.0

# --- Functions for environment.py (visuals) ---
def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

# --- Function for environment.py (DQN state) ---
def process_img(image):
    """
    Converts the CARLA camera image (800x600x4) to a
    processed (128x128x1) state for the DQN
    """
    # Get raw data and reshape
    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img = np.reshape(img, (image.height, image.width, 4))
    
    # Extract BGR and convert to grayscale
    img = img[:, :, :3] # Remove alpha channel
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to 128x128
    img_resized = cv2.resize(img_gray, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Normalize (0-255 -> 0-1) and add a channel dimension
    img_normalized = img_resized / 255.0
    img_final = np.reshape(img_normalized, (1, 128, 128))
    
    return img_final

# --- Function for train.py and DQN_main.py ---
def save_plots(episode_list, rewards_list, epsilon_list, name_prefix):
    """
    Saves plots for rewards and (optionally) epsilon decay.
    """
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
        
    print(f"Saving plots with prefix: {name_prefix}")

    # --- Plot 1: Episode vs. Total Reward ---
    plt.figure(figsize=(12, 6))
    plt.plot(episode_list, rewards_list, label='Total Reward')
    plt.title(f'{name_prefix.capitalize()} - Episode vs. Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    # Save the reward plot
    reward_plot_name = f'graphs/{name_prefix}_rewards.png'
    plt.savefig(reward_plot_name)
    print(f"Saved reward plot to {reward_plot_name}")
    plt.close()

    # --- Plot 2: Episode vs. Epsilon (only if provided) ---
    if epsilon_list is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(episode_list, epsilon_list, label='Epsilon', color='r')
        plt.title(f'{name_prefix.capitalize()} - Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.legend()
        plt.grid(True)
        
        # Save the epsilon plot
        epsilon_plot_name = f'graphs/{name_prefix}_epsilon.png'
        plt.savefig(epsilon_plot_name)
        print(f"Saved epsilon plot to {epsilon_plot_name}")
        plt.close()