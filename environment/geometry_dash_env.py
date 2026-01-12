import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import math

# --- CONSTANTS & NEON PALETTE ---
WIDTH, HEIGHT = 800, 600
FPS = 60

# Colors
BG_TOP = (15, 15, 40)
BG_BOTTOM = (40, 0, 60)
GROUND_LINE = (0, 255, 240)
GROUND_FILL = (5, 5, 10)
PLAYER_OUTLINE = (255, 180, 0)
PLAYER_FILL = (255, 255, 0)
SPIKE_DARK = (140, 0, 0)
SPIKE_LIGHT = (255, 40, 40)
BLOCK_FILL = (30, 30, 80)
BLOCK_OUTLINE = (80, 150, 255)
SAW_OUTLINE = (255, 0, 100)
SAW_INNER = (100, 0, 40)

# --- PHYSICS ---
PLAYER_SIZE = 34
GRAVITY = 1.2
JUMP_FORCE = -20.0
SCROLL_SPEED = 8.0 
GROUND_HEIGHT = HEIGHT - 110

# --- LIDAR CONFIG ---
LIDAR_RANGE = 600       
LIDAR_RES = 20          

class Player:
    def __init__(self):
        self.x = 150
        self.y = GROUND_HEIGHT - PLAYER_SIZE
        self.vel_y = 0
        self.on_ground = True
        self.dead = False
        self.rect = pygame.Rect(self.x, self.y, PLAYER_SIZE, PLAYER_SIZE)
        self.cube_rot = 0
        self.surf = pygame.Surface((PLAYER_SIZE, PLAYER_SIZE), pygame.SRCALPHA)
        self._draw_face()

    def _draw_face(self):
        pygame.draw.rect(self.surf, PLAYER_OUTLINE, (0, 0, PLAYER_SIZE, PLAYER_SIZE))
        m = 4
        pygame.draw.rect(self.surf, PLAYER_FILL, (m, m, PLAYER_SIZE-m*2, PLAYER_SIZE-m*2))
        eye_s = 6; eye_y = 10
        pygame.draw.rect(self.surf, (40, 40, 40), (8, eye_y, eye_s, eye_s))
        pygame.draw.rect(self.surf, (40, 40, 40), (20, eye_y, eye_s, eye_s))
        pygame.draw.rect(self.surf, (40, 40, 40), (10, 22, 14, 4))

    def jump(self):
        if self.on_ground:
            self.vel_y = JUMP_FORCE
            self.on_ground = False
            
    def update(self):
        self.vel_y += GRAVITY
        self.y += self.vel_y
        
        # Ground collision
        if self.y >= GROUND_HEIGHT - PLAYER_SIZE:
            self.y = GROUND_HEIGHT - PLAYER_SIZE
            self.vel_y = 0
            self.on_ground = True
            
        self.rect.y = int(self.y)
        
        # Rotation visual
        if not self.on_ground: self.cube_rot -= 12
        else: self.cube_rot = round(self.cube_rot / 90) * 90

    def draw(self, screen):
        rotated_surf = pygame.transform.rotate(self.surf, self.cube_rot)
        new_rect = rotated_surf.get_rect(center=self.rect.center)
        screen.blit(rotated_surf, new_rect.topleft)

class Obstacle:
    def __init__(self, x, obs_type, width=None, y_offset=0):
        self.x = x
        self.type = obs_type
        self.y_offset = y_offset
        self.rotation = 0 
        self.passed = False # Flag pentru reward system
        
        if obs_type == "spike":
            self.width = 32; self.height = 36
            self.y = GROUND_HEIGHT - self.height - y_offset
        elif obs_type == "saw":
            self.width = 44; self.height = 44
            self.y = GROUND_HEIGHT - 38 - y_offset
        elif obs_type == "platform" or obs_type == "block":
            self.width = width if width else 100
            self.height = 30 if obs_type == "platform" else 40
            self.y = GROUND_HEIGHT - (60 if obs_type == "platform" else 40) - y_offset
            
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update_rect(self):
        self.rect.x = int(self.x)
        if self.type == "saw": self.rotation -= 10

    def draw(self, screen):
        if self.type == "spike":
            p1 = (self.rect.left, self.rect.bottom)
            p2 = (self.rect.centerx, self.rect.top)
            p3 = (self.rect.right, self.rect.bottom)
            pygame.draw.polygon(screen, SPIKE_DARK, [p1, p2, p3])
            m = 4
            ip1 = (self.rect.left + m, self.rect.bottom - 2)
            ip2 = (self.rect.centerx, self.rect.top + m + 2)
            ip3 = (self.rect.right - m, self.rect.bottom - 2)
            pygame.draw.polygon(screen, SPIKE_LIGHT, [ip1, ip2, ip3])
            pygame.draw.polygon(screen, (0,0,0), [p1, p2, p3], 2)
            
        elif self.type == "saw":
            center = self.rect.center
            radius = self.width // 2
            pygame.draw.circle(screen, SAW_INNER, center, radius)
            pygame.draw.circle(screen, SAW_OUTLINE, center, radius, 3)
            
            angle_rad = math.radians(self.rotation)
            for offset in [0, math.pi/2, math.pi, 3*math.pi/2]:
                end_pos = (center[0] + math.cos(angle_rad + offset) * radius,
                           center[1] + math.sin(angle_rad + offset) * radius)
                pygame.draw.line(screen, SAW_OUTLINE, center, end_pos, 3)
            pygame.draw.circle(screen, (255, 255, 255), center, 6)
            
        elif self.type == "platform" or self.type == "block":
            pygame.draw.rect(screen, BLOCK_FILL, self.rect)
            pygame.draw.rect(screen, BLOCK_OUTLINE, self.rect, 2)
            pygame.draw.line(screen, (200, 230, 255), self.rect.topleft, self.rect.topright, 3)

class ObstacleGenerator:
    def __init__(self, seed=None):
        if seed: random.seed(seed)
        self.obstacles = []
        self.next_spawn_x = WIDTH
        self.distance = 0
        self.last_pattern_idx = -1
        
    def update(self):
        self.distance += SCROLL_SPEED
        for obs in self.obstacles:
            obs.x -= SCROLL_SPEED
            obs.update_rect()
        
        self.obstacles = [o for o in self.obstacles if o.x + o.width > -100]
        
        self.next_spawn_x -= SCROLL_SPEED
        if self.next_spawn_x < WIDTH:
            self.spawn_sequence()
            
    def spawn_sequence(self):
        start_x = self.next_spawn_x
        if self.obstacles:
            last = self.obstacles[-1]
            gap = random.randint(180, 250)
            start_x = max(start_x, last.x + last.width + gap) 

        patterns = [
            self.pat_simple_mix,    # 0
            self.pat_stairs,        # 1
            self.pat_tunnel,        # 2
            self.pat_pillars,       # 3
            self.pat_saw_trap       # 4
        ]
        
        weights = [25, 15, 15, 20, 25]
        
        idx = random.choices(range(len(patterns)), weights=weights)[0]
        if idx == self.last_pattern_idx and random.random() < 0.5:
             idx = random.choices(range(len(patterns)), weights=weights)[0]
        self.last_pattern_idx = idx
        
        end_x = patterns[idx](start_x)
        self.next_spawn_x = end_x

    # --- PATTERNS ---
    def pat_simple_mix(self, x):
        count = random.choice([1, 2, 3])
        current_x = x
        for _ in range(count):
            self.obstacles.append(Obstacle(current_x, "spike"))
            current_x += random.choice([32, 48]) 
        return current_x + 64

    def pat_saw_trap(self, x):
        count = random.choice([1, 2])
        current_x = x
        for _ in range(count):
            self.obstacles.append(Obstacle(current_x, "saw"))
            current_x += random.randint(180, 240)
        return current_x

    def pat_stairs(self, x):
        h1 = 0
        h2 = random.choice([40, 50])
        h3 = random.choice([80, 90])
        
        self.obstacles.append(Obstacle(x, "block", width=50, y_offset=h1))
        self.obstacles.append(Obstacle(x + 150, "block", width=50, y_offset=h2))
        self.obstacles.append(Obstacle(x + 300, "block", width=50, y_offset=h3))
        self.obstacles.append(Obstacle(x + 225, "spike"))
        return x + 350
        
    def pat_tunnel(self, x):
        length = random.randint(350, 500)
        self.obstacles.append(Obstacle(x, "platform", width=length, y_offset=95))
        
        num_dangers = length // 150
        for i in range(num_dangers):
            offset = 80 + i * 140
            obs_type = random.choice(["spike", "saw"])
            self.obstacles.append(Obstacle(x + offset, obs_type))
        return x + length

    def pat_pillars(self, x):
        h1 = random.randint(0, 40)
        h2 = random.randint(60, 90)
        
        self.obstacles.append(Obstacle(x, "block", width=40, y_offset=h1))
        self.obstacles.append(Obstacle(x + 200, "block", width=40, y_offset=h2))
        self.obstacles.append(Obstacle(x + 100, "saw")) 
        return x + 240

class ImpossibleGameEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}
    
    def __init__(self, render_mode=None, max_steps=10000):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(2) # 0 = Run, 1 = Jump
        
        # --- FIX IMPORTANT: OBS DIMENSIONS ---
        # LIDAR 2D: Avem nevoie de Tip (Ce e?) + Inaltime (Unde e?)
        self.num_scans = int(LIDAR_RANGE / LIDAR_RES)
        self.obs_dim = 3 + (self.num_scans * 2) 
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("GEOMETRY DASH AI - RADAR HUD EDITION")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 40)
            self.bg_surface = self._create_bg_gradient()
            
        self.reset()

    def _create_bg_gradient(self):
        bg = pygame.Surface((WIDTH, HEIGHT))
        for y in range(HEIGHT):
            ratio = y / HEIGHT
            r = BG_TOP[0] * (1 - ratio) + BG_BOTTOM[0] * ratio
            g = BG_TOP[1] * (1 - ratio) + BG_BOTTOM[1] * ratio
            b = BG_TOP[2] * (1 - ratio) + BG_BOTTOM[2] * ratio
            pygame.draw.line(bg, (int(r), int(g), int(b)), (0, y), (WIDTH, y))
        return bg

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player = Player()
        self.generator = ObstacleGenerator(seed)
        self.steps = 0
        self.score = 0
        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1
        if action == 1: self.player.jump()
        
        self.player.update()
        self.generator.update()
        
        terminated = False
        
        # 1. REWARD MODIFICAT (Anti-Farming)
        # Recompensa ceva mai mare pentru supraviețuire (0.01 -> 0.05)
        reward = 0.05 
        
        # --- COLLISION LOGIC ---
        player_rect = self.player.rect
        hitbox = player_rect.inflate(-10, -10) 
        
        for obs in self.generator.obstacles:
            if hitbox.colliderect(obs.rect):
                if obs.type in ["platform", "block"]:
                    # Daca cadem pe ea (vel_y pozitiva) si suntem deasupra
                    if self.player.vel_y >= 0 and self.player.rect.bottom <= obs.rect.top + 18:
                         self.player.y = obs.rect.top - PLAYER_SIZE
                         self.player.vel_y = 0
                         self.player.on_ground = True
                    else:
                         # 2. PENALIZARE MAI BLANDA PENTRU MOARTE (-50 -> -10)
                         terminated = True; reward = -10.0
                elif obs.type in ["spike", "saw"]:
                    terminated = True; reward = -10.0
        
        # 3. BONUS MARE PENTRU OBSTACOLE DEPASITE (+5 -> +10)
        if not terminated:
            player_front_x = self.player.x
            for obs in self.generator.obstacles:
                if not obs.passed and (obs.x + obs.width < player_front_x):
                    obs.passed = True
                    reward += 10.0 

        if self.steps >= self.max_steps: terminated = True
        self.score = self.steps // 10
        
        return self._get_observation(), reward, terminated, False, {'score': self.score}

    def _get_observation(self):
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        # 1. Info Player
        obs[0] = self.player.y / HEIGHT
        obs[1] = self.player.vel_y / 20.0
        obs[2] = 1.0 if self.player.on_ground else -1.0
        
        # 2. LIDAR 2D (Type + Height)
        start_type = 3
        start_height = 3 + self.num_scans
        
        player_front_x = self.player.x + PLAYER_SIZE
        player_bottom_y = self.player.rect.bottom
        
        relevant = [o for o in self.generator.obstacles 
                   if o.x + o.width > player_front_x and o.x < player_front_x + LIDAR_RANGE]
        
        for i in range(self.num_scans):
            scan_x = player_front_x + (i * LIDAR_RES)
            
            found_type = 0.0
            found_height = 0.0 
            
            for o in relevant:
                if o.x <= scan_x <= o.x + o.width:
                    if o.type in ["spike", "saw"]: 
                        found_type = 1.0 
                    elif o.type in ["platform", "block"]: 
                        found_type = 0.5 
                    
                    found_height = (o.rect.top - player_bottom_y) / HEIGHT
                    break 
            
            obs[start_type + i] = found_type
            obs[start_height + i] = found_height
            
        return obs

    def render(self):
        if self.render_mode == "human":
            self.screen.blit(self.bg_surface, (0, 0))
            
            # 1. Ground & Glow
            glow = pygame.Surface((WIDTH, 10), pygame.SRCALPHA)
            pygame.draw.rect(glow, (*GROUND_LINE, 50), (0, 0, WIDTH, 10))
            self.screen.blit(glow, (0, GROUND_HEIGHT - 5))
            pygame.draw.line(self.screen, GROUND_LINE, (0, GROUND_HEIGHT), (WIDTH, GROUND_HEIGHT), 3)
            pygame.draw.rect(self.screen, GROUND_FILL, (0, GROUND_HEIGHT+2, WIDTH, HEIGHT-GROUND_HEIGHT))
            
            # 2. Draw Game Objects
            self.player.draw(self.screen)
            for obs in self.generator.obstacles:
                obs.draw(self.screen)
                
            # --- 3. HUD / RADAR (Stânga Sus) ---
            panel_w = 320
            panel_h = 100
            panel_x = 20
            panel_y = 20
            
            radar_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            radar_surf.fill((0, 0, 0, 180)) 
            pygame.draw.rect(radar_surf, (100, 100, 100), (0,0, panel_w, panel_h), 2) 
            
            mid_y = panel_h // 2
            pygame.draw.line(radar_surf, (50, 50, 50), (0, mid_y), (panel_w, mid_y), 1)
            
            obs_data = self._get_observation()
            start_type_idx = 3
            start_height_idx = 3 + self.num_scans
            dot_spacing = panel_w / self.num_scans
            
            for i in range(self.num_scans):
                val_type = obs_data[start_type_idx + i]
                val_height = obs_data[start_height_idx + i]
                
                if val_type != 0:
                    color = (100, 100, 100) 
                    if val_type == 1.0: color = (255, 50, 50)     
                    elif val_type == 0.5: color = (50, 150, 255)  
                    
                    rx = i * dot_spacing
                    ry = mid_y + (val_height * 150) 
                    ry = max(5, min(panel_h - 5, ry))
                    
                    pygame.draw.circle(radar_surf, color, (int(rx), int(ry)), 3)
                else:
                    rx = i * dot_spacing
                    pygame.draw.circle(radar_surf, (30, 30, 30), (int(rx), mid_y), 1)

            self.screen.blit(radar_surf, (panel_x, panel_y))
            
            score_s = f"SCORE: {self.score}"
            self.screen.blit(self.font.render(score_s, True, (0,0,0)), (WIDTH - 200 + 2, 22))
            self.screen.blit(self.font.render(score_s, True, (255, 255, 0)), (WIDTH - 200, 20))
            
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        pygame.quit()