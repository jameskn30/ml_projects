import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tqdm import tqdm 
import random
from .agent import QLearningAgent
import pygame
import time
import sys
import os

def default_map():
    return 'SFFH\nH'

def GET_MAP(map):
    if map == "4x4": 
        return [
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
        ]

    if map == "8x8": 
        return [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ],

class GridWorldEnv(gym.Env):
    def __init__(self, map, reward_dict, **kwargs):

        if type(map) == type(""):
            if map == '4x4' or map == '8x8':
                map = GET_MAP(map)

        #Check if map has the same dimension as rows and cols
        rows, cols, map = self._process_map(map)

        self.map = map
        nS = rows * cols
        nA = 4
        self.rows, self.cols = rows, cols
        self.agent_pos = [0,0]
        self.goal_pos = [rows - 1, cols - 1]

        #save reward dict
        self.reward_dict = reward_dict

        #default state
        self.state = np.zeros((self.rows, self.cols))
        self.set_state(self.agent_pos, self.goal_pos)

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        #time step used to track how long this agent performing
        #if it's to long, terminate early
        self.timestep = 0
        self.max_timestep = int(kwargs['max_timestep']) if kwargs['max_timestep'] != None else 100

        #termination status
        self.terminated = False

        #render using pygame
        self.renderer = GridWorldRenderer(rows, cols)
    
    #Reset environmentj
    def reset(self):
        self.agent_pos = [0,0]
        self.goal_pos = [self.rows - 1, self.cols - 1]
        self.set_state(self.agent_pos, self.goal_pos)
        self.timestep = 0
        return 0

    def _process_map(self, map_data):
        rows = map_data
        rows_n = len(rows)
        cols_n = len(rows[0])
        map = [['' for _ in range(cols_n)] for _ in range(rows_n)]
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                map[i][j] = val 

        return rows_n, cols_n, map

    def set_state(self, agent_pos, goal_pos):
        self.state = np.zeros((self.rows, self.cols))
        self.state[tuple(agent_pos)] = 1
        self.state[tuple(goal_pos)] = 0.5
        observation = self.state.flatten()
        return observation
    
    def reached_goal(self, pos):
        x,y = pos
        return True if self.map[x][y] == 'G'else False
    
    def get_reward(self, pos):
        x,y = pos
        val = self.map[x][y]
        return self.reward_dict[val] if val in self.reward_dict else 0

    #Step function: agent take step in env
    def step(self, action):
        """
        Take a step in this environemnt
        @params:
            action: int
                0: down
                1:up 
                2:right
                3:left
        @return
            new_state: int
            reward: int
            terminated: boolean
            info: dict

        """

        if action == 0: 
            self.agent_pos[0] += 1
        elif action == 1: 
            self.agent_pos[0] -= 1
        elif action == 2: 
            self.agent_pos[1] += 1
        elif action == 3: 
            self.agent_pos[1] -= 1

        #clip the agent position to avoid out of bound
        self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.cols - 1)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.rows - 1)

        self.set_state(self.agent_pos, self.goal_pos)
        observation = self.state.flatten()

        #Check if the agent takes too long to go to goal
        self.timestep += 1
        self.terminated = True if self.timestep > self.max_timestep else False

        #Define your reward function
        reward = self.get_reward(self.agent_pos)

        if np.array_equal(self.agent_pos, self.goal_pos):
            self.terminated = True
            reward = 1
        
        if self.reached_goal(self.agent_pos):
            self.terminated = True
        
        info = {}
        #return:
        #next state, argmax to get the new state of agent, np.argmax([0,0,1,0,0,0,0.5]) = 2
        #reward
        #done or not
        #extra infomation
        return np.argmax(observation), reward, self.terminated, info
    
    def render(self, agent: QLearningAgent) -> None:
        #Put a renderer here
        self.reset()
        self.renderer.start(self.state)
        self.renderer.update(self.state)
        max_iter = 100
        iter =0 
        curr_state = 0
        while self.terminated == False and iter < max_iter:
            observation = self.state.flatten()
            curr_state = np.argmax(observation)
            action = agent.get_action(curr_state)
            curr_state, _,_,_ = self.step(action)
            self.renderer.update(self.state)
            pygame.time.wait(500)
            iter += 1

        self.renderer.update(self.state)
        pygame.time.wait(1000)
        self.renderer.end()
        print("Terminated")
        
    def __str__(self):
        print("=" * 20)
        print('ENV MAP')
        print("=" * 20)
        for row in self.map:
            print(row)
        print("=" * 20)
        print('ENV STATE')
        print("=" * 20)
        print(self.state)
        print("=" * 20)
        print("ENV DESCRIPTION")
        print("=" * 20)
        print(f'reward dict = {self.reward_dict}')
        print(f'timestep = {self.timestep}, max_timestep = {self.max_timestep},')
        print(f'obs space = {self.observation_space.n}')
        print(f'actions = {self.action_space.n}')
        print(f'agent position = {self.agent_pos}')
        print(f'goal position = {self.goal_pos}')
        print(f'terminated = {self.terminated}')
        return ''

colors = {
    "black": (0,0,0),
    "white": (255,255,255),
    "light_white": (200,200,200),
    "blue": (0,0,255),
    "green": (51, 204, 51),
    "grass": (34,177, 76)
}
class GridWorldRenderer():
    def __init__(self, rows, columns, map, cell_size = 50):
        self.rows = rows 
        self.columns = columns
        self.cell_size = cell_size
        self.map = map
        
        self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        print(self.CURR_PATH)

        pygame.init()

        self._window_width = self.columns * self.cell_size
        self._window_height = self.rows * self.cell_size

        self.screen = pygame.display.set_mode((self._window_width, self._window_height))
        pygame.display.set_caption(f"Grid World {self.rows}x{self.columns}" )
        self.clock = pygame.time.Clock()
        self.screen.fill(colors['white'])

        self.border_color = colors['black']
        self.state = None

        # self.update(state)
        self._load_assets()

    def _resolve_path(self, rel_path):
        return os.path.join(self.CURR_PATH, rel_path)

    def _load_assets(self):
        self.agent = self._resolve_path('assets/agent.png')
        self.battery =  self._resolve_path('assets/battery.png')
        self.coin =     self._resolve_path('assets/coin.png')
        self.grass =    self._resolve_path('assets/grass.png') 
        self.rock =     self._resolve_path('assets/rock.png') 
        self.shit =     self._resolve_path('assets/shit.png') 

    # def _drawgrid(self):
    #     for i, x in enumerate(range(0, self._window_width, self.cell_size)):
    #         for j, y in enumerate(range(0, self._window_height, self.cell_size)):
    #             color = colors['white']

    #             rect = pygame.Rect(x,y, self.cell_size, self.cell_size)

    #             pygame.draw.rect(self.screen, color, rect)
    #             if self.test_image and i == 0 and j == 0 :
    #                 self.draw_robot(self.screen, x,y)
    #             if i == 1 and j == 0 :
    #                 self.draw_battery(self.screen, x,y)
    #             if i == 0 and j == 1 :
    #                 self.draw_crap(self.screen, x,y)

    #             border = pygame.Rect(x,y, self.cell_size, self.cell_size)
    #             pygame.draw.rect(self.screen, self.border_color, border, 1)

    def _drawstate(self):
        self.screen.fill(colors["grass"])
        for i, x in enumerate(range(0, self._window_width, self.cell_size)):
            for j, y in enumerate(range(0, self._window_height, self.cell_size)):
                if self.map[i][j] == 'A':
                    self.draw_agent(self.screen, x,y)
                    
                elif self.map[i][j] == 'G':
                    self.draw_coin(self.screen, x,y)

                elif self.map[i][j] == 'R':
                    self.draw_rock(self.screen, x,y)

                elif self.map[i][j] == 'B':
                    self.draw_battery(self.screen, x,y)

                elif self.map[i][j] == 'S':
                    self.draw_shit(self.screen, x,y)

                else:
                    self.draw_grass(self.screen, x,y)

    def _draw_object(self, screen, img, x,y):
        img = pygame.image.load(img)
        img = pygame.transform.scale(img, (self.cell_size, self.cell_size))
        screen.blit(img, (x,y))
        border = pygame.Rect(x,y, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.border_color, border, 2)

    def draw_agent(self, screen, x, y):
        self._draw_object(screen, self.agent, x, y)

    def draw_shit(self, screen, x, y):
        self._draw_object(screen, self.shit, x, y)

    def draw_battery(self, screen, x, y):
        self._draw_object(screen, self.battery, x, y)

    def draw_grass(self, screen, x, y):
        self._draw_object(screen, self.grass, x, y)

    def draw_rock(self, screen, x, y):
        self._draw_object(screen, self.rock, x, y)

    def draw_coin(self, screen, x, y):
        self._draw_object(screen, self.coin, x, y)
    
    def update(self, new_state):
        #clear screen
        self.clock.tick(60)
        self.screen.fill(colors['white'])
        
        self.state = new_state
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.end()
            

        self._drawstate()
        pygame.display.update()
        
    def end(self):
        print('exit')
        pygame.quit()
        sys.exit()

    
    #call a while  loop to update pygame drawing
    def run(self):
        default_state = np.zeros((self.rows, self.columns))
        default_state[(0,0)] = 1
        while True:
            self.update(default_state)
