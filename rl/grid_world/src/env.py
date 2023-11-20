import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tqdm import tqdm 
import random
from .agent import QLearningAgent, SarsaAgent
import pygame
import time
import sys
import os
# from collections import set

GAME_SPEED = 40

def default_map():
    return 'SFFH\nH'

def GET_MAP(map):
    if map == "4x4": 
        return [
            "A---",
            "--S-",
            "B---",
            "R--G"
        ]

    if map == "8x8": 
        return [
            "A-------",
            "--------",
            "---R----",
            "------S-",
            "----S---",
            "R---B---",
            "-B----S-",
            "--B--S-G",
        ],

def GET_RANDOM_MAP(rows, cols):
    def gen_obj():
        choices = ['R', 'S', 'B', '-']
        weights =  [0.05, 0.05, 0.05, 0.85]
        return np.random.choice(choices, p = weights)

    map = [[gen_obj() for _ in range(rows)] for _ in range(cols)]
    map[0][0] = 'A'
    map[rows - 1][cols - 1] = 'G'

    map = [''.join(row) for row in map]
    return map

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
        self.max_timestep = int(self._get(kwargs, 'max_timestep', (self.rows * self.cols) // 2))

        #termination status
        self.terminated = False
        #done status
        self.done = False

        #known path
        self.visited_state = set()
    
    def _get(self, dict, key, default):
        return dict[key] if key in dict != None else default


    
    #Reset environmentj
    def reset(self):
        self.agent_pos = [0,0]
        self.goal_pos = [self.rows - 1, self.cols - 1]
        self.set_state(self.agent_pos, self.goal_pos)
        self.timestep = 0
        self.terminated = False
        self.done = False
        self.visited_state = set()
        
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

        #agent is penalized for out of bound
        if x < 0 or x == self.cols or y < 0 or y == self.rows:
            return self.reward_dict['out_of_bound']
        
        if (x,y) in self.visited_state:
            return self.reward_dict['visited_state']

        self.visited_state.add((x,y))
        val = self.map[x][y]

        instant_reward = self.reward_dict[val] if val in self.reward_dict else 0
        return instant_reward

    #Step function: agent take step in env
    def step(self, action):
        """
        Take a step in this environemnt
        @params:
            action: int
                0:  up 
                1:  right 
                2:  down 
                3:  left
        @return
            new_state: int
            reward: int
            done: boolean
            terminated: boolean
            info: dict
        """

        if action == 0: 
            self.agent_pos[0] -= 1
        elif action == 1: 
            self.agent_pos[1] += 1
        elif action == 2: 
            self.agent_pos[0] += 1
        elif action == 3: 
            self.agent_pos[1] -= 1

        #Define your reward function
        reward = self.get_reward(self.agent_pos)

        #clip the agent position to avoid out of bound
        self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.cols - 1)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.rows - 1)

        self.set_state(self.agent_pos, self.goal_pos)
        observation = self.state.flatten()

        #Check if the agent takes too long to go to goal
        self.timestep += 1
        if self.timestep > self.max_timestep:
            self.terminated = True
            reward = self.reward_dict['terminated']
        
        if self.reached_goal(self.agent_pos):
            self.done = True
            reward = self.reward_dict['G']
        
        info = {}
        #return:
        #next state, argmax to get the new state of agent, np.argmax([0,0,1,0,0,0,0.5]) = 2
        #reward
        #done or not
        #terminated for exceeding time steps or not
        #extra infomation
        return np.argmax(observation), reward, self.done, self.terminated, info
    
    def render_simple(self, agent: QLearningAgent):
        actions = {
            0: 'up',
            1: 'right',
            2: 'down',
            3: 'left',
        }
        #if agent takes too long to complete, terminate
        max_iter = 20 
        iter =0 
        print('=' * 20)
        print('MAP')
        self.reset()
        for row in self.map:
            print(row)
        print('=' * 20)
        state = 0
        print(self.state)
        while iter < max_iter:
            iter += 1
            action = agent.get_action(state)
            state,reward,done,terminated,info = self.step(action)
            print(f'took action = {action}({actions[action]}) to state {state} done = {done}, terminated = {terminated}')
            print(self.state)
            print('=' * 20)
            if done or terminated:
                break
            time.sleep(0.5)
        print("exit")
        
    
    def render(self, agent: QLearningAgent, debug:bool = False) -> None:
        #Put a renderer here
        #render using pygame
        self.renderer = GridWorldRenderer(self.rows, self.cols, map = self.map)
        self.reset()
        self.renderer.update(self.state)
        pygame.time.wait(1000)
        max_iter = self.rows * self.cols
        iter =0 
        curr_state = 0
        done = terminated = False
        while True:
            if done or terminated:
                continue
            observation = self.state.flatten()
            curr_state = np.argmax(observation)
            action = agent.get_action(curr_state)
            curr_state, reward,done,terminated,info = self.step(action)
            if self.renderer.update(self.state) == False:
                break
            # if debug:
            #     print(f'curr_state = {curr_state}, took action ={action}')

            # pygame.time.wait(500)
            iter += 1
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

#  Visual Render using Pygame

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

    def _drawstate(self):
        self.screen.fill(colors["grass"])
        for i, y in enumerate(range(0, self._window_height, self.cell_size)):
            for j, x in enumerate(range(0, self._window_width, self.cell_size)):
                if self.map[i][j] == 'G':
                    self.draw_coin(self.screen, x,y)

                elif self.map[i][j] == 'R':
                    self.draw_rock(self.screen, x,y)

                elif self.map[i][j] == 'B':
                    self.draw_battery(self.screen, x,y)

                elif self.map[i][j] == 'S':
                    self.draw_shit(self.screen, x,y)
                else:
                    self.draw_grass(self.screen, x,y)
                
                if self.state[i][j] == 1:
                    self.draw_agent(self.screen, x,y)

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
                return False

        self._drawstate()
        pygame.display.update()
        self.clock.tick(GAME_SPEED)
        return True
        
    def end(self):
        print('exit')
        pygame.display.quit()
        pygame.quit()
        # sys.exit(1)

    
    #call a while  loop to update pygame drawing
    # def run(self):
    #     default_state = np.zeros((self.rows, self.columns))
    #     default_state[(0,0)] = 1
    #     while True:
    #         self.update(default_state)
