import numpy as np
import random
import copy
import pygame
import time

"""Constants"""
BLACK = (0, 0, 0)
GRAY = (192, 192, 192)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
LIGHT_STEEL_BLUE = (167, 190, 211)
PRUSSIAN_BLUE = (13, 44, 84)
EMERALD = (111, 208, 140)
WINE = (115, 44, 44)
APRICOT = (255, 202, 175)
ORANGE_YELLOW = (245, 183, 0)

NOOP_ALLOWED = True

"""Experiment Conditions"""
EXP_CONDS = {
    "NR": [],
    "GR": ['GR'],
    "VG": ['explanation', 'viable goals'],
}
EXPERIMENTS = {
    "split": [
        10, 10,
        [[7, 4], [7, 8]],
        0,
        [[2, 1], [8, 1]],
        [0, 6],
        [5, 1],
        [600, 600]],
    "clusters": [
        10, 10,
        [[7, 0], [8, 0], [9, 0], [2, 9], [1, 9], [0, 9]],
        5,
        [[3, 3], [5, 4], [5, 3], [5, 2], [3, 4], [3, 2]],
        [4, 3],
        [4, 6],
        [600, 600]],
    "deceive-0 (flipped)": [
        10, 10,
        [[4, 1], [4, 0], [6, 0]],
        1,
        [[0, 9], [9, 9], [9, 8]],
        [4, 9],
        [3, 9],
        [600, 600]],
    "deceive-1": [
        10, 10,
        [[1, 8], [1, 2], [2, 9], [6, 9]],
        2,
        [[8, 0], [6, 0], [8, 6], [6, 7]],
        [4, 2],
        [5, 2],
        [600, 600]],
    "deceive-2": [
        10, 10,
        [[4, 9], [5, 9], [8, 0], [5, 8]],
        1,
        [[9, 0], [0, 9], [0, 8], [9, 1]],
        [1, 1],
        [0, 1],
        [600, 600]],
}
TUTORIALS = {
    # Station at every corner of square, worker in middle (for instructions)
    "1": [
        10, 10,
        [[0, 0], [9, 0], [0, 9], [9, 9]],
        0,
        [[4, 6], [4, 5], [4, 3], [4, 4]],
        [5, 4],
        [0, 4],
        [600, 600],
        17, ],
    "2": [
        10, 10,
        [[4, 8], [4, 9], [6, 9]],
        1,
        [[0, 0], [9, 0], [9, 1]],
        [4, 0],
        [3, 0],
        [600, 600],
        27],
}


# Replaces np.max
def maxElement(array):
    maxElem = 0
    for elem in array:
        if elem > maxElem:
            maxElem = elem

    return maxElem


# Replaces np.argmax
def maxIndex(array):
    maxElem = 0
    maxIndex = -1
    curIndex = 0
    for elem in array:
        if elem > maxElem:
            maxElem = elem
            maxIndex = curIndex
        curIndex += 1

    return maxIndex


# Replaces np.array_equal
def checkEqual(a1, a2):
    if len(a1) != len(a2):
        return False

    for i in range(len(a1)):
        if a1[i] != a2[i]:
            return False
    return True


# Replaces np.logical_and
# input is two arrays with values 0 and 1
def logicalAnd(a1, a2):
    result = np.array([0] * 4)
    for i in range(len(a1)):
        result[i] = a1[i] * a2[i]
    return result


def get_valid_actions(obs, agent):
    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    valid_actions = np.array([1] * 4)  # NOOP is always valid
    for stn in range(len(s_pos)):
        if agent.probs[stn] == 0:
            continue
        tool_valid_actions = np.array([1] * 4)
        if f_pos[0] <= t_pos[stn][0]:
            tool_valid_actions[1] = 0  # Left
        if f_pos[0] >= t_pos[stn][0]:
            tool_valid_actions[0] = 0  # Right
        if f_pos[1] >= t_pos[stn][1]:
            tool_valid_actions[2] = 0  # Down
        if f_pos[1] <= t_pos[stn][1]:
            tool_valid_actions[3] = 0  # Up
        valid_actions = logicalAnd(valid_actions, tool_valid_actions)
    return valid_actions


def never_query(obs, agent):
    return None




class FetcherQueryPolicy:
    """
    Basic Fetcher Policy for querying, follows query_policy function argument (defaults to never query)
    Assumes all tools are in same location
    """

    def __init__(self, query_policy=never_query, prior=None, epsilon=0.0):
        self.query_policy = query_policy
        self._prior = prior
        self.probs = copy.deepcopy(self._prior)
        self.query = None
        self.prev_w_pos = None
        self._epsilon = epsilon

    def reset(self):
        self.probs = copy.deepcopy(self._prior)
        self.query = None
        self.prev_w_pos = None

    def make_inference(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.prev_w_pos is None:
            return
        if w_action == 5:
            for i, stn in enumerate(s_pos):
                if not checkEqual(stn, self.prev_w_pos):
                    self.probs[i] *= self._epsilon
        elif w_action == 0:
            for i, stn in enumerate(s_pos):
                if stn[0] <= self.prev_w_pos[0]:
                    self.probs[i] *= self._epsilon
        elif w_action == 1:
            for i, stn in enumerate(s_pos):
                if stn[0] >= self.prev_w_pos[0]:
                    self.probs[i] *= self._epsilon
        elif w_action == 3:
            for i, stn in enumerate(s_pos):
                if stn[1] >= self.prev_w_pos[1]:
                    self.probs[i] *= self._epsilon
        elif w_action == 2:
            for i, stn in enumerate(s_pos):
                if stn[1] <= self.prev_w_pos[1]:
                    self.probs[i] *= self._epsilon

        self.probs /= np.sum(self.probs)

    def action_to_goal(self, pos, goal):
        actions = []
        if pos[0] < goal[0]:
            actions.append(0)
        elif pos[0] > goal[0]:
            actions.append(1)
        if pos[1] > goal[1]:
            actions.append(3)
        elif pos[1] < goal[1]:
            actions.append(2)
        if len(actions) == 0:
            return 4
        return random.choice(actions)

    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.probs is None:
            self.probs = np.ones(len(s_pos))
            self.probs /= np.sum(self.probs)
        if answer is not None:
            if answer:
                for stn in range(len(s_pos)):
                    if stn not in self.query:
                        self.probs[stn] = 0
            else:
                for stn in self.query:
                    self.probs[stn] = 0
            self.probs /= np.sum(self.probs)
        else:
            self.make_inference(obs)
        self.prev_w_pos = np.array(w_pos)

        self.query = self.query_policy(obs, self)
        if self.query is not None:
            return 5, self.query

        if maxElement(self.probs) < (1 - self._epsilon):
            # dealing with only one tool position currently
            if checkEqual(f_pos, t_pos[0]):
                return 4, None
            else:
                return self.action_to_goal(f_pos, t_pos[0]), None
        else:
            if f_tool != maxIndex(self.probs):
                if checkEqual(f_pos, t_pos[0]):
                    return 6, maxIndex(self.probs)
                else:
                    return self.action_to_goal(f_pos, t_pos[0]), None
            return self.action_to_goal(f_pos, s_pos[maxIndex(self.probs)]), None




class FetcherAltPolicy(FetcherQueryPolicy):
    """
    More Complicated Fetcher Policy, allows for multiple tool locations
    """

    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.probs is None:
            self.probs = np.ones(len(s_pos))
            self.probs /= np.sum(self.probs)
        if answer is not None:
            if answer:
                for stn in range(len(s_pos)):
                    if stn not in self.query:
                        self.probs[stn] = 0
            else:
                for stn in self.query:
                    self.probs[stn] = 0
            self.probs /= np.sum(self.probs)
        else:
            self.make_inference(obs)

        self.prev_w_pos = np.array(w_pos)

        # One station already guaranteed. No querying needed.
        if maxElement(self.probs) >= (1 - self._epsilon):
            target = maxIndex(self.probs)
            if f_tool != target:
                if checkEqual(f_pos, t_pos[target]):
                    return 6, target
                else:
                    return self.action_to_goal(f_pos, t_pos[target]), None
            return self.action_to_goal(f_pos, s_pos[target]), None

        self.query = self.query_policy(obs, self)

        if self.query is not None:
            return 5, self.query

        valid_actions = get_valid_actions(obs, self)

        if np.any(valid_actions):
            p = valid_actions / np.sum(valid_actions)
            i = 0
            num_valid_actions = []
            for x in valid_actions:
                if x:
                    num_valid_actions.append(i)
                i += 1
            action_idx = random.choice(num_valid_actions)
            return action_idx, None
        else:
            return 4, None




class FetcherYotamPolicy(FetcherQueryPolicy):
    """
    More Complicated Fetcher Policy, allows for multiple tool locations
    """

    def __int__(self):
        self.inferred_goals = None

    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.probs is None:
            self.probs = np.ones(len(s_pos))
            self.probs /= np.sum(self.probs)

        self.make_inference(obs)

        self.prev_w_pos = np.array(w_pos)

        # One station already guaranteed. No querying needed.
        if maxElement(self.probs) >= (1 - self._epsilon):
            self.inferred_goals = [maxIndex(self.probs)]
            target = maxIndex(self.probs)
            if f_tool != target:
                if checkEqual(f_pos, t_pos[target]):
                    return 6, target
                else:
                    return self.action_to_goal(f_pos, t_pos[target]), None
            return self.action_to_goal(f_pos, s_pos[target]), None
        else:
            self.inferred_goals = [i for i in range(len(self.probs)) if
                                   self.probs[i] == maxElement(self.probs)]

        # actions that are optimal even when the goal is unknown
        valid_actions = self.get_relevant_actions(obs)

        if np.any(valid_actions):
            p = valid_actions / np.sum(valid_actions)
            i = 0
            num_valid_actions = []
            for x in valid_actions:
                if x:
                    num_valid_actions.append(i)
                i += 1
            action_idx = random.choice(num_valid_actions)
            return action_idx, None
        else:
            return 4, None

    # Returns list of valid actions that brings fetcher closer to all relevant tools
    def get_relevant_actions(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

        valid_actions = np.array([1] * 4)  # NOOP is always valid
        for stn in self.inferred_goals:
            if self.probs[stn] == 0:
                continue
            tool_valid_actions = np.array([1] * 4)
            if f_pos[0] <= t_pos[stn][0]:
                tool_valid_actions[1] = 0  # Left
            if f_pos[0] >= t_pos[stn][0]:
                tool_valid_actions[0] = 0  # Right
            if f_pos[1] >= t_pos[stn][1]:
                tool_valid_actions[2] = 0  # Down
            if f_pos[1] <= t_pos[stn][1]:
                tool_valid_actions[3] = 0  # Up
            valid_actions = logicalAnd(valid_actions, tool_valid_actions)
        return valid_actions   


class GUI:

    def __init__(self, num_cols, num_rows, stn_pos, goal_stn, tool_pos, worker_pos, fetcher_pos,
                 tutorial, condition, size, current_scenrio, current_repetition):

        pygame.init()
        self.running = True
        self.clock = pygame.time.Clock()
        self.pause_screen = True
        self.tutorial = tutorial

        self.num_game_rows = num_rows
        num_rows += 1  # -- Yotam: The top row will be the explanations interface.

        # Dimensions and sizes
        self.size = self.width, self.height = size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.box_width = self.width / num_cols
        self.box_height = self.height / num_rows
        self.x_margin = self.box_width / 10
        self.y_margin = self.box_height / 10
        self.radius = self.box_width / 3.2

        # Stations
        self.stn_pos = stn_pos
        self.tool_pos = tool_pos
        self.goal_stn = goal_stn

        # Worker
        self.user = [worker_pos[0], worker_pos[1]]
        self.prev_user = [worker_pos[0], worker_pos[1]]
        self.arrived = False

        # Fetcher
        self.robot = [fetcher_pos[0], fetcher_pos[1]]
        self.prev_robot = [fetcher_pos[0], fetcher_pos[1]]
        self.pickup_tool = -1
        self.holding_tool = None
        self.robot_stay = False

        # Font
        self.font = pygame.font.SysFont("lucidaconsole", int(10 * self.height / 1080))

        # Colors --Yotam
        rgb_list = [
            GREEN,  # Green
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (192, 192, 192),  # Silver
            (128, 128, 128)  # Gray
        ]
        # random.shuffle(rgb_list)  # if you want random color assignment
        # self.colors = [list(np.random.choice(range(256), size=3)) for _ in stn_pos]
        assert len(stn_pos) <= len(rgb_list), "More stations that colors defined"
        self.colors = rgb_list[:len(stn_pos)]

        # Experiment condition --Yotam
        self.condition = condition

        self.scenario = current_scenrio
        self.iteration = current_repetition
        self.steps = 0

        if (self.on_init() == False):
            self.running = False

    def on_init(self):
        # Set screen to windowed size
        # self.screen =  pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        # Set screen to fullscreen
        self.screen = pygame.display.set_mode(self.size)

        self.draw_pause_screen()
        # self.draw_experiment_screen()
        self.running = True

    # Rectangular station
    def render_station(self, color, stn, tool=False, goal=False):
        rect = [self.box_width * stn[0] + self.x_margin,
                self.box_height * (self.num_rows - 1 - stn[1]) + self.y_margin,
                self.box_width - (self.x_margin * 2),
                self.box_height - (self.y_margin * 2)]
        width = 5 if tool else 0
        pygame.draw.rect(
            self.screen,
            color,
            rect,
            width
        )
        if goal:
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                rect,
                5
            )

    def render_all_stations(self):
        # Worker stations
        for i in range(len(self.stn_pos)):
            stn = self.stn_pos[i]
            tool = self.tool_pos[i]
            color = self.colors[i]
            if i == self.goal_stn:
                self.render_station(color, stn, goal=True)
            else:
                self.render_station(color, stn)
            self.render_text(str(i + 1), stn[0], stn[1])
            self.render_station(color, tool, tool=True)
            self.render_text("T", tool[0], tool[1])

    # Circular agent
    def render_agent(self, circle_x, circle_y, color, tool_color=False):
        gui_x = circle_x * self.box_width + (self.box_width / 2)
        gui_y = (self.num_rows - 1 - circle_y) * self.box_height + (self.box_height / 2)
        pygame.draw.circle(self.screen, color, (int(gui_x), int(gui_y)), int(self.radius))
        if tool_color:
            start_x, end_x = circle_x * self.box_width, (circle_x + 1) * self.box_width
            for i in range(-1, 3):
                pygame.draw.line(self.screen, tool_color, (start_x + 20, gui_y + 10 + i),
                                 (end_x - 20, gui_y + 10 + i))

    # Text within station or agent
    def render_text(self, textString, box_x, box_y, color=BLACK):
        text_x = box_x * self.box_width + self.x_margin * 3
        text_y = (self.num_rows - 1 - box_y) * self.box_height + self.y_margin * 3

        text = self.font.render(textString, True, color)
        self.screen.blit(text,
                         (text_x, text_y)
                         )

    # Pause screen
    def draw_pause_screen(self):
        self.font = pygame.font.SysFont("lucidaconsole", 20)
        self.screen.fill(GRAY)
        txt_pos_y, txt_pos_x, spacing, = 25, self.width / 2 - 110, 50
        strng = "Game number {}".format(self.scenario + 1)
        text = self.font.render(strng, True, WHITE)
        self.screen.blit(text, (self.width / 2 - 110, txt_pos_y))
        txt_pos_y += spacing
        strng = "goal station is number {}".format(self.goal_stn + 1)
        text = self.font.render(strng, True, BLUE)
        self.screen.blit(text, (txt_pos_x, txt_pos_y))
        txt_pos_y += spacing
        text = self.font.render("Tab - Pause/Unpause", True, BLACK)
        self.screen.blit(text, (txt_pos_x, txt_pos_y))
        txt_pos_y += spacing
        text = self.font.render("Up - Move up", True, BLACK)
        self.screen.blit(text, (txt_pos_x, txt_pos_y))
        txt_pos_y += spacing
        text = self.font.render("Left - Move left", True, BLACK)
        self.screen.blit(text, (txt_pos_x, txt_pos_y))
        txt_pos_y += spacing
        text = self.font.render("Down - Move down", True, BLACK)
        self.screen.blit(text, (txt_pos_x, txt_pos_y))
        txt_pos_y += spacing
        text = self.font.render("Right - Move right", True, BLACK)
        self.screen.blit(text, (txt_pos_x, txt_pos_y))
        txt_pos_y += spacing
        text = self.font.render("Space - Work (at station)", True, BLACK)
        self.screen.blit(text, (txt_pos_x, txt_pos_y))
        txt_pos_y += spacing
        text = self.font.render("Enter - Wait (don't move)", True, BLACK)
        self.screen.blit(text, (txt_pos_x, txt_pos_y))
        txt_pos_y += spacing
        text = self.font.render("Press Tab to go to the game", True, RED)
        self.screen.blit(text, (txt_pos_x, txt_pos_y))
        pygame.display.flip()

    def render_explanation(self, color):
        rect = [0,
                0,
                self.width,
                self.box_height]

        pygame.draw.rect(
            self.screen,
            color,
            rect
        )

    def draw_explanation_row(self):
        self.render_explanation(WHITE)  # Remove old explanation

    def draw_explanation(self, condition, inferred_goals=[]):
        self.font = pygame.font.SysFont("lucidaconsole", 20)
        goals = " ,".join([str(x + 1) for x in inferred_goals])
        self.render_text("Relevant stations: {}".format(goals), 0, self.num_rows - 1)

    def draw_steps(self):
        self.font = pygame.font.SysFont("lucidaconsole", 20)
        self.render_text("Steps: {}".format(self.steps), self.num_cols - 3, self.num_rows - 1)

    # Button
    def render_button(self):
        x = 200
        y = 200
        width = 100
        height = 50
        color = (255, 0, 0)
        text = "Reveal" 
        # Draw the button
        pygame.draw.rect(color, (x, y, width, height), 0)
        font = pygame.font.SysFont('comicsans', 60)
        text = font.render(self.text, 1, (0,0,0))
        self.screen.blit(text, (self.x + round(self.width/2) - round(text.get_width()/2),
                        self.y + round(self.height/2) - round(text.get_height()/2)))

    def is_over_button(self, pos):
        # Pos is the mouse position or a tuple of (x,y) coordinates
        if pos[0] > self.x and pos[0] < self.x + self.width:
            if pos[1] > self.y and pos[1] < self.y + self.height:
                return True
        return False

    def draw_experiment_screen(self):
        self.screen.fill(WHITE)
        vert_line_start = self.box_height
        self.draw_explanation_row()
        self.draw_steps()
        if "explanation" in self.condition:
            all_stations = list(range(len(self.stn_pos)))
            self.draw_explanation(self.condition, inferred_goals=all_stations)

        self.font = pygame.font.SysFont("lucidaconsole", int(self.height / self.num_cols * 0.35))

        # Grid lines
        for x in range(1, self.num_cols):
            point1 = (x * self.box_width, vert_line_start)
            point2 = (x * self.box_width, self.height)
            pygame.draw.line(self.screen, BLACK, point1, point2)

        for y in range(1, self.num_rows):
            point1 = (0, y * self.box_height)
            point2 = (self.width, y * self.box_height)
            pygame.draw.line(self.screen, BLACK, point1, point2)

        # Stations
        self.render_all_stations()

        # Worker
        self.render_agent(self.prev_user[0], self.prev_user[1], BLUE)
        self.render_text("W", self.prev_user[0], self.prev_user[1])

        # Fetcher
        self.render_agent(self.prev_robot[0], self.prev_robot[1], GRAY)
        self.render_text("F", self.prev_robot[0], self.prev_robot[1])

        # Tutorial Text
        if self.tutorial:
            text = "You don't need to go to the toolbox" if self.goal_stn == 0 else "You can move through stations"
            self.render_text(text, 0, 5, BLACK)
        
        # button
        

            
        pygame.display.flip()

    # Render drawing
    def on_render(self, inferred_goals):
        if self.running and not self.pause_screen:

            self.render_station(WHITE, self.prev_user)  # Remove old user agent
            self.render_station(WHITE, self.prev_robot)  # Remove old robot agent
            self.render_all_stations()  # If agent overlay

            # User
            self.render_agent(self.user[0], self.user[1], BLUE)
            self.render_text("W", self.user[0], self.user[1])

            # Robot
            if self.robot_stay:
                self.robot_stay = False

            fetcher_color = GRAY
            if 'GR' in self.condition and len(inferred_goals) == 1:
                fetcher_color = self.colors[inferred_goals[0]]

            if self.holding_tool:
                self.render_agent(self.robot[0], self.robot[1], fetcher_color, self.holding_tool)
            else:
                self.render_agent(self.robot[0], self.robot[1], fetcher_color)

            self.render_text("F", self.robot[0], self.robot[1])

            if self.tutorial:
                text = "You don't need to go to the Tool station" if self.goal_stn == 0 else "You can pass through stations"
                self.render_text(text, 0, 5, BLACK)

            self.draw_explanation_row()
            self.draw_steps()
            # explanation
            if "explanation" in self.condition:  # -- Yotam
                self.draw_explanation(self.condition, inferred_goals)
                self.font = pygame.font.SysFont("lucidaconsole",
                                                int(self.height / self.num_cols * 0.35))
                
            # draw button
                # self.button.draw()    
            pygame.display.flip()

    # Close pygame when finished
    def on_cleanup(self):
        self.font = pygame.font.SysFont("lucidaconsole", 30)
        self.screen.fill(GRAY)

        text = self.font.render("Task complete!", True, WHITE)
        self.screen.blit(text, (self.width / 2 - 100, 100))

        # text = self.font.render("Please upload the downloaded file", True, BLUE)
        # self.screen.blit(text, (self.width / 2 - 230, 250))
        # text = self.font.render("to the survey to progress", True, BLUE)
        # self.screen.blit(text, (self.width / 2 - 230, 300))
        text = self.font.render("Click on the button below", True, RED)
        self.screen.blit(text, (self.width / 2 - 230, 400))
        text = self.font.render("to copy your completion code", True, RED)
        self.screen.blit(text, (self.width / 2 - 230, 450))
        # text = self.font.render("27384632", True, RED)
        # self.screen.blit(text, (self.width / 2 - 100, 300))
        pygame.display.flip()
        pygame.quit()

    def on_end_level(self, game_best_score, tutorial, optimal_steps, trajectory_analysis=False):
        self.font = pygame.font.SysFont("lucidaconsole", 25)
        self.screen.fill(GRAY)
        strng = "Task completed in {} steps".format(self.steps)
        text = self.font.render(strng, True, WHITE)
        self.screen.blit(text, (self.width / 2 - 150, 100))

        if tutorial:
            optimal = "YES" if self.steps <= optimal_steps else "NO"
            strng = "Completed in optimal number of steps: {}".format(optimal)
            text = self.font.render(strng, True, WHITE)
            self.screen.blit(text, (self.width / 2 - 250, 200))
        else:
            strng = "Current best score: {}".format(game_best_score)
            text = self.font.render(strng, True, WHITE)
            self.screen.blit(text, (self.width / 2 - 150, 200))

        text = self.font.render("Press Tab to continue", True, WHITE)
        self.screen.blit(text, (self.width / 2 - 150, 400))
        pygame.display.flip()

        if trajectory_analysis:
            time.sleep(2)
            return

        #  Wait for TAB to continue to next level
        while self.running:
            self.clock.tick()
            e = pygame.event.wait()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_TAB:  # Pause / unpause
                    return

    # Move fetcher agent (robot)
    def _move_agent(self, other_agent_move, inferred_goals):
        self.prev_robot[0] = self.robot[0]
        self.prev_robot[1] = self.robot[1]
        move = other_agent_move[0]

        if move == 0:  # Right
            self.robot[0] += 1
        elif move == 1:  # Left
            self.robot[0] -= 1
        elif move == 2:  # Up
            if (self.robot[1] + 1) < self.num_game_rows:
                self.robot[1] += 1
        elif move == 3:  # Down
            self.robot[1] -= 1
        elif move == 4:  # NOOP
            self.robot_stay = True
        elif move == 6:  # pickup
            self.pickup_tool = other_agent_move[1]
            self.holding_tool = self.colors[inferred_goals[0]]
        # else:
        #     print("move agent 5")

    # Pygame event (key down)
    def on_event(self, e):

        # Experiment screen
        if not self.pause_screen:
            self.prev_user[0] = self.user[0]
            self.prev_user[1] = self.user[1]

            if e.key == pygame.K_LEFT:
                if (self.user[0] - 1) >= 0:
                    self.user[0] -= 1
                return 1
            elif e.key == pygame.K_RIGHT:
                if (self.user[0] + 1) < self.num_cols:
                    self.user[0] += 1
                return 0
            elif e.key == pygame.K_DOWN:
                if (self.user[1] - 1) >= 0:
                    self.user[1] -= 1
                return 3
            elif e.key == pygame.K_UP:
                if (self.user[1] + 1) < self.num_game_rows:
                    self.user[1] += 1
                return 2
            elif e.key == pygame.K_SPACE:  # Work
                self.arrived = self.user == self.stn_pos[self.goal_stn]
                return 5
            elif e.key == pygame.K_RETURN and NOOP_ALLOWED:  # NOOP
                return 4

        # Valid input for both pause screen and experiment screen
        if e.key == pygame.K_TAB:  # Pause / unpause
            self.pause_screen = not self.pause_screen  # Switch pause screen/experiment screen
            if self.pause_screen:
                self.draw_pause_screen()
            else:
                self.draw_experiment_screen()
        elif e.key == pygame.K_BACKSPACE:  # End simulation
            return -1

    # Move fetcher and get user action
    def on_execute(self, other_agent_move, inferred_goals):
        self._move_agent(other_agent_move, inferred_goals)
        action = None

        while self.running:
            self.clock.tick()
            # User input
            if not self.arrived:
                e = pygame.event.wait()
                if e.type == pygame.KEYDOWN:
                    action = self.on_event(e)

                # pos = pygame.mouse.get_pos()
                # if e.type == pygame.MOUSEBUTTONDOWN:
                #     if self.button.is_over(pos):
                #         print("Button Clicked!")
                             
            else:
                action = 5
                time.sleep(0.2)  # yotam added
            # Got input, return action, worker_pos, and fetcher_pos
            if action != None:
                self.steps += 1
                self.on_render(inferred_goals)
                return action, self.user, self.robot

        return -1, self.user, self.robot


def write_file(worker_action, fetcher_action, time):
    worker_actions = {0: "RIGHT", 1: "LEFT", 2: "UP", 3: "DOWN", 4: "NOOP", 5: "WORK"}
    fetcher_actions = {0: "RIGHT", 1: "LEFT", 2: "UP", 3: "DOWN", 4: "NOOP", 6: "PICKUP"}

    print("{0:15} {1:15} {2:15f}".format(
        worker_actions[worker_action],
        fetcher_actions[fetcher_action],
        time
    )
    )


def run_exp(condition, tutorial=False):
    """Run the experiment"""
    print("date")  # Prints date to output file
    condition = EXP_CONDS[condition]
    repetitions = 3
    """ Environments: [Num Cols, Num Rows, Stations, Goal, Tool, Worker, Fetcher] """
    exp = {"tutorial_"+tutorial: TUTORIALS[tutorial]} if tutorial else EXPERIMENTS

    for i, scenario in enumerate(exp.items()):
        scenario_name, scenario_values = scenario
        game_best_score = float("inf")
        for r in range(repetitions):
            cols = scenario_values[0]
            rows = scenario_values[1]
            stn_pos = scenario_values[2]
            goal_stn = scenario_values[3]
            tool_pos = scenario_values[4]
            worker_pos = scenario_values[5]
            fetcher_pos = scenario_values[6]
            size = scenario_values[7]
            if tutorial:
                optimal_steps = scenario_values[8]
            # Set up pygame gui
            gui = GUI(cols, rows, stn_pos, goal_stn, tool_pos, worker_pos, fetcher_pos, False,
                      condition, size, i, r)
            print("EXPERIMENT #{num}".format(num=i))
            print("Scenario name: {}".format(scenario_name))
            print("{0:15} {1:15} {2:15}\n".format("WORKER ACTION", "FETCHER ACTION",
                                                  "TIME ELAPSED"))

            # Set up fetcher robot
            # fetcher = FetcherQueryPolicy()
            fetcher = FetcherYotamPolicy(epsilon=0.05)

            # Observation state
            f_obs = [worker_pos, fetcher_pos, stn_pos, tool_pos, None, None, None, None]
            done = False

            # Loop actions until expreiment is complete
            while not done:
                # Get fetcher move
                fetcher_move = fetcher(f_obs)
                inferred_goals = fetcher.inferred_goals

                gui.on_render(inferred_goals)

                # Get user action
                t0 = time.time()
                action, worker_pos, fetcher_pos = gui.on_execute(fetcher_move, inferred_goals)
                t1 = time.time()
                # Escape (backspace button)
                if action == -1:
                    done = True
                    break

                # Write actions to file
                write_file(action, fetcher_move[0], t1 - t0)

                # working and finished
                if (action == 5 and
                        fetcher_pos == worker_pos and
                        gui.pickup_tool == goal_stn and
                        worker_pos == stn_pos[goal_stn]):
                    done = True

                # Move pickup tool
                if gui.pickup_tool != -1:
                    modified_tool_pos = copy.deepcopy(tool_pos)
                    modified_tool_pos[gui.pickup_tool] = fetcher_pos
                    f_obs[3] = modified_tool_pos
                    f_obs[4] = gui.pickup_tool

                # Modify observation state
                f_obs[0] = worker_pos
                f_obs[1] = fetcher_pos
                f_obs[5] = action
                f_obs[6] = fetcher_move[0]

            game_best_score = gui.steps if gui.steps < game_best_score else game_best_score
            optimal_steps = scenario_values[8] if tutorial else False
            gui.on_end_level(game_best_score, tutorial, optimal_steps)
            print("finished in {} steps".format(gui.steps))
            gui.steps = 0
            print("done")

    print("complete")
    gui.screen.fill(pygame.Color("white"))
    pygame.display.update()
    gui.on_cleanup()


if __name__ == '__main__':
    run_exp("VG", "2")
    print()
