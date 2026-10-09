"""
AI AUTONOMOUS SNAKE - Deep Q-Network (PyTorch) + Turtle cyberpunk UI
--------------------------------------------------------------------
Install:  pip install torch numpy
Run:      python snake_ai.py

Controls (click the game window first):
    F          toggle TURBO (fast training, no drawing)
    UP / DOWN  faster / slower playback
    S          save the model now

The snake learns by trial and error. Rewards are shaped so it prefers the
SHORTEST route to food and finishes as fast as it can:
    +10   eat food
    -10   die (wall, itself, or wandering too long)
    +0.2  step that moves closer to the food
    -0.3  step that moves away from the food
"""

import os
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import turtle
from turtle import Screen, Turtle

# ----------------------------------------
# 1. SETTINGS & CYBERPUNK PALETTE
# ----------------------------------------
BG_COLOR = "#0A0B10"
GRID_COLOR = "#161824"
BORDER_COLOR = "#00FFEA"
SNAKE_HEAD = "#00FFEA"
SNAKE_TAIL = "#00604F"
FOOD_COLOR = "#FF007F"
FOOD_GLOW = "#4A0030"
HUD_COLOR = "#A0A5C0"
LASER_COLOR = "#FF007F"

CELL = 20
COLS = ROWS = 30                 # 30 x 30 board = 600 x 600 px
OX, OY = -450, -300              # bottom-left corner of the board on screen

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
MODEL_PATH = os.path.join("model", "snake_dqn.pth")

SETTINGS = {"turbo": False, "delay": 0.04, "save_now": False}

# Directions, clockwise (grid y points UP): RIGHT, DOWN, LEFT, UP
DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1)]


def lerp_color(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def cell_to_px(gx, gy):
    return OX + gx * CELL + CELL / 2, OY + gy * CELL + CELL / 2


# ----------------------------------------
# 2. GAME ENVIRONMENT
# ----------------------------------------
class SnakeGameAI:
    def __init__(self):
        self.screen = Screen()
        self.screen.title("AI AUTONOMOUS BRAIN v3.0 - Deep Q-Network")
        self.screen.bgcolor(BG_COLOR)
        self.screen.setup(width=1000, height=720)
        self.screen.tracer(0)
        self._draw_board()

        self.art = Turtle()          # redrawn every frame: snake, food, laser
        self.art.hideturtle()
        self.art.penup()
        self.art.speed(0)

        self.hud = Turtle()
        self.hud.hideturtle()
        self.hud.penup()
        self.hud.color(HUD_COLOR)

        self.info = {"gen": 0, "record": 0, "eps": 1.0, "conf": "-", "q": "[0, 0, 0]"}
        self._tick = 0

        self.screen.listen()
        self.screen.onkey(lambda: SETTINGS.update(turbo=not SETTINGS["turbo"]), "f")
        self.screen.onkey(lambda: SETTINGS.update(delay=max(0.0, SETTINGS["delay"] - 0.01)), "Up")
        self.screen.onkey(lambda: SETTINGS.update(delay=min(0.3, SETTINGS["delay"] + 0.01)), "Down")
        self.screen.onkey(lambda: SETTINGS.update(save_now=True), "s")

        self.reset()

    # ---- static board art ----
    def _draw_board(self):
        g = Turtle()
        g.hideturtle()
        g.speed(0)
        g.pensize(1)
        g.color(GRID_COLOR)
        for i in range(COLS + 1):
            x = OX + i * CELL
            g.penup(); g.goto(x, OY); g.pendown(); g.goto(x, OY + ROWS * CELL)
        for j in range(ROWS + 1):
            y = OY + j * CELL
            g.penup(); g.goto(OX, y); g.pendown(); g.goto(OX + COLS * CELL, y)
        g.penup()
        g.color(BORDER_COLOR)
        g.pensize(3)
        g.goto(OX, OY)
        g.pendown()
        for p in [(OX + COLS * CELL, OY), (OX + COLS * CELL, OY + ROWS * CELL),
                  (OX, OY + ROWS * CELL), (OX, OY)]:
            g.goto(p)
        g.penup()

    # ---- game logic ----
    def reset(self):
        cx, cy = COLS // 2, ROWS // 2
        self.direction = 0                                  # RIGHT
        self.head = (cx, cy)
        self.snake = [self.head, (cx - 1, cy), (cx - 2, cy)]
        self.score = 0
        self.frame = 0
        self.steps_since_food = 0
        self.food_steps = []
        self._place_food()

    def _place_food(self):
        free = [(x, y) for x in range(COLS) for y in range(ROWS) if (x, y) not in self.snake]
        self.food = random.choice(free) if free else self.head

    def is_collision(self, pt=None):
        pt = pt or self.head
        if pt[0] < 0 or pt[0] >= COLS or pt[1] < 0 or pt[1] >= ROWS:
            return True
        return pt in self.snake[1:]

    def _food_dist(self):
        return abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

    def play_step(self, action):
        """action = [straight, right_turn, left_turn] -> (reward, game_over, score)"""
        self.frame += 1
        self.steps_since_food += 1
        dist_before = self._food_dist()

        # turn relative to current heading
        if action[1] == 1:
            self.direction = (self.direction + 1) % 4
        elif action[2] == 1:
            self.direction = (self.direction - 1) % 4
        dx, dy = DIRS[self.direction]
        self.head = (self.head[0] + dx, self.head[1] + dy)
        self.snake.insert(0, self.head)

        # death: wall, self, or dithering for too long
        if self.is_collision() or self.frame > 100 * len(self.snake):
            self._render(dead=True)
            return -10, True, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.food_steps.append(self.steps_since_food)
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()
            reward = 0.2 if self._food_dist() < dist_before else -0.3

        self._render()
        return reward, False, self.score

    # ---- drawing ----
    def _render(self, dead=False):
        self._tick += 1
        if SETTINGS["turbo"] and not dead:
            if self._tick % 100 == 0:
                self.screen.update()        # keep keyboard events alive
            return

        a = self.art
        a.clear()

        # laser sight to the food
        hx, hy = cell_to_px(*self.head)
        fx, fy = cell_to_px(*self.food)
        a.pensize(1)
        a.color(LASER_COLOR)
        a.goto(hx, hy)
        a.pendown()
        a.goto(fx, fy)
        a.penup()

        # food: glow + apple + leaf
        a.goto(fx, fy)
        a.dot(28, FOOD_GLOW)
        a.dot(17, FOOD_COLOR)
        a.goto(fx + 2, fy + 8)
        a.dot(6, "#39FF14")

        # snake body: gradient tail -> head, rounded segments
        n = len(self.snake)
        for i in range(n - 1, 0, -1):
            x, y = cell_to_px(*self.snake[i])
            t = 1 - i / n
            a.goto(x, y)
            a.dot(CELL - 2, lerp_color(SNAKE_TAIL, SNAKE_HEAD, t * 0.8))

        # head + eyes
        head_color = "#FF2020" if dead else SNAKE_HEAD
        a.goto(hx, hy)
        a.dot(CELL + 2, head_color)
        dx, dy = DIRS[self.direction]
        px, py = -dy, dx
        for s in (-1, 1):
            ex = hx + dx * 4 + px * 4 * s
            ey = hy + dy * 4 + py * 4 * s
            a.goto(ex, ey); a.dot(6, "white")
            a.goto(ex + dx, ey + dy); a.dot(3, "black")

        self._draw_hud(dead)
        self.screen.update()
        if not dead and SETTINGS["delay"] > 0:
            time.sleep(SETTINGS["delay"])

    def _draw_hud(self, dead):
        h, i = self.hud, self.info
        h.clear()
        h.goto(-450, 315)
        h.write(f"[AI AUTONOMOUS BRAIN v3.0]    GEN: {i['gen']}    SCORE: {self.score}"
                f"    RECORD: {i['record']}", font=("Courier", 14, "bold"))
        avg = (sum(self.food_steps) / len(self.food_steps)) if self.food_steps else 0
        lines = [
            ("[ LIVE AI METRICS ]", "bold"),
            (f"> Mode: {'TURBO' if SETTINGS['turbo'] else 'WATCH'}", "normal"),
            (f"> Epsilon (explore): {i['eps']:.2f}", "normal"),
            (f"> Confidence: {i['conf']}", "normal"),
            (f"> Q-values: {i['q']}", "normal"),
            (f"> Snake length: {len(self.snake)}", "normal"),
            (f"> Avg steps/food: {avg:.1f}", "normal"),
            (f"> Steps this life: {self.frame}", "normal"),
            ("", "normal"),
            ("[System: CRITICAL FAILURE]" if dead else "[System: LEARNING]", "bold"),
            ("", "normal"),
            ("F turbo | UP/DOWN speed | S save", "normal"),
        ]
        y = 220
        for text, weight in lines:
            if text:
                h.goto(190, y)
                h.write(text, font=("Courier", 11 if weight == "bold" else 10, weight))
            y -= 28


# ----------------------------------------
# 3. STATE (11 values)
# ----------------------------------------
def get_state(game):
    head = game.head
    d = game.direction
    straight = DIRS[d]
    right = DIRS[(d + 1) % 4]
    left = DIRS[(d - 1) % 4]

    def ahead(v):
        return (head[0] + v[0], head[1] + v[1])

    state = [
        game.is_collision(ahead(straight)),     # danger straight
        game.is_collision(ahead(right)),        # danger right
        game.is_collision(ahead(left)),         # danger left
        d == 2, d == 0, d == 3, d == 1,         # dir L, R, U, D
        game.food[0] < head[0],                 # food left
        game.food[0] > head[0],                 # food right
        game.food[1] > head[1],                 # food up
        game.food[1] < head[1],                 # food down
    ]
    return np.array(state, dtype=int)


# ----------------------------------------
# 4. NEURAL NETWORK + TRAINER
# ----------------------------------------
class LinearQNet(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

    def save(self, extra):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save({"model": self.state_dict(), **extra}, MODEL_PATH)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if state.dim() == 1:
            state, next_state = state.unsqueeze(0), next_state.unsqueeze(0)
            action, reward = action.unsqueeze(0), reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone().detach()
        for i in range(len(done)):
            q_new = reward[i]
            if not done[i]:
                q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            target[i][torch.argmax(action[i]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


# ----------------------------------------
# 5. AGENT
# ----------------------------------------
class Agent:
    def __init__(self):
        self.n_games = 0
        self.record = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, LR, GAMMA)
        if os.path.exists(MODEL_PATH):                       # resume training
            ckpt = torch.load(MODEL_PATH)
            self.model.load_state_dict(ckpt["model"])
            self.n_games = ckpt.get("n_games", 0)
            self.record = ckpt.get("record", 0)
            print(f"Loaded saved brain: {self.n_games} games, record {self.record}")

    @property
    def epsilon(self):
        return max(0.01, 0.8 * (0.97 ** self.n_games))

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def train_long_memory(self):
        batch = random.sample(self.memory, BATCH_SIZE) if len(self.memory) > BATCH_SIZE else self.memory
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, s, a, r, ns, done):
        self.trainer.train_step(s, a, r, ns, done)

    def get_action(self, state):
        """Epsilon-greedy. Returns (one-hot action, confidence text, q text)."""
        move = [0, 0, 0]
        with torch.no_grad():
            q = self.model(torch.tensor(state, dtype=torch.float))
            probs = torch.softmax(q, dim=0)
        q_text = "[" + ", ".join(f"{v:.1f}" for v in q.tolist()) + "]"
        if random.random() < self.epsilon:
            move[random.randint(0, 2)] = 1
            return move, "EXPLORING", q_text
        idx = int(torch.argmax(q).item())
        move[idx] = 1
        return move, f"{probs[idx].item() * 100:.1f}%", q_text


# ----------------------------------------
# 6. MAIN TRAINING LOOP
# ----------------------------------------
def train():
    agent = Agent()
    game = SnakeGameAI()
    state_old = get_state(game)
    total_score = 0

    while True:
        action, conf, q_text = agent.get_action(state_old)
        game.info.update(gen=agent.n_games, record=agent.record, eps=agent.epsilon,
                         conf=conf, q=q_text)

        reward, done, score = game.play_step(action)
        state_new = get_state(game)

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)
        state_old = state_new

        if SETTINGS["save_now"]:
            SETTINGS["save_now"] = False
            agent.model.save({"n_games": agent.n_games, "record": agent.record})

        if done:
            game.reset()
            state_old = get_state(game)
            agent.n_games += 1
            agent.train_long_memory()
            total_score += score
            if score > agent.record:
                agent.record = score
                agent.model.save({"n_games": agent.n_games, "record": agent.record})
            print(f"Game {agent.n_games:4d} | Score {score:3d} | Record {agent.record:3d} "
                  f"| Mean {total_score / agent.n_games:.2f} | eps {agent.epsilon:.2f}")


if __name__ == "__main__":
    try:
        train()
    except (turtle.Terminator, KeyboardInterrupt):
        print("Window closed - bye!")