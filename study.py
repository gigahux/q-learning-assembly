import os
import time
import random
import numpy as np


class MazeEnv:
    def __init__(self):
        self.maze = [
            ['.', '.', '.', '.', '.', 'S'],
            ['.', 'X', 'X', '.', 'X', '.'],
            ['.', '.', 'X', '.', 'X', '.'],
            ['X', '.', 'X', '.', '.', '.'],
            ['X', '.', 'X', 'X', 'X', '.'],
            ['.', '.', '.', 'G', 'X', 'X']
        ]
        self.start_pos = (0, 5)  # 'S' start position
        self.goal_pos = (5, 3)  # 'G' goal position
        self.state = self.start_pos
        self.steps_taken = 0  # Track steps taken by the agent

    def reset(self):
        self.state = self.start_pos
        self.steps_taken = 0
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # Up
            new_state = (x - 1, y)
        elif action == 1:  # Down
            new_state = (x + 1, y)
        elif action == 2:  # Left
            new_state = (x, y - 1)
        elif action == 3:  # Right
            new_state = (x, y + 1)
        else:
            new_state = self.state  # Invalid action

        # Check boundaries and walls
        if new_state[0] < 0 or new_state[0] >= len(self.maze) or \
                new_state[1] < 0 or new_state[1] >= len(self.maze[0]) or \
                self.maze[new_state[0]][new_state[1]] == 'X':  # Blocked by wall
            return self.state, -1, False  # Invalid move, -1 reward

        self.state = new_state
        self.steps_taken += 1
        if self.state == self.goal_pos:
            return self.state, 1, True  # Reached goal
        else:
            return self.state, -0.1, False  # Normal move, small negative reward

    def render(self, success_ratio=0, lifetime=0):
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear console properly
        maze_copy = [row.copy() for row in self.maze]
        x, y = self.state
        maze_copy[x][y] = '@'  # Show agent's current position
        for row in maze_copy:
            print("".join(row))
        print(f"Lifetime: {lifetime}")
        print(f"Success Ratio: {success_ratio:.2f}%")


# Q-learning agent with visual feedback
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.q_table = np.zeros((len(env.maze), len(env.maze[0]), 4))  # Q-table: state x action
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.success_count = 0
        self.total_episodes = 0

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2, 3])  # Explore
        else:
            return np.argmax(self.q_table[state[0], state[1], :])  # Exploit

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1], :])
        td_target = reward + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.alpha * td_error

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            lifetime = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
                lifetime += 1
                # Render maze with agent position
                success_ratio = (self.success_count / (episode + 1)) * 100
                self.env.render(success_ratio=success_ratio, lifetime=lifetime)
                time.sleep(0.1)  # Small delay to mimic the movement of the agent

                if done and reward == 1:
                    self.success_count += 1

            self.total_episodes += 1
            if (episode + 1) % 100 == 0:
                print(f'Episode: {episode + 1}')


# Training the agent with real-time visualization
env = MazeEnv()
agent = QLearningAgent(env)
agent.train(episodes=1000)
