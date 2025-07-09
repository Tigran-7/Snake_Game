import pygame
import random
import numpy as np
from collections import deque, defaultdict
import time
import os

# Pygame Settings
GRID_SIZE = 20
CELL_SIZE = 20
INFO_PANEL_HEIGHT = 50
WIDTH = GRID_SIZE * CELL_SIZE
GAME_HEIGHT = GRID_SIZE * CELL_SIZE
TOTAL_HEIGHT = GAME_HEIGHT + INFO_PANEL_HEIGHT 

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255) 
GREY = (100, 100, 100) 

pygame.font.init()
INFO_FONT = pygame.font.Font(None, 28)

class SnakeGame:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.action_space_n = 4
        self.reset()

    def reset(self):
        self.snake = deque([(self.grid_size // 2, self.grid_size // 2)])
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        self._place_food()
        self.score = 0
        self.done = False
        self.last_dist_to_food = self._dist_to_food()
        return self._get_state()

    def _place_food(self):
        while True:
            self.food = (random.randint(0, self.grid_size - 1),
                         random.randint(0, self.grid_size - 1))
            if self.food not in self.snake:
                break

    def _get_head_pos(self):
        return self.snake[0]

    def _dist_to_food(self):
        head_x, head_y = self._get_head_pos()
        food_x, food_y = self.food
        return abs(head_x - food_x) + abs(head_y - food_y)

    def _get_state(self):
        head_x, head_y = self._get_head_pos()
        food_x, food_y = self.food

        food_dir_x = 0
        if food_x < head_x: food_dir_x = -1
        elif food_x > head_x: food_dir_x = 1

        food_dir_y = 0
        if food_y < head_y: food_dir_y = -1
        elif food_y > head_y: food_dir_y = 1
        
        point_n = (head_x, head_y - 1)
        point_s = (head_x, head_y + 1)
        point_w = (head_x - 1, head_y)
        point_e = (head_x + 1, head_y)

        obstacle_n = (point_n[1] < 0 or point_n in self.snake)
        obstacle_s = (point_s[1] >= self.grid_size or point_s in self.snake)
        obstacle_w = (point_w[0] < 0 or point_w in self.snake)
        obstacle_e = (point_e[0] >= self.grid_size or point_e in self.snake)
        
        return (
            food_dir_x, food_dir_y,
            int(obstacle_n), int(obstacle_s),
            int(obstacle_w), int(obstacle_e)
        )

    def step(self, action):
        current_dx, current_dy = self.direction
        if action == 0 and current_dy == 1: action = 1 
        elif action == 1 and current_dy == -1: action = 0
        elif action == 2 and current_dx == 1: action = 3 
        elif action == 3 and current_dx == -1: action = 2 

        if action == 0: new_direction = (0, -1)
        elif action == 1: new_direction = (0, 1)
        elif action == 2: new_direction = (-1, 0)
        else: new_direction = (1, 0)
        self.direction = new_direction

        head_x, head_y = self._get_head_pos()
        new_head_x = head_x + self.direction[0]
        new_head_y = head_y + self.direction[1]

        reward = -0.1
        self.done = False

        if (new_head_x < 0 or new_head_x >= self.grid_size or
            new_head_y < 0 or new_head_y >= self.grid_size or
            (new_head_x, new_head_y) in self.snake):
            self.done = True
            reward = -100
            return self._get_state(), reward, self.done, {"score": self.score}

        self.snake.appendleft((new_head_x, new_head_y))

        if (new_head_x, new_head_y) == self.food:
            self.score += 1
            reward = 50
            self._place_food()
            self.last_dist_to_food = self._dist_to_food()
        else:
            self.snake.pop()
            current_dist_to_food = self._dist_to_food()
            if current_dist_to_food < self.last_dist_to_food:
                reward += 1
            elif current_dist_to_food > self.last_dist_to_food:
                reward -= 2
            self.last_dist_to_food = current_dist_to_food
            
        next_state = self._get_state()
        return next_state, reward, self.done, {"score": self.score}

    def render(self, screen, title_info="", score_info="", epsilon_info=""):
        game_area_rect = pygame.Rect(0, 0, WIDTH, GAME_HEIGHT)
        screen.fill(BLACK, game_area_rect)

        for i, segment in enumerate(self.snake):
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = BLUE if i == 0 else GREEN
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, WHITE, rect, 1)

        food_rect = pygame.Rect(self.food[0] * CELL_SIZE, self.food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, RED, food_rect)

        info_panel_rect = pygame.Rect(0, GAME_HEIGHT, WIDTH, INFO_PANEL_HEIGHT)
        screen.fill(GREY, info_panel_rect)

        title_surf = INFO_FONT.render(title_info, True, WHITE)
        score_surf = INFO_FONT.render(score_info, True, WHITE)
        eps_surf = INFO_FONT.render(epsilon_info, True, WHITE)

        screen.blit(title_surf, (10, GAME_HEIGHT + 10))
        screen.blit(score_surf, (WIDTH // 2 - score_surf.get_width() // 2, GAME_HEIGHT + 10))
        screen.blit(eps_surf, (WIDTH - eps_surf.get_width() - 10, GAME_HEIGHT + 10))
        
        pygame.display.flip()

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, action_space_n, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.9995, min_exploration_rate=0.01):
        self.action_space_n = action_space_n
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_n))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_space_n)
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = [action for action, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state]) if not done else 0
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)

    def save_q_table(self, filename="q_table.npy"):
        save_dict = {k: v.tolist() for k, v in self.q_table.items()}
        np.save(filename, save_dict, allow_pickle=True)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename="q_table.npy"):
        if not os.path.exists(filename):
            print(f"Q-table file '{filename}' not found. Starting fresh or ensure file exists.")
            return False
        try:
            loaded_dict_list = np.load(filename, allow_pickle=True).item()
            self.q_table = defaultdict(lambda: np.zeros(self.action_space_n))
            for k, v_list in loaded_dict_list.items():
                 self.q_table[k] = np.array(v_list)
            print(f"Q-table loaded from {filename}")
            self.epsilon = self.min_epsilon
            return True
        except Exception as e:
            print(f"Error loading Q-table: {e}. Starting fresh.")
            self.q_table = defaultdict(lambda: np.zeros(self.action_space_n))
            return False


# Training Loop
def train_agent(episodes=20000,
                load_model=False,
                visualize_continuously=False,
                training_cont_fps=30,
                visualize_episodic_every_n=1000,
                training_episodic_fps=10
                ):
    pygame.init()
    screen = None
    clock = pygame.time.Clock()

    env = SnakeGame()
    agent = QLearningAgent(action_space_n=env.action_space_n)

    if load_model:
        agent.load_q_table()

    scores = []
    avg_scores_list = []

    if visualize_continuously:
        screen = pygame.display.set_mode((WIDTH, TOTAL_HEIGHT))
        pygame.display.set_caption("Snake RL - Training (Live)")

    print(f"Starting training for {episodes} episodes...")
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        current_score = 0
        
        visualize_this_specific_episode = False
        if not visualize_continuously and (episode % visualize_episodic_every_n == 0 or episode == episodes):
            visualize_this_specific_episode = True
            if screen is None:
                screen = pygame.display.set_mode((WIDTH, TOTAL_HEIGHT))
            pygame.display.set_caption(f"Snake RL - Training Episode {episode}")

        while not done:
            if screen: 
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if screen: pygame.quit()
                        agent.save_q_table()
                        print("Training interrupted by user. Model saved.")
                        return agent 
            
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            current_score = info["score"]

            if screen and (visualize_continuously or visualize_this_specific_episode):
                fps_to_use = training_cont_fps if visualize_continuously else training_episodic_fps
                env.render(screen,
                           title_info=f"Episode: {episode}/{episodes}",
                           score_info=f"Score: {current_score}",
                           epsilon_info=f"Epsilon: {agent.epsilon:.3f}")
                clock.tick(fps_to_use)

        agent.decay_epsilon()
        scores.append(current_score)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_scores_list.append(avg_score)
            print(f"Ep: {episode}, Score: {current_score}, AvgScr: {avg_score:.2f}, Eps: {agent.epsilon:.4f}, QSize: {len(agent.q_table)}")
            if episode % 1000 == 0:
                 agent.save_q_table()

    if screen:
        pygame.quit()
    
    print("Training finished.")
    agent.save_q_table()
    return agent

# Play with Trained Agent
def play_with_agent(agent, num_games=5, play_fps=10):
    if not agent.q_table and not agent.load_q_table():
        print("Cannot play: Q-table is empty and could not be loaded.")
        print("Please train a model first or ensure 'q_table.npy' exists.")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, TOTAL_HEIGHT))
    pygame.display.set_caption("Snake RL - Trained Agent")
    clock = pygame.time.Clock()

    env = SnakeGame()
    agent.epsilon = 0 

    for game_num in range(1, num_games + 1):
        state = env.reset()
        done = False
        print(f"\n--- Game {game_num} ---")
        current_score = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print("Playback stopped by user.")
                    return

            action = agent.choose_action(state)
            next_state, _, done, info = env.step(action)
            state = next_state
            current_score = info["score"]

            env.render(screen,
                       title_info=f"Game: {game_num}/{num_games}",
                       score_info=f"Score: {current_score}",
                       epsilon_info="Mode: Play")
            clock.tick(play_fps)

            if done:
                print(f"Game {game_num} Over! Score: {current_score}")
                env.render(screen,
                           title_info=f"Game: {game_num}/{num_games} - END",
                           score_info=f"Final Score: {current_score}",
                           epsilon_info="Mode: Play")
                pygame.display.flip()
                time.sleep(2)
    pygame.quit()


if __name__ == "__main__":
    # CHOOSE MODE (train or play)
    MODE = "train"
    LOAD_EXISTING_MODEL = False

    if MODE == "train":
        NUM_EPISODES_TRAIN = 2000

        VISUALIZE_TRAINING_CONTINUOUSLY = True
        TRAINING_CONT_FPS = 15

        VISUALIZE_TRAINING_EPISODIC_EVERY_N = 500
        TRAINING_EPISODIC_FPS = 10

        print("Selected Mode: TRAIN")
        trained_agent = train_agent(
            episodes=NUM_EPISODES_TRAIN,
            load_model=LOAD_EXISTING_MODEL,
            visualize_continuously=VISUALIZE_TRAINING_CONTINUOUSLY,
            training_cont_fps=TRAINING_CONT_FPS,
            visualize_episodic_every_n=VISUALIZE_TRAINING_EPISODIC_EVERY_N,
            training_episodic_fps=TRAINING_EPISODIC_FPS
        )
        print("\n--- Training Complete ---")
        if trained_agent:
             print("Now, let's see the trained agent play!")
             play_with_agent(trained_agent, num_games=5, play_fps=10)

    elif MODE == "play":
        NUM_GAMES_PLAY = 10
        PLAY_FPS = 10

        print("Selected Mode: PLAY")
        agent_to_play = QLearningAgent(action_space_n=SnakeGame().action_space_n) 
        if not LOAD_EXISTING_MODEL:
            print("Warning: LOAD_EXISTING_MODEL is False for play mode. Attempting to load 'q_table.npy' by default.")
        
        play_with_agent(agent_to_play, num_games=NUM_GAMES_PLAY, play_fps=PLAY_FPS)