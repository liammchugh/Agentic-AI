import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import tkinter as tk
from tkinter import Scale
import math
import random

# Policy Network with LSTM for memory retention
class ActuatorPolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=4):
        super(ActuatorPolicyNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # LSTM layer for memory
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_mean = nn.Linear(hidden_size, output_size)
        self.action_log_std = nn.Parameter(torch.zeros(output_size))
    
    def forward(self, x, hidden):
        # LSTM output
        x, hidden = self.lstm(x.unsqueeze(0), hidden)
        x = x.squeeze(0)  # Remove the batch dimension for single-step processing
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        return action_mean, action_log_std, hidden

    def init_hidden(self, device):
        # Initialize the hidden state of the LSTM (both hidden and cell states)
        return (torch.zeros(1, 1, self.hidden_size).to(device),
                torch.zeros(1, 1, self.hidden_size).to(device))
    

class RLHFSystem:
    def __init__(self, policy_net, optimizer, gamma=0.99, device='cpu'):
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []
        self.human_feedback = 1.0  # Neutral feedback
        self.device = device
        self.hidden = self.policy_net.init_hidden(self.device)  # Initialize hidden states

    def select_action(self, state):
        state = state.unsqueeze(0).to(self.device)  # Ensure the state has batch dimension
        action_mean, action_log_std, self.hidden = self.policy_net(state, self.hidden)
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action = torch.tanh(action)
        log_prob = dist.log_prob(action).sum()

        # Detach the hidden state to avoid backpropagating through time
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        return action, log_prob

    def store_outcome(self, log_prob, reward):
        adjusted_reward = self.integrate_human_feedback(reward)
        self.log_probs.append(log_prob)
        self.rewards.append(adjusted_reward)

    def integrate_human_feedback(self, reward):
        feedback_factor = self.human_feedback / 100.0
        adjusted_reward = reward * feedback_factor
        return adjusted_reward

    def compute_returns(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        return (returns - returns.mean()) / (returns.std() + 1e-5)

    def update_policy(self):
        if not self.log_probs:
            return  # Skip update if no log_probs collected
        returns = self.compute_returns()
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward(retain_graph=False)  # No need to retain the graph for future updates
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []


import math
import torch
import pygame

class BallInCageEnv:
    def __init__(self):
        self.width, self.height = 600, 600
        self.ball_pos = [300, 300]  # Ball position
        self.enemy_pos = [300, 300]  # Enemy ball (controlled by cursor)
        self.ball_radius = 15
        self.enemy_radius = 15  # Radius of enemy ball
        self.max_enemy_speed = 5  # Maximum enemy speed
        self.max_penalty = -10  # Maximum penalty for full collision
        self.proximity_penalty_range = self.ball_radius + self.enemy_radius + 20  # Effective range for increasing penalty

    def reset(self):
        self.ball_pos = [random.randint(100, 500), random.randint(100, 500)]
        self.enemy_pos = [300, 300]
        return torch.tensor(self.get_state(), dtype=torch.float32)

    def get_state(self):
        # Return the state for neural network: relative enemy position and wall positions
        rel_enemy_x = self.enemy_pos[0] - self.ball_pos[0]
        rel_enemy_y = self.enemy_pos[1] - self.ball_pos[1]
        wall_info = [
            self.ball_pos[0],  # Distance to left wall
            self.width - self.ball_pos[0],  # Distance to right wall
            self.ball_pos[1],  # Distance to top wall
            self.height - self.ball_pos[1],  # Distance to bottom wall
        ]
        return [rel_enemy_x, rel_enemy_y] + wall_info

    def update(self, action, cursor_pos):
        action = action.squeeze(0)  # Now it's a 1D tensor of shape [4]

        direction = action[:2]
        speed = action[2:]

        # Ensure the action tensor is valid
        if direction.size(0) < 2 or speed.size(0) < 2:
            print("Invalid action tensor:", action)
            return -1.0  # Error, just return a penalty

        # Convert tensors to CPU and ensure they're detached for numpy conversion
        direction = direction.cpu().detach().numpy()
        speed = speed.cpu().detach().numpy()

        # Apply movement (direction: [-1, 1], speed: [-1, 1] scaled)
        dx = speed[0] * direction[0] * 5
        dy = speed[1] * direction[1] * 5

        # Update ball position
        self.ball_pos[0] += dx
        self.ball_pos[1] += dy

        # Keep ball within boundaries
        self.ball_pos[0] = max(self.ball_radius, min(self.width - self.ball_radius, self.ball_pos[0]))
        self.ball_pos[1] = max(self.ball_radius, min(self.height - self.ball_radius, self.ball_pos[1]))

        # Update enemy ball position (controlled by cursor, limited by speed)
        enemy_dx = cursor_pos[0] - self.enemy_pos[0]
        enemy_dy = cursor_pos[1] - self.enemy_pos[1]
        distance = math.sqrt(enemy_dx**2 + enemy_dy**2)

        if distance > self.max_enemy_speed:
            enemy_dx = (enemy_dx / distance) * self.max_enemy_speed
            enemy_dy = (enemy_dy / distance) * self.max_enemy_speed

        self.enemy_pos[0] += enemy_dx
        self.enemy_pos[1] += enemy_dy

        # Calculate the proximity penalty based on distance from the enemy ball
        penalty = self.calculate_proximity_penalty(distance)

        # Return reward with the proximity penalty applied
        return 0.1 - penalty  # Small reward for avoiding enemy, reduced by penalty

    def calculate_proximity_penalty(self, distance):
        # If within collision range, apply maximum penalty
        if distance < (self.ball_radius + self.enemy_radius):
            return self.max_penalty

        # If within the proximity penalty range, apply a scaled penalty
        if distance < self.proximity_penalty_range:
            proximity_ratio = (self.proximity_penalty_range - distance) / self.proximity_penalty_range
            penalty = self.max_penalty * proximity_ratio
            return penalty

        # Reward if outside the effective range
        return 0.0

    def check_collision(self, pos1, pos2):
        dist = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        return dist < (self.ball_radius + self.enemy_radius)

    def render(self, screen):
        screen.fill((255, 255, 255))  # White background

        # Draw enemy ball (controlled by cursor)
        pygame.draw.circle(screen, (255, 0, 0), (int(self.enemy_pos[0]), int(self.enemy_pos[1])), self.enemy_radius)

        # Draw player-controlled ball
        pygame.draw.circle(screen, (0, 0, 255), (int(self.ball_pos[0]), int(self.ball_pos[1])), self.ball_radius)

        pygame.display.flip()


# GUI for human feedback
class HumanFeedbackGUI:
    def __init__(self, rlhf_system):
        self.rlhf_system = rlhf_system
        self.window = tk.Tk()
        self.window.title("Human Feedback")
        self.slider = Scale(self.window, from_=0, to=100, orient="horizontal", length=300)
        self.slider.set(100)  # Start with maximum feedback
        self.slider.pack()

    def update_feedback(self):
        self.rlhf_system.human_feedback = self.slider.get()
        self.window.update_idletasks()
        self.window.update()

# Main function to run the environment
def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption('Ball in Cage Environment')

    # GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = 6  # State space: relative enemy position (2) + wall distances (4)
    hidden_size = 64
    policy_net = ActuatorPolicyNet(input_size, hidden_size, output_size=4).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    rlhf_system = RLHFSystem(policy_net, optimizer, device=device)

    feedback_gui = HumanFeedbackGUI(rlhf_system)
    env = BallInCageEnv()

    state = env.reset().to(device)
    running = True
    clock = pygame.time.Clock()
    update_interval = 10  # Update policy every N steps
    step = 0

    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get mouse position (cursor-controlled enemy)
        cursor_pos = pygame.mouse.get_pos()

        # Get action from policy network
        state_tensor = state.clone().detach().to(device)  # Clone the state to avoid tensor warnings
        action, log_prob = rlhf_system.select_action(state_tensor)

        # Update environment
        reward = env.update(action, cursor_pos)

        # Store outcomes
        rlhf_system.store_outcome(log_prob, reward)

        # Render environment
        env.render(screen)

        # Update human feedback from GUI
        feedback_gui.update_feedback()

        # Prepare next state
        state = torch.tensor(env.get_state(), dtype=torch.float32).to(device)

        # Increment step counter
        step += 1

        # Update the policy after a defined number of steps, but without resetting the environment
        if step % update_interval == 0:
            rlhf_system.update_policy()

        # Control frame rate
        clock.tick(30)


    pygame.quit()
    feedback_gui.window.destroy()

if __name__ == "__main__":
    main()
