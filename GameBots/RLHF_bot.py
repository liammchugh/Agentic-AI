import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import tkinter as tk
from tkinter import Scale
import math
import random
import os
import numpy as np

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
        
# Environment Model Network
class EnvironmentModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(EnvironmentModel, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.next_state = nn.Linear(hidden_size, state_size)
        self.reward = nn.Linear(hidden_size, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        next_state = self.next_state(x)
        reward = self.reward(x)
        return next_state, reward

# Value Function Network
class ValueFunction(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value

class RLHFSystem:
    def __init__(
        self, policy_net, env_model, value_function,
        optimizer_policy, optimizer_env_model, optimizer_value,
        gamma=0.99, lam=0.9, device='cpu', method='MC'
    ):
        self.policy_net = policy_net
        self.env_model = env_model
        self.value_function = value_function
        self.optimizer_policy = optimizer_policy
        self.optimizer_env_model = optimizer_env_model
        self.optimizer_value = optimizer_value
        self.gamma = gamma
        self.lam = lam  # For TD(λ)
        self.log_probs = []
        self.rewards = []
        self.states = []
        self.actions = []
        self.next_states = []
        self.dones = []
        self.human_feedback = 1.0  # Neutral feedback
        self.device = device
        self.hidden = self.policy_net.init_hidden(self.device)  # Initialize hidden states
        self.method = method  # 'MC' or 'TD_lambda'

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

    def store_transition(self, state, action, log_prob, reward, next_state, done):
        adjusted_reward = self.integrate_human_feedback(reward)
        self.log_probs.append(log_prob)
        self.rewards.append(adjusted_reward)
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.dones.append(done)

    def integrate_human_feedback(self, reward):
        feedback_factor = self.human_feedback / 50.0
        adjusted_reward = reward * feedback_factor
        return adjusted_reward

    def update_policy(self):
        if self.method == 'MC':
            self.update_policy_mc()
        elif self.method == 'TD_lambda':
            self.update_policy_td_lambda()
        else:
            raise ValueError("Invalid method selected. Choose 'MC' or 'TD_lambda'.")

    def update_policy_mc(self):
        # Monte Carlo Policy Gradient Update
        returns = self.compute_returns()
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        loss = torch.stack(policy_loss).sum()

        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

        # Train the environment model
        self.train_env_model()

        # Clear stored data
        self.reset_memory()

    def compute_returns(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = self.standardize(returns)
        adj_returns = self.integrate_human_feedback(returns)
        return adj_returns

    def standardize(self, x):
        return (x - x.mean()) / (x.std() + 1e-5)

    def update_policy_td_lambda(self):
        # TD(λ) Update with Eligibility Traces
        T = len(self.states)
        values = []
        for state in self.states:
            value = self.value_function(state)
            values.append(value)
        values = torch.stack(values).squeeze()

        self.optimizer_policy.zero_grad()
        self.optimizer_value.zero_grad()

        G = torch.zeros(1).to(self.device)
        for t in reversed(range(T)):
            if t == T - 1:
                G = self.rewards[t]
            else:
                G = self.rewards[t] + self.gamma * ((1 - self.lam) * values[t + 1] + self.lam * G)
            advantage = G - values[t]
            policy_loss = -self.log_probs[t] * advantage.detach()
            value_loss = advantage.pow(2)

            policy_loss.backward(retain_graph=True)
            value_loss.backward(retain_graph=True)

        self.optimizer_policy.step()
        self.optimizer_value.step()

        # Train the environment model
        self.train_env_model()

        # Clear stored data
        self.reset_memory()

    def train_env_model(self):
        # Train the environment model using collected transitions
        states = torch.stack(self.states).to(self.device)
        actions = torch.stack(self.actions).to(self.device)
        if actions.dim() == 3:
            actions = actions.squeeze(1)

        next_states = torch.stack(self.next_states).to(self.device)
        rewards = torch.tensor(self.rewards).unsqueeze(1).to(self.device)

        # Predict next states and rewards
        pred_next_states, pred_rewards = self.env_model(states, actions)

        # Compute loss
        loss_next_state = nn.MSELoss()(pred_next_states, next_states)
        loss_reward = nn.MSELoss()(pred_rewards, rewards)

        env_model_loss = loss_next_state + loss_reward

        self.optimizer_env_model.zero_grad()
        env_model_loss.backward()
        self.optimizer_env_model.step()

    def reset_memory(self):
        self.log_probs = []
        self.rewards = []
        self.states = []
        self.actions = []
        self.next_states = []
        self.dones = []

class BallInCageEnv:
    def __init__(self):
        self.width, self.height = 600, 600
        self.ball_pos = [300, 300]  # Ball position
        self.enemy_pos = [300, 300]  # Enemy ball (controlled by cursor)
        self.ball_radius = 15
        self.enemy_radius = 15  # Radius of enemy ball
        self.max_enemy_speed = 5  # Maximum enemy speed
        self.max_penalty = 25  # Maximum penalty for full collision
        self.proximity_penalty_range = self.ball_radius + self.enemy_radius + 50  # Effective range for increasing penalty

    def reset(self):
        self.ball_pos = [random.randint(100, 500), random.randint(100, 500)]
        self.enemy_pos = [300, 300]
        return torch.tensor(self.get_state(), dtype=torch.float32)

    def get_state(self):
        # Return the state for neural network: relative enemy position and wall distances
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
        return 0.0 - penalty

    def calculate_proximity_penalty(self, distance):
        # If within collision range, apply maximum penalty
        if distance < (self.ball_radius + self.enemy_radius):
            return self.max_penalty

        # Apply a scaled penalty; the closer, the higher the penalty. Past range, penalty is zero
        proximity_ratio = (self.proximity_penalty_range - distance) / self.proximity_penalty_range
        penalty = self.max_penalty * proximity_ratio
        penalty = max(0.0, penalty)  # Ensure penalty is non-negative
        return penalty

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
        self.window.title("Cost Slider")
        self.slider = Scale(self.window, from_=0, to=100, orient="horizontal", length=300)
        self.slider.set(50)  # Start with avg cost
        self.slider.pack()

    def update_feedback(self):
        self.rlhf_system.human_feedback = self.slider.get()
        self.window.update_idletasks()
        self.window.update()

def model_params_match(model, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    model_state_dict = model.state_dict()
    # Compare the shapes of the parameters
    for name, param in checkpoint.items():
        if name not in model_state_dict:
            return False
        if model_state_dict[name].shape != param.shape:
            return False
    return True

# Main function to run the environment
def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption('Ball in Cage Environment')

    # GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = 6  # State space: relative enemy position (2) + wall distances (4)
    hidden_size = 64
    action_size = 4

    policy_net = ActuatorPolicyNet(input_size, hidden_size, output_size=action_size).to(device)
    env_model = EnvironmentModel(state_size=input_size, action_size=action_size, hidden_size=hidden_size).to(device)
    value_function = ValueFunction(state_size=input_size, hidden_size=hidden_size).to(device)

    # Load models if they exist
    if os.path.exists('models/actuator_policy_net.pth') and model_params_match(policy_net, 'models/actuator_policy_net.pth'):
        policy_net.load_state_dict(torch.load('models/actuator_policy_net.pth'))
    if os.path.exists('models/environment_model.pth') and model_params_match(env_model, 'models/environment_model.pth'):
        env_model.load_state_dict(torch.load('models/environment_model.pth'))
    if os.path.exists('models/value_function.pth') and model_params_match(value_function, 'models/value_function.pth'):
        value_function.load_state_dict(torch.load('models/value_function.pth'))

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=0.001)
    optimizer_env_model = optim.Adam(env_model.parameters(), lr=0.001)
    optimizer_value = optim.Adam(value_function.parameters(), lr=0.001)

    # Choose method: 'MC' or 'TD_lambda'
    # method = 'MC'  # For Monte Carlo
    method = 'TD_lambda'  # Uncomment for TD(λ)

    rlhf_system = RLHFSystem(
        policy_net, env_model, value_function,
        optimizer_policy, optimizer_env_model, optimizer_value,
        device=device, method=method
    )

    feedback_gui = HumanFeedbackGUI(rlhf_system)
    env = BallInCageEnv()

    state = env.reset().to(device)
    running = True
    clock = pygame.time.Clock()
    update_interval = 5  # Update policy every N steps
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

        # Prepare next state
        next_state = torch.tensor(env.get_state(), dtype=torch.float32).to(device)

        # Store transition
        done = False  # You can define a condition for episode termination
        rlhf_system.store_transition(state, action, log_prob, reward, next_state, done)

        # Render environment
        env.render(screen)

        # Update human feedback from GUI
        feedback_gui.update_feedback()

        state = next_state

        # Increment step counter
        step += 1

        # Update the policy after a defined number of steps
        if step % update_interval == 0:
            rlhf_system.update_policy()

        # Control frame rate
        clock.tick(30)
        # Create directories if they don't exist
        if not os.path.exists('models'):
            os.makedirs('models')

        # Save the trained models
        torch.save(rlhf_system.policy_net.state_dict(), 'models/actuator_policy_net.pth')
        torch.save(rlhf_system.env_model.state_dict(), 'models/environment_model.pth')
        torch.save(rlhf_system.value_function.state_dict(), 'models/value_function.pth')
    pygame.quit()
    feedback_gui.window.destroy()

if __name__ == "__main__":
    main()
