import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from lib.solar.solar import Solar
from lib.env import NodeEnv
from lib.env_paolo import BatteryEnv
from lib.env_paolo import Solar as SolarP
import random
import os

# --- Actor-Critic Model ---
class ActorCritic(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.policy_head = nn.Linear(64, 2)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x)
    
# --- Advantage Estimation ---
def compute_advantages(rewards, values, dones, gamma=0.99, gae_lambda=0):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
        gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return advantages

class PPO:
    def __init__(self, 
                 env:gym.Env,
                 learning_rate:float = 1e-3,
                 epochs:int = 10,
                 batch_size:int = 32,
                 clip_eps = 0.25,
                 gamma = 0.99,
                 gae_lambda = 0.99,
                 n_steps:int = 256,
                 max_terminated:int = 0,
                 run_folder:str=None,
                seed=None,
                device=torch.device("cuda:0")):
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Set algorithm parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.max_terminated = max_terminated
        self.run_folder = run_folder
        self.device = device

        if self.run_folder is not None and not os.path.exists(self.run_folder):
            os.mkdir(self.run_folder)

        self.env:gym.Env = env
        self.model = ActorCritic(obs_size=env.observation_space.shape[0]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Buffers
        self.obs_buf, self.act_buf, self.logp_buf, self.rew_buf, self.done_buf, self.val_buf = [], [], [], [], [], []

    def learn(self, total_steps:int):
        obs,_ = self.env.reset()
        terminated_count = 0
        best_reward = 0
        for t in range(total_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits, value = self.model(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)

            next_obs, reward, done, terminated , info = self.env.step(action.item()) 

            val = value.item()
            self.obs_buf.append(obs)
            self.act_buf.append(action.item())
            self.logp_buf.append(logp.item())
            self.rew_buf.append(reward)
            self.done_buf.append(done)
            self.val_buf.append(val)

            obs = next_obs

            if terminated:
                terminated_count+=1
                if self.max_terminated > 0 and terminated_count >= self.max_terminated:
                    print(f"Terminated {terminated_count} times")
                    break
            if done and not terminated:
                terminated_count = 0

            if done or terminated:

                print(f"{(t/total_steps):.2f}% | Episode ends in {len(self.obs_buf)} steps\t| Total reward of {sum(self.rew_buf):.1f}\t|","Terminated:",terminated,"|\tDone:",done)

                if sum(self.rew_buf) > best_reward and self.run_folder is not None:
                    best_reward = sum(self.rew_buf)
                    print("Saving model")
                    torch.save(self.model.state_dict(),os.path.join(self.run_folder,"best.pth"))

                # Bootstrapped value
                with torch.no_grad():
                    next_val = self.model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))[1].item()
                self.val_buf.append(next_val)
                adv_buf = compute_advantages(self.rew_buf, self.val_buf, self.done_buf, gamma=self.gamma, gae_lambda=self.gae_lambda)
                returns = np.array(adv_buf) + np.array(self.val_buf[:-1])

                # Convert to tensors
                obs_t = torch.tensor(np.array(self.obs_buf), dtype=torch.float32).to(self.device)
                act_t = torch.tensor(np.array(self.act_buf), dtype=torch.int64).to(self.device)
                logp_old_t = torch.tensor(np.array(self.logp_buf), dtype=torch.float32).to(self.device)
                adv_t = torch.tensor(adv_buf, dtype=torch.float32).to(self.device)
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
                ret_t = torch.tensor(returns, dtype=torch.float32).to(self.device)

                # PPO Update
                for _ in range(self.epochs):
                    idx = np.random.permutation(len(obs_t))
                    for i in range(0, len(obs_t), self.batch_size):
                        sl = idx[i:i+self.batch_size]
                        logits, values = self.model(obs_t[sl])
                        dist = torch.distributions.Categorical(logits=logits)
                        logp = dist.log_prob(act_t[sl])
                        entropy = dist.entropy()

                        ratio = torch.exp(logp - logp_old_t[sl])
                        surr1 = ratio * adv_t[sl]
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[sl]
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = ((ret_t[sl] - values.squeeze()) ** 2).mean()

                        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                # Clear buffers
                self.obs_buf, self.act_buf, self.logp_buf, self.rew_buf, self.done_buf, self.val_buf = [], [], [], [], [], []
                obs,_ = self.env.reset()



    def select_action(self, obs) -> any: #Action
        test_obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(test_obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            return torch.argmax(probs).item()

    def save(self, path:str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path:str):
        self.model.load_state_dict(torch.load(path, weights_only=True))