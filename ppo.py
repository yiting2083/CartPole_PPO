import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random
from copy import deepcopy
from dm_control import suite


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.3)

    
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        x = self.net(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.net(state)


class PPO:
    def __init__(self, domain_name='cartpole', task_name='balance'):
        self.env = suite.load(domain_name=domain_name, task_name=task_name)
        self.domain_name = domain_name
        self.task_name = task_name
        
        obs_spec = self.env.observation_spec()
        self.state_dim = sum(np.prod(spec.shape) for spec in obs_spec.values())
    
        action_spec = self.env.action_spec()
        self.action_low = action_spec.minimum
        self.action_high = action_spec.maximum
        self.action_dim = action_spec.shape[0]
        
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.old_actor = None
        
        self.clip_ratio = 0.2
        self.gamma = 0.99
        self.lam = 0.95
        self.batch_size = 64
        self.n_epochs = 10
        self.target_kl = 0.015
        self.entropy_coef = 0.001
        
        self.logged_rewards = []
        self.logged_entropies = []
        self.logged_kls = []
        self.eval_points = []
        self.eval_rewards = []
        self.cumulative_samples = 0
    
    def get_state_vector(self, time_step):
        obs = time_step.observation
        state_parts = []
        for key in sorted(obs.keys()):
            val = obs[key]
            state_parts.append(val.flatten())
        return np.concatenate(state_parts)
    
    def get_action(self, state):
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        # convert to numpy and clip to env bounds
        action_np = action.squeeze().detach().cpu().numpy()
        action_np = np.clip(action_np, self.action_low, self.action_high).astype(np.float32)

        return action_np, float(log_prob.item()), float(entropy.item())

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, advantages, returns):
        policy_losses = []
        value_losses = []
        kl_divs = []
        entropy_losses = []
        
        self.old_actor = deepcopy(self.actor)
        early_stop = False
        
        for epoch in range(self.n_epochs):
            if early_stop:
                break
            
            for idx in range(0, len(states), self.batch_size):
                batch_states = states[idx:idx + self.batch_size]
                batch_actions = actions[idx:idx + self.batch_size]
                batch_old_log_probs = old_log_probs[idx:idx + self.batch_size]
                batch_advantages = advantages[idx:idx + self.batch_size]
                batch_returns = returns[idx:idx + self.batch_size]
                
                old_mean, old_std = self.old_actor(batch_states)
                new_mean, new_std = self.actor(batch_states)
                
                old_dist = Normal(old_mean.detach(), old_std.detach())
                new_dist = Normal(new_mean, new_std)
                
                new_log_probs = new_dist.log_prob(batch_actions).sum(dim=-1)
                entropy = new_dist.entropy().sum(dim=-1).mean()
                
                kl = torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1).mean().item()
                
                if kl > 1.5 * self.target_kl:
                    early_stop = True
                    break
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                values = self.critic(batch_states).squeeze()
                value_loss = ((values - batch_returns) ** 2).mean()
                
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                kl_divs.append(kl)
                entropy_losses.append(entropy.item())
        
        return {
            'policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
            'value_loss': np.mean(value_losses) if value_losses else 0.0,
            'kl_div': np.mean(kl_divs) if kl_divs else 0.0,
            'entropy': np.mean(entropy_losses) if entropy_losses else 0.0
        }
    
    def train(self, max_episodes=200, max_steps=1000, seed=45, save_interval=50,
              eval_every=10, eval_episodes=3, eval_seed=10):
        os.makedirs('saved_models', exist_ok=True)
        
        for episode in range(max_episodes):
            time_step = self.env.reset()
            state = self.get_state_vector(time_step)
            
            episode_reward = 0
            episode_entropies = []
            
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            
            for step in range(max_steps):
                action, log_prob, entropy = self.get_action(state)
                value = self.critic(state).item()
                episode_entropies.append(entropy)
                
                time_step = self.env.step(action)
                next_state = self.get_state_vector(time_step)
                reward = time_step.reward if time_step.reward is not None else 0.0
                done = time_step.last()
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            self.cumulative_samples += len(states)
            
            # advantages and returns
            advantages, returns = self.compute_gae(
                rewards, values, self.critic(state).item(), dones
            )
            
            states = torch.FloatTensor(np.array(states))
            actions = torch.FloatTensor(np.array(actions))
            old_log_probs = torch.FloatTensor(log_probs)
            
            train_info = self.update(states, actions, old_log_probs, advantages, returns)
            
            # Log metrics
            ep_entropy = np.mean(episode_entropies) if len(episode_entropies) > 0 else 0.0
            self.logged_rewards.append(episode_reward)
            self.logged_entropies.append(ep_entropy)
            self.logged_kls.append(train_info.get('kl_div', 0.0))
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.logged_rewards[-10:])
                avg_entropy = np.mean(self.logged_entropies[-10:])
                print(f"Episode {episode}, Steps: {self.cumulative_samples}, "
                      f"Avg Reward: {avg_reward:.2f}, Entropy: {avg_entropy:.4f}")
            
            if episode % 50 == 0 or episode == max_episodes - 1:
                self._save_training_plots()
            
            if eval_every > 0 and (episode % eval_every == 0 or episode == max_episodes - 1):
                avg_eval = self._evaluate_actor(eval_seed, eval_episodes)
                self.eval_points.append(episode)
                self.eval_rewards.append(avg_eval)
            
            if episode > 0 and episode % save_interval == 0:
                self.save_model(episode)
        
        return {
            'train_rewards': self.logged_rewards,
            'train_entropies': self.logged_entropies,
            'train_kls': self.logged_kls,
            'eval_points': self.eval_points,
            'eval_rewards': self.eval_rewards
        }
    
    def _evaluate_actor(self, eval_seed, eval_episodes=3):
        env = suite.load(domain_name=self.domain_name, task_name=self.task_name)
        total = []
        
        for ep in range(eval_episodes):
            time_step = env.reset()
            state = self.get_state_vector(time_step)
            ep_reward = 0
            
            while not time_step.last():
                with torch.no_grad():
                    mean, _ = self.actor(state)
                    action = mean.squeeze().numpy()
                
                time_step = env.step(action)
                state = self.get_state_vector(time_step)
                ep_reward += time_step.reward if time_step.reward is not None else 0.0
            
            total.append(ep_reward)
        
        return float(np.mean(total))
    
    def save_model(self, episode):
        actor_path = f'saved_models/actor_ep{episode}.pth'
        critic_path = f'saved_models/critic_ep{episode}.pth'
        
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
        }, actor_path)
        
        torch.save({
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, critic_path)
        
        print(f"Models saved at episode {episode}")
    
    def _save_training_plots(self, out_dir="."):
        episodes = np.arange(1, len(self.logged_rewards) + 1)
        
        fig_r, ax_r = plt.subplots(figsize=(8, 3))
        ax_r.plot(episodes, self.logged_rewards, label='Reward')
        ax_r.set_ylabel('Reward')
        ax_r.set_xlabel('Episode')
        ax_r.legend()
        fig_r.tight_layout()
        fig_r.savefig(os.path.join(out_dir, 'training_reward.png'))
        plt.close(fig_r)
        
        fig_e, ax_e = plt.subplots(figsize=(8, 3))
        ax_e.plot(episodes, self.logged_entropies, label='Entropy', color='orange')
        ax_e.set_ylabel('Entropy')
        ax_e.set_xlabel('Episode')
        ax_e.legend()
        fig_e.tight_layout()
        fig_e.savefig(os.path.join(out_dir, 'training_entropy.png'))
        plt.close(fig_e)
        
        fig_k, ax_k = plt.subplots(figsize=(8, 3))
        ax_k.plot(episodes, self.logged_kls, label='KL Divergence', color='green')
        ax_k.set_ylabel('KL')
        ax_k.set_xlabel('Episode')
        ax_k.legend()
        fig_k.tight_layout()
        fig_k.savefig(os.path.join(out_dir, 'training_kl.png'))
        plt.close(fig_k)


def main():
    seeds = [0,1,2]#0 1 2 
    max_episodes = 310
    max_steps = 1024
    eval_every = 10
    eval_episodes = 3
    eval_seed = 10
    
    all_train_rewards = []
    all_train_entropies = []
    all_train_kls = []
    all_eval_rewards = []
    eval_x = None
    
    for s in seeds:
        set_seed(s)
        agent = PPO(domain_name='cartpole', task_name='balance')
        res = agent.train(
            seed=s, max_episodes=max_episodes, max_steps=max_steps,
            save_interval=50, eval_every=eval_every,
            eval_episodes=eval_episodes, eval_seed=eval_seed
        )
        all_train_rewards.append(res['train_rewards'])
        all_train_entropies.append(res['train_entropies'])
        all_train_kls.append(res['train_kls'])
        all_eval_rewards.append(res['eval_rewards'])
        if eval_x is None:
            eval_x = res['eval_points']
    
    train_rewards_np = np.array(all_train_rewards)
    train_ent_np = np.array(all_train_entropies)
    train_kl_np = np.array(all_train_kls)
    
    tr_mean = np.mean(train_rewards_np, axis=0)
    tr_std = np.std(train_rewards_np, axis=0)
    
    ent_mean = np.mean(train_ent_np, axis=0)
    ent_std = np.std(train_ent_np, axis=0)
    
    kl_mean = np.mean(train_kl_np, axis=0)
    kl_std = np.std(train_kl_np, axis=0)
    
    eval_rewards_np = np.array(all_eval_rewards)
    eval_mean = np.mean(eval_rewards_np, axis=0)
    eval_std = np.std(eval_rewards_np, axis=0)
    
    episodes = np.arange(1, max_episodes + 1)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, tr_mean, label='Train Reward Mean')
    ax.fill_between(episodes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.3)
    if eval_x is not None and len(eval_x) > 0:
        eval_x_arr = np.array(eval_x) + 1
        ax.plot(eval_x_arr, eval_mean, marker='o', linestyle='-', color='red', label='Eval Mean')
        ax.fill_between(eval_x_arr, eval_mean - eval_std, eval_mean + eval_std, 
                        color='red', alpha=0.25, label='Eval ±STD')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    fig.tight_layout()
    fig.savefig('learning_curve_reward.png')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(episodes, ent_mean, label='Entropy Mean', color='orange')
    ax.fill_between(episodes, ent_mean - ent_std, ent_mean + ent_std, alpha=0.3, color='orange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Entropy')
    ax.legend()
    fig.tight_layout()
    fig.savefig('learning_curve_entropy.png')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(episodes, kl_mean, label='KL Mean', color='green')
    ax.fill_between(episodes, kl_mean - kl_std, kl_mean + kl_std, alpha=0.3, color='green')
    ax.set_xlabel('Episode')
    ax.set_ylabel('KL Divergence')
    ax.legend()
    fig.tight_layout()
    fig.savefig('learning_curve_kl.png')
    plt.close(fig)
    
    print('Saved learning curves: learning_curve_reward.png, learning_curve_entropy.png, learning_curve_kl.png')


if __name__ == "__main__":
    main()