import gymnasium as gym
import torch
import numpy as np
from torch.distributions import Categorical
import time
import argparse

from ppo import Actor, Critic, PPO
 
def evaluate_policy(actor, env, num_episodes=5, render=True):

    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            if render:
                env.render()
                time.sleep(0.01) 
            
         
            with torch.no_grad():
                logits = actor(state)
                dist = Categorical(logits=logits)
                action = dist.sample()
            
          
            state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            episode_reward += reward
            step += 1

        total_rewards.append(episode_reward)
        print(f"{episode_reward:.2f}")
    
    avg_reward = np.mean(total_rewards)
    # print only average reward
    print(f"{avg_reward:.2f}")
    return avg_reward

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained PPO agent on cartpole')
    parser.add_argument('--actor_checkpoint', 
                    type=str, 
                    default='saved_models/actor_ep400.pth', 
                    help='Path to actor checkpoint')

    parser.add_argument('--critic_checkpoint', 
                        type=str, 
                        default='saved_models/critic_ep400.pth',
                        help='Path to critic checkpoint')

    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate')
    args = parser.parse_args()
    

    env = gym.make('CartPole-v1', render_mode="human")

    agent = PPO(env)

    actor_checkpoint = torch.load(args.actor_checkpoint)
    agent.actor.load_state_dict(actor_checkpoint['model_state_dict'])

    if args.critic_checkpoint:
        critic_checkpoint = torch.load(args.critic_checkpoint)
        agent.critic.load_state_dict(critic_checkpoint['model_state_dict'])

    agent.actor.eval()
    agent.critic.eval()
    evaluate_policy(agent.actor, env, num_episodes=args.episodes)
    env.close()

if __name__ == "__main__":
    main()