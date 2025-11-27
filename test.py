import torch
import numpy as np
import cv2
from dm_control import suite

from ppo import PPO


def get_state_vector(time_step):
    obs = time_step.observation
    state_parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        state_parts.append(val.flatten())
    return np.concatenate(state_parts)


def visualize_policy(actor, domain_name='cartpole', task_name='balance', num_episodes=5):
    env = suite.load(domain_name=domain_name, task_name=task_name)
    width, height = 640, 480
    
    for episode in range(num_episodes):
        time_step = env.reset()
        state = get_state_vector(time_step)
        episode_reward = 0
        step = 0
        
        while not time_step.last():
            with torch.no_grad():
                mean, std = actor(state)
                action = mean.squeeze().numpy()
            
            time_step = env.step(action)
            state = get_state_vector(time_step)
            reward = time_step.reward if time_step.reward is not None else 0.0
            episode_reward += reward
            step += 1
            
            # render
            pixels = env.physics.render(height=height, width=width, camera_id=0)
            cv2.imshow('CartPole Balance', cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key & 0xFF == ord('p'):
                cv2.waitKey(0)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={step}")
    
    cv2.destroyAllWindows()
    print(f"\nVisualization complete!")


def main():
    checkpoint_path = 'saved_models/actor_ep300.pth'
    domain_name = 'cartpole'
    task_name = 'balance'
    num_episodes = 5
    
    print(f"Loading checkpoint: {checkpoint_path}\n")
    
    agent = PPO(domain_name=domain_name, task_name=task_name)
    actor_checkpoint = torch.load(checkpoint_path)
    agent.actor.load_state_dict(actor_checkpoint['model_state_dict'])
    agent.actor.eval()
    
    visualize_policy(agent.actor, domain_name, task_name, num_episodes)


if __name__ == "__main__":
    main()