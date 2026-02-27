import torch
from torch import nn
from torch.optim import Adam
import numpy as np

from .base import Algorithm
from gail_airl_ppo.buffer import RolloutBuffer
from gail_airl_ppo.network import StateIndependentPolicy, StateFunction


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state)
        # Convert action to torch.Tensor if needed (Isaac Lab expects torch.Tensor)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        
        # Gymnasium returns 5 values: (observation, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Combine terminated and truncated into done
        if isinstance(terminated, (np.ndarray, torch.Tensor)):
            done = terminated | truncated
        else:
            done = terminated or truncated
        
        # Handle vectorized environments
        is_vectorized = isinstance(done, (np.ndarray, torch.Tensor)) and len(done.shape) > 0 and done.shape[0] > 1
        
        if is_vectorized:
            # Vectorized env: process all environments
            done_np = done if isinstance(done, np.ndarray) else done.cpu().numpy()
            reward_np = reward if isinstance(reward, np.ndarray) else (reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward)
            # log_pi from explore() should already be a numpy array in vectorized case
            if isinstance(log_pi, np.ndarray):
                log_pi_np = log_pi
            elif isinstance(log_pi, torch.Tensor):
                log_pi_np = log_pi.cpu().numpy()
            else:
                # Fallback: try to convert to numpy
                log_pi_np = np.asarray(log_pi)
            
            # Convert to numpy if needed
            state_np = state if isinstance(state, np.ndarray) else (state.cpu().numpy() if isinstance(state, torch.Tensor) else state)
            action_np = action if isinstance(action, np.ndarray) else (action.cpu().numpy() if isinstance(action, torch.Tensor) else action)
            next_state_np = next_state if isinstance(next_state, np.ndarray) else (next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state)
            
            # Calculate mask: not done and not exceeded max steps
            mask = ~done_np.astype(bool)
            if hasattr(env, '_max_episode_steps'):
                mask = mask & (np.array(t) < env._max_episode_steps)
            
            # Append each transition
            num_envs = state_np.shape[0] if len(state_np.shape) > 1 else 1
            for i in range(num_envs):
                # Extract log_pi value correctly to avoid DeprecationWarning
                # log_pi_np should be a numpy array from explore() in vectorized case
                if isinstance(log_pi_np, np.ndarray):
                    if len(log_pi_np.shape) > 0:
                        # Array with elements - extract single element using item() to avoid deprecation warning
                        elem = log_pi_np[i]
                        log_pi_val = float(elem.item() if hasattr(elem, 'item') else elem)
                    else:
                        # Scalar array
                        log_pi_val = float(log_pi_np.item())
                elif isinstance(log_pi_np, (list, tuple)):
                    # List or tuple
                    if len(log_pi_np) > i:
                        log_pi_val = float(log_pi_np[i])
                    else:
                        log_pi_val = float(log_pi_np[0]) if len(log_pi_np) > 0 else 0.0
                else:
                    # Scalar value
                    try:
                        log_pi_val = float(log_pi_np)
                    except (TypeError, ValueError):
                        log_pi_val = 0.0
                
                self.buffer.append(
                    state_np[i],
                    action_np[i] if len(action_np.shape) > 1 else action_np,
                    float(reward_np[i]) if len(reward_np.shape) > 0 else float(reward_np),
                    bool(mask[i]) if len(mask.shape) > 0 else bool(mask),
                    log_pi_val,
                    next_state_np[i] if len(next_state_np.shape) > 1 else next_state_np
                )
            
            # Update t for done environments
            t = np.where(done_np, 0, t)
            
            # For vectorized environments, Isaac Lab handles resets internally
            # We don't need to manually reset here as env.step() already returns
            # the next_state for all environments, including reset ones
        else:
            # Single env
            mask = False if t == env._max_episode_steps else (bool(done) if not isinstance(done, (np.ndarray, torch.Tensor)) else bool(done.item() if hasattr(done, 'item') else done[0]))
            self.buffer.append(state, action, reward, mask, log_pi, next_state)
            
            # For single environment, reset if done
            if done:
                t = 0
                reset_result = env.reset()
                # Handle tuple return (obs, info) from gymnasium
                if isinstance(reset_result, tuple):
                    next_state, _ = reset_result
                else:
                    next_state = reset_result

        return next_state, t

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,
                   writer, step=None):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for epoch in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, writer)
            
            # 打印進度（每 25% 或開始時）
            if step is not None and ((epoch + 1) % max(1, self.epoch_ppo // 4) == 0 or epoch == 0):
                print(f"    PPO 訓練進度: {epoch+1}/{self.epoch_ppo} 輪")

    def update_critic(self, states, targets, writer):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)

    def save_models(self, save_dir):
        """保存模型權重。
        
        Args:
            save_dir: 保存目錄路徑
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存 actor
        actor_path = os.path.join(save_dir, "actor.pth")
        torch.save(self.actor.state_dict(), actor_path)
        
        # 保存 critic（可選，測試時不需要）
        critic_path = os.path.join(save_dir, "critic.pth")
        torch.save(self.critic.state_dict(), critic_path)
