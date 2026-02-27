import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from .ppo import PPO
from gail_airl_ppo.network import GAILDiscrim


class GAIL(PPO):

    def __init__(self, 
                buffer_exp, 
                state_shape, 
                action_shape, 
                device, 
                seed,
                gamma=0.995, 
                rollout_length=50000, 
                mix_buffer=1,
                batch_size=64, 
                lr_actor=3e-4, 
                lr_critic=3e-4, 
                lr_disc=3e-4,
                units_actor=(64, 64), 
                units_critic=(64, 64),
                units_disc=(100, 100), 
                epoch_ppo=50, 
                epoch_disc=10,
                clip_eps=0.2, 
                lambd=0.97, 
                coef_ent=0.0, 
                max_grad_norm=10.0):
        super().__init__(
                state_shape, 
                action_shape, 
                device, 
                seed, 
                gamma, 
                rollout_length,
                mix_buffer, 
                lr_actor, 
                lr_critic, 
                units_actor, 
                units_critic,
                epoch_ppo, 
                clip_eps, 
                lambd, 
                coef_ent, 
                max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = GAILDiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_disc,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer, step=None):
        self.learning_steps += 1
        update_num = self.learning_steps
        
        if step is not None:
            print(f"  [更新 #{update_num}] 開始訓練 Discriminator 和 Actor...")
        
        # 訓練 Discriminator
        print(f"  [Discriminator 訓練] 開始訓練 ({self.epoch_disc} 輪)...")
        for epoch in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, actions = self.buffer.sample(self.batch_size)[:2]
            # Samples from expert's demonstrations.
            states_exp, actions_exp = \
                self.buffer_exp.sample(self.batch_size)[:2]
            # Update discriminator.
            self.update_disc(states, actions, states_exp, actions_exp, writer)
            
            if (epoch + 1) % max(1, self.epoch_disc // 5) == 0 or epoch == 0:
                print(f"    Discriminator 訓練進度: {epoch+1}/{self.epoch_disc} 輪")
        
        print(f"  [Discriminator 訓練] 完成")

        # 獲取 Actor 的完整軌跡並計算獎勵
        print(f"  [計算獎勵] 使用 Discriminator 為 Actor 軌跡打分...")
        states, actions, _, dones, log_pis, next_states = self.buffer.get()
        rewards = self.disc.calculate_reward(states, actions)
        mean_reward = rewards.mean().item()
        print(f"    平均獎勵 (Mean reward): {mean_reward:.4f}")
        print(f"    軌跡長度 (Trajectory length): {len(states):,}")

        # 訓練 Actor (PPO)
        print(f"  [Actor (PPO) 訓練] 開始訓練 ({self.epoch_ppo} 輪)...")
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer, step=step)
        print(f"  [Actor (PPO) 訓練] 完成")

    def update_disc(self, states, actions, states_exp, actions_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, actions)
        logits_exp = self.disc(states_exp, actions_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
    
    def save_models(self, save_dir):
        """保存模型權重（包括 actor, critic 和 discriminator）。
        
        Args:
            save_dir: 保存目錄路徑
        """
        import os
        # 調用父類方法保存 actor 和 critic
        super().save_models(save_dir)
        
        # 保存 discriminator
        disc_path = os.path.join(save_dir, "discriminator.pth")
        torch.save(self.disc.state_dict(), disc_path)
