import os
from time import time, sleep
from datetime import timedelta
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, 
                env, 
                env_test, 
                algo, 
                log_dir, 
                seed=0, 
                num_steps=10**5,
                eval_interval=10**3, 
                num_eval_episodes=5
        ):
        super().__init__()

        # Env to collect samples.
        self.env = env
        # Set seed if the environment supports it
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'seed'):
            self.env.unwrapped.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        # Set seed if the environment supports it
        if hasattr(self.env_test, 'seed'):
            self.env_test.seed(2**31-seed)
        elif hasattr(self.env_test, 'unwrapped') and hasattr(self.env_test.unwrapped, 'seed'):
            self.env_test.unwrapped.seed(2**31-seed)

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        
        # 打印訓練設定
        self._print_training_config()

    def _print_training_config(self):
        """打印訓練設定資訊"""
        print("\n" + "="*80)
        print("訓練設定 (Training Configuration)")
        print("="*80)
        print(f"  總訓練步數 (Total steps): {self.num_steps:,}")
        print(f"  評估間隔 (Eval interval): {self.eval_interval:,} steps")
        print(f"  評估回合數 (Eval episodes): {self.num_eval_episodes}")
        print(f"  日誌目錄 (Log directory): {self.log_dir}")
        
        # 打印演算法相關設定
        if hasattr(self.algo, 'rollout_length'):
            print(f"  Rollout 長度 (Rollout length): {self.algo.rollout_length:,}")
            print(f"  更新頻率 (Update frequency): 每 {self.algo.rollout_length:,} 步更新一次")
            print(f"  預期更新次數 (Expected updates): {self.num_steps // self.algo.rollout_length}")
        
        if hasattr(self.algo, 'epoch_disc'):
            print(f"  Discriminator 訓練輪數 (Disc epochs): {self.algo.epoch_disc}")
        if hasattr(self.algo, 'epoch_ppo'):
            print(f"  PPO 訓練輪數 (PPO epochs): {self.algo.epoch_ppo}")
        if hasattr(self.algo, 'batch_size'):
            print(f"  批次大小 (Batch size): {self.algo.batch_size}")
        if hasattr(self.algo, 'buffer_exp'):
            if hasattr(self.algo.buffer_exp, 'buffer_size'):
                print(f"  專家示範樣本數 (Expert samples): {self.algo.buffer_exp.buffer_size:,}")
        
        print("="*80 + "\n")

    def train(self):
        # Time to start training.
        self.start_time = time()
        print("\n" + "="*80)
        print("開始訓練 (Training Started)")
        print("="*80 + "\n")
        
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        reset_result = self.env.reset()
        # Handle tuple return (obs, info) from gymnasium
        if isinstance(reset_result, tuple):
            state, _ = reset_result
        else:
            state = reset_result

        last_update_step = 0
        for step in range(1, self.num_steps + 1):
            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step(self.env, state, t, step)
            
            # 顯示收集數據的進度（每 10% 或開始時打印一次）
            if hasattr(self.algo, 'rollout_length'):
                steps_since_update = step - last_update_step
                progress_interval = max(1, self.algo.rollout_length // 10)
                if step % progress_interval == 0 or (step == 1 and last_update_step == 0):
                    progress = (steps_since_update / self.algo.rollout_length) * 100
                    print(f"[階段 1: 收集數據] Step {step:,}/{self.num_steps:,} "
                          f"(進度: {progress:.1f}% / {self.algo.rollout_length:,} steps)")

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                print(f"\n{'='*80}")
                print(f"[階段 2: 模型更新] Step {step:,}/{self.num_steps:,} "
                      f"(已收集 {self.algo.rollout_length:,} 步數據)")
                print(f"{'='*80}")
                self.algo.update(self.writer, step=step)
                last_update_step = step
                print(f"[階段 2: 完成] 更新完成，繼續收集數據...\n")

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                print(f"\n{'='*80}")
                print(f"[階段 3: 評估] Step {step:,}/{self.num_steps:,}")
                print(f"{'='*80}")
                self.evaluate(step)
                self.algo.save_models(
                    os.path.join(self.model_dir, f'step{step}'))
                print(f"[階段 3: 完成] 評估完成，模型已保存\n")
        
        print("\n" + "="*80)
        print("訓練完成 (Training Completed)")
        print("="*80)
        print(f"總訓練時間: {self.time}")
        print("="*80 + "\n")

        # Wait for the logging to be finished.
        sleep(10)

    def evaluate(self, step):
        mean_return = 0.0

        for _ in range(self.num_eval_episodes):
            reset_result = self.env_test.reset()
            # Handle tuple return (obs, info) from gymnasium
            if isinstance(reset_result, tuple):
                state, _ = reset_result
            else:
                state = reset_result
            episode_return = 0.0
            done = False

            while (not done):
                action = self.algo.exploit(state)
                # Convert action to torch.Tensor if needed (Isaac Lab expects torch.Tensor)
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action).to(self.algo.device)
                elif not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=torch.float32, device=self.algo.device)
                step_result = self.env_test.step(action)
                # Handle tuple return from step (gymnasium returns 5 values)
                if len(step_result) == 5:
                    state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    state, reward, done, _ = step_result
                else:
                    raise ValueError(f"Unexpected step return format: {step_result}")
                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        self.writer.add_scalar('return/test', mean_return, step)
        print(f'  測試回報 (Test Return): {mean_return:.2f}')
        print(f'  訓練時間 (Training Time): {self.time}')
        print(f'  完成度 (Progress): {step/self.num_steps*100:.1f}% ({step:,}/{self.num_steps:,} steps)')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
