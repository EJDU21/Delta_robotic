# GAIL 訓練腳本使用說明

## 📋 概述

這個目錄包含了用於 GAIL (Generative Adversarial Imitation Learning) 訓練的腳本和工具。

## 📁 目錄結構

```
scripts/gail/
├── train.py                    # GAIL 訓練主腳本
├── test.py                     # 測試腳本
├── hdf5_buffer.py              # HDF5 專家示範資料載入器
├── obs_wrapper.py              # 觀察值包裝器（展平 dict）
├── frame_stack_wrapper.py      # 幀堆疊包裝器
├── vec_env_evaluator.py        # 向量化環境評估器
├── GAIL_for_IsaacLab/          # GAIL 核心庫
│   ├── gail_airl_ppo/          # GAIL/AIRL/PPO 演算法實現
│   └── ...
└── export_data/                # 專家示範資料目錄
```

---

## 🚀 快速開始

### 1. 準備專家示範資料

首先需要收集專家示範資料並保存為 HDF5 格式。專家示範資料應該包含：
- `state`: 觀察值（shape: `(T, frame_idx, obs_dim)`）
- `action`: 動作（shape: `(T, action_dim)`）
- `reward`: 獎勵（shape: `(T,)`）
- `done`: 終止標記（shape: `(T,)`）
- `next_state`: 下一個觀察值（shape: `(T, frame_idx, obs_dim)`）

### 2. 運行訓練

```bash
python scripts/gail/train.py \
  --task=Template-Delta-Robotic-Direct-v0 \
  --num_envs 10 \
  --seed 0 \
  --algo gail \
  --rollout_length 10000 \
  --num_steps 1000000 \
  --eval_interval 100000 \
  --video \
  --buffer /path/to/expert_demo.hdf5 \
  --max_buffer_samples 100000
```

### 3. 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--task` | 環境任務名稱 | `Template-Delta-Robotic-Direct-v0` |
| `--num_envs` | 並行環境數量 | `None` (使用配置預設值) |
| `--seed` | 隨機種子 | `None` (隨機) |
| `--algo` | 演算法（gail/airl） | `gail` |
| `--rollout_length` | Rollout 緩衝區長度 | `50000` |
| `--num_steps` | 總訓練步數 | `10000000` |
| `--eval_interval` | 評估間隔（步數） | `100000` |
| `--video` | 是否錄製影片 | `False` |
| `--video_length` | 影片長度（步數） | `200` |
| `--video_interval` | 影片錄製間隔（步數） | `2000` |
| `--buffer` | 專家示範資料路徑（.hdf5） | `None` (必需) |
| `--max_buffer_samples` | 最大載入樣本數 | `None` (載入全部) |

---

## ⚙️ 重要參數調整建議

### Rollout Length (`--rollout_length`)

- **建議值**：`10000-20000`
- **說明**：過大（如 `50000`）會消耗大量 GPU 記憶體
- **影響**：影響訓練穩定性和記憶體使用

### 環境數量 (`--num_envs`)

- **建議值**：`4-10`（根據 GPU 記憶體調整）
- **說明**：更多環境可以加快訓練，但需要更多記憶體
- **影響**：影響訓練速度和記憶體使用

### 最大緩衝區樣本數 (`--max_buffer_samples`)

- **建議值**：`100000-500000`（根據 HDF5 文件大小）
- **說明**：如果 HDF5 文件很大，可以限制載入的樣本數量以節省記憶體
- **影響**：影響記憶體使用和訓練資料多樣性

---

## 📊 訓練輸出

訓練過程中會生成以下內容：

1. **日誌目錄**：`logs/gail/Template-Delta-Robotic-Direct-v0/YYYY-MM-DD_HH-MM-SS/`
   - `env.yaml`: 環境配置
   - `env.pkl`: 環境配置（pickle）
   - `videos/`: 訓練影片（如果啟用 `--video`）

2. **訓練日誌**：包含損失、獎勵、評估結果等

---

## 🔧 環境包裝器

訓練腳本會自動應用以下包裝器：

1. **FlattenDictObsWrapper**: 將字典觀察值展平為一維陣列
2. **FrameStackWrapper**: 堆疊多個連續觀察值（如果專家資料包含歷史幀）
3. **RecordVideo**: 錄製訓練影片（如果啟用 `--video`）

---

## 📝 專家示範資料格式

HDF5 文件應該包含以下結構：

```
traj_0/
  ├── state: (T, frame_idx, obs_dim)  # 例如: (1000, 8, 23)
  ├── action: (T, action_dim)         # 例如: (1000, 8)
  ├── reward: (T,)                    # 例如: (1000,)
  ├── done: (T,)                      # 例如: (1000,)
  └── next_state: (T, frame_idx, obs_dim)  # 例如: (1000, 8, 23)

traj_1/
  └── ...
```

**重要**：
- `frame_idx` 是堆疊的幀數（例如：8 = 當前幀 + 7 個歷史幀）
- `obs_dim` 是單個觀察值的維度（例如：23）
- `action_dim` 是動作的維度（例如：8）

---

## 🐛 常見問題

### 1. 記憶體不足

**症狀**：GPU 記憶體不足錯誤

**解決方案**：
- 減少 `--rollout_length`（例如：`10000`）
- 減少 `--num_envs`（例如：`4`）
- 減少 `--max_buffer_samples`（例如：`100000`）

### 2. State Dimension 不匹配

**症狀**：警告訊息 "state_dim 不匹配"

**解決方案**：
- 檢查專家資料的 `obs_dim` 是否與環境的 `observation_space` 一致
- 檢查 `frame_idx` 是否正確（應該與 FrameStackWrapper 的 `num_frames` 一致）

### 3. 環境找不到

**症狀**：`gym.error.UnregisteredEnv`

**解決方案**：
- 確保已安裝 Delta_robotic 專案：`pip install -e source/Delta_robotic`
- 確保環境已註冊：檢查 `source/Delta_robotic/Delta_robotic/tasks/direct/delta_robotic/__init__.py`

---

## 📚 相關文檔

- [GAIL 演算法說明](GAIL_for_IsaacLab/README.md)
- [環境配置說明](../../source/Delta_robotic/Delta_robotic/tasks/direct/delta_robotic/README.md)
- [Reward 函式說明](../../scripts/REWARD_IMPLEMENTATION.md)

---

## 🔗 相關檔案

- `train.py`: 主訓練腳本
- `test.py`: 測試腳本
- `hdf5_buffer.py`: HDF5 資料載入器
- `obs_wrapper.py`: 觀察值包裝器
- `frame_stack_wrapper.py`: 幀堆疊包裝器
