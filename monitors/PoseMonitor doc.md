# PoseMonitor 使用說明

---

## 📖 這是什麼？

`PoseMonitor` 是一個**監控工具**，用來追蹤：
1. 機器人手臂末端（夾爪）的位置和姿態
2. 夾爪和目標物體（例如風扇）之間的距離和角度誤差
3. 夾爪是否成功夾住物體
4. 碰撞事件（可選）


---

## 🚀 快速開始

使用 `create_default` 快速建立 monitor，現在只需要提供 prim paths：
```python
from pose_monitor import PoseMonitor
from grasp_config import GraspDetectionConfig

# 用 factory method 一行搞定
monitor = PoseMonitor.create_default(
    robot_prim_path="/World/WorkSpace/RS_M90E7A_Left",      # 機器人的 USD 路徑
    fan_prim_path="/World/WorkSpace/Scene/Fan",              # 風扇的 USD 路徑
    ground_truth_prim_path="/World/WorkSpace/Scene/GroundTruth",  # 目標位置的 USD 路徑
)

# ⚠️ 重要！模擬開始後一定要呼叫 initialize()
monitor.initialize()

# 現在可以開始使用了！
```

### 使用自訂設定

```python
from pose_monitor import PoseMonitor
from grasp_config import GraspDetectionConfig

# 建立自訂設定
config = GraspDetectionConfig(
    grip_position_min=0.018,
    grip_position_max=0.022,
    grasp_zone_max_m=0.03,
    hold_confirm_frames=5,  # 連續 5 幀才算真正夾住
)

monitor = PoseMonitor.create_default(
    robot_prim_path="/World/WorkSpace/RS_M90E7A_Left",
    fan_prim_path="/World/WorkSpace/Scene/Fan",
    ground_truth_prim_path="/World/WorkSpace/Scene/GroundTruth",
    grasp_config=config,
)
```

---

## 📍 常用功能

### 1. 取得夾爪目前的位置和姿態

```python
ee_pose = monitor.get_end_effector_pose()

print(f"夾爪位置: {ee_pose.p}")      # 輸出: [x, y, z] 三維座標
print(f"夾爪姿態: {ee_pose.q}")      # 輸出: [w, x, y, z] 四元數
```

### 2. 計算夾爪和風扇之間的誤差

```python
error = monitor.get_ee_to_fan_error()

print(f"距離: {error.distance:.3f} 公尺")
print(f"角度誤差: {np.degrees(error.angle_error):.1f} 度")
```

**回傳的 `PoseError` 物件包含：**
| 屬性 | 說明 |
|------|------|
| `distance` | 直線距離（公尺） |
| `position_error` | 位置差向量 `[dx, dy, dz]` |
| `angle_error` | 旋轉角度誤差（弧度） |

### 3. 計算風扇和目標位置之間的誤差

```python
error = monitor.get_fan_to_ground_truth_error()

print(f"風扇到目標距離: {error.distance:.3f} 公尺")
print(f"角度誤差: {np.degrees(error.angle_error):.1f} 度")
```

### 4. 檢查夾爪是否夾住風扇

```python
if monitor.is_holding_fan():
    print("✅ 成功夾住風扇！")
else:
    print("❌ 還沒夾到")
```

#### 抓取判定 (Grasp Logic)

Monitor 會檢查三個條件：

1. **距離條件**：`grasp_zone_min` $\le$ 距離 $\le$ `grasp_zone_max`
    
2. **夾爪條件**：
    - Slider9（左手指）：`grip_min` $\le$ 值 $\le$ `grip_max`
    - Slider10（右手指）：$-$`grip_max` $\le$ 值 $\le$ $-$`grip_min`

3. **幀數確認**：連續 `hold_confirm_frames` 幀滿足上述條件才回傳 `True`

這些條件可在 `GraspDetectionConfig` 中調整。

#### 重置夾持確認

```python
# 當需要重置夾持狀態時（例如任務重新開始）
monitor.reset_holding_confirmation()
```

### 5. 取得手臂關節角度

```python
# 取得 7 軸手臂的關節位置
arm_positions = monitor.get_arm_joint_positions()
print(f"手臂關節: {arm_positions}")  # 7 個數值的陣列

# 取得夾爪的開合程度
gripper_positions = monitor.get_gripper_joint_positions()
print(f"夾爪狀態: {gripper_positions}")  # [Slider9, Slider10]
```

**夾爪關節範圍：**
- Slider9：[0, 0.05]，正值 = 張開
- Slider10：[-0.05, 0]，負值 = 張開

### 6. 取得手指和把手的距離（關節到把手TCP距離）

```python
left_dist, right_dist = monitor.get_finger_to_handle_distances()
print(f"左手指到左把手: {left_dist:.3f}m")
print(f"右手指到右把手: {right_dist:.3f}m")
```
把手的TCP是從風扇的TCP使用 offset推算的，此offset定義於config之中。

### 7. `get_finger_poses()`

取得左右手指在**世界座標系 (World Frame)** 中的實際姿態。這與 `get_end_effector_pose` 不同，後者是夾爪基座或 TCP 中心。這邊取的點是以目前手指的關節位置為準，也就是 Slider9 和 Slider10，而 rotation 則與夾爪 TCP 一致。

```python
left_finger, right_finger = monitor.get_finger_poses()

# left_finger 和 right_finger 都是 PosePq 物件
print(f"左手指世界座標: {left_finger.p}")
print(f"右手指世界座標: {right_finger.p}")
```

### 8. `get_handle_poses()`

取得風扇左右把手的虛擬 TCP 位置：

```python
left_handle, right_handle = monitor.get_handle_poses()

print(f"左把手世界座標: {left_handle.p}")
print(f"右把手世界座標: {right_handle.p}")
```

---

## ⚠️ 碰撞偵測功能

PoseMonitor 現在內建碰撞偵測功能，可以追蹤非法碰撞事件。

### 啟用碰撞偵測

```python
# 初始化後啟動碰撞事件監聽
monitor.initialize()
monitor.start_contact_events()

# 在主迴圈中檢查非法碰撞
def on_physics_step():
    report = monitor.get_illegal_contacts()
    if report.occurred:
        print(f"❌ 偵測到非法碰撞: {report.events}")
```

### 停止碰撞偵測

```python
monitor.stop_contact_events()
```

### 碰撞白名單

碰撞偵測會根據夾持狀態自動切換白名單：
- **未夾持時 (`is_holding=False`)**：忽略 cabinet ↔ fan 的碰撞
- **夾持時 (`is_holding=True`)**：忽略 robot ↔ fan 的碰撞

白名單可在 `GraspDetectionConfig` 中自訂：

```python
config = GraspDetectionConfig(
    collision_whitelist_not_holding=[
        ("/World/Cabinet", "/World/Fan"),  # 忽略 cabinet 和 fan 的碰撞
    ],
    collision_whitelist_holding=[
        ("/World/Robot", "/World/Fan"),    # 夾持時忽略 robot 和 fan 的碰撞
    ],
)
```

### 手動覆寫夾持狀態

```python
# 強制將夾持狀態設為 True（用於碰撞判定）
monitor.set_is_holding_override(True)

# 恢復自動偵測
monitor.set_is_holding_override(None)
```


---
## 所有函數

| **類別**          | **方法/屬性**                          | **功能簡述**             | **回傳型別**           |
| --------------- | ---------------------------------- | -------------------- | ------------------ |
| **Setup**       | `create_default(...)`              | 建立 Monitor (Factory) | `PoseMonitor`      |
|                 | `initialize()`                     | 初始化物理觀察者             | `None`             |
| **Robot Pose**  | `get_end_effector_pose()`          | 取得夾爪中心姿態             | `PosePq`           |
|                 | `get_arm_joint_positions()`        | 取得手臂 7 軸角度           | `np.ndarray`       |
|                 | `get_gripper_joint_positions()`    | 取得夾爪 2 軸位置           | `np.ndarray`       |
|                 | `get_finger_poses()`               | 取得左右手指世界姿態           | `(PosePq, PosePq)` |
| **Object Pose** | `get_handle_poses()`               | 取得左右把手世界姿態           | `(PosePq, PosePq)` |
| **Error/Dist**  | `get_ee_to_fan_error()`            | 夾爪與風扇的誤差             | `PoseError`        |
|                 | `get_fan_to_ground_truth_error()`  | 風扇與 GT 的誤差           | `PoseError`        |
|                 | `get_pose_error_to_target(obj)`    | 夾爪與任意物件的誤差           | `PoseError`        |
|                 | `get_finger_to_handle_distances()` | 手指到把手的個別距離           | `(float, float)`   |
| **Logic**       | `is_holding_fan()`                 | 判斷是否夾住               | `bool`             |
|                 | `reset_holding_confirmation()`     | 重置夾持幀數計數器            | `None`             |
|                 | `set_is_holding_override(value)`   | 覆寫夾持狀態（碰撞用）          | `None`             |
| **Collision**   | `start_contact_events()`           | 啟動碰撞事件監聽             | `None`             |
|                 | `stop_contact_events()`            | 停止碰撞事件監聯             | `None`             |
|                 | `get_illegal_contacts()`           | 取得並清空非法碰撞報告          | `IllegalCollisionReport` |

### 關於 `PosePq` 資料結構

很多方法回傳 `PosePq`，它是一個簡單的 Data Class：

- `.p`: `np.ndarray` (shape: 3,) - 位置向量 $[x, y, z]$
    
- `.q`: `np.ndarray` (shape: 4,) - 四元數 $[w, x, y, z]$

- `.to_T()`: 轉換成 4x4 齊次變換矩陣

### 關於 `IllegalCollisionReport` 資料結構

碰撞偵測回傳的報告物件：

- `.occurred`: `bool` - 是否有發生非法碰撞
- `.events`: `List[dict]` - 每個事件包含 `actor0`, `actor1`, `type` 等資訊

## 🔧 參數設定

### GraspDetectionConfig - 夾取偵測參數

以下是 `GraspDetectionConfig` 的所有參數：

| **參數名稱 (Attribute)**       | **預設值**   | **單位** | **說明與用途**                                                                            |
| -------------------------- | --------- | ------ | ------------------------------------------------------------------------------------ |
| **抓取邏輯 (Grasp Logic)**     |           |        | 用於 `is_holding_fan()` 判定                                                             |
| `grip_position_min`        | `0.019`   | m      | **最小夾持閉合量** (絕對值)。  <br>低於此值則視為夾空。<br>左指(Slider9)需 $\ge$ 此值，右指(Slider10)需 $\le$ 負此值。 |
| `grip_position_max`        | `0.021`   | m      | **最大夾持閉合量** (絕對值)。                                                               |
| `grasp_zone_min_m`         | `0.01415` | m      | **最小有效距離**。  <br>夾爪中心 (TCP) 與目標中心的最小距離。                                          |
| `grasp_zone_max_m`         | `0.02415` | m      | **最大有效距離**。  <br>超過此距離即使夾爪閉合，也會被視為夾空。                                            |
| `hold_confirm_frames`      | `10`      | frames | **夾持確認幀數**。  <br>連續多少幀滿足條件才判定為夾住。                                               |
| **把手幾何 (Handle Geometry)** |           |        | 用於 `get_handle_poses()` 計算                                                           |
| `handle_y_offset`          | `0.1`     | m      | **把手半寬**。  <br>從物體中心沿著 Y 軸 (抓取軸) 到左右把手的距離。左把手為 $+Y$，右把手為 $-Y$。                   |
| `handle_x_offset`          | `-0.015`  | m      | **前後偏移量**。  <br>從物體中心沿著 X 軸 (接近軸) 的偏移。                             |
| **預設路徑 (Default Paths)**   |           |        | 用於 `create_default()` 時的預設值                                                         |
| `robot_prim_path`          | `/World/WorkSpace/RS_M90E7A_Left` | - | 機器人 USD 路徑                                                                       |
| `fan_prim_path`            | (見程式碼)  | -      | 風扇 USD 路徑                                                                          |
| `ground_truth_prim_path`   | (見程式碼)  | -      | 目標位置 USD 路徑                                                                       |
| `cabinet_prim_path`        | (見程式碼)  | -      | 櫃子 USD 路徑                                                                          |
| **碰撞白名單 (Collision)**     |           |        | 用於碰撞偵測                                                                            |
| `collision_whitelist_not_holding` | `[(cabinet, fan)]` | - | 未夾持時忽略的碰撞對。設為 `[]` 可停用白名單。                                               |
| `collision_whitelist_holding`     | `[(robot, fan)]`   | - | 夾持時忽略的碰撞對。設為 `[]` 可停用白名單。                                                 |

### ApproachFrameConfig - 目標座標系設定

以下是 `ApproachFrameConfig` 的參數（此設定包含在 `GraspDetectionConfig.target_frame` 內）：

| **參數名稱**      | **預設值** | **說明**                                    |
| --------------- | --------- | ------------------------------------------ |
| `approach_axis` | `"+y"`    | 目標物體的哪個軸對應夾爪的 **+X 軸**（接近方向） |
| `grasp_axis`    | `"-x"`    | 目標物體的哪個軸對應夾爪的 **+Y 軸**（夾取方向） |

**座標系對應說明：**

這個設定用於將目標物體的 local 座標系轉換為夾爪（end effector）的座標系。

- **夾爪座標系慣例**：
  - **+X 軸**：接近方向（approach）- 夾爪向前移動的方向
  - **+Y 軸**：夾取方向（grasp）- 手指張開的方向
  - **+Z 軸**：上方向（由右手定則決定）

- **使用情境**：如果目標物體不是風扇，或座標系與預設不同時，需要設定這兩個參數。

- **可用值**：`"+x"`, `"-x"`, `"+y"`, `"-y"`, `"+z"`, `"-z"`

- **便捷 property**：可以直接透過 `config.approach_axis` 和 `config.grasp_axis` 存取/設定，不需要手動操作 `target_frame`。

```python
config = GraspDetectionConfig()
config.approach_axis = "-y"  # 自動更新 target_frame
config.grasp_axis = "+x"
```

### RobotJointConfig - 機器人關節設定

如果需要自訂機器人關節名稱或夾爪幾何：

| **參數名稱**               | **預設值**                              | **說明**                              |
| ------------------------ | ------------------------------------- | ------------------------------------ |
| `arm_joint_names`        | `("Revolute7", ..., "Revolute1")`    | 手臂關節名稱（7軸）                      |
| `gripper_joint_names`    | `("Slider9", "Slider10")`            | 夾爪關節名稱                            |
| `end_effector_prim_suffix` | `"TF_1/gripper_base_2/rail_2"`     | 末端執行器 prim 相對路徑                 |
| `end_effector_offset`    | `(0.06, 0.0, 0.0)`                   | 末端執行器偏移量                         |
| `left_finger_y_offset`   | `0.075`                              | 左手指 Y 軸偏移                          |
| `right_finger_y_offset`  | `-0.075`                             | 右手指 Y 軸偏移                          |

```python
from articulation_observer import RobotJointConfig

joint_config = RobotJointConfig(
    end_effector_offset=(0.07, 0.0, 0.0),
)

monitor = PoseMonitor.create_default(
    robot_prim_path="/World/Robot",
    fan_prim_path="/World/Fan",
    ground_truth_prim_path="/World/GT",
    joint_config=joint_config,
)
```

---

## 📦 類別總覽

```
┌─────────────────────────────────────────────────────────────┐
│                       PoseMonitor                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 組合的元件：                                           │  │
│  │  • ArticulationObserver (監控機器人關節狀態)           │  │
│  │  • TargetObject - fan (風扇物件)                       │  │
│  │  • TargetObject - ground_truth (目標位置)              │  │
│  │  • GraspDetectionStrategy (夾取偵測策略)               │  │
│  │  • CollisionDispatcher (碰撞偵測分發器)                │  │
│  │  • ContactEventSource (碰撞事件來源)                   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

什麼時候要呼叫 `initialize()`？
**A:** 在 Isaac Sim 模擬開始**之後**、第一次使用 monitor **之前**呼叫。

碰撞偵測是否預設啟用？
**A:** 否，需要手動呼叫 `start_contact_events()` 才會開始監聽碰撞事件。

---

## 📝 完整範例

```python
import numpy as np
from pose_monitor import PoseMonitor
from grasp_config import GraspDetectionConfig

# 建立自訂設定
config = GraspDetectionConfig(
    hold_confirm_frames=5,  # 降低確認幀數
)

# 建立 monitor
monitor = PoseMonitor.create_default(
    robot_prim_path="/World/WorkSpace/RS_M90E7A_Left",
    fan_prim_path="/World/WorkSpace/Scene/Fan",
    ground_truth_prim_path="/World/WorkSpace/Scene/GroundTruth",
    grasp_config=config,
)

# 模擬開始後初始化
monitor.initialize()

# 啟動碰撞偵測（可選）
monitor.start_contact_events()

# 主迴圈中使用
def on_physics_step():
    # 取得目前誤差
    error = monitor.get_ee_to_fan_error()
    
    # 印出狀態
    print(f"距離目標: {error.distance:.3f}m, 角度誤差: {np.degrees(error.angle_error):.1f}°")
    
    # 檢查是否夾住
    if monitor.is_holding_fan():
        print("夾住了！可以開始移動")
        
        # 檢查風扇到目標的誤差
        gt_error = monitor.get_fan_to_ground_truth_error()
        print(f"風扇到目標: {gt_error.distance:.3f}m")
    
    # 檢查非法碰撞
    collision_report = monitor.get_illegal_contacts()
    if collision_report.occurred:
        print(f"⚠️ 發生非法碰撞: {collision_report.events}")
    
    # 取得關節狀態（如果需要）
    arm_joints = monitor.get_arm_joint_positions()
    gripper_joints = monitor.get_gripper_joint_positions()
```

---

## 🔗 相關檔案

| 檔案 | 說明 |
|------|------|
| `pose_monitor.py` | 主要類別 |
| `articulation_observer.py` | 機器人關節狀態監控 |
| `target_object.py` | 目標物體 wrapper |
| `grasp_config.py` | 設定類別（PosePq、GraspDetectionConfig 等）|
| `collision_observer.py` | 碰撞偵測與白名單邏輯 |
| `contact_event_source.py` | PhysX 碰撞事件訂閱 |

---
