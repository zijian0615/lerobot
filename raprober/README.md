# raprober — 黑色 Cube 抓取放置工具

用一台 SO100 机械臂 + 一个**固定在顶部的相机(眼在手外 / eye-to-hand)**,自动把桌面上的黑色 cube 抓起来放到指定位置。
纯脚本化方案(经典视觉 + 逆运动学),**不需要训练**。

## 原理

```
观察位(手臂让开,固定顶部相机俯视桌面)
   └─> OpenCV 阈值分割黑色 cube  → 像素中心 (u,v)
        └─> 手眼单应 H  → 桌面基座坐标 (x,y)
             └─> placo 逆运动学  → 关节角
                  └─> 悬停 → 下降 → 合爪 → 抬起 → 移到放置点 → 开爪 → 回 home
```

- **感知**:cube 是黑色、桌面是浅木色,HSV 的 Value 通道一阈值即可分割(`perception.py`)。
- **定位**:相机固定在顶部(眼在手外),看平面桌子 → "像素→桌面平面"的单应矩阵**全局恒定、与手臂姿态无关**,标定一次即可(`calibration.py`)。观察位只负责把手臂挪开、别挡住 cube。
- **运动**:placo + SO100 URDF 做 FK/IK。SO100 只有 5 个非夹爪自由度,**完整 6-DOF 位姿是过约束的**(位置和整个姿态无法同时满足)。因此 IK 只用两个任务:**位置任务(高权重,残差 ~0mm)+ 轴对齐任务(把夹爪接近轴对到世界 -Z,朝下)**,放开绕接近轴的自转。合理桌面点位置残差 ~0mm 且夹爪朝下度 ≈1.0(`kinematics_ik.py`)。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `config.py` | 端口 / 相机 / 关键位姿 / 放置点 / 阈值,**所有硬件相关参数集中在这里** |
| `perception.py` | 黑色 cube 检测,可单独测试 |
| `kinematics_ik.py` | FK / 迭代 IK 封装 |
| `arm.py` | SO100 连接、插值移动、夹爪、torque 开关 |
| `calibration.py` | 交互式手眼标定,生成单应矩阵 |
| `pick_place.py` | 完整抓取-放置主流程 + CLI |
| `assets/SO-ARM100/…/SO100/so100.urdf` | placo 用的 SO100 URDF(已随仓库克隆) |

## 前置依赖(已就绪)

- `placo` 已安装。
- SO100 URDF 已放在 `assets/SO-ARM100/Simulation/SO100/so100.urdf`。

## 使用步骤

> 所有命令在仓库根目录 `~/lerobot` 下运行,用 `uv run`。

### 0. 确认端口/相机,给串口权限

```bash
sudo chmod 666 /dev/ttyACM0    # 执行抓取的那个臂的端口
```

打开 `config.py`,把 `arm_port`(默认 `/dev/ttyACM0`)和 `camera.index_or_path`
(顶部固定相机的设备号,如 `/dev/video4`)改成你实际用的那一对。

### 1. 电机标定(每个臂一次)

抓取依赖准确的关节读数,必须先做 lerobot 原生电机标定:

```bash
uv run lerobot-calibrate --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 --robot.id=grasper
```

`--robot.id` 要和 `config.py` 里的 `arm_id`(默认 `grasper`)一致。

### 2. 调关键位姿(home / observe)

`config.py` 里的 `home_pose` 和 `observe_pose` 是**初始猜测值**,必须按你的实际
安装调好:

- `observe_pose`:把手臂完全挪出固定相机对桌面的视野(别挡住 cube)。
- `home_pose`:安全的收起姿态。

可以用遥操作/teleoperate 或临时脚本把臂摆到位后读关节角,填回去。

### 3. 手眼标定(每次相机/观察位变动后一次)

```bash
uv run python -m raprober.calibration --port /dev/ttyACM0 --camera /dev/video0 --samples 6
```

流程(重复 6 次,cube 每次放不同位置,尽量分散):

1. 把 cube 放到视野内某处,回车 → 脚本在观察位检测到像素中心。
2. 脚本释放 torque,你**用手把夹爪末端移到 cube 中心**,回车 → 脚本用 FK 记录该点的基座坐标。
3. 收集完自动拟合单应矩阵,打印残差(mean/max),保存到 `calib_out/handeye_homography.json`。

> 残差 max 应 < 1cm,否则重采(点分散一些、末端对准 cube 中心)。

### 4. 设定放置点

`config.py` 的 `place_xy`(基座坐标,米)先写死一个点。也可以运行时覆盖:

```bash
uv run python -m raprober.pick_place --place-x 0.15 --place-y -0.15
```

### 5. 先 dry-run 验证轨迹(强烈建议)

不接电机、给定 cube 坐标,打印整条轨迹每个路点的目标 xyz、IK 残差(mm)、夹爪朝下度、是否可达:

```bash
uv run python -m raprober.pick_place --dry-run --cube-x 0.05 --cube-y -0.18 \
  --place-x 0.15 --place-y -0.12
```

- 某些路点 `UNREACHABLE` 或 `down` 远小于 1 → 调 `table_z`/`hover_z`/`lift_z` 或换放置点,直到全 `OK`。
- **若检测到的 cube 坐标离基座只有几厘米(如 `(-0.011,-0.065)`)→ 手眼标定没标好**,cube 物理上不可能在那,先重做标定。

### 6. 运行抓取

```bash
uv run python -m raprober.pick_place --port /dev/ttyACM0 --camera /dev/video0 --cycles 1
```

## 单独测试感知(不接硬件)

```bash
uv run python -m raprober.perception <某张图.png> --h-min 35 --h-max 85 --s-min 70
```

会输出检测到的像素坐标,并保存标注图。绿 cube 检不到就放宽 hue 带(`--h-min/--h-max`)或调低 `--s-min/--v-min`;若背景被误检,收紧这些阈值或在 `config.py` 里设 `cube.roi` 裁掉干扰区域。

## 关键可调参数(`config.py`)

| 参数 | 说明 |
| --- | --- |
| `cube.h_min/h_max/s_min/v_min` | 绿色判定的 HSV 阈值,检不到就放宽 |
| `cube.roi` | 限定搜索区域(cube 在画面下方,手臂在上方) |
| `grasp.table_z` | 桌面在基座坐标下的高度(米),**决定下爪深度,最关键** |
| `grasp.hover_z / lift_z` | 悬停 / 搬运高度 |
| `grasp.gripper_open / closed` | 夹爪开合的电机值(0–100) |
| `grasp.approach_axis` | 夹爪接近轴(默认局部 z),朝下抓不对时改这个 |
| `grasp.ik_align_weight` | 朝下对齐权重(默认 1.0) |
| `max_relative_target` | 单步关节位移安全上限(度),防止大跳变 |

## 已知限制 / 后续扩展

- **5-DOF 姿态受限**:夹爪能垂直朝下(靠轴对齐),但**绕竖直轴的自转不可控**——夹爪张开方向无法主动对齐 cube 的朝向,对细长物体或需要特定抓取角的场景成功率会下降(cube 近似正方体一般没问题)。工作空间边缘的点可能不可达,用 `--dry-run` 先确认。
- **固定顶部相机(眼在手外)**:单应全局有效,是最稳的配置。注意相机一旦被挪动/碰动,必须重新标定。cube 必须落在相机视野和手臂可达范围的交集内。
- **放置点写死**:已支持 `--place-x/--place-y` 覆盖;后续可接第二种颜色/marker 识别放置点。
- **首次务必慢速、手放急停附近**:`observe_pose`/`table_z` 没调好可能撞桌面。
