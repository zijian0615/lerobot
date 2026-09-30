# FANUC Cosmos Edge 动作策略

这条链路和 [`CMD.md`](../CMD.md) 里的 Cosmos Nano 感知规划是分开的。

- Nano reasoner（`cosmos-reasoner/serve.sh`）占 **8000**，给 `manipulation.run_fanuc_live` 做感知和符号规划，手臂走笛卡尔 `FRC_LinearMotion`。
- 这里是以后要后训练的 **动作策略**：绝对关节块，服务占 **8001**。`run_fanuc_live` 不会调用它。

不要把 [kabilanKB/cosmos_edge_policy_so101](https://huggingface.co/kabilanKB/cosmos_edge_policy_so101) 的权重接到 FANUC。那是 domain **22**、6 维 LeRobot `.pos`、腕部相机在上。本目录的合同是 domain **23**、7 维（J1–J6 度 + 夹爪 0/1）、只有头顶相机。

## 这台机器（2026-09-22 探测）

`python -m cosmos_edge_fanuc.probe`

| 项 | 结果 |
| --- | --- |
| 机器 | aarch64，NVIDIA GB10，compute 12.1，内存约 121 GiB |
| Python | 3.12.3。没有 `python3.13` |
| cosmos-framework | 未安装（`~/cosmos-framework` 不存在） |
| 8000 / 8001 | 探测时都没有进程在听 |

[cosmos-framework 的 pyproject](https://github.com/NVIDIA/cosmos-framework/blob/main/pyproject.toml) 把 aarch64 列为支持平台。真正加载 Edge 权重的 CUDA 组要求 **Python 3.13**。训练组里的 `torchao` 只发布了 x86_64 轮子。所以现在不能在这台机器上把 policy server 装起来，也没有去拉训练依赖。

## 数据和相机缺口

示教条数是 **0**。仓库里没有 FANUC 的 LeRobot 数据集，`fanuc_lerobot_stats.json` 的 min/max 来自 URDF 外包络，不是从回合里统计的。

关节：策略要的是示教器角度 J1–J6，加上夹爪。现有 `Fanuc.send_action` 发的是笛卡尔位姿；夹爪是端口 3（开）/ 端口 4（关），不是连续开合量。

相机：

- `/dev/video0`：Full HD webcam，`usb-NVDA8000:00-1.4.1`，标定和 live 用的头顶相机。
- `/dev/video1`：同一只摄像头的 metadata 节点。
- `/dev/video2`：Innomaker U20CAM 720P，`usb-NVDA8000:00-1.4.4`，腕部相机。FANUC 还没有这只相机的外参，仿真里先按法兰旁、对准 TCP 来放。
- `/dev/video3`：这只 Innomaker 的 metadata 节点。

策略描述是头顶加腕部两路，不用 SO-101 的「上半腕部、下半头顶」拼接。

没有这些回合之前，不要开始后训练，也不要 `--execute` 关节流。

## 后训练合同（还没训）

| 项 | 值 |
| --- | --- |
| domain | `fanuc` / 23 |
| 动作 | `joint_pos`，32 步 × 7，30 fps |
| 归一化 | minmax，[`fanuc_lerobot_stats.json`](fanuc_lerobot_stats.json) |
| 夹爪 | 不翻转。0 开，1 关 |
| 图像 | 头顶 + 腕部，各 640×360 |

J3 的 min/max 是 `J3_fanuc = J3_model - J2` 在两条 URDF 区间上的外包络（约 -189° 到 315°）。真正发运动前用模型关节限位再查一次。

```bash
cd ~/lerobot/examples
UV_NO_SYNC=1 uv run python -m cosmos_edge_fanuc.joint_client --dry-run
```

这只在本地组一帧 `FRC_JointMotion`，不连机器、不连 8001。

## 仿真数据（LeRobot v3）

任务在 [`task_scene.json`](task_scene.json)：把黑方块放进、或把笔平放在蓝盒 / 橙盒上。四个物体都是扫描或建模的 USD 资产（`scanner/objects/<name>/`，`twin/crop_object.py` 生成，原点在底面中心）。每一集随机摆放四个物体、随机选一个物体和一个盒子，`sim_teacher.py` 解出俯视抓取轨迹（夹爪开合方向垂直于笔身），物体在夹爪闭合时跟随 TCP（不做接触物理），松开后方块落到盒底、笔架在盒沿上。`record_overhead.py` 用对齐后的桌面扫描做背景，头顶和腕部相机都带真实镜头畸变。`observation.state` 和 `action` 都是 J1–J6 度加夹爪 0/1。

当前 `data/fanuc_sim_bin`（已推到 `zijian2022/fanuc_sim_bin`）：4 集、897 帧、30 fps，四种任务各一集（`SEED=2`）。

重做：

```bash
SEED=2 examples/cosmos_edge_fanuc/record_sim.sh 4
```

等有了 FANUC checkpoint，并且装好带 `--domain-name` 的 cosmos-framework 之后：

```bash
examples/cosmos_edge_fanuc/serve_policy.sh /path/to/fanuc_checkpoint
```

SO-101 路径会直接退出。框架没装、或上游服务没有 `--domain-name` 时也会退出，避免静默当成 DROID 服务。
