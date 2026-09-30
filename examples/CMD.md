# 常用命令

三个终端分开开：先起 Cosmos，再开可视化，最后跑 prompt。可视化听 UDP，不要用 `--source rmi`（RMI 同时只能有一个客户端，live 已经占着）。

`run_fanuc_live` 连上机器人后把关节角发到 `127.0.0.1:5005`。提示符输入 `go` 才开始感知和执行。Live UI：`http://127.0.0.1:8765/`。

Cosmos Nano 感知规划走下面的 `:8000`。FANUC 的 Cosmos Edge 动作策略是另一条链路（端口 `8001`，不能加载 SO-101 权重），见 `examples/cosmos_edge_fanuc/README.md`。

## 1. 开启 Cosmos 本地服务

窗口要一直开着。看到 `Uvicorn running` 后再跑任务。检查：`curl http://127.0.0.1:8000/v1/models`。

```bash
export COSMOS_BASE_URL=http://127.0.0.1:8000/v1
/home/gb2_8634/cosmos-reasoner/serve.sh
```

## 2. 执行 prompt 任务

在跑任务的终端里也要有 `COSMOS_BASE_URL`（上面那条 `export`）。改 `--instruction` 即可换任务。

```bash
cd ~/lerobot/examples
export COSMOS_BASE_URL=http://127.0.0.1:8000/v1
UV_NO_SYNC=1 uv run python -m manipulation.run_fanuc_live \
  --vlm cosmos --camera /dev/video0 --execute \
  --instruction "pick all screws and put them into the yellow container"
```

Gemini 把 `--vlm cosmos` 换成 `--vlm gemini`（需要 `GOOGLE_API_KEY` 或 `GEMINI_API_KEY`）。

## 3. 可视化

先开可视化，再跑上面的 live。画面先停着（没数据），live 连上后才跟着动。Cursor 远程终端需要 `DISPLAY=:1`。

MuJoCo 和 Omniverse 不要同时听 `5005`。只开其中一个时用下面的命令。

### MuJoCo

```bash
cd ~/lerobot/fanuc_lrmate200id_smc
DISPLAY=:1 .venv-twin/bin/python twin/twin.py --source udp --udp-port 5005
```

### Omniverse / Isaac Sim

带桌面和感知物体（俯视，跟最新一次 `perceive_*`）：

```bash
cd ~/lerobot/fanuc_lrmate200id_smc
OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
  DISPLAY=:1 ~/isaacsim-venv/bin/python twin/omni_twin.py \
  --source udp --udp-port 5005 --camera top \
  --scene '../examples/*/runs/*/perceive_*/geometric_view.json'
```

只要机械臂、斜视：

```bash
cd ~/lerobot/fanuc_lrmate200id_smc
OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
  DISPLAY=:1 ~/isaacsim-venv/bin/python twin/omni_twin.py \
  --source udp --udp-port 5005 --camera persp
```

### 两个一起看

live 仍发 `5005`。中转再分给 MuJoCo `5006`、Isaac `5007`。

```bash
cd ~/lerobot/fanuc_lrmate200id_smc
.venv-twin/bin/python twin/udp_relay.py
```

```bash
DISPLAY=:1 .venv-twin/bin/python twin/twin.py --source udp --udp-port 5006
```

```bash
OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
  DISPLAY=:1 ~/isaacsim-venv/bin/python twin/omni_twin.py \
  --source udp --udp-port 5007 --camera top \
  --scene '../examples/*/runs/*/perceive_*/geometric_view.json'
```
