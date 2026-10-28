# Cloud relay for the cell

Teams reach the cell through our VM. Neither the lab nor the teams accept inbound connections from
each other:

```
team machine --HTTPS--> VM :8443 relay_proxy.py --> VM 127.0.0.1:9099 ==SSH reverse tunnel== GB10 --> cell server 10.20.8.103:9099
```

- The relay forwards only `/api/v1/*`. The operator API (`/admin`) returns 403 there, so the cell can be
  operated only from the lab.
- Every team request still needs its run token. TLS keeps the tokens off the wire.
- The lab side makes only an outbound SSH connection, using a key that can open nothing but this one
  forward.

## VM setup (once)

Any small Linux VM with a public IP works. It needs no GPU and only Python 3.

1. Copy `relay_proxy.py` to the VM.
2. Make a TLS certificate. A self-signed one is fine; teams receive `cert.pem` and pass it to `run.py --ca`.
   Replace `<VM IP>` with the VM's public IP:
   ```bash
   openssl req -x509 -newkey rsa:2048 -nodes -days 90 -subj "/CN=<VM IP>" -addext "subjectAltName=IP:<VM IP>" -keyout key.pem -out cert.pem
   ```
3. Allow the GB10's tunnel key. Add this line, from `~/.ssh/cell_tunnel_brev.pub` on the GB10, to
   `~/.ssh/authorized_keys` on the VM. The key can then open only the reverse forward to 127.0.0.1:9099:
   ```
   restrict,port-forwarding,permitlisten="127.0.0.1:9099",command="/bin/false" ssh-ed25519 AAAA... cell-tunnel ...
   ```
4. Open TCP 8443 in the VM's firewall or security group. Port 9099 stays closed, because the tunnel binds
   only to localhost.
5. Run the relay, for example in tmux:
   ```bash
   python3 relay_proxy.py --port 8443 --cert cert.pem --key key.pem
   ```

## Current deployment: hackathon.sidclab.org

The Azure VM (`azureuser@172.208.64.66`, admin key `~/.ssh/cell_relay_admin` on the GB10) serves
`https://hackathon.sidclab.org` (DNS at GoDaddy: one A record for `hackathon`; `@` and `www` are the lab's site, leave
them alone). The VM also hosts demo.sidclab.org, so the relay sits behind its nginx:

- nginx: `/etc/nginx/sites-available/hackathon.sidclab.org` (our only file there) ends TLS and proxies everything to
  127.0.0.1:8443. Certificate: certbot webroot (`/var/www/html`), renewed by the VM's certbot timer, which then
  reloads nginx (`renew_hook` in `/etc/letsencrypt/renewal/hackathon.sidclab.org.conf`).
- `hackathon-relay.service` (systemd): `relay_proxy.py --bind 127.0.0.1 --port 8443 --behind-proxy --cache site_cache.html
  --rollouts rollouts --results results --datasets datasets --viz 127.0.0.1:7860 --static public` in `~/cell_relay`;
  log in `~/cell_relay/relay.log`. `systemctl status hackathon-relay`.
- Public paths: `/` and `/leaderboard` (the cell's `/site`, cached while the tunnel is down), `/results/<key>.json` and
  `/results/<key>/rollouts/...` (one team's runs and videos, for its My results; pushed by the cell into
  `~/cell_relay/results` and `~/cell_relay/rollouts`), `/files/<name>` (`~/cell_relay/public`), `/api/v1/*` (token),
  `/org/*` (Organizer session). `/rollouts/...` only after `operator_console.py reveal`. Everything else 403. The access
  log shows only the first 6 characters of a key.

To update the relay code: copy `relay_proxy.py` to `~/cell_relay/` and `sudo systemctl restart hackathon-relay`.

Dataset viewer (`/viz/<key>/<run>/episode_<n>`): `hackathon-viz.service` runs our fork of
[lerobot-dataset-visualizer](https://github.com/huggingface/lerobot-dataset-visualizer) (branch `manipbench`,
`~/lerobot-dataset-visualizer` on the GB10) from `~/viz` on 127.0.0.1:7860, settings in `~/viz/viz.env`
(`VIZ_BASE_PATH=/viz`, `SELF_HOSTED=1`, `DATASET_URL=https://hackathon.sidclab.org/vizdata/datasets`; read at build
*and* start). It reads each run's LeRobotDataset from `~/cell_relay/datasets/<run>` (the cell rsyncs it after every
trial) through the relay's `/vizdata/...`, which checks the key like `/results/...`; `/vizdata/download/<key>/<run>.tar`
streams the whole dataset for download. To update it:

```bash
rsync -a --delete --exclude node_modules --exclude .next --exclude .git --exclude viz.env ~/lerobot-dataset-visualizer/ azureuser@172.208.64.66:viz/
ssh azureuser@172.208.64.66 'cd ~/viz && ~/.bun/bin/bun install --frozen-lockfile && set -a && . ./viz.env && set +a && ~/.bun/bin/bun run build && sudo systemctl restart hackathon-viz'
```

**manipbench.com** (the old address, Brev VM `ubuntu@34.61.90.95`, DNS at Cloudflare) runs `relay_proxy.py
--redirect-to https://hackathon.sidclab.org`: it sends browsers to the new address (308) and still forwards
`/api/v1` to the cell for clients set up with the old URL, so it keeps its own tunnel (tmux `relaytun`) until the
event ends.

## During a slot (GB10)

```bash
uv run python fanuc_er4ia_sim/challenge/robot_server.py --backend fanuc --port 9099 ...
fanuc_er4ia_sim/challenge/tunnel_to_team.sh azureuser@172.208.64.66 9099   # outbound tunnel, reconnects (tmux relaytun_azure)
```

Teams then run `uv run run.py`: their client reaches `https://hackathon.sidclab.org` and picks up its token once you
approve their request.

The relay prints one access line per request: time, source IP, path, status, bytes and milliseconds. Stop
the tunnel (Ctrl+C) when the slot ends.

**Bandwidth:** three 640×480 PNG images are about 480 KB per observation; with `IMAGE_FORMAT = "jpeg"` they
are about 70 KB. At startup the client logs the cell's clock offset and the round-trip time.
