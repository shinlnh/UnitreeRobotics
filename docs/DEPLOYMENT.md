# Deployment

## Máy và environments

Khuyến nghị tách hai máy hoặc ít nhất hai environment:

- GPU machine: Isaac-GR00T, Python 3.12, CUDA, checkpoint, PolicyServer.
- Robot/inference machine: GEAR-SONIC Python 3.10 envs, C++ controller, camera và Unitree network.

ZMQ PolicyServer mặc định ở TCP 5550. Camera mặc định 5555. SONIC nội bộ dùng thêm các port cho action/state/keyboard; kiểm tra firewall và port conflict trước rollout.

## Setup

```bash
scripts/bootstrap_upstreams.sh --install
cd .upstream/GR00T-WholeBodyControl/gear_sonic_deploy
just build
cd ../../..
scripts/download_models.sh
```

Không cài hai upstream vào cùng virtualenv. GR00T N1.7 và SONIC inference pin Python/dependency khác nhau.

`docker/Dockerfile.cpu` chỉ đóng gói smoke/evaluation harness. Với GR00T GPU, Jetson Orin/Thor hoặc DGX Spark, dùng Dockerfiles và activation scripts theo đúng platform trong upstream Isaac-GR00T; không dùng CPU image để serve model.

## Start order

1. Camera server / MuJoCo camera.
2. GR00T PolicyServer.
3. SONIC launcher.
4. Quan sát state/camera/policy health.
5. Init pose.
6. Engage controller bằng operator.

PolicyServer:

```bash
gr00t-g1 serve --checkpoint /path/to/checkpoint --execute
```

MuJoCo:

```bash
gr00t-g1 deploy --mode sim --prompt "pick up the apple" --execute
```

Launcher tạo tmux. Các control chính:

- `p`: pause/resume VLA inference;
- `k`: start/stop C++ control loop;
- `i`: blend tới initial pose;
- `t <prompt>`: đổi language prompt;
- `O` trong C++ controller: emergency stop;
- `Ctrl+\\`: kill tmux session.

## Remote PolicyServer

Đổi:

```toml
[model]
host = "GPU_MACHINE_IP"
bind_host = "0.0.0.0"
port = 5550
```

Chỉ expose port trên trusted robot network. Nếu đi qua shared network, dùng firewall/VPN và API-token support của upstream PolicyServer; không mở unauthenticated service ra Internet.

## Checkpoint gate

`gr00t-g1 serve` từ chối:

- checkpoint rỗng;
- `nvidia/GR00T-N1.7-3B` base checkpoint dùng trực tiếp với `UNITREE_G1_SONIC`.

Đây là lỗi cấu hình, không phải thiếu flag. Cần fine-tune checkpoint post-training.

## Troubleshooting order

1. `gr00t-g1 doctor --profile sim --online`.
2. Xác nhận upstream revisions đúng config.
3. Xác nhận Cosmos gated access.
4. Xác nhận đúng SONIC encoder + decoder + observation config.
5. Ping PolicyServer và kiểm tra host/port.
6. Kiểm tra ego frame timestamp và robot state không stale.
7. Kiểm tra action max không vượt upstream bound.
8. Chạy sim không recorder để loại trừ I/O latency.
