# Architecture

## Production data plane

Ba process chính được tách ra để dependency và real-time loop không cản nhau:

1. Isaac-GR00T `PolicyServer` chạy checkpoint trên GPU và phục vụ ZMQ.
2. GEAR-SONIC `run_vla_inference.py` đọc camera/state, gửi observation, nhận action chunk và bù latency.
3. `gear_sonic_deploy` giải mã latent token và chạy whole-body controller ở 50 Hz trong C++.

Observation contract cho `UNITREE_G1_SONIC`:

| Modality | Keys |
|---|---|
| Video | `ego_view` |
| State | `left_leg`, `right_leg`, `waist`, `left_arm`, `right_arm`, `left_hand`, `right_hand`, `projected_gravity` |
| Language | `annotation.human.task_description` |

Action contract cho mỗi inference:

| Key | Shape | Ý nghĩa |
|---|---:|---|
| `motion_token` | `40 × 64` | SONIC latent whole-body reference |
| `left_hand_joints` | `40 × 7` | Left hand targets |
| `right_hand_joints` | `40 × 7` | Right hand targets |

VLA chạy xấp xỉ 2.5 Hz. Inference client phát từng action ở 50 Hz và bỏ qua các action đầu chunk đã stale dựa trên measured inference latency. Không được nối 64-dim latent trực tiếp vào Unitree SDK: SONIC decoder là lớp bắt buộc để tạo balanced full-body joint commands.

## Control plane của project

`gr00t-g1` giữ các tham số dùng chung trong `configs/project.toml` và dựng command cho upstream. Mặc định command chỉ được in ra. `--execute` mới tạo process hoặc dùng GPU.

```text
project.toml
  ├── train ───────> Isaac-GR00T launch_finetune.py
  ├── serve ───────> Isaac-GR00T run_gr00t_server.py
  ├── open-loop ───> Isaac-GR00T open_loop_eval.py
  ├── collect ─────> SONIC launch_data_collection.py
  ├── process ─────> SONIC process_dataset.py
  └── deploy ──────> SONIC launch_inference.py
```

## CPU smoke backend

Smoke backend là world 2D deterministic có random object placement. Policy heuristic nhận language task spec và phát action chunk có đúng horizon/dimensions của SONIC. Safety supervisor clip Cartesian proxy deltas và workspace. Nó xác minh:

- task selection và multi-episode generalist orchestration;
- action schema validation;
- chunk execution horizon;
- safety clip và emergency stop;
- success metrics, event log, JSONL và HTML report.

Ba dimension đầu của synthetic 64D token được simulator diễn giải như `dx/dy/dz`. Đây chỉ là test adapter, không phải latent space của checkpoint SONIC thật.

## Reproducibility

Hai upstream được pin trong `configs/project.toml`. Script bootstrap checkout detached revision. Update revision phải đi kèm:

1. đọc release notes/interface changes;
2. chạy CPU suite;
3. rebuild controller;
4. dataset validator trên sample thật;
5. open-loop evaluation;
6. closed-loop MuJoCo regression.
