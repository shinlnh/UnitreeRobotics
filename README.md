# Unitree G1 RL + GR00T trên Isaac Lab

Đây là project chạy được, không phải skeleton tài liệu: ngoài locomotion PPO trên Isaac Lab,
project có một track **GR00T N1.7 đa nhiệm cho Unitree G1** dùng chung một checkpoint cho bốn
lệnh lấy táo, lê, nho hoặc khế; có dataset audit, loader validation, fine-tune, scene bốn vật,
policy server và closed-loop whole-body demo thật.

Stack được khóa ngày **2026-08-09**:

- Isaac Sim `6.0.1.0`
- Isaac Lab `3.0.0` tại commit `8718973527ae347b6a7a1e62a77e71bd03ffa2f0`
- RSL-RL `5.4.1`
- NVIDIA GR00T `N1.7`, model `nvidia/GR00T-N1.7-3B` revision `2fc962b9...`, code commit `b9955401...`
- Dataset G1 đa nhiệm `nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1` revision `0d7bdd06...`
- ApplePnP deployment `nvidia/GR00T-N1.7-ApplePnP-V1` tại revision `1c956f1f...`
- LeApp `0.6.0`, ONNX Runtime GPU `1.23.2` (CUDA 12), GEAR WBC tại commit `1983e888...`
- Python `3.12`

Toàn bộ pin nằm trong [`configs/setup/versions.env`](configs/setup/versions.env). Bootstrap luôn checkout đúng commit nên kết quả không phụ thuộc vào nhánh `main/develop` thay đổi sau này.

## Kiến trúc chính: một checkpoint GR00T đa nhiệm

```text
Lệnh Việt/Anh ──► target guard (không cho lệnh mơ hồ)
                         │
2 RGB đầu G1 + EEF/proprio + lệnh gốc
                         │
                         ▼
          GR00T N1.7-3B REAL_G1 generalist (mặc định)
             một checkpoint, không router theo vật
                         │ chunk 40 bước @ 20 Hz
                         ├─ hai wrist EEF9D + hands 14D
                         └─ navigation 3D + base height
                                          │
                                          ▼ adapter 20 → 50 Hz
                            Pink IK (thân trên) + AGILE (chân)
                                          │
                                          ▼
                               Unitree G1 trong Isaac Sim
```

Chế độ mặc định dùng thẳng pretrain tag `REAL_G1` đã nằm trong base checkpoint: language, hai
RGB frame, EEF/joint, base height và navigation; không cần checkpoint vô-lăng và không bắt buộc
train trước. Runner Isaac dịch EEF9D base-relative sang pose cổ tay world-frame, đổi đúng thứ tự
14 joint bàn tay rồi đưa navigation/base-height vào action 32D; Pink IK và AGILE mới là tầng
control vật lý. Cả bốn vật cùng nằm trong scene, còn câu lệnh đi nguyên vẹn vào cùng một policy.
Không có đoạn `if apple: load checkpoint A` hoặc selector checkpoint theo tên vật.

Recipe fine-tune tùy chọn bên dưới tạo **một model đa nhiệm có language conditioning** từ bốn
LeRobot subdataset trộn cân bằng. Dataset public này chỉ chứa RGB + 43 joint ở 20 Hz, không chứa
EEF/navigation, nên checkpoint 43D đó được evaluate/deploy qua GEAR WBC/RoboCasa; không gắn nhãn
giả là checkpoint mobile-manipulation Isaac. Luồng Isaac mặc định dùng đúng contract `REAL_G1`
của base N1.7, vì contract đó mới có EEF và navigation.

Phạm vi đã có dữ liệu thật là bốn biến thể pick-and-place kể trên. Nó đa nhiệm hơn checkpoint
vô-lăng, nhưng **chưa phải** model có thể làm mọi việc gia dụng tùy ý. Muốn thêm chai, hộp, mở
ngăn kéo… phải bổ sung demonstration có đúng vật, scene và success criterion rồi tiếp tục joint
fine-tune. Không có checkpoint public nào tự bảo đảm mọi task G1 ngoài tập demonstration.

Demo chính chạy trong Isaac Sim 6.0/Isaac Lab 3.0 trên task locomanipulation G1: RGB camera thật
của simulator, contact/rigid-body physics, Pink IK và AGILE locomotion. RoboCasa/MuJoCo + GEAR
WBC vẫn được giữ làm backend reference cho checkpoint post-train 43D, không còn là demo mặc định.

Baseline navigation và các track cũ vẫn được giữ để nghiên cứu/so sánh:

```text
Track chính thức ApplePnP:
RGB 480×640 + 43D proprio ──► GR00T ApplePnP ONNX (30 Hz, chunk 16)
                            └─► arms + hands + waist + nav + base height
                                                │
                                                ▼ 50 Hz
                                      GEAR WBC Balance/Walk

Track task tùy biến:
RGB + proprio + language ──► GR00T UNITREE_G1_SONIC
                            └─► motion token 64D + hands 14D
                                                │
                                                ▼
                                          SONIC v1.1
```

PPO `model_2799.pt` vẫn là baseline locomotion đã kiểm chứng. ApplePnP dùng checkpoint deployment đã post-train sẵn; SONIC là đường mở rộng khi cần thu demonstration và train task khác.

Robot mặc định là **Unitree G1** vì đây là Unitree embodiment được GR00T N1.7 hỗ trợ chính thức.
Demo đa nhiệm dùng G1 29-DoF + hai bàn tay của locomanipulation stack; các task PPO locomotion
riêng dùng `G1_MINIMAL_CFG`. Cả hai lấy robot asset trực tiếp từ Isaac Lab, không copy USD robot.

## 1. Chuẩn bị

Cần máy Linux có NVIDIA GPU/driver tương thích Isaac Sim, `git`, `uv`, đủ dung lượng cho Isaac Sim + model GR00T. Chỉ chấp nhận EULA sau khi đã đọc điều khoản NVIDIA:

```bash
cp .env.example .env
export OMNI_KIT_ACCEPT_EULA=1
make setup-isaaclab
make setup-groot
make doctor
```

Hai bootstrap tạo:

- `.deps/IsaacLab/.venv`: Isaac Sim/Lab, RSL-RL và package của project.
- `.deps/Isaac-GR00T/.venv`: GR00T N1.7 độc lập.

Kiểm tra registry và smoke test trước khi train:

```bash
make list
make smoke-flat
make smoke-rough
```

### GR00T G1 đa nhiệm — quy trình chính

Dataset public của NVIDIA gồm bốn subdataset G1 thật. Downloader resume được khi mạng/Hugging
Face rate-limit; audit chỉ cho phép train khi đủ mọi Parquet, MP4, metadata, 43 joint và prompt:

```bash
hf auth login                         # nên làm để tránh rate limit tải ẩn danh
make download-multitask-dataset
make download-locomanip-dataset       # initial pose G1 chính thức cho scene Isaac (khoảng 5 MB)
make audit-multitask-dataset
make validate-multitask-dataset       # đọc sample bằng chính GR00T N1.7 loader
make smoke-multitask-scene            # spawn 4 vật + camera + Pink/AGILE trong Isaac, chưa load model 3B
```

Base generalist chạy được ngay sau khi tài khoản có quyền backbone; fine-tune dưới đây là bước
tùy chọn để tăng độ ổn định trên đúng bốn loại quả. Nó tạo một checkpoint dùng chung:

```bash
# Cần chấp nhận điều khoản nvidia/Cosmos-Reason2-2B trên Hugging Face trước.
NUM_GPUS=1 MAX_STEPS=10000 SAVE_STEPS=1000 \
GLOBAL_BATCH_SIZE=32 USE_WANDB=0 \
make train-multitask
```

Recipe chuẩn của GR00T N1.7 cần khoảng 40 GB VRAM/GPU. RTX 5070 Ti 16 GB trên máy hiện tại phù
hợp inference nhưng không đủ cho full fine-tune chính thức; script dừng trước khi OOM. Có thể chạy
lệnh train không đổi trên A100/H100/L40S 48 GB sau khi mount project/dataset. Không dùng
`ALLOW_LOW_VRAM=1` cho production nếu recipe giảm bộ nhớ chưa được đánh giá.

Chạy toàn pipeline **trong Isaac Sim** bằng base `REAL_G1` đa nhiệm:

```bash
MULTITASK_INSTRUCTION="Hãy lấy quả lê vàng bỏ lên đĩa" make demo-multitask
# bốn target có trong scene hiện tại: táo đỏ, lê vàng, nho xanh, khế vàng
```

Chạy lab liên tục (không đóng Isaac sau mỗi task), rồi nhập lệnh ngay trong terminal đã mở nó:

```bash
make lab-multitask
# Dùng cửa sổ `Unitree Mission Center` ngay trong Isaac để nhập/gửi lệnh hoặc bấm nút fruit.
# Terminal `lab>` vẫn hoạt động như một giao diện dự phòng.
# Lệnh điều khiển: help | status | stop | reset | quit
```

Lệnh mới preempt nhiệm vụ đang chạy; `stop` trả hai tay về pose trung tính, `reset` khôi phục cả
robot lẫn bốn vật mà không restart Isaac/GR00T. Các yêu cầu ngoài bốn kỹ năng fruit-fetch bị từ
chối rõ ràng thay vì cho model tạo chuyển động không có contract an toàn.
Launcher đồng thời ghi toàn bộ trạng thái runner vào `outputs/groot_multitask_runner.log`; nhãn
trong Mission Center phân biệt rõ `IDLE`, `RUNNING`, `SUCCESS`, `TIMEOUT`, `STOPPED` và `REJECTED`.

Runner có warm-up cân bằng, giới hạn workspace/tốc độ cổ tay, hành lang reach buộc tay bám đúng
vật đích, pose trung tính cho tay không hoạt động, phát hiện té, và fetch supervisor
theo pha. Khi đi, hai tay được giữ theo thân; khi gắp trái cây nhỏ, chỉ tay gần vật nhận action từ
GR00T; khi mang vật, tay lại được khóa cho tới bàn đích. Simulator chỉ in `[SUCCESS]` khi điều kiện
vật lý của Isaac xác nhận vật đã tới đích; timeout trả mã lỗi, không được tính là demo thành công.

Checkpoint base nối đúng toàn bộ interface nhưng không bảo đảm zero-shot thành công trong scene
khác miền dữ liệu. Trên máy hiện tại, rollout quả lê đã ổn định nhưng chưa gắp thành công; muốn có
tỉ lệ task đáng tin cậy cần checkpoint bốn-task fine-tune và closed-loop evaluation, không thể xem
việc model/server khởi động là bằng chứng task đã hoàn tất.

Để demo checkpoint bốn-task sau khi train xong:

```bash
GR00T_MULTITASK_MODE=finetuned \
MULTITASK_INSTRUCTION="Pick up the green grapes and place it on the plate" \
make demo-multitask-wbc
```

Nếu máy có `tmux`, script mở server và simulator trong hai cửa sổ tmux. Nếu không có, policy server
tự chạy nền (log tại `outputs/groot_multitask_policy.log`) và Isaac Sim chạy ngay trong terminal hiện
tại. Lệnh được kiểm tra để chỉ rõ đúng một vật; mode base gửi nguyên câu vào generalist. Vì checkpoint fine-tuned 43D không có EEF/navigation, demo của nó dùng
backend WBC reference, còn đánh giá open-loop MAE/MSE chạy trên cả bốn held-out trajectory:

```bash
GR00T_MULTITASK_MODE=finetuned make serve-multitask
# terminal khác:
make evaluate-multitask

# xem closed-loop 43D qua RoboCasa/GEAR WBC thay vì Isaac:
GR00T_MULTITASK_MODE=finetuned make demo-multitask-wbc
```

### Legacy/reference: checkpoint vô-lăng trong Isaac Sim

Track loco-manipulation chạy trọn pipeline trong Isaac Sim 6.0/Isaac Lab 3.0: nhận câu lệnh,
đọc RGB camera thân và state của scene, gọi checkpoint
[`nvidia/g1_locomanip_finetune`](https://huggingface.co/nvidia/g1_locomanip_finetune), rồi điều
khiển đồng thời hai cổ tay, 14 joint bàn tay, vận tốc/độ cao thân. Pink IK xử lý thân trên;
AGILE policy xử lý locomotion chân. Scene physics tự kiểm tra vật đã được nhả và đứng yên trên
bàn giao. Cài riêng runtime N1.5 tương thích task và tải model/dataset chính thức một lần:

```bash
make setup-locomanip
make download-locomanip-model
make download-locomanip-dataset
```

Sau đó chỉ cần:

```bash
export OMNI_KIT_ACCEPT_EULA=1
make demo-commanded-fetch
```

Script tự mở policy server GPU và cửa sổ Isaac Sim, reset từ các initial state được annotate chính
thức, rồi chạy tối đa ba rollout closed-loop nếu diffusion sample đầu chưa hoàn thành. Dấu kết thúc
hợp lệ là `[SUCCESS]`; log đầy đủ ở `outputs/commanded_fetch_demo.log`. Muốn đổi cách diễn đạt:

```bash
INSTRUCTION="Hãy lấy vô lăng rồi mang sang bàn giao" make demo-commanded-fetch
```

Checkpoint phát hành cho đúng skill **vô lăng: bàn lấy → bàn giao** và không có language modality;
câu lệnh hiện là admission/skill selector trung thực, không phải khả năng lấy đồ tùy ý. Muốn thêm
chai, hộp hoặc nhiệm vụ gia dụng khác phải có scene, demonstration tương ứng và fine-tune GR00T.
Đánh giá theo ô target nhỏ nguyên bản của Isaac Lab dùng
`bash scripts/locomanip/demo_commanded_fetch.sh --success-mode official-target`.

## 2. Train RL locomotion — lệnh để em chạy

Đã benchmark trực tiếp trên RTX 5070 Ti 16 GB của máy này bằng rollout PPO thật:

| Cấu hình | Throughput ổn định | VRAM đỉnh |
|---|---:|---:|
| 4096 env / 4 minibatch | 95,188 FPS | 5,410 MiB |
| 6144 env / 6 minibatch | 116,557 FPS | 6,502 MiB |
| **8192 env / 8 minibatch** | **130,817 FPS** | **7,660 MiB** |

`make train-flat` đã mặc định dùng cấu hình nhanh nhất đo được là 8192/8. VRAM còn trống là bình thường: ở quy mô này PhysX/CPU và rollout bắt đầu là nút thắt; tăng VRAM bằng cách làm policy lớn hơn không tự làm PPO hội tụ tốt hơn. Kiểm tra chế độ nguồn trước khi train:

```bash
make performance-check
```

Nếu tool phát cảnh báo, chuyển Ubuntu sang chế độ **Performance** rồi kiểm tra lại. Với `intel_pstate`, chữ `powersave` tự nó không phải lỗi khi EPP và desktop power profile đều đang ở `performance`. Flat trước, rough sau:

```bash
make train-flat
make train-rough
```

Rough mặc định dùng 8.192 environment / 8 minibatch sau khi smoke và benchmark trực tiếp
không OOM trên RTX 5070 Ti. Có thể hạ tạm bằng
`ROUGH_NUM_ENVS=4096 ROUGH_MINI_BATCHES=4 make train-rough` nếu cần dành GPU cho process khác.
Nếu job rough bị ngắt sau khi đã có checkpoint, tiếp tục bằng `make train-rough-resume`.
Xem iteration, metric, checkpoint mới nhất và ETA gọn bằng `make status-rough`.

Ở 8.192 environment, task rough cấp riêng PhysX broad-phase buffers lớn hơn mặc định. Reward
height bỏ các ray-caster no-hit sentinel trước khi lấy mean; các penalty bình phương có giới hạn
số học; rollout guard dừng ngay khi observation/reward có NaN **hoặc** infinity. Actor dùng
bounded log-standard-deviation để một minibatch xấu không thể đầu độc tham số exploration.
Các guard này nằm trong package project, nên vẫn được áp dụng sau khi bootstrap lại dependency.

Checkpoint flat hoàn chỉnh nằm tại
`logs/rsl_rl/unitree_g1_flat_robust/2026-08-08_16-38-59/model_2799.pt`. TensorBoard cuối
iteration 2799 ghi success rate `1.0`, mean episode length `1000`, linear-velocity error
`0.1183` và base-contact termination `0.00256`. Muốn tiếp tục chính checkpoint đó:

```bash
make train-flat-resume
```

Checkpoint rough hoàn chỉnh nằm tại
`logs/rsl_rl/unitree_g1_rough_robust/2026-08-09_03-08-51_production/model_4998.pt`.
Evaluation độc lập 1.024 episode đạt fall rate `0.0`, đủ `1.000` bước/episode, sai số
tracking XY `0.0815 m/s` và yaw `0.1877 rad/s`; báo cáo nằm ở
`outputs/eval-rough-final.json`.

Hoặc điều khiển trực tiếp:

```bash
.deps/IsaacLab/.venv/bin/python scripts/train.py \
  --rl_library rsl_rl \
  --task Unitree-G1-Velocity-Flat-Robust \
  --num_envs 8192 \
  --viz none \
  agent.algorithm.num_mini_batches=8
```

Resume/multi-GPU:

```bash
.deps/IsaacLab/.venv/bin/python scripts/train.py \
  --rl_library rsl_rl --task Unitree-G1-Velocity-Flat-Robust \
  --resume --checkpoint latest --viz none

.deps/IsaacLab/.venv/bin/python scripts/train_multigpu.py \
  --rl_library rsl_rl --task Unitree-G1-Velocity-Rough-Robust --viz none
```

Log và checkpoint nằm tại `logs/rsl_rl/unitree_g1_{flat,rough}_robust/<timestamp>/`. Config PPO ở [`rsl_rl_ppo_cfg.py`](source/unitree_rl_groot/unitree_rl_groot/tasks/locomotion/g1/agents/rsl_rl_ppo_cfg.py), MDP và randomization ở [`env_cfg.py`](source/unitree_rl_groot/unitree_rl_groot/tasks/locomotion/g1/env_cfg.py).

Play tự export JIT + ONNX vào thư mục `exported/` cạnh checkpoint:

```bash
make play        # flat
make play-rough  # rough
```

Target này chạy `--viz kit`, vì vậy sẽ mở cửa sổ Isaac Sim thay vì chạy headless.

Đánh giá tracking/fall rate:

```bash
make evaluate        # flat
make evaluate-rough  # rough
# hoặc thêm: --output outputs/eval-flat.json
```

## 3. Thu demonstration cho GR00T

Sau khi PPO flat đã tốt, collector tạo scene mới cho từng episode: goal xanh ngẫu nhiên, tối đa tám obstacle có collision, A* với footprint G1 được inflate, rồi PPO thực thi command vận tốc của expert waypoint follower. Nó chỉ lưu episode đến goal thành công và ghi RGB 224×224, proprio G1, command `[vx, vy, yaw_rate]`, language, goal và obstacle đồng bộ ở 10 Hz:

```bash
make collect
# tương đương mặc định: 200 episode thành công, 6 obstacle, tối đa 600 lần thử
```

Mỗi episode hoàn chỉnh được ghi atomically theo file riêng dưới `datasets/raw/g1_navigation/`; nếu dừng giữa chừng, những episode đã hoàn tất vẫn giữ nguyên. Trước khi convert, bắt buộc audit độ đa dạng và tính hợp lệ:

```bash
make audit-raw
```

Chuyển sang LeRobot v2.1 và bắt loader thật của GR00T đọc thử:

```bash
make convert
make validate-dataset
```

Bộ navigation đã thu trên máy này có 200/200 episode thành công, 26.427 frame ở 10 Hz,
6 obstacle mỗi scene và bốn instruction paraphrase. Audit strict không có quality failure;
báo cáo ở `outputs/navigation_raw_audit.json`. Dataset đã convert nằm tại
`datasets/lerobot/g1_navigation/` và đã được loader thật của GR00T đọc thành công.

Muốn xem ngay expert planner điều khiển PPO tới goal và tránh obstacle trong Isaac Sim
(đây là oracle demo, chưa phải output của GR00T):

```bash
make demo-navigation-expert
```

Converter kiểm tra dtype/shape/timestamp, ghi Parquet + MP4, `info.json`, `modality.json`, `stats.json`, `episodes.jsonl`, `tasks.jsonl`, và từ chối ghi đè một output không rỗng.

## 4. Fine-tune và chạy GR00T

Recipe chỉ tune projector và diffusion action head; LLM và vision backbone được freeze đúng theo recipe N1.7 mặc định. Lệnh được để sẵn nhưng không tự chạy:

```bash
NUM_GPUS=1 MAX_STEPS=10000 SAVE_STEPS=1000 USE_WANDB=0 make finetune-groot
```

Base checkpoint 6,5 GB đã được khóa revision và lưu tại
`checkpoints/nvidia-gr00t-n1.7-3b/`. Trước khi cấp phát model, recipe kiểm tra quyền truy cập
backbone gated `nvidia/Cosmos-Reason2-2B`. Chủ tài khoản phải chấp nhận điều khoản model và
đăng nhập máy một lần bằng `hf auth login`; thao tác chấp nhận license không thể được tự động hóa.

Nếu dùng nhiều GPU, đặt `NUM_GPUS`; chỉnh batch bằng `GLOBAL_BATCH_SIZE`. Checkpoint mặc định vào `checkpoints/groot-g1-navigation/`. Fine-tune N1.7-3B chính thức thường cần GPU khoảng 40 GB trở lên; RTX 5070 Ti 16 GB phù hợp chạy inference, còn fine-tune có thể OOM dù đã freeze backbone. Khi đó dùng GPU cloud/A100 hoặc recipe giảm bộ nhớ đã được kiểm chứng, không trộn GR00T vào virtualenv Isaac Lab.

Terminal 1 — inference server trong GR00T environment:

```bash
GR00T_CHECKPOINT=checkpoints/groot-g1-navigation/checkpoint-10000 make serve-groot
```

Terminal 2 — Isaac Lab client và PPO:

```bash
make hierarchical INSTRUCTION="Turn left while walking forward."
```

Client chỉ nhận numeric ndarray, cấm object/pickle payload, có timeout/reconnect, giới hạn vận tốc, deadband và slew rate. Hai tầng có thể đặt trên hai GPU hoặc hai máy bằng `--server-host`.

Đánh giá navigation trên layout ngẫu nhiên chưa thấy lúc train và kiểm tra GR00T có thực sự dùng camera thay vì chỉ học prior command:

```bash
make evaluate-navigation
make vision-ablation
```

`vision-ablation` so sánh action khi dùng ảnh thật, ảnh đen và frame bị xáo trộn; strict mode sẽ fail nếu action gần như không đổi.

## 5. Whole-body ApplePnP chính thức

NVIDIA đã phát hành cả dataset `GR00T-N1.7-AppleToPlate` và checkpoint deployment
`GR00T-N1.7-ApplePnP-V1`. Project khóa đúng revision, không cần train lại để thử đúng task này.
Bundle ONNX chứa video/state preprocessing, Cosmos backbone, action head và decoder:

```text
ego RGB 480×640 + G1 state 43D
                 │
                 ▼
       ApplePnP LeApp/ONNX graph
                 │ chunk 16 bước @ 30 Hz
                 ├─ arm 14D + hand 14D + waist 3D
                 └─ navigation 3D + base height 1D
                                      │
                                      ▼
                     GEAR WBC Balance/Walk @ 50 Hz
                                      │
                                      ▼
                       RoboCasa / Unitree G1 43 DoF
```

### Cài và xác minh artifact

```bash
make setup-groot
make setup-sonic
make setup-sonic-sim
make setup-applepnp-wbc
make download-fetch-dataset
make download-applepnp-model
make audit-fetch
make validate-fetch-dataset
make smoke-applepnp
```

Dataset chính thức có 402 episode, 171.625 frame, MP4 AV1 480×640 ở 30 Hz. Checkpoint
deployment có 17 file, khoảng 12 GB. `smoke-applepnp` lấy khung hình và proprioception thật
từ dataset rồi chạy toàn bộ graph trên CUDA; output phải đủ bảy action group, chunk 16 và
không có NaN/Inf.

### Chạy closed-loop GR00T → GEAR WBC

Một lệnh mở hai cửa sổ tmux: policy server và RoboCasa simulation.

```bash
make demo-applepnp
# headless regression:
APPLEPNP_HEADLESS=1 APPLEPNP_STEPS=600 make demo-applepnp
```

Task simulation là `LMPnPAppleToPlateDC_G1_gear_wbc`. Client đổi đúng joint ordering của
hai bàn tay, nội suy chunk 30 Hz sang control loop 50 Hz, bật gravity compensation cho tay,
và chỉ kết luận thành công khi success criterion vật lý apple-contact-plate của RoboCasa bật.

Checkpoint này được NVIDIA train/evaluate trên **robot G1 thật**, không phải ảnh render
RoboCasa. Trên máy này toàn bộ closed loop đã chạy ổn định (chunk đầu khoảng 8,2 giây để
khởi tạo graph, chunk sau khoảng 0,06 giây), nhưng episode synthetic seed 0 chưa đặt được
apple lên plate. Đây là domain gap real-camera → synthetic-camera, không được ghi nhận giả là
task success. Dùng model trực tiếp phù hợp để kiểm tra integration và chuẩn bị real-robot
deployment; muốn success trong simulation cần thêm demonstration synthetic rồi post-train.

Một chi tiết quan trọng của chính dataset: `action.navigate_command` bằng 0 trong toàn bộ
402 episode. Vì vậy checkpoint ApplePnP này là manipulation tại chỗ, chưa phải planner tổng
quát kiểu “đi qua phòng lấy đồ”. Locomotion/navigation tổng quát vẫn là pipeline PPO + GR00T
navigation ở phần 3–4 hoặc phải thu thêm dữ liệu whole-body có chuyển động chân.

## 6. Train task whole-body tùy biến

Đường tùy biến vẫn dùng action contract `UNITREE_G1_SONIC` 78D và PICO/Isaac Teleop. Setup:

```bash
make setup-sonic-teleop
make setup-sonic-data
make setup-sonic-inference
make setup-sonic-deploy
make download-sonic-model
make fetch-collection-preflight
make smoke-fetch-stack
```

Tạo curriculum/scene rồi thu demonstration thành công thật; smoke đứng yên không được đưa
vào dataset train:

```bash
make validate-fetch-scenes
make generate-fetch-curriculum

FETCH_MODE=sim \
FETCH_OBJECT="red soda can" FETCH_OBJECT_PROFILE=red_soda_can \
FETCH_SOURCE="table" FETCH_SOURCE_PROFILE=table \
FETCH_DESTINATION="delivery bin" FETCH_DESTINATION_PROFILE=delivery_bin \
FETCH_DATASET_NAME=g1_fetch_red_can_bin \
make collect-fetch
```

Trong tmux, `c` bắt đầu/kết thúc và lưu, `x` discard, `r` reset/randomize. Sau collection:

```bash
RAW_SONIC_FETCH_DATASET=.deps/GR00T-WholeBodyControl/outputs/g1_fetch_red_can_bin \
make process-fetch
make audit-fetch
make validate-fetch-dataset
```

Fine-tune GR00T N1.7 cần truy cập gated backbone `nvidia/Cosmos-Reason2-2B`. Chấp nhận điều
khoản trên Hugging Face rồi đăng nhập máy một lần bằng `hf auth login`. Recipe chuẩn của
NVIDIA yêu cầu tối thiểu khoảng 40 GB VRAM trên **mỗi GPU**; RTX 5070 Ti 16 GB không đủ cho
full fine-tune chuẩn. Script kiểm tra cả quyền truy cập và VRAM trước khi tạo job:

```bash
NUM_GPUS=1 \
MAX_STEPS=20000 SAVE_STEPS=5000 \
GLOBAL_BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS=1 \
USE_WANDB=1 \
make finetune-fetch
```

Checkpoint tùy biến nằm trong `checkpoints/groot-g1-sonic-fetch/`. Khi có checkpoint hợp lệ,
chạy `make serve-fetch` và `make demo-fetch-sim`. Không dùng checkpoint ApplePnP ONNX làm
đầu vào SONIC: hai action contract này khác nhau.

## Tasks

| Gym ID | Mục đích |
|---|---|
| `Unitree-G1-Velocity-Flat-Robust` | PPO locomotion phẳng, domain randomization |
| `Unitree-G1-Velocity-Rough-Robust` | PPO locomotion địa hình procedural + privileged critic |
| `Unitree-G1-GR00T-Navigation` | Một G1, camera đầu, deterministic closed-loop GR00T |

Actor chỉ thấy noisy deployable observations. Critic thấy privileged base velocity, effort, contact force và height scan. Randomization gồm friction/restitution, torso mass, actuator gains, push disturbances; rough task giữ curriculum địa hình của Isaac Lab.

## Kiểm thử và cấu trúc

```bash
make test
make lint       # sau khi setup Isaac Lab
bash -n scripts/setup/*.sh scripts/groot/*.sh
```

```text
configs/                 version lock và GR00T modality
source/unitree_rl_groot/ external Isaac Lab extension/package
scripts/                 train, play, evaluate, doctor
scripts/groot/           collect, convert, fine-tune, serve, closed loop
scripts/sonic/           fetch curriculum, audit, fine-tune, serve, simulation
scripts/applepnp/        official ONNX server, real-data smoke, GEAR-WBC demo
scripts/multitask/       four-task G1 audit, train, server, scene, WBC closed loop
assets/multitask/        project-local pear/grapes/starfruit collision + visual MJCF
tools/lerobot/           converter LeRobot v2.1
tests/                    unit/schema/reproducibility tests CPU
outputs/                  benchmark và báo cáo evaluation dạng JSON
```

Các environment Isaac Lab, GR00T và GEAR-WBC được giữ riêng; không trộn Torch/CUDA dependency giữa chúng.

## Lưu ý khi chuyển sang robot thật

Không đưa checkpoint simulator thẳng lên G1 ở nơi có người. Trước real deployment cần kiểm tra joint ordering, PD gains, torque/velocity limit, state estimator, watchdog/E-stop, command timeout, latency và test treo robot/low-speed. Runtime hiện là simulation integration; nó không tự nhận quyền điều khiển hardware Unitree.
