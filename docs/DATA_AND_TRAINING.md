# Data and training

## Dataset design cho generalist policy

Một base VLA không tự biến thành policy general cho G1 chỉ nhờ đổi prompt. `UNITREE_G1_SONIC` là post-training embodiment; cần demonstrations khớp robot, gripper, camera, controller variant và môi trường.

Để một checkpoint xử lý nhiều object/task bằng language:

- trộn nhiều task family trong cùng training run;
- dùng prompt cụ thể và nhất quán cho từng episode;
- randomize object identity, pose, table height, lighting và distractors;
- giữ success demonstrations sạch, ưu tiên first-attempt behavior;
- có validation split theo object/placement chưa thấy;
- giữ tỷ lệ task cân bằng hoặc dùng dataset mixture weights;
- ghi đúng SONIC controller variant đã dùng khi collect.

Khởi đầu thực dụng là 50–100 demonstration sạch cho mỗi task hẹp. Mixed-object/generalist behavior thường cần lớn hơn đáng kể; số lượng không bù được trajectory lỗi hoặc prompt mơ hồ.

## Collect

GEAR-SONIC exporter tạo LeRobot v2.1:

```text
dataset/
├── data/*.parquet
├── videos/observation.images.ego_view/*.mp4
└── meta/
    ├── info.json
    ├── modality.json
    ├── episodes.jsonl
    └── tasks.jsonl
```

Trong recording, đánh dấu success/failure/discard đúng. Không đưa episode có stale SMPL frame, pause dài, failed grasp correction hoặc wrong prompt vào train split.

## Clean và validate

```bash
gr00t-g1 process-dataset --dataset RAW --output CLEAN --execute
gr00t-g1 dataset-check CLEAN
```

Validator của project fail fast trước khi chạy GPU. Sau đó vẫn nên dùng loader upstream để đọc thử nhiều episode và kiểm tra video/action alignment.

## Fine-tune

Preview command:

```bash
gr00t-g1 train --dataset CLEAN --output checkpoints/experiment
```

Config mặc định:

- base: `nvidia/GR00T-N1.7-3B`;
- embodiment: `UNITREE_G1_SONIC`;
- 4 GPU;
- global batch size 32;
- 20k steps;
- save mỗi 5k steps;
- color jitter để tăng visual robustness.

Các giá trị là starting point, không phải guarantee. Theo dõi train/validation loss, nhưng chọn checkpoint bằng open-loop plots và closed-loop success rate chứ không chỉ loss thấp nhất.

## Evaluation ladder

Không bỏ qua cấp nào:

1. Schema: `dataset-check`.
2. Loader: sample episode load không lỗi.
3. Open-loop: predicted trajectory vs ground truth, action range và NaN/Inf.
4. MuJoCo closed-loop: randomized seeds, success rate, fall/contact events.
5. Hardware tether/gantry: low-risk workspace, một object mềm.
6. Full trial matrix: object × pose × table height × lighting.

Mọi report phải ghi checkpoint revision, dataset revision, controller variant, prompt, seed và failure taxonomy.

## Controller/checkpoint compatibility

Project mặc định dùng SONIC v1.1:

```text
policy/sonic_v1_1/model_encoder.onnx
policy/sonic_v1_1/model_decoder.onnx
policy/sonic_v1_1/observation_config.yaml
```

Không mix encoder, decoder hoặc observation config giữa `release`, `low_latency` và `sonic_v1_1`. Checkpoint VLA phải được train từ demonstrations dùng cùng latent/controller convention.
