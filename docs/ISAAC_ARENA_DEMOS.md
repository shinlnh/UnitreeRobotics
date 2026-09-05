# Demo GR00T thật trong Isaac Sim

Hai launcher trong thư mục `scripts/` chạy Isaac Sim 6.0.1 ở chế độ headless, gọi
GR00T qua remote policy server và ghi video từ camera đầu robot. Supervisor, model
server và Isaac runner chạy tách khỏi terminal/VS Code, nên việc đóng cửa sổ editor
không làm mất rollout.

## Kết quả đã kiểm chứng trên máy này

Ngày chạy: 2026-08-24. GPU: NVIDIA RTX 5070 Ti 16 GB.

| Tác vụ | Policy | Episode hoàn tất | Success | Object moved | Kết luận |
|---|---|---:|---:|---:|---|
| Apple → đĩa | GR00T N1.7 static/WBC Agile | 3 | 2/3 (66.7%) | 3/3 | Pass |
| Hộp nâu → thùng xanh | GR00T N1.6 loco-manipulation/WBC | 1 | 1/1 (100%) | 1/1 | Pass |
| Banana → đĩa, zero-shot OOD | checkpoint apple N1.7 | 2 | 0/2 (0%) | 2/2 | Không generalize được |

Tác vụ banana cố ý đổi cả object và câu lệnh nhưng vẫn dùng checkpoint đã
post-train cho apple. Robot tương tác và làm vật thể di chuyển, nhưng không hoàn tất
predicate đặt vật lên đĩa. Kết quả này minh họa đúng nghĩa của “generalist”:
GR00T có thể chia sẻ representation và thích nghi nhiều task, không đảm bảo thực thi
một task/embodiment bất kỳ chỉ từ một câu lệnh nếu checkpoint chưa được post-train
với phân phối dữ liệu phù hợp.

## Chạy lại

Apple → đĩa:

```bash
scripts/run_arena_static_apple.sh start
scripts/run_arena_static_apple.sh status
```

Mở trực tiếp cửa sổ Isaac Sim/Omniverse Kit:

```bash
ARENA_NUM_STEPS=100000 scripts/run_arena_static_apple.sh gui
```

Để xem đúng một lần gắp trong thời gian thực, rồi giữ nguyên tư thế cuối trong cửa sổ Isaac Sim:

```bash
scripts/run_arena_static_apple_demo_once.sh
```

Lệnh này chạy đúng một instance trong cgroup riêng (RAM mềm 22 GB, tối đa 24 GB)
để bảo vệ VS Code/Codex. Dừng toàn bộ demo bằng
`systemctl --user stop arena-static-apple-demo.service`.

Để quan sát hành vi tự nhiên của GR00T trên một timeline vật lý liên tục, launcher
chạy đúng một episode. Khi episode kết thúc vì thành công, thất bại hoặc timeout,
robot, vật thể, vận tốc, trạng thái policy và action chunk đều được giữ nguyên;
không có teleport/reset hay đổi vị trí apple. Camera tiếp tục phát frame từ cùng
simulation runtime vô hạn cho tới khi người dùng đóng cửa sổ hoặc dừng service:

```bash
scripts/run_arena_static_apple_observe_after_success.sh
```

Dừng experiment này bằng
`systemctl --user stop arena-static-apple-continuous.service`.

Đóng cửa sổ Isaac Sim hoặc nhấn `Ctrl+C` trong terminal để dừng model server và giải phóng GPU.

Các USD, texture và dependency tải từ Omniverse được giữ lâu dài tại
`.cache/isaaclab-assets/` thay vì `/tmp`. Lần chạy đầu vẫn cần mạng; các lần sau dùng
cache local. Có thể đổi vị trí bằng `ARENA_ASSET_CACHE_DIR=/duong/dan/khac`.

Có thể tải trước toàn bộ asset mà không mở GUI:

```bash
scripts/prefetch_arena_static_apple_assets.sh
```

Nếu chỉ cần tải nhanh bốn USD gốc để mở demo trước, còn texture phụ tải sau:

```bash
scripts/prefetch_arena_static_apple_assets.sh --root-only
```

Nhấn `Ctrl+C` trong terminal để đóng cả Isaac Sim và GR00T server. GUI tắt ghi MP4
để tiết kiệm VRAM. Nếu chỉ muốn chạy đúng rollout đã kiểm chứng rồi tự đóng, bỏ
`ARENA_NUM_STEPS=100000`; mặc
định là 600 bước.

GUI dùng timeout GR00T 120 giây vì lần inference đầu có thể lâu hơn mức mặc định
15 giây khi Isaac Sim và policy dùng chung một GPU. Có thể đổi bằng biến
`ARENA_REMOTE_TIMEOUT_MS` nếu cần.

N1.7 được stream thẳng lên GPU bằng `device_map`, `bfloat16` và chế độ low-memory.
Không nạp toàn bộ checkpoint ở CPU/`float32` rồi mới cast, vì cách đó tạo đỉnh RAM
đủ để Ubuntu `systemd-oomd` đóng VS Code trên máy 32 GB. Khi checkpoint chưa nằm
trong filesystem cache, SSD ngoài có thể làm bước khởi động kéo dài khoảng 1–2 phút;
đợi đến khi terminal báo `Opening Isaac Sim GUI`.

Hộp nâu → thùng xanh:

```bash
scripts/run_arena_loco_box.sh start
scripts/run_arena_loco_box.sh status
```

Thử object/instruction khác với static checkpoint:

```bash
ARENA_RUN_NAME=static_banana_ood \
ARENA_OBJECT=banana_ycb_robolab \
ARENA_LANGUAGE_INSTRUCTION="move the banana to the plate" \
scripts/run_arena_static_apple.sh start

ARENA_RUN_NAME=static_banana_ood \
scripts/run_arena_static_apple.sh status
```

Mỗi run ghi `gr00t_server.log`, `isaac_headless.log`, JSONL episode metrics, HTML
report và MP4 vào `artifacts/isaac_arena/<run-name>/`.

## Các artifact của ba lần chạy

- Apple: `artifacts/isaac_arena/static_apple/outputs/2026-08-24_23-22-37/`
- Hộp: `artifacts/isaac_arena/loco_box/outputs/2026-08-24_23-36-15/`
- Banana OOD: `artifacts/isaac_arena/static_banana_ood/outputs/2026-08-24_23-42-07/`

## Vì sao dùng headless

SSD ngoài đã đọc tuần tự khoảng 770 MB/s và kernel không ghi nhận I/O error,
USB reset hay lỗi ext4. Các lần Isaac/VS Code thoát lại trùng với cảnh báo NVIDIA
khi Isaac tạo Vulkan context. Chạy Isaac GUI, inference model và VS Code GPU trên
cùng một RTX 5070 Ti làm tăng áp lực VRAM/context; vì vậy launcher dùng `--viz none`
nhưng vẫn render camera và video. Đây là cấu hình ổn định đã dùng để lấy các kết quả
trên.

Nếu cần cửa sổ GUI trực tiếp, nên giải quyết driver/Vulkan trước hoặc chuyển GR00T
server sang GPU/máy khác. Không nên xem việc giảm tốc hay thay SSD là hướng sửa
chính khi kernel chưa có bằng chứng lỗi lưu trữ.
