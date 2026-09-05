# Safety

Whole-body humanoid control có thể gây ngã, va chạm, kẹp tay và hỏng cơ cấu. Không coi software interlock là thay thế cho hardware E-stop hoặc safety operator.

## Bắt buộc trước robot thật

- closed-loop MuJoCo pass với đúng checkpoint, controller variant và prompt;
- open-loop actions không có NaN/Inf, không vượt bound và bám dataset;
- dọn vùng an toàn tối thiểu 3 m quanh G1;
- một operator chỉ phụ trách hardware E-stop và phím `O`;
- network/camera/state ổn định, không packet loss bất thường;
- robot gắn gantry/tether ở các buổi đầu;
- bắt đầu bằng object mềm, bàn thấp-risk và tốc độ giới hạn;
- initial pose lấy từ demonstration tương thích checkpoint;
- có checklist shutdown và người chịu trách nhiệm rõ ràng.

## Emergency controls

- `O`/`o` trong terminal C++: immediate stop và exit;
- hardware E-stop: lớp cuối cùng khi workstation/input stack không phản hồi;
- PICO A+B+X+Y: stop stream phía teleop khi dùng PICO;
- `Ctrl+\\`: dừng toàn bộ tmux session, nhưng không thay cho E-stop.

Mỗi operator phải tập stop trong simulation và xác nhận đúng terminal focus trước trial.

## Project interlocks

Real `collect`/`deploy` chỉ execute khi có literal acknowledgement:

```text
--acknowledge-real-robot-risk I_UNDERSTAND_G1_CAN_MOVE
```

Flag chỉ chứng minh người chạy đã đi qua checklist; nó không tự kiểm tra gantry, con người hay vật cản. `doctor --profile real --online` kiểm tra software/network prerequisites nhưng cũng không certify an toàn vật lý.

## Abort conditions

Stop ngay nếu:

- camera/state timestamp stale hoặc policy timeout;
- robot rung, foot slip, joint tracking error hay unexpected contact;
- prompt/action không khớp task;
- latent/action magnitude warning;
- operator mất line of sight;
- người/vật vào safety zone;
- sim và real instances chạy đồng thời trên cùng control interface.

Sau abort, không resume mù. Lưu log, xác định root cause, tái hiện trong simulation và quay lại từ đầu checklist.
