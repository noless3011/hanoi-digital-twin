# Projection instead of the detector on the simulation side (17 Nov 2026)

For method 3.6 (image-space distance), 5.6 (computational cost) and the limitations in Chapter 6.

## Decision

Every score compares the tracked video with the simulation **projected** into the camera image: each
simulated vehicle's 3D box (length, width, height by kind) through the camera matrix Unity renders with
(`twin/measure/image.py`, `from_result`). The simulation is not rendered and passed through RF-DETR +
TrackTrack. The code that did that (`twin/calibrate/evaluate.py` with Unity batch rendering,
`feed/emulate_camera.py`, `TrajectoryVideoRenderer -jobs`) was removed; it is in git before 17 Nov 2026.

## Why: cost

Measured on the project's machine (one GTX 1650; all CPU cores for simulation):

| Step, one 3-minute clip | Detector path | Projection |
| --- | --- | --- |
| Render in Unity | about 1 min | not needed |
| RF-DETR + TrackTrack | about 5 min | not needed |
| Score | 0.1 s | 0.1 s |
| **Per simulated clip** | **about 6 min** | **0.1 s** (plus 2–3 s of simulation, either way) |

| Task | Simulations | Detector path | Projection |
| --- | --- | --- | --- |
| Final evaluation: 5 configurations × 12 clips × 3 seeds | 180 | about 17 h (5 h for one seed) | 3 min (measured) |
| One fit: choice model, 25 generations × 12 candidates × 6 clips | 1,800 | about 7 days | 18 min (measured) |
| One fit: priority model, 80 generations × 20 × 6 clips | 9,600 | about 40 days | 44 min (measured) |
| Morris analysis, choice model | about 2,500 | about 10 days | 77 min (measured) |

A search through the detector is out of reach, and even the final evaluation could not be run as one
uninterrupted 17-hour job on the available machine. All reported numbers therefore use projection.

## What it costs: the detector is on one side only

The video's boxes carry the detector's errors: it misses small and far vehicles, merges vehicles that
overlap into one box, jitters boxes and breaks tracks (1.7 tracks per vehicle on average, table
`dataset`). Projected boxes have none of these. Two checks were made before the decision
(`docs/twin.md` sections 4 and 5):

- **Size of the gap.** The constant-speed simulation of segment 0010, rendered and tracked, scored 0.20
  against its own projection (0.26 with the real camera's irregular frame timing); the detector found
  74–85% of the projected vehicles in every distance band. This is as large as the differences between
  models (0.05–0.09), so every absolute score carries a floor of this size.
- **Ranking, one clip.** Segment 0010 through the detector: fitted choice model 0.31, constant speed
  0.28. By projection: 0.26 and 0.33. The order is reversed, mostly through *spacing*: the detector
  merges overlapping vehicles, so the constant-speed run's overlaps, which projection penalises (0.32),
  are invisible through the detector (0.10). One clip with one seed is inside the noise floor (0.07–0.10
  per clip), so this neither confirms nor refutes the ranking.

So: comparisons between configurations are fair to each other (all scored the same way), absolute scores
are not "how far the behaviour is off", and whether the ranking holds through a real camera is open.

## Results by projection (`runs/twin/results/report.json`)

Held-out day (8 Nov), means over 6 clips × 3 seeds; full table in `03-experiments.md` 5.4.

| Configuration | Score | Contacts per vehicle | Seed spread |
| --- | --- | --- | --- |
| Constant speed | 0.315 | 1.14 | 0 |
| Choice, defaults | 0.262 | 0.12 | 0.017 |
| Choice, fitted | 0.260 | 0.12 | 0.019 |
| Priority, start | 0.303 | 0.16 | 0.006 |
| Priority, fitted | 0.271 | 0.08 | 0.004 |

## Text for the thesis (Vietnamese, adapt freely)

> Phía mô phỏng, khung bao của mỗi phương tiện được tính trực tiếp bằng cách chiếu hộp 3D của xe qua
> ma trận camera ảo, thay vì kết xuất video mô phỏng rồi chạy lại RF-DETR và TrackTrack như phía video
> thực. Lý do là chi phí tính toán: kết xuất và nhận dạng một đoạn 3 phút mất khoảng 6 phút trên GPU
> GTX 1650, nên riêng bước đánh giá cuối (180 lượt mô phỏng) đã cần khoảng 17 giờ, còn một lần hiệu
> chỉnh cần hàng nghìn lượt mô phỏng. Phép chiếu chỉ mất khoảng 0,1 giây mỗi đoạn. Đổi lại, sai số của
> bộ nhận dạng (bỏ sót xe, gộp các xe chồng lấn, rung khung bao, đứt quỹ đạo) chỉ có ở phía video. Một
> lần kiểm tra cho thấy chênh lệch do bộ nhận dạng gây ra vào khoảng 0,20–0,26, nên giá trị tuyệt đối
> của khoảng cách không phản ánh trực tiếp mức sai lệch về hành vi; các cấu hình được so sánh với nhau
> trên cùng một thước đo.

For the limitations (Chapter 6):

> Mô phỏng được so sánh với video qua phép chiếu khung bao chứ không qua cùng bộ nhận dạng, do giới hạn
> tài nguyên tính toán. Việc đánh giá lại các cấu hình cuối qua RF-DETR là hướng phát triển tiếp theo.
