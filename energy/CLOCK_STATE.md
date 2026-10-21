# Clock / governor state — mục XIV

**Ghi nhận:** 2026-08-16 ~20:04, tức **sau** chiến dịch đo chính (14:08–17:40).
Máy **không reboot** giữa chừng (uptime liên tục, chiến dịch chạy dưới `nohup` xuyên suốt),
nên trạng thái governor ghi sau vẫn phản ánh đúng lúc đo. **Nói rõ đây là ghi sau.**

## `jetson_clocks --show` — KHÔNG chạy được không có sudo

```
$ jetson_clocks --show
SOC family:tegra234  Machine:NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
Error: Run this script(/usr/bin/jetson_clocks) as a root user

$ sudo -n jetson_clocks --show
sudo: a password is required
```

Phiên làm việc dùng `BatchMode=yes` (không tương tác) nên không cấp được mật khẩu sudo.

## Trạng thái suy ra từ sysfs đọc được — kết luận chắc chắn, không cần sudo

| Node | Giá trị |
|---|---|
| `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` | **schedutil** |
| `/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq` | 1.267.200 kHz |
| `/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq` | 1.497.600 kHz |
| GPU devfreq `governor` | **nvhost_podgov** |
| GPU devfreq `min_freq` | 306.000.000 Hz |
| GPU devfreq `max_freq` | 612.000.000 Hz |
| GPU devfreq `cur_freq` | 612.000.000 Hz (đọc lúc GPU đang tải a=0,5) |

## Kết luận

**`jetson_clocks` KHÔNG được áp dụng.** Khi `jetson_clocks` bật, nó ghim mọi xung nhịp ở
mức tối đa và chuyển governor sang chế độ tĩnh (performance/userspace). Ở đây:

- CPU governor là **schedutil** — bộ điều khiển động, và `cur_freq < max_freq` xác nhận CPU
  không bị ghim;
- GPU governor là **nvhost_podgov** — bộ điều khiển động của Tegra, với `min_freq` (306 MHz)
  thấp hơn `max_freq` (612 MHz), tức dải tần vẫn mở.

Vậy trong toàn bộ chiến dịch đo, xung nhịp CPU/GPU do **governor động** điều khiển, ở chế độ
nguồn **15 W (`nvpmodel` mode 0)**.

## Hệ quả cho phép đo

Đây là cấu hình **mặc định như xuất xưởng**, tức đúng cấu hình một thiết bị triển khai thật
sẽ chạy — hợp lý cho một neo năng lượng mang tính triển khai. Nhưng nó cũng có nghĩa là
xung nhịp **tự do biến thiên theo tải và theo nhiệt**, nên nó là một cơ chế nữa (bên cạnh
quạt) mà qua đó activation ảnh hưởng tới công suất bo mạch. Điều này củng cố cách đọc ở
`ENERGY_REPORT.md` §6.2: nhiệt/xung nhịp/quạt đều nằm trên **đường nhân quả từ `a`**, nên
đại lượng cần báo cáo là **tổng hiệu ứng (M1)**.

Nếu muốn bản ghi chính chủ từ `jetson_clocks --show`, chạy trong phiên tương tác:

```
! ssh dat@100.72.207.8 'sudo jetson_clocks --show'
```
