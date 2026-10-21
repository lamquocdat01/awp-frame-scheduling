# ENERGY_REPORT — AwP activation-energy anchor, Jetson Orin Nano

**Đo:** 2026-08-16, 14:08–17:40 (giờ máy đo) · **Máy:** `dat@100.72.207.8` (`dat-desktop`), Tailscale
**Đường:** A (bộ đo mới, riêng cho AwP; **không** dùng lại số của Deep3)
**Mục đích đo (khác Deep3):** đo **công suất bo mạch biên trên một đơn vị activation** (κ), để một ngân sách năng lượng triển khai quy đổi thành `a` khả dụng, rồi thành `K=⌈1/a⌉` và `β=min(1,(W+1)/K)`. Deep3 C1 đo `(E_idle, E_active)` để chứng nhận độ trễ worst-case của lịch oblivious — khác đại lượng, khác mục đích.

> **Mọi con số dưới đây truy được về `run.log` hoặc `raw/*.csv` trong chính thư mục này. Không con số nào gõ tay.**

---

## 1. Thiết lập (xác minh trên máy sau khi đo)

| Mục | Giá trị |
|---|---|
| Bo mạch | Jetson Orin Nano 8GB |
| L4T | R36 REVISION 5.0, build 2026-01-16 |
| Kernel | 5.15.185-tegra, aarch64 |
| Chế độ nguồn | `nvpmodel -q` → **NV Power Mode: 15W** (mode 0). Không thay đổi. |
| `jetson_clocks` | **KHÔNG đọc được** — `jetson_clocks --show` cần sudo, phiên dùng BatchMode không có. Trạng thái giữ nguyên như tìm thấy, không đụng vào. ⚠️ Đây là một mục metadata prompt yêu cầu mà tôi **không lấy được**. |
| Cảm biến công suất | INA3221 on-board, `hwmon1`, rail **VDD_IN** (tổng bo) = `in1_input` mV × `curr1_input` mA, lấy mẫu 10 Hz |
| Detector | YOLO26s ONNX, `/home/dat/topic30/models/yolo26s.onnx` (38.290.649 bytes) |
| Runtime / precision | **onnxruntime-gpu 1.24.0, CUDAExecutionProvider** (không TensorRT). Providers khả dụng: TRT/CUDA/CPU |
| Độ phân giải đầu vào | **640 × 640** |
| OpenCV | 4.13.0 |
| Nguồn frame | CDnet2014 **highway**, 500 frame, lặp vòng, pace **30 fps** |
| Ngữ nghĩa frame bị gate | decode + preprocess vẫn chạy, **không** infer — đúng như một gate triển khai |
| Stack AAP | **để nguyên đang chạy.** Đo tải thực trước khi đo: CPU tổng 0,28%, GPU load 0, loadavg 0,02; công suất bo 4,43–4,51 W ≈ nền static AAP-free của C1 (4,446 W). Không dừng gì của anh. |
| Nhiệt | ghi `temp_start`/`temp_end` mỗi run; cổng nguội 60 °C, tối đa 240 s |

## 2. Ma trận đo

`a ∈ {0; 0,05; 0,1; 0,2; 0,333; 0,5}` (a = 1/k, a=0 là idle-skip), **3 lần lặp mỗi mức**, tổng **18 run**.
Warmup 45 s mỗi run. Cửa sổ đo: **300 s** cho a=0 (yêu cầu idle ≥ 5 phút), **150 s** cho các mức sweep.
Tổng 18 raw power log × ~1500–3000 mẫu.

## 3. Dữ liệu thô — tổng hợp theo mức

Từ `E_of_a.csv` (sinh bởi `analyze_awp_energy.py` từ `run.log`):

| a | k | runs | W_mean (W) | W_sd | fps | infer_fps | E (mJ/frame) | T_start (°C) | T_end (°C) | drift lặp đầu→cuối |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,0000 | 1 | 3 | 4,9272 | 0,0114 | 30,00 | 0,000 | 164,24 | 60,21 | 60,47 | −0,13 % |
| 0,0500 | 20 | 3 | 5,2813 | 0,0071 | 29,99 | 1,506 | 176,04 | 60,47 | 61,03 | +0,20 % |
| 0,1000 | 10 | 3 | 5,6487 | 0,0057 | 29,99 | 3,005 | 188,29 | 60,70 | 61,49 | +0,09 % |
| 0,2000 | 5 | 3 | 6,4001 | 0,0056 | 29,99 | 6,004 | 213,34 | 61,26 | 62,56 | +0,09 % |
| 0,3333 | 3 | 3 | 7,2951 | 0,0026 | 29,99 | 10,002 | 243,17 | 61,88 | 63,72 | +0,08 % |
| 0,5000 | 2 | 3 | 8,3657 | 0,0060 | 30,00 | 15,000 | 278,86 | 62,63 | 65,04 | −0,17 % |

Lặp lại rất tốt: độ lệch chuẩn giữa 3 run ≤ 0,0114 W ở mọi mức. Pace giữ đúng 30 fps ở mọi mức (average per-frame work < 33 ms) nên năng lượng mỗi frame tỉ lệ thẳng với công suất bo.

**Luật dừng "trôi nhiệt > 5%": ĐẠT** — trôi giữa lần lặp đầu và cuối ≤ 0,20 % ở mọi mức.

## 4. Hồi quy — HAI mô hình

**M1 (mô hình của bài):** `W = W₀ + κ·a`

| | giá trị |
|---|---|
| W₀ | **4,9571 ± 0,0144 W** |
| κ | **6,9089 ± 0,0549 W / đơn vị a** |
| R² | **0,99899** |

**M2 (nhiệt là hiệp biến nhiễu):** `W = W₀ + κ·a + γ·T_start`

| | giá trị |
|---|---|
| W₀ | **−6,1151 ± 7,0901 W** |
| κ | **6,0129 ± 0,5761 W / đơn vị a** |
| γ | **+0,1838 ± 0,1177 W / °C** |
| R² | 0,99913 |

**corr(T_start, a) = +0,9958** · **κ lệch M1↔M2 = −12,97 %**

## 5. ⛔ LUẬT DỪNG MỤC II ĐÃ KÍCH HOẠT

`|corr| = 0,996 > 0,7` **VÀ** `|Δκ| = 12,97 % > 3 %` → **DỪNG, BÁO CÁO.**

## 6. Đọc kết quả — ba điều phải nói cùng nhau

### 6.1 M2 không phải là bản sửa đáng tin, vì gần cộng tuyến hoàn toàn

`corr(T_start, a) = 0,996`. T và a mang gần như cùng một thông tin, nên M2 **không định danh được** tách bạch κ và γ. Ba dấu hiệu:

- `W₀ = −6,12 ± 7,09 W` — hệ số chặn **âm**, tức công suất bo âm khi a=0 và T=0. Phi vật lý.
- `γ = 0,1838 ± 0,1177` — sai số chuẩn bằng **64 %** giá trị; γ không có ý nghĩa thống kê.
- `SE(κ)` nở từ **0,055** (M1) lên **0,576** (M2), gấp **10,5 lần** — đúng dấu hiệu variance inflation của cộng tuyến.

Nói cách khác, M2 không "sửa" M1; nó chia một lượng biến thiên chung cho hai biến gần trùng nhau, và phép chia đó không ổn định.

### 6.2 Nhiệt ở đây là **biến trung gian**, không phải biến gây nhiễu

Chuỗi nhân quả đo được:

```
a ↑  →  GPU làm việc nhiều hơn  →  nhiệt ↑  →  quạt PWM ↑  →  công suất quạt ↑  →  VDD_IN ↑
```

Bằng chứng: `corr(PWM, nhiệt) = +0,914`, và `corr(nhiệt, a) = +0,996`.

Điều này có hệ quả trực tiếp: **kiểm soát theo T là kiểm soát theo một biến trung gian**, tức over-adjustment. Nó bóc đi đúng phần chi phí năng lượng mà activation *gây ra* qua đường nhiệt/quạt. Với câu hỏi của bài — *"một ngân sách năng lượng mua được a bằng bao nhiêu?"* — thiết bị triển khai **vẫn phải trả** phần công suất quạt đó. Đại lượng cần là **tổng hiệu ứng** (M1), không phải hiệu ứng trực tiếp trừ đường nhiệt (M2).

### 6.3 ⚠️ Nhưng thiết kế hiện tại KHÔNG phân biệt được trung gian với gây nhiễu

Vì chiến dịch chạy `a` **tăng dần**, "nhiệt do a gây ra" và "nhiệt do thời gian/môi trường trôi" trùng nhau hoàn toàn. Lập luận 6.2 **hợp lý nhưng chưa được chứng minh bằng dữ liệu này**. Chỉ chiến dịch **đảo thứ tự ngẫu nhiên** mới tách được.

## 7. Mục V — trọng tài ngoài **KHÔNG dùng được**

Prompt đề nghị dùng C1 làm trọng tài vì "C1 đo với lịch sử nhiệt khác". **Đối chiếu artifact cho thấy điều đó không đúng.** Từ `c1_orin_results.csv`:

| C1 tag | a | temp_start |
|---|---:|---:|
| idle_yolo26s ×3 | 0,000 | 59,00 / 59,31 / 59,47 |
| sweep_yolo26s_k10 | 0,100 | 60,59 |
| sweep_yolo26s_k5 | 0,200 | 60,94 |
| sweep_yolo26s_k2 | 0,500 | 62,34 |

Nhiệt của C1 **cũng tăng đơn điệu theo a**, đúng cùng cấu trúc confound. Nên:

| | slope | so C1 |
|---|---:|---:|
| C1 | 6,9155 | — |
| **M1 (của tôi)** | **6,9089** | **−0,10 %** |
| M2 | 6,0129 | −13,05 % |

M1 tái lập C1 tới **0,10 %** sau một tháng. Đó là bằng chứng **rất mạnh về khả năng tái lập của bàn đo**, nhưng **không** phải bằng chứng rằng nhiệt không nạp vào κ — vì C1 mang y hệt confound. Hai phép đo cùng lệch một kiểu thì khớp nhau là chuyện đương nhiên.

## 8. Mục III — quạt

| | |
|---|---|
| Node | `/sys/class/hwmon/hwmon0/pwm1` (`pwmfan`), RPM ở `hwmon2` |
| PWM quan sát | **96, 99, 101, 102, 103, 106, 107, 110, 111, 114, 118, 122, 126** (thang 255) |
| RPM | 2260 → 3084 |
| corr(PWM, nhiệt) | **+0,914** |
| Phủ | logger bật từ run 6; run 1–5 (a ∈ {0; 0,05}) không có dữ liệu quạt |

**Quạt ramp thật, không phải hằng số.** Tôi đã thử ước lượng chi phí một bậc PWM bằng cách so công suất trung bình theo PWM trong dải nhiệt hẹp 59,5–61,5 °C, nhưng kết quả **không đơn điệu** (pwm=106 cho 4,664 W trong khi pwm=103 cho 5,212 W) vì trong cùng dải nhiệt đó `a` vẫn biến thiên. **Nên con số "+0,377 W cho PWM 103→111" mà script in ra là KHÔNG đáng tin và tôi không dùng nó.** Tách công suất quạt khỏi công suất detector đòi hỏi hoặc ép PWM thủ công ở `a` cố định, hoặc chính chiến dịch ngẫu nhiên hoá.

## 9. Một phát hiện thêm: có độ cong nhẹ mà R² = 0,999 che mất

Độ dốc cục bộ giữa các mức kề nhau:

| khoảng a | dW/da |
|---|---:|
| 0,000 → 0,050 | 7,082 |
| 0,050 → 0,100 | 7,347 |
| 0,100 → 0,200 | 7,514 |
| 0,200 → 0,333 | 6,715 |
| 0,333 → 0,500 | 6,422 |

Phần dư M1 theo mức: −0,030 / −0,021 / +0,001 / +0,061 / +0,035 / −0,046 W — một hình vòng cung hệ thống, biên độ ~0,06 W trên dải 3,44 W (**≈1,8 %**). Nhỏ, nhưng **có hệ thống**, không phải nhiễu. Nếu κ được đưa vào bài thì điều này thuộc §Threats to validity.

## 10. Khuyến nghị

Chạy lại bằng `awp_energy_orchestrate_v2.py` (đã viết, đã pass syntax check, đã nằm trên máy đo):

- thứ tự `(a, lần lặp)` **đảo ngẫu nhiên**, seed cố định **20260816**, ghi `run_order.json`
- cổng nhiệt **tương đối**: `T ≤ T_baseline_idle + 1,5 °C`, `T_baseline_idle` đo 60 s ngay trước chiến dịch, ghi vào `metadata.json`
- **telemetry quạt từng run** (PWM/RPM trước và sau)
- vá luôn lỗi parser `SUMMARY` của v1

Chi phí: ~2 giờ máy. Sau đó `corr(T_start, a)` sẽ về gần 0, và:
- nếu κ vẫn ≈ 6,91 → nhiệt là trung gian, M1 đúng, dùng được, và ta có bằng chứng chứ không phải lập luận;
- nếu κ tụt về ≈ 6,0 → nhiệt/quạt thật sự là nhiễu và κ đúng là ~6,0.

**Không thể quyết bằng dữ liệu hiện có.**

---

## Phụ lục — tệp artifact

| Tệp | Nội dung |
|---|---|
| `run.log` | Log điều phối + **18 dòng `SUMMARY {json}`** — nguồn gốc của mọi con số |
| `raw/awp_a*_run*.csv` | 18 log công suất thô 10 Hz: `t_rel_s, phase, mv, ma, watt, temp_max_c` |
| `fan_trace.csv` | 10.743 mẫu 1 Hz: `epoch, pwm, rpm, temp_max_c, watt` |
| `E_of_a.csv` | Bảng tổng hợp ở §3 |
| `fit.json` | Toàn bộ hệ số M1/M2, corr, cờ luật dừng |
| `awp_energy_orchestrate.py` | Bộ điều phối v1 (bản đã chạy) |
| `awp_energy_orchestrate_v2.py` | Bản v2 đã sửa, **chưa chạy** |

**Lỗi đã gặp và cách xử lý:** parser của v1 lọc dòng bắt đầu bằng `{` nên bỏ sót tiền tố `SUMMARY ` của `c1_run.py`; `results.jsonl` rỗng và log in "DONE 0 runs". **Không mất dữ liệu** — toàn bộ 18 summary nằm nguyên văn trong `run.log` (stdout con được echo lại) và 18 log công suất thô còn đủ. Phân tích đọc thẳng `run.log`, và `a` được **tái dựng từ `(mode, k)`** chứ không từ tag, nên chính xác tuyệt đối.
