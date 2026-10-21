# -*- coding: utf-8 -*-
"""gen_report.py — Báo cáo song ngữ EN/VN giải thích toàn bộ bài báo (Word 1 cột),
3 mức cho từng ý: Trực giác / Ví dụ số / Hình thức; nhúng biểu đồ; ~3x chi tiết."""
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = ("/sessions/bold-wizardly-cannon/mnt/PhD Program/02. Working/"
       "03_AwP_Frame_Scheduling/03_Report_Paper_Bilingual_EN_VN.docx")
PF = "/sessions/bold-wizardly-cannon/mnt/PhD Program/02. Working/03_AwP_Frame_Scheduling/proof_figs/"
RS = "/sessions/bold-wizardly-cannon/mnt/PhD Program/02. Working/03_AwP_Frame_Scheduling/results/"
doc = Document()
b = doc.styles["Normal"]; b.font.name = "Times New Roman"; b.font.size = Pt(11)
GREEN = RGBColor(0x1E,0x7A,0x33); AMBER = RGBColor(0xB8,0x6A,0x00); RED = RGBColor(0xB0,0x22,0x22)
NAVY = RGBColor(0x33,0x33,0x99); PURP = RGBColor(0x6A,0x1B,0x9A)

def H(t, l=1): doc.add_heading(t, level=l)
def EN(t):
    p = doc.add_paragraph(); p.add_run("EN  ").bold = True; p.add_run(t)
def VN(t):
    p = doc.add_paragraph(); r0 = p.add_run("VI  "); r0.bold = True; r0.font.color.rgb = NAVY
    p.add_run(t).italic = True
def LEADC(kind):
    return {"Intuition":AMBER,"Trực giác":AMBER,"Worked example":GREEN,"Ví dụ số":GREEN,
            "Why it matters":PURP,"Vì sao quan trọng":PURP,"Formal":NAVY,"Hình thức":NAVY,
            "Pitfall":RED,"Lỗi thường gặp":RED,"Takeaway":GREEN,"Ghi nhớ":GREEN}.get(kind, AMBER)
def PED(kind_en, kind_vn, en, vn):
    p = doc.add_paragraph(); p.add_run("EN  ").bold = True
    r = p.add_run(kind_en + ". "); r.bold = True; r.font.color.rgb = LEADC(kind_en); p.add_run(en)
    p = doc.add_paragraph(); r0 = p.add_run("VI  "); r0.bold = True; r0.font.color.rgb = NAVY
    r = p.add_run(kind_vn + ". "); r.bold = True; r.font.color.rgb = LEADC(kind_vn); p.add_run(vn).italic = True
def STAT(label, color):
    p = doc.add_paragraph(); r = p.add_run("STATUS: " + label); r.bold = True; r.font.color.rgb = color
def FIG(path, cap_en, cap_vn, width=5.6):
    try:
        doc.add_picture(path, width=Inches(width)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    except Exception as e:
        doc.add_paragraph("[figure missing: %s]" % e)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(cap_en + "  |  " + cap_vn); r.italic = True; r.font.size = Pt(9)

# ===================== TITLE / HOW TO READ =====================
doc.add_heading("A Student's Companion to “Learning-Augmented Frame Scheduling”", 0)
doc.add_heading("Báo cáo giải thích bài báo (song ngữ EN–VN, cho sinh viên)", 1)
PED("How to read this report", "Cách đọc báo cáo",
    "This report explains the paper section by section at three levels. Each idea is given first as an "
    "INTUITION in plain words, then as a WORKED EXAMPLE with concrete numbers you can check by hand, "
    "and finally as the FORMAL statement (and proof, where useful). Diagrams accompany the hard parts. "
    "You do not need prior knowledge of online algorithms; we define everything. Symbols: $a$ budget, "
    "$W$ window, $K=\\lceil1/a\\rceil$, $\\beta=\\min(1,(W{+}1)/K)$, $\\eta$ prediction error, $\\lambda$ "
    "trust dial. (We write math in plain text, e.g. beta, eta, lambda, for readability.)",
    "Báo cáo giải thích bài báo theo từng mục, ở 3 mức. Mỗi ý được trình bày trước bằng TRỰC GIÁC "
    "(lời thường), rồi VÍ DỤ SỐ (số cụ thể để bạn tự kiểm), và cuối cùng là phát biểu HÌNH THỨC (kèm "
    "chứng minh khi cần). Có biểu đồ cho phần khó. Bạn không cần biết trước về thuật toán trực tuyến; "
    "mọi khái niệm đều được định nghĩa. Ký hiệu: a = ngân sách, W = cửa sổ, K = ⌈1/a⌉, β = "
    "min(1,(W+1)/K), η = sai số dự đoán, λ = núm tin cậy. (Toán viết dạng chữ thường: beta, eta, "
    "lambda cho dễ đọc.)")
PED("The one-sentence summary", "Tóm tắt một câu",
    "A cheap, blind ‘fire the detector every K-th frame’ schedule is provably hard to beat for catching "
    "events on energy-limited cameras; a content-aware gate that trusts a motion cue can do better only "
    "when the cue is genuinely informative, and can do far worse when it is not—so the right design "
    "keeps a guaranteed ‘backbone’ and treats the cue as a tunable bonus.",
    "Một lịch ‘mù’ rẻ tiền kiểu ‘bắn detector mỗi K khung’ rất khó bị vượt qua khi bắt sự kiện trên "
    "camera hạn chế năng lượng; một gate nhận-biết-nội-dung tin vào cue chuyển động chỉ tốt hơn khi cue "
    "thật sự hữu ích, và có thể tệ hơn NHIỀU khi cue dở—nên thiết kế đúng là giữ một ‘xương sống’ có "
    "bảo đảm và coi cue như phần thưởng cộng thêm có thể chỉnh.")
print("chunk0 ok:", len(doc.paragraphs), "paragraphs")
doc.save(OUT)

# ===================== 1. THE PROBLEM =====================
H("1. The problem: catching events on an energy-starved camera", 1)
PED("Intuition", "Trực giác",
    "Imagine a solar-powered wildlife camera. Running the neural-network object detector on a frame "
    "costs precious battery, so you can only afford to run it on, say, 1 in 10 frames. Events (an "
    "animal appears) are brief. Question: which frames should you spend the detector on, so that you "
    "‘catch’ as many events as possible? Two philosophies compete. (1) BLIND/PERIODIC: just fire every "
    "10th frame, like a metronome. (2) CONTENT-AWARE: use a cheap hint—how much the picture is moving—"
    "to fire on frames that look eventful. Intuition screams that (2) is smarter. The paper proves "
    "intuition is wrong in the regime that matters.",
    "Hãy hình dung một camera động vật hoang dã chạy bằng pin mặt trời. Chạy mạng nơ-ron phát hiện vật "
    "thể trên một khung tốn pin quý giá, nên bạn chỉ ‘bắn’ được detector trên, ví dụ, 1/10 khung. Sự "
    "kiện (một con vật xuất hiện) rất ngắn. Câu hỏi: nên tiêu detector vào khung nào để ‘bắt’ được "
    "nhiều sự kiện nhất? Có hai triết lý. (1) MÙ/TUẦN HOÀN: cứ bắn mỗi 10 khung như máy đếm nhịp. (2) "
    "NHẬN-BIẾT-NỘI-DUNG: dùng một gợi ý rẻ tiền—mức độ chuyển động của ảnh—để bắn vào khung trông có sự "
    "kiện. Trực giác bảo (2) khôn hơn. Bài báo chứng minh trực giác SAI ở vùng quan trọng.")
PED("The surprising finding (the motivation)", "Phát hiện bất ngờ (động lực)",
    "A large measurement study (called ‘P-A’) tested this honestly across 121 videos and 3,713 events. "
    "At equal energy (activation a = 0.20, i.e. 20% of frames), the periodic schedule caught 98.4% of "
    "events, while a hand-crafted motion gate caught only 28.4% and a learned random-forest gate only "
    "20.0%. Even worse, when you train/tune the gate on some scenes and test on NEW scenes (the honest "
    "‘leave-one-dataset-out’ test), the ranking of methods flips completely. So the gate’s apparent "
    "wins were artifacts of testing on the same scenes it was tuned on.",
    "Một nghiên cứu đo lường lớn (gọi là ‘P-A’) đã kiểm tra trung thực trên 121 video và 3.713 sự kiện. "
    "Cùng mức năng lượng (kích hoạt a = 0.20, tức 20% số khung), lịch tuần hoàn bắt được 98.4% sự kiện, "
    "trong khi gate chuyển động thủ công chỉ bắt 28.4% và gate random-forest học được chỉ 20.0%. Tệ "
    "hơn, khi huấn luyện/chỉnh gate trên vài cảnh rồi kiểm trên cảnh MỚI (kiểm trung thực ‘bỏ-một-bộ-"
    "dữ-liệu-ra’), thứ hạng các phương pháp đảo ngược hoàn toàn. Vậy ‘chiến thắng’ của gate chỉ là ảo "
    "do kiểm trên đúng những cảnh đã chỉnh.")
PED("Why it matters", "Vì sao quan trọng",
    "This is not academic. Always-on cameras for safety, conservation, and industry run for months on "
    "tiny power; a scheduler that silently skips event frames loses the very incidents the system "
    "exists to catch—and the failure hides until cross-scene deployment, after the benchmark already "
    "blessed the gate. The paper’s job is to turn this empirical surprise into a THEOREM that tells a "
    "deployer, in advance, whether to ship a learned gate or a one-line periodic loop.",
    "Đây không phải chuyện hàn lâm. Camera luôn-bật cho an ninh, bảo tồn, công nghiệp chạy hàng tháng "
    "bằng nguồn điện nhỏ; một bộ lập lịch âm thầm bỏ qua khung sự kiện sẽ đánh mất chính những sự cố mà "
    "hệ thống sinh ra để bắt—và lỗi này ẩn cho tới khi triển khai cross-scene, sau khi benchmark đã "
    "‘chấp thuận’ gate. Nhiệm vụ của bài báo là biến bất ngờ thực nghiệm này thành một ĐỊNH LÝ, nói "
    "trước cho người triển khai biết nên dùng gate học được hay một vòng lặp tuần hoàn một dòng.")

# ===================== 2. THE MODEL =====================
H("2. The model: turning the camera problem into math", 1)
PED("Intuition (the filmstrip picture)", "Trực giác (hình ảnh dải phim)",
    "Think of the video as a filmstrip of frames numbered t = 1, 2, …, T. Now and then an EVENT starts "
    "(its ‘onset’). We do not need to photograph the whole event—just catch it ONCE, promptly: a fire "
    "within W frames of the onset counts as a ‘recall’. That window [onset, onset+W] is the green band "
    "in Fig. 1. Energy limits us to B = a·T fires total (a = budget fraction). RECALL R = the fraction "
    "of events we caught in time. The oracle (a magic scheduler that knows all onsets) gets R = 1.",
    "Hãy xem video là dải phim gồm các khung đánh số t = 1, 2, …, T. Thỉnh thoảng một SỰ KIỆN bắt đầu "
    "(điểm ‘khởi phát’/onset). Ta không cần chụp cả sự kiện—chỉ cần bắt nó MỘT lần, kịp thời: một lần "
    "bắn trong vòng W khung kể từ onset được tính là ‘thu hồi’ (recall). Cửa sổ [onset, onset+W] là dải "
    "xanh trong Hình 1. Năng lượng giới hạn ta ở B = a·T lần bắn (a = tỉ lệ ngân sách). RECALL R = tỉ "
    "lệ sự kiện ta bắt kịp. Oracle (bộ lập lịch ‘thần’ biết mọi onset) đạt R = 1.")
FIG(PF+"pfig1_model.png",
    "Fig. 1 — Windowed recall and the periodic certificate.",
    "Hình 1 — Recall theo cửa sổ và chứng chỉ tuần hoàn.")
PED("Worked example", "Ví dụ số",
    "Say T = 100 frames and the budget is a = 0.1, so we may fire B = 10 times. Suppose there are m = 5 "
    "events. If our 10 fires land so that each event has a fire within W frames of its onset, R = 5/5 = "
    "1. If only 2 events get a timely fire, R = 2/5 = 0.4. The whole game is: place 10 fires to cover "
    "as many of the 5 onset-windows as possible, without knowing the future.",
    "Giả sử T = 100 khung và ngân sách a = 0.1, nên ta được bắn B = 10 lần. Giả sử có m = 5 sự kiện. "
    "Nếu 10 lần bắn rơi sao cho mỗi sự kiện có một lần bắn trong vòng W khung kể từ onset, thì R = 5/5 "
    "= 1. Nếu chỉ 2 sự kiện được bắn kịp, R = 2/5 = 0.4. Cả bài toán là: đặt 10 lần bắn để phủ càng "
    "nhiều trong 5 cửa sổ-onset càng tốt, mà KHÔNG biết tương lai.")
PED("The cue and the prediction error eta", "Cue và sai số dự đoán eta",
    "The content-aware gate does not see onsets; it sees a CUE x_t each frame (e.g. how much motion). "
    "It fires when the cue is high. How good is the cue? We summarize it by one number, the prediction "
    "error eta in [0,1]: eta = 1 − (onset-coverage AUC of the cue). AUC = 0.5 means the cue is useless "
    "at locating onsets (coin flip), so eta = 0.5; AUC = 1 means perfect, eta = 0. Crucially eta is "
    "MEASURABLE offline on labeled scenes—an engineer can compute it before shipping.",
    "Gate nhận-biết-nội-dung không thấy onset; mỗi khung nó thấy một CUE x_t (vd. mức chuyển động). Nó "
    "bắn khi cue cao. Cue tốt đến đâu? Ta tóm tắt bằng một số, sai số dự đoán eta ∈ [0,1]: eta = 1 − "
    "(AUC phủ-onset của cue). AUC = 0.5 nghĩa là cue vô dụng trong việc định vị onset (tung đồng xu), "
    "nên eta = 0.5; AUC = 1 nghĩa là hoàn hảo, eta = 0. Quan trọng: eta ĐO ĐƯỢC ngoại tuyến trên cảnh "
    "có nhãn—kỹ sư tính được trước khi triển khai.")
PED("The key distinction: non-adaptive gates", "Phân biệt then chốt: gate không thích nghi",
    "A gate is NON-ADAPTIVE if it decides where to fire from the cue alone, NOT from the detector's "
    "feedback (whether past fires hit or missed). All deployed gates—threshold, random-forest, even the "
    "RL gate as used—are non-adaptive in this sense. This matters a lot: the paper's sharpest theorems "
    "are about non-adaptive gates, and a later result shows that USING feedback can break the limits.",
    "Một gate là KHÔNG THÍCH NGHI nếu nó quyết định bắn ở đâu CHỈ từ cue, KHÔNG dùng phản hồi của "
    "detector (các lần bắn trước trúng hay trượt). Mọi gate đang triển khai—ngưỡng, random-forest, kể "
    "cả gate RL như đang dùng—đều không thích nghi theo nghĩa này. Điều này rất quan trọng: các định lý "
    "sắc nhất của bài là về gate không thích nghi, và một kết quả sau cho thấy DÙNG phản hồi có thể phá "
    "vỡ giới hạn.")
# notation table
H("Notation at a glance / Bảng ký hiệu", 2)
tb = doc.add_table(rows=10, cols=3); tb.style = "Light Grid Accent 1"
rows = [("Symbol", "Meaning (EN)", "Ý nghĩa (VI)"),
        ("T, t", "horizon; frame index", "độ dài; chỉ số khung"),
        ("o_j, m", "onset of event j; number of events", "onset của sự kiện j; số sự kiện"),
        ("W", "staleness window (recall if fire in [o, o+W])", "cửa sổ trễ (recall nếu bắn trong [o,o+W])"),
        ("a, B", "budget fraction; B = floor(aT) fires", "tỉ lệ ngân sách; B = floor(aT) lần bắn"),
        ("K", "ceil(1/a), the periodic stride", "ceil(1/a), bước nhảy tuần hoàn"),
        ("beta", "min(1,(W+1)/K), oblivious coverage rate", "min(1,(W+1)/K), tỉ lệ phủ ‘mù’"),
        ("eta", "prediction error = 1 − onset-AUC", "sai số dự đoán = 1 − onset-AUC"),
        ("c, r", "consistency (eta=0) & robustness (eta=1)", "consistency (eta=0) & robustness (eta=1)"),
        ("lambda", "HEDGE-GATE trust dial (backbone budget share)", "núm tin cậy HEDGE-GATE (phần ngân sách backbone)")]
for i, (a, e, v) in enumerate(rows):
    c0 = tb.rows[i].cells[0]; c0.text=""; rr=c0.paragraphs[0].add_run(a)
    c1 = tb.rows[i].cells[1]; c1.text=""; c1.paragraphs[0].add_run(e)
    c2 = tb.rows[i].cells[2]; c2.text=""; rv=c2.paragraphs[0].add_run(v); rv.italic=True
    if i==0:
        rr.bold=True
print("chunk1 ok:", len(doc.paragraphs))

# ===================== 3. LEMMA 1 =====================
H("3. Lemma 1 — a steady rhythm gives recall ‘for free’", 1)
STAT("PROVEN (fully rigorous)", GREEN)
PED("Intuition", "Trực giác",
    "If you take a photo on a steady beat that is FASTER than the window is long, you can never miss: "
    "every window of W+1 consecutive frames is guaranteed to contain one of your beats. If your beat is "
    "SLOWER than the window, you only catch a fraction (W+1)/K of randomly-placed onsets. This "
    "guaranteed fraction is called beta. The beauty: beta needs no cue, no learning—it is the recall "
    "you get just by being regular.",
    "Nếu bạn chụp theo nhịp đều NHANH hơn độ dài cửa sổ, bạn không bao giờ trượt: mọi cửa sổ gồm W+1 "
    "khung liên tiếp chắc chắn chứa một nhịp của bạn. Nếu nhịp CHẬM hơn cửa sổ, bạn chỉ bắt được tỉ lệ "
    "(W+1)/K số onset đặt ngẫu nhiên. Tỉ lệ bảo đảm này gọi là beta. Cái hay: beta không cần cue, không "
    "cần học—đó là recall bạn có chỉ nhờ đều đặn.")
PED("Worked example", "Ví dụ số",
    "Take W = 2, so the window has W+1 = 3 frames. Case A: a = 1/3 gives K = 3 ≤ 3. Fire at t = 0,3,6,… "
    "Any 3 consecutive frames hold a fire, so EVERY onset is caught: R = 1. Case B: a = 1/6 gives K = 6 "
    "> 3. Fire at 0,6,12,… Now a 3-frame window meets a fire in only 3 of every 6 phases, so on average "
    "E[R] = (W+1)/K = 3/6 = 0.5 = beta. (See Fig. 1: top row all check-marks; bottom row half crosses.)",
    "Lấy W = 2, cửa sổ có W+1 = 3 khung. Trường hợp A: a = 1/3 cho K = 3 ≤ 3. Bắn tại t = 0,3,6,… Mọi 3 "
    "khung liên tiếp đều có một lần bắn, nên MỌI onset bị bắt: R = 1. Trường hợp B: a = 1/6 cho K = 6 > "
    "3. Bắn tại 0,6,12,… Giờ một cửa sổ 3 khung chỉ gặp một lần bắn ở 3 trên mỗi 6 pha, nên trung bình "
    "E[R] = (W+1)/K = 3/6 = 0.5 = beta. (Xem Hình 1: hàng trên toàn dấu ✓; hàng dưới nửa dấu ✗.)")
PED("Formal", "Hình thức",
    "Lemma 1. (a) If K ≤ W+1, the deterministic periodic schedule recalls every event (R = 1). (b) If K "
    "> W+1, the random-phase periodic schedule recalls each event with probability (W+1)/K, so E[R] = "
    "(W+1)/K. In both cases E[R] ≥ beta.",
    "Bổ đề 1. (a) Nếu K ≤ W+1, lịch tuần hoàn tất định thu hồi mọi sự kiện (R = 1). (b) Nếu K > W+1, "
    "lịch tuần hoàn pha-ngẫu-nhiên thu hồi mỗi sự kiện với xác suất (W+1)/K, nên E[R] = (W+1)/K. Cả hai "
    "trường hợp E[R] ≥ beta.")
PED("Proof in plain words", "Chứng minh bằng lời",
    "(a) Consecutive fires are exactly K apart; if K ≤ W+1 then between any frame and W frames later "
    "(that is W+1 slots) there must be a fire—so the onset window always contains one. (b) Slide the "
    "fire-grid by a random phase U. A fixed window of W+1 slots is hit exactly when the phase lands in "
    "one of W+1 of the K possible offsets; probability (W+1)/K. Average over events ⇒ E[R] = (W+1)/K. "
    "Done.",
    "(a) Hai lần bắn liên tiếp cách đúng K; nếu K ≤ W+1 thì giữa một khung và W khung sau đó (tức W+1 "
    "khe) bắt buộc có một lần bắn—nên cửa sổ onset luôn chứa một lần. (b) Trượt lưới-bắn theo một pha "
    "ngẫu nhiên U. Một cửa sổ cố định gồm W+1 khe bị trúng đúng khi pha rơi vào một trong W+1 trên K "
    "offset; xác suất (W+1)/K. Lấy trung bình trên các sự kiện ⇒ E[R] = (W+1)/K. Xong.")
PED("Takeaway", "Ghi nhớ",
    "beta is the ‘free’ recall floor. Everything later is about whether a cue can beat beta (and by how "
    "little) or, when adversarial, drag you BELOW beta.",
    "beta là ‘sàn’ recall miễn phí. Mọi thứ sau này xoay quanh việc liệu một cue có thể vượt beta (và "
    "vượt được bao nhiêu) hay, khi đối kháng, kéo bạn XUỐNG DƯỚI beta.")

# ===================== 4. THEOREM 1 =====================
H("4. Theorem 1 — if the cue mislabels events, you lose about eta", 1)
STAT("PROVEN (counting argument)", GREEN)
PED("Intuition", "Trực giác",
    "Suppose the cue is mostly right but ‘mislabels’ a fraction eta of events—pointing the gate at the "
    "wrong place for those. A gate that pours its budget where the cue points will catch the (1−eta) it "
    "labeled correctly, but the eta it mislabeled get no targeted fire. Those can only be saved by "
    "whatever small ‘blind’ part of the budget is left. So recall is at most about (1−eta): you lose "
    "roughly eta to the mislabels.",
    "Giả sử cue phần lớn đúng nhưng ‘gán nhãn sai’ một tỉ lệ eta số sự kiện—chỉ gate vào sai chỗ cho "
    "những sự kiện đó. Một gate dồn ngân sách vào nơi cue chỉ sẽ bắt được (1−eta) phần nó gán đúng, "
    "nhưng phần eta gán sai không có lần bắn nhắm trúng. Những sự kiện đó chỉ được cứu bởi phần ngân "
    "sách ‘mù’ nhỏ còn lại. Nên recall tối đa khoảng (1−eta): bạn mất xấp xỉ eta vì gán sai.")
PED("Worked example", "Ví dụ số",
    "10 events; the cue mislabels eta = 0.3 of them (3 events) and the gate commits all budget to the "
    "cue (delta = 0). Then those 3 mislabeled events get no targeted fire and are missed; at best the 7 "
    "correctly-labeled are caught, so R ≤ 7/10 = 0.7 = 1 − eta. If you keep a tiny blind fraction delta "
    "= 0.1, it can rescue at most 1 extra event, lifting the bound to 1 − eta + delta.",
    "10 sự kiện; cue gán sai eta = 0.3 (3 sự kiện) và gate dồn toàn bộ ngân sách vào cue (delta = 0). "
    "Khi đó 3 sự kiện gán sai không có lần bắn nhắm trúng và bị bỏ lỡ; tốt nhất 7 sự kiện gán đúng được "
    "bắt, nên R ≤ 7/10 = 0.7 = 1 − eta. Nếu giữ một phần ‘mù’ nhỏ delta = 0.1, nó cứu thêm tối đa 1 sự "
    "kiện, nâng cận lên 1 − eta + delta.")
PED("Formal", "Hình thức",
    "Theorem 1. On disjoint ‘blocks’ with one event each and budget B = m, a gate that puts at least "
    "(1−delta)B fires on cue-marked blocks, where the cue mislabels an eta fraction, has recall R ≤ "
    "(1−eta) + delta, hence R ≤ 1 − eta(1−(W+1)/K) + delta. A pure-cue gate (delta = 0) gets R ≤ 1−eta.",
    "Định lý 1. Trên các ‘khối’ rời, mỗi khối một sự kiện, ngân sách B = m, một gate đặt ít nhất "
    "(1−delta)B lần bắn vào khối được cue đánh dấu, với cue gán sai tỉ lệ eta, có recall R ≤ (1−eta) + "
    "delta, do đó R ≤ 1 − eta(1−(W+1)/K) + delta. Gate thuần-cue (delta = 0) đạt R ≤ 1−eta.")
PED("Proof in plain words", "Chứng minh bằng lời",
    "Split events into ‘correctly labeled’ (fraction 1−eta) and ‘mislabeled’ (fraction eta). The "
    "mislabeled ones sit in blocks with NO cue-fire, so only a blind fire can catch them. There are at "
    "most delta·B blind fires, and since blocks are disjoint each fire catches at most one event—so at "
    "most delta·B mislabeled events are saved. Add the at-most (1−eta)·B correctly labeled. Divide by m "
    "= B: R ≤ (1−eta) + delta.",
    "Chia sự kiện thành ‘gán đúng’ (tỉ lệ 1−eta) và ‘gán sai’ (tỉ lệ eta). Phần gán sai nằm ở khối "
    "KHÔNG có lần bắn theo cue, nên chỉ lần bắn ‘mù’ mới cứu được. Có tối đa delta·B lần bắn mù, và vì "
    "các khối rời nhau, mỗi lần bắn cứu tối đa một sự kiện—nên tối đa delta·B sự kiện gán sai được cứu. "
    "Cộng với tối đa (1−eta)·B sự kiện gán đúng. Chia cho m = B: R ≤ (1−eta) + delta.")
PED("Pitfall", "Lỗi thường gặp",
    "Theorem 1 is only INFORMATIVE when K > W+1 (low budget). When K ≤ W+1 the term (W+1)/K ≥ 1 makes "
    "the bound say ‘R ≤ 1’, which is useless—yet gates STILL collapse there. That is exactly why we "
    "need a second, stronger bound (Theorem 1′) for the high-budget regime.",
    "Định lý 1 chỉ CÓ ÍCH khi K > W+1 (ngân sách thấp). Khi K ≤ W+1, số hạng (W+1)/K ≥ 1 làm cận thành "
    "‘R ≤ 1’, vô dụng—dù gate VẪN sụp ở đó. Chính vì vậy ta cần một cận thứ hai, mạnh hơn (Định lý 1′) "
    "cho vùng ngân sách cao.")
print("chunk2 ok:", len(doc.paragraphs))

# ===================== 5. THEOREM 1' =====================
H("5. Theorem 1′ — fully trusting a bad cue can cost EVERYTHING", 1)
STAT("PROVEN (the headline impossibility)", GREEN)
PED("Intuition", "Trực giác",
    "Theorem 1 lost ‘about eta’. Theorem 1′ is scarier: in the high-budget regime, an adversarial cue "
    "can point ENTIRELY at busy-but-empty places (waving trees, camera shake) that contain no events. A "
    "gate that fully trusts such a cue spends 100% of its budget there and catches ZERO events—worse "
    "than random! The cure is insurance: reserve a slice of budget for a blind backbone that fires on a "
    "steady beat no matter what the cue says. That slice guarantees a floor of lambda·beta.",
    "Định lý 1 mất ‘khoảng eta’. Định lý 1′ đáng sợ hơn: ở vùng ngân sách cao, một cue đối kháng có thể "
    "chỉ HOÀN TOÀN vào chỗ nhộn-nhịp-mà-rỗng (cây đung đưa, rung máy) không chứa sự kiện. Một gate tin "
    "tuyệt đối vào cue đó dồn 100% ngân sách vào đấy và bắt được KHÔNG sự kiện nào—tệ hơn ngẫu nhiên! "
    "Thuốc chữa là bảo hiểm: dành một phần ngân sách cho backbone ‘mù’ bắn theo nhịp đều bất kể cue nói "
    "gì. Phần đó bảo đảm một sàn lambda·beta.")
FIG(PF+"pfig2_thm1prime.png",
    "Fig. 2 — (a) pure-cue gate fires clutter → R=0; (b) backbone guarantees floor lambda·beta.",
    "Hình 2 — (a) gate thuần-cue bắn nhiễu → R=0; (b) backbone bảo đảm sàn lambda·beta.")
PED("Worked example", "Ví dụ số",
    "T = 100, W = 2, budget B = 10 (a = 0.1), m = 5 events, K = 10, beta = min(1,3/10) = 0.3. Non-window "
    "frames = 100 − 5·3 = 85 ≥ 10, so the adversary puts the 10 highest cue values on 10 clutter frames "
    "with no events. The pure-cue gate fires exactly those → 0 events → R = 0. Now HEDGE-GATE with "
    "lambda = 0.5 spends 5 fires on a blind rate-0.05 backbone; each event is caught with probability "
    "min(1, 0.05·3) = 0.15, so E[R] ≥ lambda·beta = 0.5·0.3 = 0.15, GUARANTEED, no matter the cue.",
    "T = 100, W = 2, ngân sách B = 10 (a = 0.1), m = 5 sự kiện, K = 10, beta = min(1,3/10) = 0.3. Khung "
    "ngoài cửa sổ = 100 − 5·3 = 85 ≥ 10, nên kẻ đối kháng đặt 10 giá trị cue cao nhất lên 10 khung "
    "nhiễu không có sự kiện. Gate thuần-cue bắn đúng những khung đó → 0 sự kiện → R = 0. Giờ HEDGE-GATE "
    "với lambda = 0.5 tiêu 5 lần bắn cho backbone mù nhịp 0.05; mỗi sự kiện được bắt với xác suất "
    "min(1, 0.05·3) = 0.15, nên E[R] ≥ lambda·beta = 0.5·0.3 = 0.15, BẢO ĐẢM, bất kể cue.")
PED("Formal", "Hình thức",
    "Theorem 1′. (i) A pure-advice gate (fires top-B cue frames) has, in the sparse regime m(W+1) ≤ "
    "T−B, an instance with onset-AUC = 1/2 on which R = 0. (ii) HEDGE-GATE(lambda) satisfies E[R] ≥ "
    "lambda·beta on EVERY instance and EVERY cue—the advice arm adds zero worst-case recall; a gate is "
    "no better than its blind fraction alone.",
    "Định lý 1′. (i) Gate thuần-dự-đoán (bắn B khung cue cao nhất) — trong chế độ thưa m(W+1) ≤ T−B — "
    "có một thể hiện với onset-AUC = 1/2 khiến R = 0. (ii) HEDGE-GATE(lambda) thỏa E[R] ≥ lambda·beta "
    "trên MỌI thể hiện và MỌI cue—nhánh dự đoán cộng thêm 0 recall xấu nhất; gate không hơn gì phần mù "
    "của nó.")
PED("Proof in plain words", "Chứng minh bằng lời",
    "(i) There is room (≥ B non-window frames) to hide B high-cue clutter frames away from all events; "
    "the gate fires them and recalls nothing. (ii) The backbone is a blind random-phase periodic "
    "schedule at rate lambda·a, chosen WITHOUT looking at the cue; by Lemma 1 it catches each event "
    "with probability lambda·beta, and the adversary cannot steer a blind schedule. So the floor holds "
    "regardless of how bad the cue is.",
    "(i) Có chỗ (≥ B khung ngoài cửa sổ) để giấu B khung nhiễu cue-cao tránh xa mọi sự kiện; gate bắn "
    "chúng và không thu hồi gì. (ii) Backbone là lịch tuần hoàn pha-ngẫu-nhiên ‘mù’ nhịp lambda·a, chọn "
    "mà KHÔNG nhìn cue; theo Bổ đề 1 nó bắt mỗi sự kiện với xác suất lambda·beta, và kẻ đối kháng không "
    "lái được lịch mù. Nên sàn giữ vững bất kể cue tệ đến đâu.")
PED("Why it matters", "Vì sao quan trọng",
    "This is the paper's first ‘impossibility’ result for content-aware sampling, and it explains the "
    "measured collapse to 20–28%: deployed cues are roughly adversarial cross-scene, so the trusting "
    "gate forfeits its floor. HEDGE-GATE's lambda·beta is the insurance that the deployed gates lacked.",
    "Đây là kết quả ‘bất khả thi’ đầu tiên cho lấy mẫu nhận-biết-nội-dung, và nó giải thích sự sụp đổ "
    "đo được về 20–28%: cue triển khai gần như đối kháng cross-scene, nên gate tin tưởng đánh mất sàn. "
    "lambda·beta của HEDGE-GATE chính là bảo hiểm mà các gate triển khai thiếu.")

# ===================== 6. THEOREM 2 =====================
H("6. Theorem 2 — the exact trade-off: you can't be both fully trusting and fully safe", 1)
STAT("PROVEN — tight (achievability = converse), for non-adaptive gates", GREEN)
PED("Intuition", "Trực giác",
    "Two virtues pull apart. CONSISTENCY c = how well you do when the cue is PERFECT (eta=0): you'd "
    "want to spend everything on the cue. ROBUSTNESS r = how well you do when the cue is ADVERSARIAL "
    "(eta=1): you'd want to spend everything on the blind backbone. Every fire you give the cue is a "
    "fire NOT on the backbone—so more of one means less of the other. The paper shows the trade-off is "
    "a straight LINE, and HEDGE-GATE's dial lambda slides exactly along it.",
    "Hai phẩm chất kéo ngược. CONSISTENCY c = làm tốt ra sao khi cue HOÀN HẢO (eta=0): bạn muốn dồn hết "
    "cho cue. ROBUSTNESS r = làm tốt ra sao khi cue ĐỐI KHÁNG (eta=1): bạn muốn dồn hết cho backbone "
    "mù. Mỗi lần bắn dành cho cue là một lần KHÔNG dành cho backbone—nên tăng cái này giảm cái kia. Bài "
    "báo cho thấy đánh đổi là một ĐƯỜNG THẲNG, và núm lambda của HEDGE-GATE trượt đúng trên đó.")
FIG(PF+"pfig3_frontier.png",
    "Fig. 3 — the tight frontier r = beta(1−c)/(1−beta); HEDGE-GATE points sit on it.",
    "Hình 3 — biên chặt r = beta(1−c)/(1−beta); các điểm HEDGE-GATE nằm trên đó.")
PED("Worked example", "Ví dụ số",
    "Let W = 7, a = 0.1, so K = 10 and beta = min(1,8/10) = 0.8. Pick lambda = 0.25. Then consistency c "
    "= (1−0.25) + 0.25·0.8 = 0.95 and robustness r = 0.25·0.8 = 0.20. Check the frontier formula: "
    "beta(1−c)/(1−beta) = 0.8·0.05/0.2 = 0.20. Exactly on the line! Reading: to GUARANTEE recall 0.20 "
    "when the cue is useless, you accept consistency 0.95 (not 1.0) when the cue is perfect. Demand c = "
    "1 and you are forced to r = 0.",
    "Lấy W = 7, a = 0.1, nên K = 10 và beta = min(1,8/10) = 0.8. Chọn lambda = 0.25. Khi đó consistency "
    "c = (1−0.25) + 0.25·0.8 = 0.95 và robustness r = 0.25·0.8 = 0.20. Kiểm công thức biên: "
    "beta(1−c)/(1−beta) = 0.8·0.05/0.2 = 0.20. Đúng trên đường! Đọc: để BẢO ĐẢM recall 0.20 khi cue vô "
    "dụng, bạn chấp nhận consistency 0.95 (không phải 1.0) khi cue hoàn hảo. Đòi c = 1 thì buộc r = 0.")
PED("Formal", "Hình thức",
    "Theorem 2. For K > W+1 (beta < 1), every non-adaptive randomized gate with consistency c and "
    "robustness r satisfies r ≤ beta(1−c)/(1−beta); HEDGE-GATE(lambda) attains it with c = (1−lambda) + "
    "lambda·beta and r = lambda·beta. Theorem 3: HEDGE-GATE is therefore Pareto-optimal in this class.",
    "Định lý 2. Với K > W+1 (beta < 1), mọi gate không-thích-nghi ngẫu nhiên với consistency c và "
    "robustness r thỏa r ≤ beta(1−c)/(1−beta); HEDGE-GATE(lambda) đạt tới với c = (1−lambda) + "
    "lambda·beta và r = lambda·beta. Định lý 3: do đó HEDGE-GATE tối ưu Pareto trong lớp này.")
PED("The converse, in plain words (the clever part)", "Phần đảo, bằng lời (chỗ tinh tế)",
    "Why can't a non-adaptive gate cheat the line? Because the cue looks IDENTICAL whether it is "
    "trustworthy or adversarial—so the gate, seeing only the cue, must commit the SAME firing plan p_b "
    "(chance it fires in block b) in both worlds. To be consistent it piles probability on the cue-"
    "marked blocks; the budget (sum of p_b ≤ B) then leaves little for the others. The adversary simply "
    "hides the events on the blocks with the SMALLEST p_b. The only firing it cannot dodge is the blind "
    "backbone—exactly lambda·beta. Fig. 4 shows this bar-chart picture.",
    "Vì sao gate không-thích-nghi không ‘gian lận’ được đường biên? Vì cue trông Y HỆT dù đáng tin hay "
    "đối kháng—nên gate, chỉ thấy cue, phải cam kết CÙNG kế hoạch bắn p_b (khả năng bắn ở khối b) ở cả "
    "hai thế giới. Để consistent, nó dồn xác suất vào các khối được cue đánh dấu; ràng buộc ngân sách "
    "(tổng p_b ≤ B) khiến phần còn lại rất ít. Kẻ đối kháng chỉ việc giấu sự kiện ở các khối có p_b NHỎ "
    "NHẤT. Lần bắn duy nhất nó không né được là backbone mù—đúng lambda·beta. Hình 4 minh hoạ bằng biểu "
    "đồ cột.")
FIG(PF+"pfig5_converse.png",
    "Fig. 4 — converse: fixed p_b; adversary hides events on the smallest non-marked bars.",
    "Hình 4 — phần đảo: p_b cố định; kẻ đối kháng giấu sự kiện ở các cột không-được-đánh-dấu nhỏ nhất.")
PED("A concrete converse computation", "Một tính toán phần đảo cụ thể",
    "N = 10 blocks, B = 2 events, so beta = B/N = 0.2. Suppose c = 0.9. Budget: sum of p_b ≤ 2, and the "
    "cue-marked blocks hold sum = c·B = 1.8, leaving ≤ 0.2 over the 8 non-marked blocks. The adversary "
    "puts the 2 events on the 2 smallest of those, total ≤ 2·(0.2/8), giving r ≤ 0.025. The formula "
    "agrees: beta(1−c)/(1−beta) = 0.2·0.1/0.8 = 0.025. A 90%-consistent gate guarantees only 2.5% "
    "recall against a bad cue.",
    "N = 10 khối, B = 2 sự kiện, nên beta = B/N = 0.2. Giả sử c = 0.9. Ngân sách: tổng p_b ≤ 2, và các "
    "khối được cue đánh dấu giữ tổng = c·B = 1.8, còn ≤ 0.2 cho 8 khối không-đánh-dấu. Kẻ đối kháng đặt "
    "2 sự kiện vào 2 khối nhỏ nhất trong đó, tổng ≤ 2·(0.2/8), cho r ≤ 0.025. Công thức khớp: "
    "beta(1−c)/(1−beta) = 0.2·0.1/0.8 = 0.025. Gate 90%-consistent chỉ bảo đảm 2.5% recall trước cue dở.")
PED("Corollary (ratio form)", "Hệ quả (dạng tỉ số)",
    "If you push consistency to c = 1 − epsilon (almost optimal when the cue is good), the robustness "
    "RATIO beta/r blows up like (1−beta)/epsilon = Omega(1/epsilon). In words: the closer you insist on "
    "being to the oracle when the cue is perfect, the worse your worst-case guarantee, and it diverges.",
    "Nếu đẩy consistency tới c = 1 − epsilon (gần tối ưu khi cue tốt), thì TỈ SỐ robustness beta/r phình "
    "lên như (1−beta)/epsilon = Omega(1/epsilon). Nói cách khác: càng đòi gần oracle khi cue hoàn hảo, "
    "bảo đảm xấu-nhất càng tệ, và phân kỳ.")
print("chunk3 ok:", len(doc.paragraphs))

# ===================== 7. HEDGE-GATE =====================
H("7. HEDGE-GATE — the algorithm, and the dial as a certificate", 1)
STAT("Constructive (achieves the frontier)", GREEN)
PED("Intuition", "Trực giác",
    "HEDGE-GATE is almost embarrassingly simple: split the budget. Give a fraction lambda to a blind "
    "periodic backbone (for safety) and the remaining (1−lambda) to the cue (for upside). lambda = 1 is "
    "pure periodic; lambda = 0 is the pure (fragile) gate; in between you get both a guaranteed floor "
    "and a shot at the cue's bonus. The dial lambda turns a vague ‘how much should I trust the model?’ "
    "into a one-line worst-case guarantee.",
    "HEDGE-GATE đơn giản đến phát ngượng: chia ngân sách. Dành tỉ lệ lambda cho backbone tuần hoàn mù "
    "(để an toàn) và phần (1−lambda) còn lại cho cue (để có upside). lambda = 1 là tuần hoàn thuần; "
    "lambda = 0 là gate thuần (dễ vỡ); ở giữa bạn vừa có sàn bảo đảm vừa có cơ hội hưởng phần thưởng từ "
    "cue. Núm lambda biến câu hỏi mơ hồ ‘nên tin mô hình bao nhiêu?’ thành một bảo đảm xấu-nhất một dòng.")
PED("Algorithm 1 (HEDGE-GATE)", "Thuật toán 1 (HEDGE-GATE)",
    "Input: budget a, window W, dial lambda. Step 1 (backbone): fire a blind periodic schedule at rate "
    "lambda·a (one fire every ceil(1/(lambda·a)) frames). Step 2 (advice arm): with the remaining "
    "(1−lambda)·a budget, fire the highest-cue frames not already fired. Guarantee: E[R] ≥ lambda·beta "
    "always; on the worst-case family the point (c, r) = ((1−lambda)+lambda·beta, lambda·beta) lies on "
    "the frontier.",
    "Đầu vào: ngân sách a, cửa sổ W, núm lambda. Bước 1 (backbone): bắn lịch tuần hoàn mù nhịp lambda·a "
    "(một lần bắn mỗi ceil(1/(lambda·a)) khung). Bước 2 (nhánh dự đoán): với ngân sách (1−lambda)·a còn "
    "lại, bắn các khung cue cao nhất chưa bị bắn. Bảo đảm: E[R] ≥ lambda·beta luôn luôn; trên họ xấu "
    "nhất, điểm (c, r) = ((1−lambda)+lambda·beta, lambda·beta) nằm trên biên.")
PED("The dial recipe (use this in practice)", "Công thức núm (dùng thực tế)",
    "To guarantee worst-case recall rho no matter the cue, pick the smallest lambda with lambda·beta ≥ "
    "rho, i.e. lambda ≥ rho/beta, and give the rest to the cue. Example: beta = 0.8 and you need rho = "
    "0.6 → lambda ≥ 0.75. Below a = 1/(W+1) (here 0.125) the trade-off is real; above it beta = 1 and "
    "you can give the WHOLE budget to the cue with no robustness loss.",
    "Để bảo đảm recall xấu-nhất rho bất kể cue, chọn lambda nhỏ nhất với lambda·beta ≥ rho, tức lambda "
    "≥ rho/beta, và dành phần còn lại cho cue. Ví dụ: beta = 0.8 và bạn cần rho = 0.6 → lambda ≥ 0.75. "
    "Dưới a = 1/(W+1) (ở đây 0.125) đánh đổi mới thật sự; trên ngưỡng đó beta = 1 và bạn có thể dồn TOÀN "
    "BỘ ngân sách cho cue mà không mất robustness.")
PED("Cost", "Chi phí",
    "HEDGE-GATE adds essentially nothing: one integer counter (the backbone) and one threshold (the cue "
    "it already computes). No training, no model to store; O(1) work per frame. On a microcontroller it "
    "is one comparison and one increment per frame on top of the cue.",
    "HEDGE-GATE gần như không thêm gì: một bộ đếm số nguyên (backbone) và một ngưỡng (cue vốn đã tính). "
    "Không huấn luyện, không mô hình để lưu; O(1) công mỗi khung. Trên vi điều khiển chỉ là một phép so "
    "sánh và một phép tăng mỗi khung, cộng thêm cue.")

# ===================== 8. PROPOSITION 4 =====================
H("8. Proposition 4 — performance fades on a straight line, not a cliff", 1)
STAT("PROVEN under Assumption A (per-event independence)", AMBER)
PED("Intuition", "Trực giác",
    "As the cue worsens (eta from 0 to 1), HEDGE-GATE's recall slides DOWN A STRAIGHT LINE from its "
    "best (good cue) to its guaranteed floor lambda·beta (useless cue). No sudden collapse: the "
    "backbone holds the bottom, and the cue arm contributes in proportion to how often it is right, "
    "namely (1−eta). Smoothness is reassuring for operators—small cue degradation means small recall "
    "loss.",
    "Khi cue xấu dần (eta từ 0 đến 1), recall của HEDGE-GATE trượt XUỐNG THEO ĐƯỜNG THẲNG từ mức tốt "
    "nhất (cue tốt) về sàn bảo đảm lambda·beta (cue vô dụng). Không sụp đột ngột: backbone giữ đáy, và "
    "nhánh cue đóng góp tỉ lệ với tần suất nó đúng, tức (1−eta). Tính trơn này trấn an người vận hành—"
    "cue xấu đi một chút thì recall mất một chút.")
FIG(PF+"pfig4_smoothness.png",
    "Fig. 5 — recall is affine in eta: E[R] = lambda·beta + (1−eta)·kappa·(1−lambda·beta).",
    "Hình 5 — recall affine theo eta: E[R] = lambda·beta + (1−eta)·kappa·(1−lambda·beta).", 5.0)
PED("Worked example", "Ví dụ số",
    "lambda = 0.5, beta = 0.8 → floor lambda·beta = 0.4. Suppose at eta = 0 the cue arm covers kappa = "
    "0.85 of the remaining headroom. Then E[R](eta) = 0.4 + (1−eta)·0.85·(1−0.4) = 0.4 + 0.51·(1−eta). "
    "So eta = 0 → 0.91, eta = 0.5 → 0.655, eta = 1 → 0.40. A straight line that never drops below the "
    "floor 0.40.",
    "lambda = 0.5, beta = 0.8 → sàn lambda·beta = 0.4. Giả sử tại eta = 0 nhánh cue phủ kappa = 0.85 "
    "phần dư địa còn lại. Khi đó E[R](eta) = 0.4 + (1−eta)·0.85·(1−0.4) = 0.4 + 0.51·(1−eta). Vậy eta = "
    "0 → 0.91, eta = 0.5 → 0.655, eta = 1 → 0.40. Một đường thẳng không bao giờ xuống dưới sàn 0.40.")
PED("Honesty note (why ‘under Assumption A’)", "Ghi chú trung thực (vì sao ‘dưới Giả thiết A’)",
    "The clean straight line assumes per-event errors are INDEPENDENT. Real cues are correlated across "
    "nearby events (a scene that fools the cue fools it for a burst), so the line is only approximate "
    "without the assumption. Empirically the fit is excellent (Pearson r up to −0.99). We label this "
    "honestly: PROVEN under the assumption, otherwise an empirically-supported claim.",
    "Đường thẳng đẹp giả định sai số theo từng sự kiện là ĐỘC LẬP. Cue thật tương quan giữa các sự kiện "
    "gần nhau (một cảnh đánh lừa cue sẽ lừa cho cả một cụm), nên đường thẳng chỉ gần đúng khi bỏ giả "
    "thiết. Thực nghiệm khớp rất tốt (Pearson r tới −0.99). Ta gắn nhãn trung thực: ĐÃ CHỨNG MINH dưới "
    "giả thiết, nếu không thì là khẳng định có hỗ trợ thực nghiệm.")

# ===================== 9. PROPOSITION 5 =====================
H("9. Proposition 5 — using detector feedback breaks the limit (a positive result)", 1)
STAT("PROVEN (delimits the impossibility)", GREEN)
PED("Intuition", "Trực giác",
    "The impossibility (Theorem 2) was for NON-ADAPTIVE gates. But a smarter gate can USE the detector's "
    "hit/miss feedback: fire one ‘probe’ at a cue-marked spot; if it HITS, the cue is trustworthy—keep "
    "trusting it; if it MISSES, the cue is bad—fall back to the blind backbone. This ‘test-then-"
    "fallback’ gate gets near-perfect consistency AND near-beta robustness at once, beyond the non-"
    "adaptive line. The practical lesson: verify the cue with a few probes before betting your whole "
    "budget.",
    "Bất khả thi (Định lý 2) là cho gate KHÔNG-THÍCH-NGHI. Nhưng một gate khôn hơn có thể DÙNG phản hồi "
    "trúng/trượt của detector: bắn một lần ‘thử’ vào chỗ cue đánh dấu; nếu TRÚNG, cue đáng tin—tiếp tục "
    "tin; nếu TRƯỢT, cue dở—lui về backbone mù. Gate ‘thử-rồi-lui’ này đạt gần như consistency hoàn hảo "
    "VÀ robustness gần beta cùng lúc, vượt qua đường không-thích-nghi. Bài học thực tế: kiểm chứng cue "
    "bằng vài lần thử trước khi đặt cược toàn bộ ngân sách.")
PED("Formal", "Hình thức",
    "Proposition 5. On the canonical family with B > m, the test-then-fallback gate achieves c = 1 and "
    "r ≥ (1−1/B)·beta simultaneously, strictly above the non-adaptive frontier (which forces r = 0 at c "
    "= 1).",
    "Định lý 5. Trên họ chuẩn với B > m, gate thử-rồi-lui đạt đồng thời c = 1 và r ≥ (1−1/B)·beta, vượt "
    "hẳn biên không-thích-nghi (vốn ép r = 0 tại c = 1).")
PED("Proof in plain words", "Chứng minh bằng lời",
    "When eta = 0 the cue-marked blocks ARE the events, so firing them recalls everything: c = 1. When "
    "eta = 1 the first probe lands in an empty marked block (a miss), revealing the bad cue; the gate "
    "then runs the remaining B−1 fires as a blind backbone, catching each event with probability ≥ "
    "(1−1/B)·beta by Lemma 1. One cheap probe buys both worlds.",
    "Khi eta = 0, các khối được cue đánh dấu CHÍNH LÀ sự kiện, nên bắn chúng thu hồi tất cả: c = 1. Khi "
    "eta = 1, lần thử đầu rơi vào một khối đánh dấu rỗng (trượt), lộ ra cue dở; gate sau đó chạy B−1 "
    "lần bắn còn lại như backbone mù, bắt mỗi sự kiện với xác suất ≥ (1−1/B)·beta theo Bổ đề 1. Một lần "
    "thử rẻ tiền mua được cả hai thế giới.")
print("chunk4 ok:", len(doc.paragraphs))

# ===================== 10. EMPIRICAL =====================
H("10. The experiments: does the theory match reality?", 1)
PED("How the evidence is layered", "Bằng chứng xếp tầng thế nào",
    "The paper checks the theory at three layers: (A) MEASURED anchors reused from the P-A study (real "
    "detectors, real videos); (B) a calibrated SYNTHETIC sweep that can dial eta across the full range "
    "0→1 (impossible to do cleanly on real video); (C) a REAL-DATA run where motion cues and events are "
    "recomputed from raw frames of 121 videos. Together they pin down both the extremes and the "
    "in-between behaviour.",
    "Bài kiểm tra lý thuyết ở ba tầng: (A) các neo ĐO ĐƯỢC tái dùng từ nghiên cứu P-A (detector thật, "
    "video thật); (B) một quét SYNTHETIC hiệu chỉnh có thể chỉnh eta khắp dải 0→1 (không thể làm sạch "
    "trên video thật); (C) một lần chạy DỮ LIỆU THẬT trong đó cue chuyển động và sự kiện được tính lại "
    "từ khung gốc của 121 video. Cùng nhau, chúng ghim cả hai cực lẫn hành vi ở giữa.")
H("10.1 Measured anchors (real detectors) / Neo đo được (detector thật)", 2)
FIG(RS+"fig1_anchor_frontier_PA.png",
    "Fig. 6 — measured recall vs energy: periodic dominates all gates below a=0.5.",
    "Hình 6 — recall đo theo năng lượng: periodic vượt mọi gate khi a<0.5.", 5.0)
PED("What it shows", "Cho thấy gì",
    "The periodic curve sits near the top; the score-gate (S) and random-forest gate (RF) are far "
    "below, and at a = 0.20 they recover only 28% and 20% versus periodic's 98%. This is the eta→1 "
    "robustness arm of Theorem 1′ realized by REAL deployed cues: cross-scene, the cue is roughly "
    "adversarial, so the trusting gate forfeits its floor.",
    "Đường periodic nằm gần đỉnh; gate điểm số (S) và gate random-forest (RF) thấp hơn nhiều, và tại a "
    "= 0.20 chúng chỉ thu hồi 28% và 20% so với 98% của periodic. Đây là nhánh robustness eta→1 của "
    "Định lý 1′ hiện thực bởi cue triển khai THẬT: qua cảnh khác, cue gần như đối kháng, nên gate tin "
    "tưởng đánh mất sàn.")
H("10.2 Synthetic sweep (the controlled lab) / Quét synthetic (phòng thí nghiệm có kiểm soát)", 2)
FIG(RS+"fig2_noise_smoothness.png",
    "Fig. 7 — uninformative noise: HEDGE-GATE recall is a straight line in eta (Prop. 4).",
    "Hình 7 — nhiễu vô định hướng: recall HEDGE-GATE là đường thẳng theo eta (Mệnh đề 4).", 6.2)
FIG(RS+"fig4_adversarial_floor.png",
    "Fig. 8 — adversarial cue: the pure gate collapses below random; HEDGE-GATE holds lambda·beta.",
    "Hình 8 — cue đối kháng: gate thuần sụp dưới ngẫu nhiên; HEDGE-GATE giữ lambda·beta.", 6.2)
PED("What it shows", "Cho thấy gì",
    "Two failure modes are separated. Under merely UNINFORMATIVE noise (Fig. 7), recall slides smoothly "
    "down to the random level beta—bad but bounded, and the straight lines confirm Proposition 4 "
    "(Pearson r up to −0.99). Under ADVERSARIAL advice (Fig. 8), the pure gate (lambda=0) falls BELOW "
    "beta toward 0, while HEDGE-GATE with lambda>0 stays flat at its floor lambda·beta—exactly Theorem "
    "1′. Deployed gates live in this second, sub-beta regime, which is why they LOSE to periodic rather "
    "than merely tie.",
    "Hai chế độ hỏng được tách ra. Dưới nhiễu chỉ VÔ ĐỊNH HƯỚNG (Hình 7), recall trượt trơn về mức ngẫu "
    "nhiên beta—xấu nhưng có chặn, và các đường thẳng xác nhận Mệnh đề 4 (Pearson r tới −0.99). Dưới "
    "dự đoán ĐỐI KHÁNG (Hình 8), gate thuần (lambda=0) rơi DƯỚI beta về 0, trong khi HEDGE-GATE với "
    "lambda>0 phẳng ở sàn lambda·beta—đúng Định lý 1′. Gate triển khai sống ở chế độ dưới-beta thứ hai "
    "này, nên chúng THUA periodic chứ không chỉ hòa.")
FIG(RS+"fig3_consistency_robustness.png",
    "Fig. 9 — the measured (c,r) points trace the tight frontier of Theorem 2.",
    "Hình 9 — các điểm (c,r) đo được vạch đúng biên chặt của Định lý 2.", 4.8)
H("10.3 Real-data run on raw video (121 videos) / Chạy dữ liệu thật (121 video)", 2)
FIG(RS+"fig5_real_corruption.png",
    "Fig. 10 — real cues sit at eta≈0.5; gate-heavy policies trail the periodic certificate.",
    "Hình 10 — cue thật ở eta≈0.5; chính sách thiên-gate tụt dưới chứng chỉ periodic.", 6.2)
PED("What it shows", "Cho thấy gì",
    "Real motion cues were recomputed from frames of BMC, LASIESTA and CDnet2014 (121 videos, 262 "
    "events). The cue's onset-AUC is 0.51—right at the ‘coin-flip’ level, i.e. near-uninformative about "
    "onsets. As the theory predicts in this high-eta regime, the deployed gate trails periodic at every "
    "budget: at a=0.1, gate 0.36 vs periodic 0.78 (a 0.42 gap with NON-overlapping confidence "
    "intervals). The cue is informative about MOTION, not about event ONSETS, and cross-scene motion is "
    "not events.",
    "Cue chuyển động thật được tính lại từ khung của BMC, LASIESTA và CDnet2014 (121 video, 262 sự "
    "kiện). Onset-AUC của cue là 0.51—đúng mức ‘tung đồng xu’, tức gần như vô dụng về onset. Đúng như "
    "lý thuyết dự báo ở vùng eta cao này, gate triển khai tụt dưới periodic ở mọi ngân sách: tại a=0.1, "
    "gate 0.36 so với periodic 0.78 (chênh 0.42 với khoảng tin cậy KHÔNG chồng lấp). Cue hữu ích về "
    "CHUYỂN ĐỘNG, không phải về ONSET sự kiện, và chuyển động cross-scene không phải là sự kiện.")
FIG(RS+"fig6_percategory.png",
    "Fig. 11 — per-category gap at a=0.2: the gate loses in ALL 11 groups.",
    "Hình 11 — chênh theo từng nhóm tại a=0.2: gate THUA ở CẢ 11 nhóm.", 5.6)
PED("The per-category clincher", "Bằng chứng theo từng nhóm",
    "Breaking the result down by dataset and CDnet category, the gate loses to periodic in EVERY one of "
    "the 11 groups (gap −0.27 to −1.0). The categories where the cue is below chance at onsets "
    "(thermal, intermittent-object-motion) collapse the gate to 0. This rules out ‘pooling artifact’: "
    "the sign of the gap is invariant, exactly as Theorem 1′ predicts category by category.",
    "Tách kết quả theo bộ dữ liệu và category CDnet, gate THUA periodic ở MỖI một trong 11 nhóm (chênh "
    "−0.27 đến −1.0). Các category nơi cue dưới mức ngẫu nhiên ở onset (thermal, intermittent-object-"
    "motion) làm gate sụp về 0. Điều này loại bỏ ‘ảo do gộp’: dấu của chênh bất biến, đúng như Định lý "
    "1′ dự báo theo từng category.")
print("chunk5 ok:", len(doc.paragraphs))

# ===================== 11. DISCUSSION =====================
H("11. Big-picture lessons", 1)
PED("Why periodic wins (the unifying view)", "Vì sao periodic thắng (góc nhìn thống nhất)",
    "The single mechanism behind every experiment: deployment cues track MOTION, not event ONSETS. A "
    "motion-following gate therefore spends budget on motion-rich but event-poor frames—shaking "
    "branches, lighting changes—which cross-scene is indistinguishable from adversarial advice. "
    "Periodic wins not because it is clever but because it is OBLIVIOUS: its coverage beta cannot be "
    "steered by the scene. HEDGE-GATE keeps that un-steerable floor and treats the cue as a bonus.",
    "Cơ chế duy nhất đứng sau mọi thí nghiệm: cue triển khai bám CHUYỂN ĐỘNG, không phải ONSET sự kiện. "
    "Một gate bám chuyển động vì thế tiêu ngân sách vào khung nhiều chuyển động mà ít sự kiện—cành cây "
    "lay, ánh sáng đổi—mà cross-scene thì không phân biệt được với dự đoán đối kháng. Periodic thắng "
    "không vì khôn mà vì nó MÙ: độ phủ beta không bị cảnh lái được. HEDGE-GATE giữ cái sàn không-lái-"
    "được đó và coi cue như phần thưởng.")
PED("When DOES content-awareness help?", "Khi nào nhận-biết-nội-dung CÓ ích?",
    "This is not a blanket ‘never adapt’. The cue arm strictly improves recall when eta is genuinely "
    "small—a same-scene calibrated cue, or a strong learned predictor whose onset-AUC stays well above "
    "0.5 at deployment. The theorem just insists the benefit be EARNED: keep the floor, scale trust by "
    "the measured eta, and (with feedback) verify before committing. Content-awareness helps exactly "
    "when the cue's held-out onset-AUC materially beats chance.",
    "Đây không phải ‘đừng bao giờ thích nghi’. Nhánh cue cải thiện recall khi eta thật sự nhỏ—một cue "
    "hiệu chỉnh cùng cảnh, hoặc một bộ dự đoán học mạnh có onset-AUC vẫn cao hơn hẳn 0.5 khi triển "
    "khai. Định lý chỉ đòi lợi ích phải ĐƯỢC KIẾM: giữ sàn, chia tin cậy theo eta đo được, và (có phản "
    "hồi) kiểm chứng trước khi cam kết. Nhận-biết-nội-dung có ích đúng khi onset-AUC trên tập giữ-lại "
    "của cue vượt hẳn mức ngẫu nhiên.")
PED("beta as a ‘coverage capacity’", "beta như ‘dung lượng phủ’",
    "It helps to read beta = min(1,(W+1)/K) as the CAPACITY of the blind channel: with one fire every K "
    "frames and tolerance W+1, a fraction (W+1)/K of arbitrary onsets is unavoidably covered. The cue "
    "arm tries to raise effective capacity by aiming; Theorem 1′ says aiming is worthless under bad "
    "advice, and Theorem 2 says under good advice the gain is bounded by exactly how much budget was "
    "diverted from the guaranteed channel.",
    "Nên đọc beta = min(1,(W+1)/K) như DUNG LƯỢNG của kênh mù: với một lần bắn mỗi K khung và dung sai "
    "W+1, một tỉ lệ (W+1)/K số onset bất kỳ bị phủ không thể tránh. Nhánh cue cố nâng dung lượng hiệu "
    "dụng bằng cách nhắm; Định lý 1′ nói nhắm là vô ích khi dự đoán dở, và Định lý 2 nói khi dự đoán "
    "tốt, lợi ích bị chặn bởi đúng phần ngân sách rút khỏi kênh bảo đảm.")
PED("A bias–variance reading", "Một cách đọc bias–variance",
    "HEDGE-GATE's dial is a bias–variance knob. The blind backbone is high-bias, zero-variance (ignores "
    "data, cannot be fooled); the cue arm is low-bias, high-variance (follows the cue, but the cue is "
    "scene-dependent). The optimal mix is the straight-line trade-off of Theorem 2, and lambda·beta is "
    "the variance floor the operator refuses to cross. Feedback (Prop. 5) is the explore–exploit "
    "analogue: a cheap probe reduces variance about the cue's reliability before committing.",
    "Núm của HEDGE-GATE là một nút bias–variance. Backbone mù là bias-cao, variance-0 (bỏ qua dữ liệu, "
    "không bị lừa); nhánh cue là bias-thấp, variance-cao (bám cue, nhưng cue phụ thuộc cảnh). Hỗn hợp "
    "tối ưu là đánh đổi đường thẳng của Định lý 2, và lambda·beta là sàn variance người vận hành không "
    "vượt. Phản hồi (Định lý 5) là tương tự explore–exploit: một lần thử rẻ giảm variance về độ tin cậy "
    "của cue trước khi cam kết.")
PED("Threats to validity (being honest)", "Đe dọa tính hợp lệ (trung thực)",
    "Three caveats. (i) Cue choice: we use a generic motion cue; a stronger learned cue could raise the "
    "onset-AUC, but P-A found even learned gates in the same 0.49–0.79 band, and Theorem 1′ caps any "
    "non-adaptive cue at high eta. (ii) Frame cap: capping CDnet at 300 ROI-aligned frames trades a bit "
    "of statistical power for tractability; the gap signs and beta-floor are cap-independent. (iii) "
    "Detector idealization: recall assumes a reliable detector on fired frames; a uniform miss-rate "
    "scales all policies equally, leaving the comparison intact.",
    "Ba điểm dè dặt. (i) Chọn cue: ta dùng cue chuyển động chung; một cue học mạnh hơn có thể nâng "
    "onset-AUC, nhưng P-A thấy ngay cả gate học cũng ở dải 0.49–0.79, và Định lý 1′ chặn mọi cue không-"
    "thích-nghi ở eta cao. (ii) Cắt frame: cắt CDnet ở 300 khung ROI đổi một chút năng lực thống kê lấy "
    "tính khả thi; dấu chênh và sàn-beta không phụ thuộc việc cắt. (iii) Lý tưởng hóa detector: recall "
    "giả định detector tin cậy trên khung được bắn; tỉ lệ trượt đồng đều co giãn mọi chính sách như "
    "nhau, giữ nguyên so sánh.")
PED("Beyond video", "Ngoài video",
    "Nothing is specific to vision. The same frontier governs any WINDOWED-RECALL SAMPLING problem: a "
    "limited number of probes must each catch a short-lived event within a tolerance window, guided by "
    "an untrusted score. Examples: packet sampling for network anomalies, periodic health-sensor "
    "polling, active monitoring of rare faults. In each, beta is the oblivious floor and Theorem 2 "
    "bounds what trust can buy.",
    "Không có gì riêng cho thị giác. Cùng biên đó chi phối mọi bài toán LẤY MẪU THU-HỒI-THEO-CỬA-SỔ: "
    "một số hữu hạn lần thử phải bắt mỗi sự kiện ngắn ngủi trong một cửa sổ dung sai, dẫn dắt bởi một "
    "điểm số không đáng tin. Ví dụ: lấy mẫu gói tin cho bất thường mạng, thăm dò cảm biến sức khỏe định "
    "kỳ, giám sát chủ động lỗi hiếm. Mỗi nơi, beta là sàn mù và Định lý 2 chặn điều mà tin cậy mua được.")
print("chunk6 ok:", len(doc.paragraphs))

# ===================== 12. CASE STUDY =====================
H("12. A worked deployment case study", 1)
PED("Scenario", "Kịch bản",
    "A solar-powered trail camera at 30 fps can afford the detector on 10% of frames (a = 0.1, K = 10) "
    "and tolerates 0.23 s staleness (W = 7). So beta = 0.8. The operator must guarantee recall rho = "
    "0.6 across unknown scenes. Decision: lambda = rho/beta = 0.6/0.8 = 0.75. Three-quarters of the "
    "budget runs a blind backbone (one fire every ceil(1/(0.75·0.1)) ≈ 14 frames); the rest follows the "
    "motion cue.",
    "Một camera mòi chạy pin mặt trời 30 fps có thể chạy detector trên 10% khung (a = 0.1, K = 10) và "
    "chịu trễ 0.23 s (W = 7). Nên beta = 0.8. Người vận hành phải bảo đảm recall rho = 0.6 trên các "
    "cảnh chưa biết. Quyết định: lambda = rho/beta = 0.6/0.8 = 0.75. Ba phần tư ngân sách chạy backbone "
    "mù (một lần bắn mỗi ceil(1/(0.75·0.1)) ≈ 14 khung); phần còn lại theo cue chuyển động.")
PED("Outcomes", "Kết quả",
    "If the cue turns out informative in some scene (eta small), recall rises toward the oracle for "
    "free. If the cue is the near-chance one we measured (eta ≈ 0.5), recall stays near the 0.6 "
    "guarantee. Even an adversarial cue cannot push recall below lambda·beta = 0.6. A pure gate "
    "(lambda=0) would give no guarantee and, at the measured eta, deliver only ≈0.36. The decision in "
    "three numbers: compute beta, measure eta, set lambda = rho/beta.",
    "Nếu cue hóa ra hữu ích ở vài cảnh (eta nhỏ), recall tăng về phía oracle miễn phí. Nếu cue là loại "
    "gần ngẫu nhiên ta đo (eta ≈ 0.5), recall giữ gần bảo đảm 0.6. Ngay cả cue đối kháng cũng không kéo "
    "recall xuống dưới lambda·beta = 0.6. Gate thuần (lambda=0) không cho bảo đảm và, ở eta đo được, "
    "chỉ đạt ≈0.36. Quyết định trong ba con số: tính beta, đo eta, đặt lambda = rho/beta.")

# ===================== 13. FAQ =====================
H("13. Frequently asked questions (for students)", 1)
PED("Q: Isn't a smart model always better than a dumb loop?",
    "H: Mô hình thông minh chẳng phải luôn tốt hơn vòng lặp ngốc nghếch sao?",
    "Not for worst-case guarantees. A smart model that COMMITS to a possibly-wrong signal inherits that "
    "signal's worst case. The dumb loop's strength is that nothing in the scene can mislead it. The "
    "paper makes this precise and shows the loop's guarantee beta is exactly what the smart gate "
    "forfeits when it over-trusts.",
    "Không, về bảo đảm xấu nhất. Một mô hình thông minh CAM KẾT theo một tín hiệu có thể sai sẽ thừa "
    "hưởng trường hợp xấu nhất của tín hiệu đó. Sức mạnh của vòng lặp ngốc là không gì trong cảnh đánh "
    "lừa được nó. Bài báo làm rõ điều này và cho thấy bảo đảm beta của vòng lặp đúng là cái mà gate "
    "thông minh đánh mất khi tin quá mức.")
PED("Q: Why measure the cue by ‘onset-AUC’ and not accuracy?",
    "H: Vì sao đo cue bằng ‘onset-AUC’ chứ không phải độ chính xác?",
    "Because what matters is whether the cue ranks ONSET frames above non-onset frames—that is exactly "
    "AUC at onsets. A cue can be a great motion detector (high motion-AUC) yet useless at onsets (AUC "
    "≈0.5), which is precisely the deployed failure. eta = 1 − onset-AUC captures the relevant error in "
    "one number you can compute on a validation set.",
    "Vì điều quan trọng là cue có xếp khung ONSET cao hơn khung không-onset hay không—đó đúng là AUC ở "
    "onset. Một cue có thể là máy dò chuyển động giỏi (motion-AUC cao) nhưng vô dụng ở onset (AUC "
    "≈0.5), chính là kiểu hỏng khi triển khai. eta = 1 − onset-AUC tóm sai số liên quan trong một con "
    "số bạn tính được trên tập kiểm.")
PED("Q: Is the impossibility the end of adaptive scheduling?",
    "H: Bất khả thi có phải dấu chấm hết cho lập lịch thích nghi?",
    "No—Proposition 5 shows that USING detector feedback escapes it. The open research question is the "
    "optimal feedback-aware gate. The impossibility is specifically about gates that decide from the "
    "cue alone, which is what is deployed today.",
    "Không—Định lý 5 cho thấy DÙNG phản hồi detector thoát được. Câu hỏi nghiên cứu mở là gate có-phản-"
    "hồi tối ưu. Bất khả thi chỉ về gate quyết định từ cue đơn thuần, vốn là cái đang triển khai hôm "
    "nay.")

# ===================== 14. GLOSSARY =====================
H("14. Glossary / Thuật ngữ", 1)
gl = [("Term", "Meaning (EN)", "Ý nghĩa (VI)"),
 ("activation a", "fraction of frames the detector runs on", "tỉ lệ khung detector chạy"),
 ("staleness window W", "an event is recalled if a fire lands within W of its onset",
  "sự kiện được thu hồi nếu có lần bắn trong W kể từ onset"),
 ("recall R", "fraction of events caught in time", "tỉ lệ sự kiện bắt kịp"),
 ("oblivious / blind", "decides without looking at the cue", "quyết định mà không nhìn cue"),
 ("periodic schedule", "fire every K-th frame (a metronome)", "bắn mỗi K khung (máy đếm nhịp)"),
 ("beta", "the guaranteed oblivious recall, min(1,(W+1)/K)", "recall mù bảo đảm, min(1,(W+1)/K)"),
 ("cue", "a cheap per-frame signal (e.g. motion)", "tín hiệu rẻ mỗi khung (vd. chuyển động)"),
 ("prediction error eta", "1 − onset-AUC of the cue, in [0,1]", "1 − onset-AUC của cue, trong [0,1]"),
 ("consistency c", "recall when the cue is perfect (eta=0)", "recall khi cue hoàn hảo (eta=0)"),
 ("robustness r", "recall when the cue is adversarial (eta=1)", "recall khi cue đối kháng (eta=1)"),
 ("non-adaptive gate", "fires from the cue only, ignoring feedback", "bắn chỉ từ cue, bỏ qua phản hồi"),
 ("HEDGE-GATE(lambda)", "split budget: lambda blind backbone + (1−lambda) cue", "chia ngân sách: lambda backbone mù + (1−lambda) cue"),
 ("Pareto-optimal", "no policy is better on both c and r at once", "không chính sách nào tốt hơn đồng thời cả c và r")]
tb = doc.add_table(rows=len(gl), cols=3); tb.style = "Light Grid Accent 1"
for i, (a, e, v) in enumerate(gl):
    c0=tb.rows[i].cells[0]; c0.text=""; rr=c0.paragraphs[0].add_run(a)
    c1=tb.rows[i].cells[1]; c1.text=""; c1.paragraphs[0].add_run(e)
    c2=tb.rows[i].cells[2]; c2.text=""; rv=c2.paragraphs[0].add_run(v); rv.italic=True
    if i==0: rr.bold=True

# ===================== 15. SUMMARY =====================
H("15. One-page summary / Tóm tắt một trang", 1)
PED("The five things to remember", "Năm điều cần nhớ",
    "(1) A blind periodic schedule already guarantees recall beta = min(1,(W+1)/K)—free, un-foolable. "
    "(2) A gate that fully trusts a cue can be driven to recall 0 by an adversarial cue (Theorem 1′). "
    "(3) For deployed (non-adaptive) gates the consistency–robustness trade-off is a TIGHT line r = "
    "beta(1−c)/(1−beta) (Theorem 2), achieved by HEDGE-GATE. (4) HEDGE-GATE’s dial lambda is a "
    "certificate: lambda·beta is the guaranteed floor; set lambda = rho/beta for a target rho. (5) "
    "Real motion cues sit at eta≈0.5, the regime where the theory forbids gains—so periodic wins, and "
    "the honest advice is ‘keep a floor, trust the cue only as much as its measured onset-AUC earns, "
    "and verify with feedback before committing.’",
    "(1) Một lịch tuần hoàn mù đã bảo đảm recall beta = min(1,(W+1)/K)—miễn phí, không lừa được. (2) "
    "Một gate tin tuyệt đối vào cue có thể bị kéo về recall 0 bởi cue đối kháng (Định lý 1′). (3) Với "
    "gate triển khai (không-thích-nghi), đánh đổi consistency–robustness là một đường CHẶT r = "
    "beta(1−c)/(1−beta) (Định lý 2), đạt bởi HEDGE-GATE. (4) Núm lambda của HEDGE-GATE là một chứng "
    "chỉ: lambda·beta là sàn bảo đảm; đặt lambda = rho/beta cho mục tiêu rho. (5) Cue chuyển động thật "
    "ở eta≈0.5, vùng mà lý thuyết cấm hưởng lợi—nên periodic thắng, và lời khuyên trung thực là ‘giữ "
    "một sàn, chỉ tin cue đúng mức onset-AUC đo được cho phép, và kiểm chứng bằng phản hồi trước khi "
    "cam kết.’")
PED("Status of the results (honest labels)", "Trạng thái kết quả (nhãn trung thực)",
    "PROVEN: Lemma 1; Theorem 1; Theorem 1′; Theorem 2 & 3 (for non-adaptive gates); Proposition 5. "
    "PROVEN UNDER ASSUMPTION: Proposition 4 (per-event independence). OPEN / future work: the optimal "
    "feedback-aware adaptive gate. All proofs should be independently verified before publication.",
    "ĐÃ CHỨNG MINH: Bổ đề 1; Định lý 1; Định lý 1′; Định lý 2 & 3 (cho gate không-thích-nghi); Định lý "
    "5. CHỨNG MINH DƯỚI GIẢ THIẾT: Mệnh đề 4 (độc lập theo sự kiện). MỞ / tương lai: gate thích nghi "
    "có-phản-hồi tối ưu. Mọi chứng minh nên được kiểm chứng độc lập trước khi công bố.")
doc.save(OUT)
print("FINAL saved:", OUT, "| paragraphs:", len(doc.paragraphs), "tables:", len(doc.tables), "images:", len(doc.inline_shapes))

# ===================== APPENDIX A: PRIMER =====================
H("Appendix A — Background primer for students", 1)
PED("A1. What is an ‘online’ algorithm?", "A1. Thuật toán ‘trực tuyến’ là gì?",
    "An OFFLINE algorithm sees the whole input first, then decides; an ONLINE algorithm must decide step "
    "by step, as data arrives, WITHOUT knowing the future. Our scheduler is online: at each frame it "
    "must choose to fire or not, before seeing whether an event is coming. We judge an online algorithm "
    "by its COMPETITIVE RATIO—its performance compared to the best offline (oracle) choice. Here the "
    "oracle, knowing all onsets, gets recall 1; the question is how close an online gate can stay.",
    "Thuật toán NGOẠI TUYẾN thấy toàn bộ đầu vào rồi mới quyết; thuật toán TRỰC TUYẾN phải quyết từng "
    "bước khi dữ liệu đến, KHÔNG biết tương lai. Bộ lập lịch của ta là trực tuyến: mỗi khung phải chọn "
    "bắn hay không, trước khi biết có sự kiện sắp tới. Ta đánh giá thuật toán trực tuyến bằng TỈ SỐ "
    "CẠNH TRANH—hiệu năng so với lựa chọn ngoại tuyến (oracle) tốt nhất. Ở đây oracle biết mọi onset "
    "đạt recall 1; câu hỏi là gate trực tuyến giữ gần đến đâu.")
PED("A2. Algorithms with predictions (AwP)", "A2. Thuật toán với dự đoán (AwP)",
    "Classic online analysis is pessimistic—it assumes the worst future. AwP softens this: you are "
    "given a PREDICTION (here, the motion cue) that may be right or wrong, and you want two things. "
    "CONSISTENCY: be near-optimal when the prediction is correct. ROBUSTNESS: keep a worst-case "
    "guarantee when the prediction is arbitrary. The art is combining a ‘trust the prediction’ policy "
    "with a ‘safe baseline’ so you get both. HEDGE-GATE is exactly this combination for frame "
    "scheduling.",
    "Phân tích trực tuyến cổ điển bi quan—giả định tương lai xấu nhất. AwP làm dịu điều này: bạn được "
    "cho một DỰ ĐOÁN (ở đây là cue chuyển động) có thể đúng hoặc sai, và bạn muốn hai điều. CONSISTENCY: "
    "gần tối ưu khi dự đoán đúng. ROBUSTNESS: giữ bảo đảm xấu-nhất khi dự đoán tùy ý. Nghệ thuật là kết "
    "hợp chính sách ‘tin dự đoán’ với ‘nền an toàn’ để có cả hai. HEDGE-GATE đúng là kết hợp này cho "
    "lập lịch khung hình.")
PED("A3. ROC, AUC, and ‘onset-AUC’", "A3. ROC, AUC và ‘onset-AUC’",
    "AUC (area under the ROC curve) measures how well a score separates two classes. Concretely, AUC = "
    "the probability that a random POSITIVE example scores higher than a random NEGATIVE one. AUC = 1 is "
    "perfect, AUC = 0.5 is a coin flip. Onset-AUC applies this with POSITIVE = onset-window frames, "
    "NEGATIVE = other frames: it asks ‘does the cue rank event-onset frames above the rest?’ Tiny "
    "example: cue values [onset: 0.9, 0.8; non-onset: 0.4, 0.7]. Pairs (0.9>0.4,0.9>0.7,0.8>0.4,0.8>0.7) "
    "= 4/4 → AUC = 1; if instead non-onsets were [0.95,0.85], AUC drops toward 0. Then eta = 1 − AUC.",
    "AUC (diện tích dưới đường ROC) đo mức một điểm số phân tách hai lớp. Cụ thể, AUC = xác suất một ví "
    "dụ DƯƠNG ngẫu nhiên có điểm cao hơn một ví dụ ÂM ngẫu nhiên. AUC = 1 hoàn hảo, AUC = 0.5 là tung "
    "đồng xu. Onset-AUC áp dụng với DƯƠNG = khung trong cửa sổ onset, ÂM = khung khác: nó hỏi ‘cue có "
    "xếp khung onset cao hơn phần còn lại không?’ Ví dụ nhỏ: điểm cue [onset: 0.9, 0.8; không-onset: "
    "0.4, 0.7]. Các cặp (0.9>0.4,0.9>0.7,0.8>0.4,0.8>0.7) = 4/4 → AUC = 1; nếu không-onset là "
    "[0.95,0.85] thì AUC giảm về 0. Khi đó eta = 1 − AUC.")
PED("A4. Bootstrap confidence intervals", "A4. Khoảng tin cậy bootstrap",
    "How sure are we that ‘gate 0.36 vs periodic 0.78’ is real, not luck? VIDEO-LEVEL BOOTSTRAP: "
    "resample the set of videos WITH REPLACEMENT many times (B = 1000), recompute the mean each time, "
    "and take the 2.5th–97.5th percentiles as a 95% confidence interval. If the gate's interval and the "
    "periodic's interval do NOT overlap, the difference is statistically solid. Resampling whole videos "
    "(not frames) accounts for the big source of variability—differences BETWEEN scenes.",
    "Làm sao chắc ‘gate 0.36 vs periodic 0.78’ là thật, không phải may? BOOTSTRAP CẤP VIDEO: lấy lại "
    "mẫu tập video CÓ HOÀN LẠI nhiều lần (B = 1000), tính lại trung bình mỗi lần, và lấy phân vị "
    "2.5–97.5 làm khoảng tin cậy 95%. Nếu khoảng của gate và khoảng của periodic KHÔNG chồng lấp, khác "
    "biệt là vững về mặt thống kê. Lấy lại mẫu cả video (không phải khung) tính đến nguồn biến thiên "
    "lớn—khác biệt GIỮA các cảnh.")
PED("A5. What does ‘adversary’ mean in a proof?", "A5. ‘Kẻ đối kháng’ trong chứng minh nghĩa là gì?",
    "To prove a WORST-CASE bound we imagine an adversary who, knowing our policy, picks the hardest "
    "input. For lower bounds we often use Yao's idea: instead of one hard input, the adversary picks a "
    "DISTRIBUTION over inputs that is hard ON AVERAGE for any deterministic policy. In our converse, the "
    "‘hardest input’ is simply: hide the events wherever the gate is least likely to fire. The "
    "adversary is a proof device, not a real attacker—it certifies that NO policy can do better.",
    "Để chứng minh một cận XẤU-NHẤT, ta tưởng tượng một kẻ đối kháng biết chính sách của ta và chọn đầu "
    "vào khó nhất. Với cận dưới, ta thường dùng ý của Yao: thay vì một đầu vào khó, kẻ đối kháng chọn "
    "một PHÂN PHỐI trên đầu vào khó TRUNG BÌNH cho mọi chính sách tất định. Trong phần đảo của ta, ‘đầu "
    "vào khó nhất’ chỉ là: giấu sự kiện ở nơi gate ít khả năng bắn nhất. Kẻ đối kháng là công cụ chứng "
    "minh, không phải kẻ tấn công thật—nó chứng nhận KHÔNG chính sách nào tốt hơn được.")
print("chunkA ok:", len(doc.paragraphs))

# ===================== APPENDIX B: DEEP DIVES =====================
H("Appendix B — Deep dives and common misconceptions", 1)
PED("B1. Deriving HEDGE-GATE's (c, r) step by step", "B1. Suy ra (c, r) của HEDGE-GATE từng bước",
    "Set up the worst-case block family: N blocks, one event per event-block, budget B = m, the cue "
    "marks B blocks (set A). HEDGE-GATE fires lambda·B blind backbone + (1−lambda)·B on A. CONSISTENCY "
    "(eta=0, events = A): the advice arm covers (1−lambda)·B of the B events directly; the backbone, "
    "being blind, additionally covers a beta-fraction of the remaining lambda·B events; so c = "
    "(1−lambda) + lambda·beta. ROBUSTNESS (eta=1, events NOT in A): the advice arm fires inside empty A "
    "and catches nothing; only the backbone helps, covering a beta-fraction of all events; so r = "
    "lambda·beta. Now eliminate lambda: from r = lambda·beta we get lambda = r/beta; plug into c: c = "
    "1 − (r/beta)(1−beta), i.e. r = beta(1−c)/(1−beta). That is the frontier.",
    "Dựng họ khối xấu nhất: N khối, mỗi khối-sự-kiện một sự kiện, ngân sách B = m, cue đánh dấu B khối "
    "(tập A). HEDGE-GATE bắn lambda·B backbone mù + (1−lambda)·B vào A. CONSISTENCY (eta=0, sự kiện = "
    "A): nhánh dự đoán phủ trực tiếp (1−lambda)·B trong B sự kiện; backbone mù phủ thêm beta phần của "
    "lambda·B sự kiện còn lại; nên c = (1−lambda) + lambda·beta. ROBUSTNESS (eta=1, sự kiện KHÔNG ở A): "
    "nhánh dự đoán bắn trong A rỗng, bắt 0; chỉ backbone giúp, phủ beta phần của mọi sự kiện; nên r = "
    "lambda·beta. Giờ khử lambda: từ r = lambda·beta ta có lambda = r/beta; thay vào c: c = 1 − "
    "(r/beta)(1−beta), tức r = beta(1−c)/(1−beta). Đó là biên.")
PED("B2. Why ‘disjoint blocks’ is the hardest case", "B2. Vì sao ‘khối rời’ là trường hợp khó nhất",
    "You might worry the proof cheats by using a special structure. It does the opposite: disjoint "
    "windows are the WORST case for the scheduler, so a bound proved there holds more generally. Why? "
    "Overlapping windows only HELP—a single fire can then cover several onsets at once. Clustering "
    "events only REDUCES the number of distinct targets to cover. So if the gate already fails on clean "
    "disjoint blocks, it fails at least as badly on messier real overlapping events. Empirically, "
    "relaxing disjointness leaves the gap signs unchanged.",
    "Bạn có thể lo chứng minh ‘gian lận’ bằng cấu trúc đặc biệt. Thực ra ngược lại: cửa sổ rời là "
    "trường hợp XẤU NHẤT cho bộ lập lịch, nên cận chứng minh ở đó còn đúng tổng quát hơn. Vì sao? Cửa "
    "sổ chồng lấp chỉ GIÚP—một lần bắn khi đó phủ được nhiều onset cùng lúc. Cụm sự kiện chỉ GIẢM số "
    "mục tiêu phân biệt cần phủ. Nên nếu gate đã hỏng trên khối rời sạch sẽ, nó hỏng ít nhất tệ bằng "
    "trên sự kiện thật chồng lấp lộn xộn. Thực nghiệm: bỏ tính rời không đổi dấu chênh.")
PED("B3. Misconception: ‘more model capacity will fix the gate’",
    "B3. Hiểu lầm: ‘thêm năng lực mô hình sẽ sửa được gate’",
    "Random-forest and DQN gates have more capacity than a threshold, yet they lose just as much (RF "
    "20% at a=0.2; DQN cannot even reach low activation). Theorem 2 caps ALL non-adaptive gates by the "
    "SAME frontier. A bigger model only fits the within-scene cue more tightly; it cannot manufacture "
    "onset information the cue does not carry cross-scene. Capacity is not the bottleneck—the cue's "
    "onset-AUC is.",
    "Gate random-forest và DQN có năng lực hơn ngưỡng, vẫn thua như nhau (RF 20% tại a=0.2; DQN còn "
    "không đạt nổi kích hoạt thấp). Định lý 2 chặn MỌI gate không-thích-nghi bởi CÙNG một biên. Mô hình "
    "lớn hơn chỉ khớp cue trong-cảnh chặt hơn; nó không tạo ra được thông tin onset mà cue không mang "
    "cross-scene. Năng lực không phải nút thắt—onset-AUC của cue mới là.")
PED("B4. Misconception: ‘eta=1 means the cue is random’",
    "B4. Hiểu lầm: ‘eta=1 nghĩa là cue ngẫu nhiên’",
    "Subtle but important. A RANDOM cue (AUC=0.5, eta=0.5) drags recall down only to the random level "
    "beta—bad but bounded. An ADVERSARIAL cue (anti-correlated with onsets, pushing AUC below 0.5 "
    "toward eta=1) is worse: it actively steers fires AWAY from events, dropping recall BELOW beta "
    "toward 0. Deployed cross-scene cues behave adversarially because they chase motion that is "
    "anti-correlated with onsets in the new scene—hence the sub-beta collapse.",
    "Tinh tế nhưng quan trọng. Một cue NGẪU NHIÊN (AUC=0.5, eta=0.5) chỉ kéo recall xuống mức ngẫu "
    "nhiên beta—xấu nhưng có chặn. Một cue ĐỐI KHÁNG (phản tương quan với onset, đẩy AUC xuống dưới 0.5 "
    "về eta=1) tệ hơn: nó chủ động lái các lần bắn TRÁNH XA sự kiện, kéo recall DƯỚI beta về 0. Cue "
    "triển khai cross-scene hành xử đối kháng vì chúng đuổi theo chuyển động phản tương quan với onset "
    "ở cảnh mới—do đó sụp dưới-beta.")

# ===================== APPENDIX C: EXERCISES =====================
H("Appendix C — Exercises (with solutions)", 1)
PED("Exercise 1", "Bài tập 1",
    "W = 4 and a = 0.1. Compute K and beta. Is the consistency–robustness trade-off active? SOLUTION: "
    "K = ceil(1/0.1) = 10; beta = min(1,(W+1)/K) = min(1,5/10) = 0.5 < 1, so YES, the trade-off is "
    "active (the cue could help or hurt).",
    "W = 4 và a = 0.1. Tính K và beta. Đánh đổi consistency–robustness có hoạt động không? LỜI GIẢI: K "
    "= ceil(1/0.1) = 10; beta = min(1,(W+1)/K) = min(1,5/10) = 0.5 < 1, nên CÓ, đánh đổi hoạt động (cue "
    "có thể giúp hoặc hại).")
PED("Exercise 2", "Bài tập 2",
    "With beta = 0.5, you need a worst-case recall guarantee rho = 0.4. What lambda do you pick, and "
    "what consistency do you sacrifice? SOLUTION: lambda = rho/beta = 0.4/0.5 = 0.8. Then c = (1−0.8) + "
    "0.8·0.5 = 0.2 + 0.4 = 0.6, and r = 0.8·0.5 = 0.4 = rho. You spend 80% on the backbone to secure the "
    "0.4 floor, leaving 20% to chase the cue.",
    "Với beta = 0.5, bạn cần bảo đảm recall xấu-nhất rho = 0.4. Chọn lambda nào, và hi sinh consistency "
    "bao nhiêu? LỜI GIẢI: lambda = rho/beta = 0.4/0.5 = 0.8. Khi đó c = (1−0.8) + 0.8·0.5 = 0.2 + 0.4 = "
    "0.6, và r = 0.8·0.5 = 0.4 = rho. Bạn tiêu 80% cho backbone để giữ sàn 0.4, còn 20% đuổi theo cue.")
PED("Exercise 3", "Bài tập 3",
    "A cue has onset-AUC 0.95 on held-out scenes. Should you set lambda small or large, and why? "
    "SOLUTION: eta = 1 − 0.95 = 0.05 is small, so the cue is trustworthy; set lambda small (mostly trust "
    "the cue) to harvest its upside, keeping just enough backbone for your required floor. If instead "
    "AUC were 0.52 (eta ≈ 0.48), set lambda ≈ 1 (mostly periodic).",
    "Một cue có onset-AUC 0.95 trên cảnh giữ-lại. Nên đặt lambda nhỏ hay lớn, vì sao? LỜI GIẢI: eta = 1 "
    "− 0.95 = 0.05 nhỏ, nên cue đáng tin; đặt lambda nhỏ (chủ yếu tin cue) để hưởng upside, chỉ giữ đủ "
    "backbone cho sàn yêu cầu. Nếu thay vào đó AUC = 0.52 (eta ≈ 0.48), đặt lambda ≈ 1 (chủ yếu "
    "periodic).")
PED("Exercise 4", "Bài tập 4",
    "Explain in one sentence why a DQN gate with huge capacity still loses to periodic cross-scene. "
    "SOLUTION: because it is non-adaptive, Theorem 2 caps it by the same frontier r = beta(1−c)/(1−beta), "
    "and its extra capacity only overfits the within-scene cue—it cannot create onset information the "
    "cue lacks in a new scene.",
    "Giải thích trong một câu vì sao gate DQN năng lực khổng lồ vẫn thua periodic cross-scene. LỜI GIẢI: "
    "vì nó không-thích-nghi, Định lý 2 chặn nó bởi cùng biên r = beta(1−c)/(1−beta), và năng lực thừa "
    "chỉ overfit cue trong-cảnh—không tạo được thông tin onset mà cue thiếu ở cảnh mới.")
PED("Exercise 5 (harder)", "Bài tập 5 (khó hơn)",
    "Show that at a = 0.2, W = 7, the trade-off vanishes. SOLUTION: K = ceil(1/0.2) = 5; beta = "
    "min(1,8/5) = min(1,1.6) = 1. Since beta = 1, the frontier r ≤ beta(1−c)/(1−beta) is undefined "
    "(1−beta = 0) — operationally, periodic already achieves recall 1, so both consistency and "
    "robustness can be 1 at once; you may give the whole budget to the cue with no robustness loss.",
    "Chứng tỏ tại a = 0.2, W = 7, đánh đổi biến mất. LỜI GIẢI: K = ceil(1/0.2) = 5; beta = min(1,8/5) = "
    "min(1,1.6) = 1. Vì beta = 1, biên r ≤ beta(1−c)/(1−beta) không xác định (1−beta = 0) — về vận "
    "hành, periodic đã đạt recall 1, nên cả consistency và robustness đều có thể = 1 cùng lúc; bạn có "
    "thể dồn toàn bộ ngân sách cho cue mà không mất robustness.")

H("End of report / Hết báo cáo", 1)
PED("Closing", "Kết",
    "If you remember one image: a metronome (periodic) versus a nervous photographer chasing every "
    "movement (the cue). On a stage full of distractions, the metronome reliably catches the actors' "
    "entrances; the photographer, fooled by the distractions, misses them. The wise director keeps the "
    "metronome running and lets the photographer add shots only when proven trustworthy.",
    "Nếu nhớ một hình ảnh: một máy đếm nhịp (periodic) so với một nhiếp ảnh gia bồn chồn đuổi theo mọi "
    "chuyển động (cue). Trên một sân khấu đầy thứ gây nhiễu, máy đếm nhịp bắt được lúc diễn viên ra "
    "đáng tin cậy; nhiếp ảnh gia bị nhiễu đánh lừa, bỏ lỡ. Đạo diễn khôn ngoan giữ máy đếm nhịp chạy và "
    "chỉ để nhiếp ảnh gia thêm ảnh khi đã chứng minh đáng tin.")
doc.save(OUT)
print("FINAL v2 saved | paragraphs:", len(doc.paragraphs), "tables:", len(doc.tables), "images:", len(doc.inline_shapes))

# ===================== APPENDIX D: RUNNING STORY =====================
H("Appendix D — One numeric story through the whole theory", 1)
PED("The setting we will carry throughout", "Bối cảnh xuyên suốt",
    "Fix one concrete world and trace every result through it. Horizon T = 1000 frames. Budget a = 0.1, "
    "so B = 100 detector fires and K = ceil(1/0.1) = 10. Window W = 7, so a window is W+1 = 8 frames "
    "long and beta = min(1,(W+1)/K) = min(1,8/10) = 0.8. There are m = 20 events. Keep these numbers in "
    "mind: B=100, K=10, beta=0.8, m=20.",
    "Cố định một thế giới cụ thể và truy mọi kết quả qua nó. Độ dài T = 1000 khung. Ngân sách a = 0.1, "
    "nên B = 100 lần bắn detector và K = ceil(1/0.1) = 10. Cửa sổ W = 7, nên một cửa sổ dài W+1 = 8 "
    "khung và beta = min(1,(W+1)/K) = min(1,8/10) = 0.8. Có m = 20 sự kiện. Nhớ các số: B=100, K=10, "
    "beta=0.8, m=20.")
PED("Step 1 — what periodic alone gives (Lemma 1)", "Bước 1 — periodic đơn độc cho gì (Bổ đề 1)",
    "Fire every K=10 frames. Since K=10 > W+1=8, we are in the ‘slower than the window’ case, so each "
    "event is caught with probability (W+1)/K = 8/10 = 0.8. Expected recall = beta = 0.8: out of 20 "
    "events, about 16 are caught for free, with no cue and no learning.",
    "Bắn mỗi K=10 khung. Vì K=10 > W+1=8, ta ở trường hợp ‘chậm hơn cửa sổ’, nên mỗi sự kiện được bắt "
    "với xác suất (W+1)/K = 8/10 = 0.8. Recall kỳ vọng = beta = 0.8: trong 20 sự kiện, khoảng 16 được "
    "bắt miễn phí, không cue không học.")
PED("Step 2 — a perfectly trusting gate when the cue is good (consistency)",
    "Bước 2 — gate tin tuyệt đối khi cue tốt (consistency)",
    "If the cue is perfect (eta=0), a pure gate (lambda=0) aims all 100 fires at the 20 events and "
    "catches all 20: consistency c = 1.0. Great—when the cue is good, trusting it reaches the oracle.",
    "Nếu cue hoàn hảo (eta=0), gate thuần (lambda=0) nhắm toàn bộ 100 lần bắn vào 20 sự kiện và bắt cả "
    "20: consistency c = 1.0. Tuyệt—khi cue tốt, tin nó đạt oracle.")
PED("Step 3 — the same gate when the cue is adversarial (Theorem 1′)",
    "Bước 3 — cũng gate đó khi cue đối kháng (Định lý 1′)",
    "Now the cue is adversarial (eta=1): it points the 100 fires at 100 clutter frames with no events "
    "(there is room: 1000 − 20·8 = 840 ≥ 100 non-window frames). The pure gate catches 0 of 20: "
    "robustness r = 0. The SAME gate that was perfect in Step 2 is now catastrophic. That is the "
    "consistency–robustness tension made concrete.",
    "Giờ cue đối kháng (eta=1): nó chỉ 100 lần bắn vào 100 khung nhiễu không có sự kiện (có chỗ: 1000 − "
    "20·8 = 840 ≥ 100 khung ngoài cửa sổ). Gate thuần bắt 0 trong 20: robustness r = 0. CÙNG gate hoàn "
    "hảo ở Bước 2 giờ thảm họa. Đó là căng thẳng consistency–robustness được cụ thể hóa.")
PED("Step 4 — HEDGE-GATE hedges (Theorem 2 frontier)", "Bước 4 — HEDGE-GATE phòng hộ (biên Định lý 2)",
    "Choose lambda = 0.5. Backbone = 50 blind fires at rate 0.05 → floor lambda·beta = 0.5·0.8 = 0.4; "
    "advice arm = 50 cue fires. When eta=0: c = (1−0.5) + 0.5·0.8 = 0.9 (catch ~18 of 20). When eta=1: "
    "r = lambda·beta = 0.4 (catch ~8 of 20, GUARANTEED). Check the frontier: beta(1−c)/(1−beta) = "
    "0.8·0.1/0.2 = 0.4 = r. The point (0.9, 0.4) is exactly on the line. We traded 0.1 of consistency "
    "(1.0→0.9) to buy 0.4 of guaranteed robustness (0→0.4).",
    "Chọn lambda = 0.5. Backbone = 50 lần bắn mù nhịp 0.05 → sàn lambda·beta = 0.5·0.8 = 0.4; nhánh dự "
    "đoán = 50 lần bắn theo cue. Khi eta=0: c = (1−0.5) + 0.5·0.8 = 0.9 (bắt ~18/20). Khi eta=1: r = "
    "lambda·beta = 0.4 (bắt ~8/20, BẢO ĐẢM). Kiểm biên: beta(1−c)/(1−beta) = 0.8·0.1/0.2 = 0.4 = r. "
    "Điểm (0.9, 0.4) đúng trên đường. Ta đổi 0.1 consistency (1.0→0.9) lấy 0.4 robustness bảo đảm "
    "(0→0.4).")
PED("Step 5 — smoothness in between (Proposition 4)", "Bước 5 — trơn ở khoảng giữa (Mệnh đề 4)",
    "What if the cue is mediocre, eta=0.5? With lambda=0.5, floor 0.4, and (say) kappa=0.8: E[R] = 0.4 "
    "+ (1−0.5)·0.8·(1−0.4) = 0.4 + 0.24 = 0.64. So as eta goes 0→0.5→1, recall goes 0.88→0.64→0.40 "
    "along a straight line—no cliff. The operator can predict performance from the measured eta.",
    "Nếu cue tầm tầm, eta=0.5? Với lambda=0.5, sàn 0.4, và (giả sử) kappa=0.8: E[R] = 0.4 + "
    "(1−0.5)·0.8·(1−0.4) = 0.4 + 0.24 = 0.64. Vậy khi eta đi 0→0.5→1, recall đi 0.88→0.64→0.40 theo "
    "một đường thẳng—không vách. Người vận hành dự đoán được hiệu năng từ eta đo được.")
PED("Step 6 — if feedback is allowed (Proposition 5)", "Bước 6 — nếu cho phép phản hồi (Định lý 5)",
    "Allow the gate to see detector hits/misses. It probes 1 cue-marked frame. If hit (cue good), keep "
    "trusting → c near 1. If miss (cue bad), revert the other 99 fires to a blind backbone → r ≥ "
    "(1−1/100)·0.8 = 0.792. So with feedback the gate gets c≈1 AND r≈0.79 at once—above the non-adaptive "
    "frontier, which forced r=0 at c=1. The cost was a single wasted probe.",
    "Cho phép gate thấy trúng/trượt của detector. Nó thử 1 khung được cue đánh dấu. Nếu trúng (cue "
    "tốt), tiếp tục tin → c gần 1. Nếu trượt (cue dở), chuyển 99 lần bắn còn lại sang backbone mù → r ≥ "
    "(1−1/100)·0.8 = 0.792. Vậy có phản hồi, gate đạt c≈1 VÀ r≈0.79 cùng lúc—trên biên không-thích-"
    "nghi, vốn ép r=0 tại c=1. Cái giá là một lần thử lãng phí.")
PED("Moral of the story", "Bài học của câu chuyện",
    "Same world, six results, one arc: free floor 0.8 → trust gives 1.0 (good cue) or 0.0 (bad cue) → "
    "hedging buys a guaranteed 0.4 at small consistency cost → smooth in between → feedback gets nearly "
    "the best of both. The numbers are not magic; they all come from beta=0.8 and the budget split "
    "lambda.",
    "Cùng thế giới, sáu kết quả, một mạch: sàn miễn phí 0.8 → tin cho 1.0 (cue tốt) hoặc 0.0 (cue dở) → "
    "phòng hộ mua được 0.4 bảo đảm với chi phí consistency nhỏ → trơn ở giữa → phản hồi đạt gần như tốt "
    "nhất cả hai. Các con số không phải phép màu; tất cả đến từ beta=0.8 và cách chia ngân sách lambda.")

# ===================== APPENDIX E: HOW TO READ FIGURES =====================
H("Appendix E — How to read each figure yourself", 1)
PED("Figures 6–9 (synthetic & anchors)", "Hình 6–9 (synthetic & neo)",
    "x-axis is usually energy (activation a) or prediction error eta; y-axis is recall. The dashed "
    "‘periodic cert’ line is beta; the dotted ‘oblivious random’ line is the random-sampling level. A "
    "curve BELOW the dotted line means the policy is worse than random (adversarial regime). The colored "
    "curves are HEDGE-GATE for different lambda; lambda=1 is flat (advice-independent), lambda=0 dives "
    "under adversarial advice. The shaded vertical band marks the eta of real deployed cues.",
    "Trục x thường là năng lượng (kích hoạt a) hoặc sai số dự đoán eta; trục y là recall. Đường gạch "
    "‘periodic cert’ là beta; đường chấm ‘oblivious random’ là mức lấy mẫu ngẫu nhiên. Một đường DƯỚI "
    "đường chấm nghĩa là chính sách tệ hơn ngẫu nhiên (chế độ đối kháng). Các đường màu là HEDGE-GATE "
    "cho các lambda khác nhau; lambda=1 phẳng (không phụ thuộc dự đoán), lambda=0 lao xuống khi dự đoán "
    "đối kháng. Dải đứng tô bóng đánh dấu eta của cue triển khai thật.")
PED("Figure 10–11 (real data)", "Hình 10–11 (dữ liệu thật)",
    "Figure 10's narrow x-range (eta ≈ 0.49–0.51) is itself a finding: real cues barely move off "
    "coin-flip, so they live in the regime where gates cannot help. Figure 11 is a bar chart of the "
    "gate-minus-periodic gap per category; every bar points left (negative), meaning the gate loses in "
    "all 11 groups—the strongest single picture of the paper's message.",
    "Dải x hẹp của Hình 10 (eta ≈ 0.49–0.51) tự nó là một phát hiện: cue thật gần như không nhích khỏi "
    "tung-đồng-xu, nên chúng sống ở vùng mà gate không giúp được. Hình 11 là biểu đồ cột của chênh "
    "gate-trừ-periodic theo category; mọi cột chỉ sang trái (âm), nghĩa là gate thua ở cả 11 nhóm—bức "
    "tranh đơn mạnh nhất cho thông điệp của bài.")
doc.save(OUT)
print("FINAL v3 saved | paragraphs:", len(doc.paragraphs), "tables:", len(doc.tables), "images:", len(doc.inline_shapes))
