

# 🏸 PLAYSIGHTAI

AI-powered real-time badminton shot detection and performance analysis using Computer Vision and Deep Learning.

---

## 📌 Overview

The **Badminton Analysis System** analyzes badminton match videos to:

* Detect the badminton court
* Track shuttle movement
* Estimate player pose
* Classify shot types
* Grade shot quality (Good / Average / Bad)
* Generate match heatmaps
* Extract highlight clips automatically

The system combines **YOLO object detection**, **pose estimation**, and **biomechanical heuristics** to deliver intelligent sports analytics.

---

# 🚀 Key Features

* 🎯 Shot Detection (Smash, Clear, Drop, Drive, Lift, Netshot)
* 🏸 Shuttle Tracking using YOLOv8
* 🏟 Court Boundary Detection
* 💪 Elbow Angle & Biomechanics Analysis
* 📊 Shot Quality Grading (Good / Average / Bad)
* 🔥 Match Heatmap Generation
* 🎥 Highlight Clip Extraction
* ⚡ Multi-frame Confirmation for stable predictions

---

# 🤖 Model Performance

## 🏟 Court Detection Model

The court detection model identifies badminton court boundaries in each frame.

### 🔹 Details

* Model: YOLO-based custom-trained detector
* Dataset: Custom annotated court images
* Metric: mAP (mean Average Precision)

### ✅ Accuracy: **95.5%**

### 🎯 Importance

High court detection accuracy ensures:

* Accurate zone mapping
* Reliable heatmap generation
* Correct shot position analysis
* Stable spatial understanding of the match

Court detection is highly accurate because:

* Court lines are static
* Large object features
* Strong contrast patterns

---

## 🏸 Shuttle Detection Model

The shuttle detection model tracks the shuttlecock across frames.

### 🔹 Details

* Model: YOLOv8
* Dataset: Custom shuttle image dataset
* Metric: mAP

### ✅ Accuracy: **88.4%**

### 🎯 Importance

Shuttle tracking enables:

* Shot classification
* Direction estimation (UP / DOWN / FLAT)
* Speed approximation
* Quality grading

### ⚠ Why Accuracy is Lower?

Shuttle detection is more challenging because:

* Small object size
* High velocity
* Motion blur
* Occlusion during fast rallies

Despite these challenges, 88.4% accuracy provides strong real-time performance.

---

# 🧠 Shot Quality Evaluation Logic

Shot quality is evaluated using:

* Wrist height
* Elbow extension angle
* Swing velocity
* Shuttle direction
* Shuttle speed
* Court zone

### Example – Smash

| Condition                                   | Result     |
| ------------------------------------------- | ---------- |
| Jump + Elbow > 155° + Fast Downward Shuttle | 🟢 GOOD    |
| Moderate Extension                          | 🟡 AVERAGE |
| Weak Swing / Low Contact                    | 🔴 BAD     |

Multi-frame confirmation ensures stable grading.

---

# 📂 Project Structure

```
PlaySightAI/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── detected_shots/
│   ├── smashes/
│   └── lifts/
│
├── frames/
├── highlights/
├── models/
│   └── shuttle_best.pt
│
├── utils/
│   ├── math_utils.py
│   ├── check_files.py
│   └── config.yaml
│
├── extract_frames.py
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/rohitmotwani2004/PlaySightAI.git
cd PlaySightAI
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python main.py
```

Optional:

```bash
python main.py --video input.mp4
```

Press `q` to exit video window.

---

# 📊 Output

The system generates:

* Shot logs in terminal
* Classified shot folders
* Highlight video clips
* `match_heatmap.png`
* Performance grading

---

# 🔮 Future Improvements

* ML-based shot quality scoring
* LSTM swing pattern recognition
* Player performance dashboard
* Mobile application integration
* Advanced trajectory analytics

---

# 🏆 Applications

* Player performance analysis
* Coaching assistance
* Sports biomechanics research
* AI in sports analytics

---

# 👨‍💻 Author

**Rohit Motwani**
AI & Computer Vision Enthusiast
Combining Badminton and Artificial Intelligence


