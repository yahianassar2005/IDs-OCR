# 🪪 Egyptian National ID OCR & Automatic Extraction Pipeline

An end-to-end Computer Vision and Optical Character Recognition (OCR) pipeline designed to detect, crop, and extract structured personal details from Egyptian National Identity Cards. 

The pipeline combines a **fine-tuned YOLOv8 model** for layout field localization, **OpenCV** image processing, **PaddleOCR (arabic_PP-OCRv5)** for high-accuracy Arabic text recognition, and **SQLite** for structured database storage.

---

## 🌟 Key Features

* **Custom Field Localization**: Pinpoint accuracy for 5 core ID card fields (`first_name`, `full_name`, `address_1`, `address_2`, `national_id`).
* **Synthetic Data Generator Engine**: Includes a automated dataset generation pipeline (`synthetic_generator.py`) using PIL and OpenCV to render realistic Arabic text, Eastern Arabic numerals (`٠١٢٣٤٥٦٧٨٩`), and auto-formatted bounding boxes in YOLO TXT format.
* **Precision OCR Recognition**: Leverages PaddleOCR's `arabic_PP-OCRv5_mobile_rec` engine to parse cursive Arabic text and isolated 14-digit National ID strings without layout distortion.
* **Database Integration**: Automatically cleans, formats, and commits extracted record payloads to a lightweight SQLite relational database.

---

## 📐 System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────────┐
│ Raw ID Image    │ ──> │ YOLOv8 Field     │ ──> │ OpenCV Image           │
│ Input           │     │ Localization     │     │ Preprocessing & Padding│
└─────────────────┘     └──────────────────┘     └────────────────────────┘
                                                             │
                                                             ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────────┐
│ Structured      │ <── │ SQLite Database  │ <── │ PaddleOCR Text         │
│ Data Output     │     │ Storage Engine   │     │ Recognition Engine     │
└─────────────────┘     └──────────────────┘     └────────────────────────┘
```

---

## 🚀 Performance Metrics

### YOLOv8 Field Localization Model
* **mAP@50**: `99.2%`
* **mAP@50-95**: `76.2%`
* **Precision**: `97.5%`
* **Recall**: `100.0%`

| Class Field | Precision | Recall | mAP@50 |
| :--- | :---: | :---: | :---: |
| `first_name` | 0.946 | 1.000 | 0.993 |
| `full_name` | 0.992 | 1.000 | 0.995 |
| `address_1` | 0.996 | 1.000 | 0.995 |
| `address_2` | 0.990 | 1.000 | 0.995 |
| `national_id` | 0.949 | 1.000 | 0.982 |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/egyptian-national-id-ocr.git](https://github.com/your-username/egyptian-national-id-ocr.git)
cd egyptian-national-id-ocr
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Running the Pipeline

### Execute Pipeline Inference
To process an ID image, crop the detected fields, run Arabic OCR, and save the record to SQLite:

```bash
python main.py --image_path "path/to/national_id_sample.jpg" --model_path "models/best_yolov8_id.pt"
```

### Generate Synthetic Training Data
To generate synthetic document samples with auto-generated YOLO annotations:

```bash
python data/synthetic_generator.py --num_samples 100 --output_dir "data/processed/"
```

---

## 🔍 Field Localization & OCR Extraction Results

Below is the end-to-end extraction pipeline showcasing the **YOLOv8 crop inputs** alongside the resulting **PaddleOCR Arabic text extractions**:

<img src="assets/annotated_id_detection2.png" width="180"/>

| Target Field | Isolated YOLOv8 Crop | PaddleOCR Extracted Text | Status |
| :--- | :---: | :---: | :---: |
| **First Name** (`first_name`) | <img src="assets/paddle_crop_0_first_name.png" width="180"/> | `محمود` | ✅ Exact Match |
| **Full Name** (`full_name`) | <img src="assets/paddle_crop_1_full_name.png" width="280"/> | `عبدالتواب مصطفى سالم` | ✅ Exact Match |
| **Address Line 1** (`address_1`) | <img src="assets/paddle_crop_2_address_1.png" width="220"/> | `منية قلين` | ✅ Exact Match |
| **Address Line 2** (`address_2`) | <img src="assets/paddle_crop_3_address_2.png" width="280"/> | `مركز قلين كفر الشيخ` | ✅ Exact Match |
| **National ID** (`national_id`) | <img src="assets/paddle_crop_4_national_id2.png" width="300"/> | `٢٥٤٠٥٢١١٥٠****` | ✅ 14 Digits Valid |
---

## 📜 License

This project is open-source under the [MIT License](LICENSE).
