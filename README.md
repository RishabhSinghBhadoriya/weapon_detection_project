# 🔍 Weapon Detection System using YOLO and Django

A web-based weapon detection system that uses a custom-trained YOLO model to detect weapons in uploaded images. This app is built using Django and allows users to upload images, see detection results, and browse detection history.

---

## 📌 Features

- 🚀 YOLO-based image detection (trained on a custom dataset)
- 🖼️ Upload image and detect weapons instantly
- 📂 View detection history
- 🧠 Integrated with a pre-trained local YOLO model
- 💾 SQLite for lightweight local database
- 🖥️ Web interface built with Django and Bootstrap
- 🔴 Live detection using webcams and camera modules

---

## 🛠️ Tech Stack

- **Backend**: Django (Python)
- **Model**: YOLOv8 / YOLOv10 (Custom-trained)
- **Frontend**: HTML, CSS (Bootstrap)
- **Database**: SQLite (default Django DB)
- **Deployment**: Render (Free Tier)

---

## 🧠 YOLO Model Info

- Model trained on a custom dataset of weapon images
- Detection classes include: *gun, knife*
- Training done using Ultralytics YOLO
- Model file is stored locally (`/weapon_detection_app`)

---

## 📷 Usage

1. Upload an image via the web interface
2. The YOLO model runs locally and detects weapons
3. Results are displayed with bounding boxes
4. All uploads (temporary) are shown in the "History" section

---

## ⚙️ Setup & Installation

```bash
# Clone the repo
git clone "link of repository here"
cd weapon-detection-app

# Create virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Run server
python manage.py runserver
```
- The model (`best.pt`) was trained by the author using datasets collected from multiple open sources. Redistribution of the trained model is not permitted without explicit permission.
- The user interface,model training and platform were developed as part of a research project focused on centralized threat detection using YOLO and CNN. Please cite the original research if using or referencing this work.
- The model architecture is based on [YOLO](https://github.com/ultralytics/ultralytics), which is open-source under GPL-3.

Please respect dataset licenses where applicable.

'''to do: add link to research here'''