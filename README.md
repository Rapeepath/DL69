# DL69 — Wk09 Perceptron Training Dashboard

รหัสนักศึกษา: 67114540439
ชื่อ-นามสกุล: รพีภัทร ขั้นทอง

Dashboard สำหรับฝึกสอนเพอร์เซปตรอน (Linear + Sigmoid) และแสดงผล loss / accuracy ทีละ epoch แบบ real-time ผ่าน Server-Sent Events

## วิธีรัน

```bash
gh repo clone Rapeepath/DL69
cd DL69
git checkout wk09
uv sync
uv run manage.py runserver
```

เปิด http://127.0.0.1:8000/ แล้วกดปุ่ม "เริ่มฝึกสอน"

## โครงสร้างสำคัญ

- `dashboard/ml/train.py` — ฟังก์ชัน `train()` ฝึกสอนเพอร์เซปตรอน 200 epoch โดยอัปเดต weight ด้วยมือ และเปรียบเทียบ learning rate 0.001, 0.1, 1.0
- `dashboard/views.py` — `index` (หน้าแรก) และ `train_stream` (SSE endpoint `/train/`)
- `dashboard/templates/dashboard/index.html` — หน้า dashboard (Alpine.js + canvas)
