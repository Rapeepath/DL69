"""ฝึกสอนเพอร์เซปตรอนชั้นเดียว (Linear + Sigmoid) สำหรับ binary classification

อัปเดต weight ด้วยมือ (ไม่ใช้ optimizer) และเปรียบเทียบ learning rate 3 ค่า
ผลลัพธ์ถูกบันทึกไว้ที่ dashboard/static/dashboard/
"""
import copy
import csv
from pathlib import Path

import torch
from matplotlib.figure import Figure

SEED = 42
NUM_SAMPLES = 200
NUM_EPOCHS = 200
LEARNING_RATES = [0.001, 0.1, 1.0]
# lr ที่รายงานความคืบหน้าผ่าน on_progress (อีก 2 ค่าใช้เปรียบเทียบตอนจบ)
STREAM_LR = 0.1

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "static" / "dashboard"
LOSS_CURVE_PATH = OUTPUT_DIR / "loss_curve.png"
LR_TABLE_PATH = OUTPUT_DIR / "lr_comparison.csv"


def make_data():
    """สร้างข้อมูล binary classification: 200 จุดใน 2 มิติ แบ่งด้วยเส้นตรง 1.5x + y > 0.5

    คืนค่า (X, y) โดย X มี shape [200, 2] และ y มี shape [200, 1] (ค่า 0.0 หรือ 1.0)
    """
    X = torch.randn(NUM_SAMPLES, 2)
    y = ((X[:, 0] * 1.5 + X[:, 1] - 0.5) > 0).float().unsqueeze(1)
    return X, y


def build_model():
    """สร้างเพอร์เซปตรอนชั้นเดียว: Linear(2, 1) + Sigmoid"""
    return torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid(),
    )


def fit(model, X, y, lr, num_epochs=NUM_EPOCHS, on_progress=None):
    """ฝึกสอน model ด้วย learning rate ค่าเดียว โดยอัปเดต weight ด้วยมือ

    คืนค่า dict {"loss": [...], "accuracy": [...]} เก็บค่าของทุก epoch
    """
    loss_fn = torch.nn.BCELoss()
    history = {"loss": [], "accuracy": []}

    for epoch in range(num_epochs):
        # forward
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        # backward
        loss.backward()

        # update weights ด้วยมือ: w = w - lr * gradient
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
            model.zero_grad()

        # คำนวณ accuracy: ตัดที่ 0.5 แล้วนับสัดส่วนที่ทำนายถูก
        acc = ((y_hat > 0.5).float() == y).float().mean().item()

        history["loss"].append(loss.item())
        history["accuracy"].append(acc)

        # ส่งข้อมูลผ่าน SSE
        if on_progress:
            on_progress(epoch, loss.item(), acc)

    return history


def save_loss_curve(histories, path=LOSS_CURVE_PATH):
    """บันทึกกราฟ loss ของทุก learning rate เป็นไฟล์ PNG

    histories คือ dict ที่ key เป็น lr และ value เป็น history จาก fit()
    """
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    for lr, history in histories.items():
        ax.plot(history["loss"], label=f"lr = {lr}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss curve")
    ax.legend()
    fig.savefig(path)


def save_lr_table(results, path=LR_TABLE_PATH):
    """บันทึกตารางเปรียบเทียบ learning rate เป็นไฟล์ CSV"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lr", "loss", "accuracy"])
        writer.writeheader()
        writer.writerows(results)


def train(on_progress=None):
    """ฝึกสอนเพอร์เซปตรอนด้วย learning rate 3 ค่า แล้วบันทึกผลลัพธ์

    on_progress(epoch, loss, acc) → เรียกทุก epoch ของรอบที่ lr = STREAM_LR
    เพื่อส่งข้อมูลผ่าน SSE ไปยังเบราว์เซอร์

    คืนค่า list ของ dict {"lr", "loss", "accuracy"} คือ loss และ accuracy
    ของ epoch สุดท้ายของแต่ละ learning rate
    """
    torch.manual_seed(SEED)

    # 1. สร้างข้อมูล (binary classification)
    X, y = make_data()

    # 2. สร้างโมเดล (perceptron)
    base_model = build_model()

    # 3. Training loop: ทุก lr เริ่มจาก weight ชุดเดียวกัน เพื่อให้เปรียบเทียบกันได้
    histories = {}
    for lr in LEARNING_RATES:
        model = copy.deepcopy(base_model)
        callback = on_progress if lr == STREAM_LR else None
        histories[lr] = fit(model, X, y, lr, on_progress=callback)

    # 4. บันทึกผลลัพธ์
    results = [
        {
            "lr": lr,
            "loss": history["loss"][-1],
            "accuracy": history["accuracy"][-1],
        }
        for lr, history in histories.items()
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_loss_curve(histories)
    save_lr_table(results)
    return results
