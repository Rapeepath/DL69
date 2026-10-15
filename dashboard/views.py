import json
import queue
import threading
from time import sleep

from django.http import StreamingHttpResponse
from django.shortcuts import render

from .ml.train import train

# หน่วงเวลาต่อ epoch เพื่อให้เห็นค่าเปลี่ยนบนหน้าเว็บทัน (การฝึกสอนจริงเร็วกว่านี้มาก)
EPOCH_DELAY = 0.05


def index(request):
    """หน้า landing page"""
    return render(request, 'dashboard/index.html')


def train_stream(request):
    """SSE endpoint: เรียก ml.train() แล้วส่งผลของแต่ละ epoch ให้เบราว์เซอร์แบบ real-time

    train() เป็นฟังก์ชันที่ทำงานจนจบแล้วค่อยคืนค่า จึงรันไว้ใน thread แยก
    และส่งข้อมูลกลับมาทาง queue เพื่อให้ response ทยอยส่งออกไปได้ระหว่างที่ยังฝึกสอนอยู่
    """
    events = queue.Queue()

    def on_progress(epoch, loss, acc):
        events.put(('message', dict(epoch=epoch, loss=loss, accuracy=acc)))
        sleep(EPOCH_DELAY)

    def worker():
        try:
            results = train(on_progress)
            events.put(('done', dict(results=results)))
        finally:
            events.put(None)    # บอก generate() ว่าจบ stream แล้ว

    def generate():
        threading.Thread(target=worker, daemon=True).start()
        while (item := events.get()) is not None:
            event, data = item
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"

    response = StreamingHttpResponse(
        generate(),
        content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    return response
