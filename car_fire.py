# 智能小车 实时火焰检测 + 蜂鸣器报警
import torch
import cv2
import time
import RPi.GPIO as GPIO

# ====================== 蜂鸣器设置 ======================
BUZZER = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER, GPIO.OUT)
GPIO.output(BUZZER, GPIO.LOW)

# ====================== 加载火焰模型 ======================
print("⏳ 正在加载火焰模型...")
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="/home/best.pt",  # 这里已改成你文件的真实路径
    trust_repo=True
)
model.conf = 0.4
model.classes = [0]  # 只检测火焰

# ====================== 打开摄像头 ======================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ 小车火焰实时检测已启动！")

# ====================== 主循环 ======================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 摄像头读取失败")
            break

        # 火焰检测
        results = model(frame)

        # 判断是否有火焰
        fire_detected = len(results.xyxy[0]) > 0

        # 检测到火焰 → 蜂鸣器响
        if fire_detected:
            print("🔥 发现火焰！正在报警！")
            GPIO.output(BUZZER, GPIO.HIGH)
        else:
            GPIO.output(BUZZER, GPIO.LOW)

        time.sleep(0.05)

finally:
    GPIO.output(BUZZER, GPIO.LOW)
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 程序已退出")