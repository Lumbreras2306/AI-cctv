from typing import Any


from ultralytics import YOLO
import cv2
import time
from dotenv import load_dotenv
from os import getenv

load_dotenv()

RTSP_URL = getenv("RTSP_URL")

MODEL_PATH = getenv("MODEL_PATH")


def main():
    print("Cargando modelo YOLO11n...")
    model = YOLO(MODEL_PATH)

    print(f"Abriendo stream RTSP: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("❌ No se pudo abrir el stream. Revisa la URL RTSP.")
        return

    last_infer = 0.0
    interval = 0.5  # segundos entre inferencias (~5 fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Error leyendo frame. Reintentando en 1s...")
            time.sleep(1)
            continue

        now = time.time()
        if now - last_infer < interval:
            continue
        last_infer = now

        # Inferencia con YOLO11
        results = model(
            frame,
            imgsz=640,
            conf=0.5,
            verbose=False,
        )

        # Dibujar detecciones en el frame
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]

                if name == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{name} {conf:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        cv2.imshow("YOLO11 - DVR", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
