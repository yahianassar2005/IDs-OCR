import cv2
import numpy as np
from ultralytics import YOLO

CLASS_TO_KEY = {
    0: "first_name",
    1: "full_name",
    2: "address_1",
    3: "address_2",
    4: "national_id"
}

class FieldDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_and_crop(self, img_bgr: np.ndarray, pad_x: int = 15, pad_y: int = 3) -> dict:
        h, w, _ = img_bgr.shape
        results = self.model.predict(source=img_bgr, conf=self.conf_threshold, verbose=False)
        boxes = results[0].boxes.cpu()

        best_predictions = {}
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].numpy()

            if cls_id not in best_predictions or conf > best_predictions[cls_id][0]:
                best_predictions[cls_id] = (conf, xyxy)

        crops = {}
        for cls_id, (conf, xyxy) in best_predictions.items():
            key = CLASS_TO_KEY.get(cls_id)
            if not key:
                continue

            xmin, ymin, xmax, ymax = map(int, xyxy)
            xmin_padded = max(0, xmin - pad_x)
            ymin_padded = max(0, ymin - pad_y)
            xmax_padded = min(w, xmax + pad_x)
            ymax_padded = min(h, ymax + pad_y)

            crop = img_bgr[ymin_padded:ymax_padded, xmin_padded:xmax_padded]
            crops[key] = {
                "class_id": cls_id,
                "confidence": conf,
                "bbox": [xmin_padded, ymin_padded, xmax_padded, ymax_padded],
                "crop": crop
            }

        return crops
