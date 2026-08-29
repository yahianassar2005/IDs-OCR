import os
import cv2
import numpy as np

# Disable oneDNN CPU instruction set crashes before importing PaddleOCR
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import TextRecognition

class ArabicOCREngine:
    def __init__(self, model_name: str = "arabic_PP-OCRv5_mobile_rec"):
        self.rec_engine = TextRecognition(model_name=model_name)

    def preprocess_crop(self, crop: np.ndarray, target_height: int = 120) -> np.ndarray:
        crop_h, crop_w = crop.shape[:2]
        scale_ratio = target_height / max(1, crop_h)
        target_width = int(crop_w * scale_ratio)
        return cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    def recognize_text(self, crop: np.ndarray, is_numeric: bool = False) -> str:
        processed = self.preprocess_crop(crop)
        output = self.rec_engine.predict(input=processed, batch_size=1)

        extracted_text = ""
        for res in output:
            if hasattr(res, 'res') and 'rec_text' in res.res:
                extracted_text = res.res['rec_text']
            elif isinstance(res, dict):
                extracted_text = res.get('rec_text', res.get('text', ''))

        if is_numeric:
            return str(extracted_text).replace(" ", "").strip()
        return str(extracted_text).strip()
