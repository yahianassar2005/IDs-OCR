import os
import cv2
from src.detector import FieldDetector
from src.ocr_engine import ArabicOCREngine
from src.database import IDDatabase

class EgyptianIDPipeline:
    def __init__(self, yolo_model_path: str, db_path: str = "national_ids.db"):
        self.detector = FieldDetector(model_path=yolo_model_path)
        self.ocr_engine = ArabicOCREngine()
        self.db = IDDatabase(db_path=db_path)

    def process_image(self, image_path: str, output_debug_dir: str = None) -> dict:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image at {image_path}")

        if output_debug_dir:
            os.makedirs(output_debug_dir, exist_ok=True)

        detected_crops = self.detector.detect_and_crop(img_bgr)
        extracted_data = {
            "first_name": None,
            "full_name": None,
            "address_1": None,
            "address_2": None,
            "national_id": None
        }

        annotated_image = img_bgr.copy()

        for field_name, field_info in detected_crops.items():
            crop = field_info["crop"]
            is_numeric = (field_name == "national_id")

            if output_debug_dir:
                crop_save_path = os.path.join(output_debug_dir, f"crop_{field_name}.png")
                cv2.imwrite(crop_save_path, crop)

            text = self.ocr_engine.recognize_text(crop, is_numeric=is_numeric)
            extracted_data[field_name] = text

            # Annotate visual verification
            bbox = field_info["bbox"]
            cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        if output_debug_dir:
            cv2.imwrite(os.path.join(output_debug_dir, "annotated_result.png"), annotated_image)

        self.db.insert_record(extracted_data, source_image=image_path)
        return extracted_data
