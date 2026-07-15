import os
import cv2
from ultralytics import YOLO

# 1. Configuration Paths
model_path = r"C:\Users\National\CS\Projects\IDs\runs\detect\train-2\weights\best.pt"  # Path to your freshly baked weights
test_image_path = r"c:\Users\National\Downloads\examples\input\55.jpg"
output_dir = r"c:\Users\National\Pictures\ocr_test_results\2"

os.makedirs(output_dir, exist_ok=True)

# 2. Load your custom fine-tuned YOLO model
model = YOLO(model_path)

# 3. Run Inference
# We set save_crop=True so YOLO automatically crops out each text zone into separate folders!
results = model.predict(
    source=test_image_path,
    conf=0.075,           # Confidence threshold (adjust higher/lower if needed)
    save=True,           # Saves the full image with bounding boxes drawn
    save_crop=True,      # Automatically crops out every text segment it finds
    project=output_dir,
    name="prediction"
)

print(f"\nInference completed! Check your results directory here: {output_dir}")