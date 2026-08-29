import argparse
from src.pipeline import EgyptianIDPipeline

def main():
    parser = argparse.ArgumentParser(description="Egyptian National ID Extraction Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input ID card image")
    parser.add_argument("--model", type=str, default="models/best_yolov8_id.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--output", type=str, default="output/", help="Directory to save crop inspection images")
    parser.add_argument("--db", type=str, default="national_ids.db", help="Path to SQLite database")

    args = parser.parse_args()

    print("Initializing Pipeline...")
    pipeline = EgyptianIDPipeline(yolo_model_path=args.model, db_path=args.db)

    print(f"Processing: {args.image}")
    results = pipeline.process_image(args.image, output_debug_dir=args.output)

    print("\n--- Extracted Data ---")
    for k, v in results.items():
        print(f"[{k.upper()}]: {v}")
    print(f"\nRecord saved to {args.db}. Visual crops saved to {args.output}")

if __name__ == "__main__":
    main()
