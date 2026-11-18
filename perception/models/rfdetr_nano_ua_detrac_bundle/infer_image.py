import argparse
from pathlib import Path
from rfdetr.detr import RFDETR

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--device", default="cuda")
parser.add_argument("--threshold", type=float, default=0.35)
args = parser.parse_args()
weights = Path(__file__).resolve().with_name("checkpoint_best_total.pth")
model = RFDETR.from_checkpoint(str(weights), device=args.device)
detections = model.predict(args.image, threshold=args.threshold)
for index, box in enumerate(detections.xyxy):
    print(detections.data["class_name"][index],
          float(detections.confidence[index]), box.tolist())
