import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

model = YOLO("../yolo11n.pt")

results = model.train(
    data='../datasets/yolo_dataset/data.yaml', 
    epochs=50, 
    imgsz=640,
    batch=16,
    name='steel_defect_yolo11n',
    project='../checkpoints'
)

metrics = model.val(split='test')
print(f"mAP50-95: {metrics.box.map}") # Chỉ số độ chính xác trung bình