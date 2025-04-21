import torch
from ultralytics import YOLO, NAS
from thop import profile


model = YOLO("yolo113n.yaml")
# print(model)

model.info(detailed=False)

# Load a COCO-pretrained YOLO-NAS-s model
# model = NAS("yolo_nas_s.pt")

# # Display model information (optional)
# model.info()