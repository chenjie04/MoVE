from ultralytics import YOLO

model = YOLO("yolo113n.yaml")
# print(model)
print("模型复杂度:")
model.info(detailed=False)

# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLO('runs/yolo11_voc_ab/113n_no_move/weights/best.pt')


metrics = model.val(data='VOC.yaml', batch=1)

print("mAP50-95:", metrics.box.map)  # mAP50-95
print("mAP50:", metrics.box.map50)  # mAP50
print("mAP75:",metrics.box.map75)  # mAP75
print("list of mAP50-95 for each category:", metrics.box.maps)  # list of mAP50-95 for each category