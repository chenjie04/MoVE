# import os
# os.environ['MKL_SERVICE_FORCE_INTEL'] = 'GNU'


from ultralytics import YOLO

# Load a model
model = YOLO("yolo113n.yaml")
# model = YOLO("runs/yolo11_VOC/113n35/weights/last.pt")

# Train the model
train_results = model.train(
    # resume=True,
    data="VOC.yaml",  # path to dataset YAML
    # data="coco.yaml",
    epochs=500,  # number of training epochs
    batch=128,
    imgsz=640,  # training image size
    device=[0, 1],  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    project="runs/yolo11_voc_ab",
    name="113n",
)
