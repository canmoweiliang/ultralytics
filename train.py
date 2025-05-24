from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("myyolo11n.yaml")

# Train the model on the COCO8 example dataset for 100 epochs
if __name__=='__main__':
    results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)