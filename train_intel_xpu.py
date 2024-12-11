from ultralytics import YOLO
import torch
import intel_extension_for_pytorch as ipex

model = YOLO("yolo11n.yaml")  # train model from yaml
# model = YOLO("yolov11n.pt")  # resume training from a pretrained model

#! Edit here -------------------------------------------------
## Training parameters
# device = "xpu:1"
device = "xpu"
epochs = 100
data = "coco128.yaml"
#! -----------------------------------------------------------

try:
    print("[Preamble] Display model information:")
    model.info()
    print(f"[Preamble] Loading device: '{device}' to model")
    results = model.to(device)
except Exception as e:
    print(f"[Preamble] --- Error: Fail to load device {device}")
    print(e)
    exit(1)

try:
    print("[1] Training model ============================================")
    results = model.train(data=data, epochs=epochs)  # train the model, default lr=0.01, 
except Exception as e:
    print("[1] --- Error: Fail to train model")
    print(e)
    exit(1)

try:
    print("[2] Validate model performance ================================")
    results = model.val()  # evaluate model performance on the validation set
except Exception as e:
    print("[2] --- Error: Fail to validate training")
    print(e)
    exit(1)

try:
    print("[3] Inferencing image =========================================")
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
except Exception as e:
    print("[3] --- Error: Fail to inference model on an image")
    print(e)
    exit(1)

try:
    print("[4] Export to ONNX ============================================")
    results = model.export(format="onnx")  # export the model to ONNX format
except Exception as e:
    print("[4] --- Error: Fail to export trained model to ONNX format")
    print(e)
    exit(1)
print("\nDone")
