import torch
import torchvision
import matplotlib.pyplot as plt

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Images
imgs = ['D:/TFG/resources/Test_Training_p2p/test_pretrained/test/scene05/img_5.jpg']  # batch of images

# Inference
results = model(imgs)

results.show()