import torch

# Load YOLOv5 model just for people
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
model.classes = [0]

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Images
#imgs = ['./new_repo/assets/image_and_annotations_repo/base/images/94.jpg']  # batch of images

imgs = ['C:\\Users\\Usuario\\Downloads\\Imaganes wapas web\\2021061714471884302.jpg']  # batch of images

# Inference
results = model(imgs)

results.show()