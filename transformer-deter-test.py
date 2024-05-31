import torch
from PIL import Image
import requests
from transformers import DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the image
image_path = "path_to_image.jpg"
image = Image.open(image_path)

# Load the DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Perform object detection
outputs = model(**inputs)

# Extract boxes and scores
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9
)[0]

# Plot the image and bounding boxes
plt.figure(figsize=(8, 8))
plt.imshow(image)
ax = plt.gca()

# Draw bounding boxes and labels
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = box.detach().numpy()
    x, y, w, h = box
    ax.add_patch(
        plt.Rectangle((x, y), w - x, h - y, fill=False, color="red", linewidth=2)
    )
    ax.text(
        x,
        y,
        f"{model.config.id2label[label.item()]}: {score:.2f}",
        fontsize=12,
        bbox=dict(facecolor="yellow", alpha=0.5),
    )

plt.axis("off")
plt.show()
