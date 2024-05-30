from turtle import title
import gradio as gr
from transformers import pipeline
import numpy as np
from PIL import Image
import torch
from torch import nn
import cv2

from matplotlib import pyplot as plt
from segmentation_mask_overlay import overlay_masks
from transformers import (
    CLIPSegProcessor,
    CLIPSegForImageSegmentation,
    AutoProcessor,
    AutoConfig,
)

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
classes = list()


def create_rgb_mask(mask):
    color = tuple(np.random.choice(range(0, 256), size=3))
    gray_3_channel = cv2.merge((mask, mask, mask))
    gray_3_channel[mask == 255] = color
    return gray_3_channel.astype(np.uint8)


def detect_using_clip(image, prompts=[], threshould=0.4):
    predicted_masks = list()
    inputs = processor(
        text=prompts,
        images=[image] * len(prompts),
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():  # Use 'torch.no_grad()' to disable gradient computation
        outputs = model(**inputs)
    # preds = outputs.logits.unsqueeze(1)
    preds = nn.functional.interpolate(
        outputs.logits.unsqueeze(1),
        size=(image.shape[0], image.shape[1]),
        mode="bilinear",
    )
    threshold = 0.1

    flat_preds = torch.sigmoid(preds.squeeze()).reshape((preds.shape[0], -1))

    # Initialize a dummy "unlabeled" mask with the threshold
    flat_preds_with_treshold = torch.full(
        (preds.shape[0] + 1, flat_preds.shape[-1]), threshold
    )
    flat_preds_with_treshold[1 : preds.shape[0] + 1, :] = flat_preds

    # Get the top mask index for each pixel
    inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape(
        (preds.shape[-2], preds.shape[-1])
    )
    predicted_masks = []

    for i in range(1, len(prompts) + 1):
        mask = np.where(inds == i, 255, 0)
        predicted_masks.append(mask)

    return predicted_masks


def visualize_images(image, predicted_images, brightness=15, contrast=1.8):
    alpha = 0.7
    image_resize = cv2.resize(image, (352, 352))
    resize_image_copy = image_resize.copy()

    # for mask_image in predicted_images:
    #     resize_image_copy = cv2.addWeighted(resize_image_copy,alpha,mask_image,1-alpha,10)

    return cv2.convertScaleAbs(resize_image_copy, alpha=contrast, beta=brightness)


def shot(alpha, beta, image, labels_text):
    print(labels_text)

    if "," in labels_text:
        prompts = labels_text.split(",")
    else:
        prompts = [labels_text]
    print(prompts)

    prompts = list(map(lambda x: x.strip(), prompts))

    mask_labels = [f"{prompt}_{i}" for i, prompt in enumerate(prompts)]
    cmap = plt.cm.tab20(np.arange(len(mask_labels)))[..., :-1]

    predicted_masks = detect_using_clip(image, prompts=prompts)
    bool_masks = [predicted_mask.astype("bool") for predicted_mask in predicted_masks]
    category_image = overlay_masks(
        image,
        np.stack(bool_masks, -1),
        labels=mask_labels,
        colors=cmap,
        alpha=alpha,
        beta=beta,
    )

    return category_image


iface = gr.Interface(
    fn=shot,
    inputs=[
        gr.Slider(
            0.1, 1, value=0.3, step=0.1, label="alpha", info="Choose between 0.1 to 1"
        ),
        gr.Slider(
            0.1, 1, value=0.7, step=0.1, label="beta", info="Choose between 0.1 to 1"
        ),
        "image",
        "text",
    ],
    outputs="image",
    description="Add an Image and  labels to be detected separated by commas(atleast 2)",
    title="Zero-shot Image Segmentation with Prompt",
    examples=[
        [
            0.4,
            0.7,
            "images/room.jpg",
            "chair, plant , flower pot , white cabinet , paintings , decorative plates , books",
        ],
        [0.4, 0.7, "images/seats.jpg", "door,table,chairs"],
        [
            0.3,
            0.8,
            "images/vegetables.jpg",
            "carrot,white radish,brinjal,basket,potato",
        ],
        [0.4, 0.7, "images/dashcam.jpeg", "car,sky,road,grassland,trees"],
    ],
    # allow_flagging=False,
    # analytics_enabled=False,
)
iface.launch()
