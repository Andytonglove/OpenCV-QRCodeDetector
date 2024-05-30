import argparse
import cv2
import numpy as np
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image

import gradio as gr

# 初始化模型和处理器
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")


def get_args():
    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="image1.png", help="Image path")
    parser.add_argument("--prompt", type=str, default="QR code", help="Prompt")
    args = parser.parse_args()
    return args


def detect_qr_code(image_path, prompt="QR code"):
    # 读取图像
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # 准备输入
    inputs = processor(
        text=[prompt], images=[image], padding="max_length", return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # 获取分割结果
    preds = torch.nn.functional.interpolate(
        outputs.logits.unsqueeze(1),
        size=(image_np.shape[0], image_np.shape[1]),
        mode="bilinear",
    )

    # 转换为掩码
    preds = torch.sigmoid(preds).squeeze().cpu().numpy()
    mask = (preds > 0.5).astype(np.uint8) * 255

    return image_np, mask


def overlay_mask(image, mask, alpha=0.5):
    # 将mask转换为彩色图像
    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    # 叠加图像和掩码
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlay


def crop_to_mask(image, mask):
    # 找到mask的非零像素的边界
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    # 裁剪图像
    cropped_image = image[y : y + h, x : x + w]

    return cropped_image


def solo_crop_to_mask(image, mask, padding=10):
    # 查找mask中的所有独立轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cropped_images = []

    for contour in contours:
        # 获取每个轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 添加padding
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, image.shape[1] - x)
        h = min(h + 2 * padding, image.shape[0] - y)

        # 裁剪图像
        cropped_image = image[y : y + h, x : x + w]
        cropped_images.append(cropped_image)

    return cropped_images


def main():
    args = get_args()
    prompt = args.prompt
    image_path = args.image

    # 示例使用
    # image_path = "image1.png"
    image, mask = detect_qr_code(image_path)

    if mask is not None:
        # 叠加掩码到原图像
        overlay_image = overlay_mask(image, mask)

        # 根据掩码裁剪图像
        cropped_image = crop_to_mask(image, mask)

        # 单独裁剪图像
        solo_cropped_images = solo_crop_to_mask(image, mask, padding=10)

        # 显示原图、掩码和叠加后的图像
        # cv2.imshow("Original Image", image)
        # cv2.imshow("QR Code Mask", mask)
        cv2.imshow("Overlay Image", overlay_image)
        cv2.imshow("Cropped Image", cropped_image)

        print(f"total detected QR Code Masks: {len(solo_cropped_images)}")
        if len(solo_cropped_images) > 1:
            for i, cropped_image in enumerate(solo_cropped_images):
                cv2.imshow(f"Cropped Image {i+1}", cropped_image)

        # 保存裁剪后的图像
        # cv2.imwrite("Overlay_image.jpg", overlay_image)
        # cv2.imwrite("cropped_image.jpg", cropped_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# 为gradio使用
def grad_detector(image, prompt="QR code"):
    # 准备输入
    inputs = processor(
        text=[prompt], images=[image], padding="max_length", return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # 获取分割结果
    preds = torch.nn.functional.interpolate(
        outputs.logits.unsqueeze(1),
        size=(image.shape[0], image.shape[1]),
        mode="bilinear",
    )

    # 转换为掩码
    preds = torch.sigmoid(preds).squeeze().cpu().numpy()
    mask = (preds > 0.5).astype(np.uint8) * 255

    # 将mask转换为彩色图像
    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    # 叠加图像和掩码
    annotated_image = cv2.addWeighted(image, 0.5, colored_mask, 0.5, 0)

    # 查找mask中的所有独立轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_codes = [
        f"QR Code {i+1}: {cv2.contourArea(contour)} pixels"
        for i, contour in enumerate(contours)
    ]

    return annotated_image, "\n".join(detected_codes)


if __name__ == "__main__":
    main()
    # python segment-clip-test.py --image image3.jpg

    # 一个gradio实例，效果远差于命令行版本
    interface = gr.Interface(
        fn=grad_detector,
        inputs=gr.Image(type="numpy", label="Upload an image"),
        outputs=[
            gr.Image(type="numpy", label="Annotated Image"),
            gr.Textbox(label="Detected QR Codes"),
        ],
        title="QR Code Segmentation",
        description="Segment QR Codes from Images",
    )
    # interface.launch()
