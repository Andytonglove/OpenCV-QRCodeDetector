import argparse
import copy
import os
import time
import numpy as np
import cv2 as cv
from pyzbar import pyzbar
from pyzxing import BarCodeReader
from PIL import Image
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import gradio as gr

# 初始化CLIPSeg模型和处理器
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)
    parser.add_argument(
        "--input",
        type=str,
        default="image1.png",
        help="path to input image or video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="path to save output image or video file",
        default="output.jpg",
    )
    parser.add_argument(
        "--save-txt",
        type=str,
        help="path to save recognized QR code texts",
    )
    parser.add_argument(
        "--input-dir", type=str, help="path to input directory with images"
    )

    args = parser.parse_args()

    return args


# 通过pyzbar库识别二维码
def decode_qrcode_with_pyzbar(image):
    decoded_objects = pyzbar.decode(image)
    qrcodes = []
    points = []

    for obj in decoded_objects:
        qrcodes.append(obj.data.decode("utf-8"))
        points.append([(int(point.x), int(point.y)) for point in obj.polygon])

    return qrcodes, points


# 集成pyzxing库识别二维码
def pyzxing_decode(image):
    reader = BarCodeReader()
    results = reader.decode_array(image)
    qrcodes = [item.get("parsed") for item in results]
    points = [item.get("points") for item in results]

    return qrcodes, points


def preprocess_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    gamma = 1.2
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    gamma_corrected = cv.LUT(equalized, lookUpTable)
    blurred = cv.GaussianBlur(gamma_corrected, (5, 5), 0)
    adaptive_thresh = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
    )
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    top_hat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    black_hat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    compensated = cv.add(cv.subtract(gray, black_hat), top_hat)

    return compensated


def detect_qr_codes_with_opencv(image):
    qrcode_detector = cv.wechat_qrcode_WeChatQRCode(
        "model/detect.prototxt",
        "model/detect.caffemodel",
        "model/sr.prototxt",
        "model/sr.caffemodel",
    )

    result = qrcode_detector.detectAndDecode(image)
    if not result[0]:
        result = decode_qrcode_with_pyzbar(image)
    return result


def detect_qr_codes_with_clipseg(image_path, prompt="QR code"):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    inputs = processor(
        text=[prompt], images=[image], padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.nn.functional.interpolate(
        outputs.logits.unsqueeze(1),
        size=(image_np.shape[0], image_np.shape[1]),
        mode="bilinear",
    )
    preds = torch.sigmoid(preds).squeeze().cpu().numpy()
    mask = (preds > 0.5).astype(np.uint8) * 255
    return image_np, mask


def draw_tags(image, qrcode_result, elapsed_time, total_qrcodes, current_qrcodes):
    for i in range(len(qrcode_result[0])):
        text = qrcode_result[0][i]
        corner = qrcode_result[1][i]

        corner_01 = (int(corner[0][0]), int(corner[0][1]))
        corner_02 = (int(corner[1][0]), int(corner[1][1]))
        corner_03 = (int(corner[2][0]), int(corner[2][1]))
        corner_04 = (int(corner[3][0]), int(corner[3][1]))

        # 各边勾画
        cv.line(
            image,
            (corner_01[0], corner_01[1]),
            (corner_02[0], corner_02[1]),
            (255, 0, 0),
            2,
        )
        cv.line(
            image,
            (corner_02[0], corner_02[1]),
            (corner_03[0], corner_03[1]),
            (255, 0, 0),
            2,
        )
        cv.line(
            image,
            (corner_03[0], corner_03[1]),
            (corner_04[0], corner_04[1]),
            (0, 255, 0),
            2,
        )
        cv.line(
            image,
            (corner_04[0], corner_04[1]),
            (corner_01[0], corner_01[1]),
            (0, 255, 0),
            2,
        )

        # 文本
        cv.putText(
            image,
            str(text),
            (corner_01[0], corner_01[1] - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

    # 处理时间
    cv.putText(
        image,
        "Elapsed Time:" + "{:.1f}".format(elapsed_time * 1000) + "ms",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        "Total QR Codes: " + str(total_qrcodes),
        (10, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        "Current QR Codes: " + str(current_qrcodes),
        (10, 90),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    return image


def overlay_mask(image, mask, alpha=0.5):
    colored_mask = cv.applyColorMap(mask, cv.COLORMAP_JET)
    overlay = cv.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlay


def crop_to_mask(image, mask):
    coords = cv.findNonZero(mask)
    x, y, w, h = cv.boundingRect(coords)
    cropped_image = image[y : y + h, x : x + w]
    return cropped_image


def solo_crop_to_mask(image, mask, padding=10):
    # 查找mask中的所有独立轮廓
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cropped_images = []

    for contour in contours:
        # 获取每个轮廓的边界框
        x, y, w, h = cv.boundingRect(contour)

        # 添加padding
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, image.shape[1] - x)
        h = min(h + 2 * padding, image.shape[0] - y)

        # 裁剪图像
        cropped_image = image[y : y + h, x : x + w]
        cropped_images.append(cropped_image)

    return cropped_images


# TODO: 第三次识别函数
def preprocess_for_qr(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    gamma = 1.2
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    gamma_corrected = cv.LUT(equalized, lookUpTable)
    blurred = cv.GaussianBlur(gamma_corrected, (5, 5), 0)
    adaptive_thresh = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
    )
    return adaptive_thresh


def second_stage_detection(cropped_images):
    second_stage_qrcodes = set()
    for cropped_image in cropped_images:
        preprocessed_image = preprocess_for_qr(cropped_image)
        # 使用不同的方法进行识别
        qrcodes_pyzbar, _ = decode_qrcode_with_pyzbar(preprocessed_image)
        qrcodes_pyzxing, _ = pyzxing_decode(preprocessed_image)

        # 合并结果
        second_stage_qrcodes.update(qrcodes_pyzbar)
        second_stage_qrcodes.update(qrcodes_pyzxing)

    return second_stage_qrcodes


def main():
    args = get_args()
    input_path = args.input
    output_path = args.output

    seen_qrcodes = set()  # 存储已识别的二维码内容× 去重后的
    current_qrcodes = set()  # 存储当前帧识别的二维码内容 所有二维码
    cnt_qrcodes = 0

    # Step 1: 使用第一段代码进行二维码检测
    image = cv.imread(input_path)
    result = detect_qr_codes_with_opencv(image)
    for text in result[0]:
        if text:
            cnt_qrcodes += 1
            current_qrcodes.add(text)
            if text not in seen_qrcodes:
                seen_qrcodes.add(text)
                print(f"New QR Code detected: {text}")
            else:
                print(f"Old QR Code detected: {text}")
    print(f"times detected QR Code: {cnt_qrcodes}")
    print(f"total QR Code detected: {len(seen_qrcodes)}")

    debug_image = draw_tags(image, result, 0, len(seen_qrcodes), len(result[0]))
    cv.imshow("Debug Image", debug_image)

    # Step 2: 使用第二段代码进行二次检测和裁剪
    _, mask = detect_qr_codes_with_clipseg(input_path)
    if mask is not None:
        overlay_image = overlay_mask(image, mask)
        cropped_image = crop_to_mask(image, mask)
        solo_cropped_images = solo_crop_to_mask(image, mask, padding=10)

        cv.imshow("Overlay Image", overlay_image)
        cv.imshow("Cropped Image", cropped_image)

        print(f"total detected QR Code Masks: {len(solo_cropped_images)}")
        if len(solo_cropped_images) > 1:
            for i, cropped_image in enumerate(solo_cropped_images):
                cv.imshow(f"Cropped Image {i+1}", cropped_image)

        cv.waitKey(0)
        cv.destroyAllWindows()

        if output_path:
            print(f"Output path of overlay_image: {output_path}. No saving this time.")
            # cv.imwrite(output_path, overlay_image)

    # TODO: Step 3: 对裁剪后提取的做二次识别，考虑效果策略
    maybe_ignore_cnt = len(solo_cropped_images) - cnt_qrcodes
    print(f"Maybe ignore QR Codes: {maybe_ignore_cnt if maybe_ignore_cnt > 0 else 0}.")

    # 做二次识别
    print("\n")
    sec_cnt_qrcodes = 0
    sec_seen_qrcodes = set()
    sec_current_qrcodes = set()

    second_stage_qrcodes = second_stage_detection(solo_cropped_images)
    for text in second_stage_qrcodes:
        if text:
            sec_cnt_qrcodes += 1
            sec_current_qrcodes.add(text)
            if text not in sec_seen_qrcodes:
                sec_seen_qrcodes.add(text)
                print(f"New QR Code detected in second stage: {text}")
            else:
                print(f"Old QR Code detected in second stage: {text}")

    # 对两个set执行去重，merge
    all_detected_qrcodes = seen_qrcodes.union(sec_seen_qrcodes)
    print(all_detected_qrcodes)

    sec_new_detect = len(sec_seen_qrcodes) - len(seen_qrcodes)
    if sec_new_detect > 0:
        print(f"New QR Code detected in second stage: {sec_new_detect}")
    else:
        print(f"No new QR Code detected in second stage, cnt = {sec_new_detect}")


if __name__ == "__main__":
    main()
    # python detect-mult-test.py --input image1.png
