#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import os
import numpy as np

from pyzbar import pyzbar  # 导入pyzbar库
from pyzxing import BarCodeReader  # 导入pyzxing库

from PIL import Image
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import gradio as gr

print(cv.__version__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)
    parser.add_argument("--input", type=str, help="path to input image or video file")
    parser.add_argument(
        "--output",
        type=str,
        help="path to save output image or video file",
        default="output.jpg",  # 默认输出文件名.avi也可
    )
    parser.add_argument(
        "--save-txt",
        type=str,
        # default="result.txt",
        help="path to save recognized QR code texts",
    )
    parser.add_argument(
        "--input-dir", type=str, help="path to input directory with images"
    )

    args = parser.parse_args()

    return args


# 通过pyzbar库识别二维码
def decode_qrcode_with_pyzbar(image):
    # 给pyzbar增加一个断言参数，防止出错？ymbols=[pyzbar.ZBarSymbol.QRCODE]
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
    for item in results:
        filename = (
            item.get("filename").decode("utf-8", errors="replace")
            if isinstance(item.get("filename"), bytes)
            else item.get("filename")
        )
        format = (
            item.get("format").decode("utf-8", errors="replace")
            if isinstance(item.get("format"), bytes)
            else item.get("format")
        )
        type_ = (
            item.get("type").decode("utf-8", errors="replace")
            if isinstance(item.get("type"), bytes)
            else item.get("type")
        )
        raw = (
            item.get("raw").decode("utf-8", errors="replace")
            if isinstance(item.get("raw"), bytes)
            else item.get("raw")
        )
        parsed = (
            item.get("parsed").decode("utf-8", errors="replace")
            if isinstance(item.get("parsed"), bytes)
            else item.get("parsed")
        )
        points = item.get("points")

        print(f"Filename: {filename}")
        print(f"Format: {format}")
        print(f"Type: {type_}")
        print(f"Raw: {raw}")
        print(f"Parsed: {parsed}")
        print(f"Points: {points}")
        print("\n")

    # 返回格式和pyzbar一样
    qrcodes = [item.get("parsed") for item in results]
    points = [item.get("points") for item in results]
    # # qrcodes中的内容是字节串，需要解码为字符串，注意解码
    # qrcodes = [
    #     qrcode.decode("utf-8", errors="replace") if qrcode is not None else ""
    #     for qrcode in qrcodes
    # ]

    return qrcodes, points


# 采用CLIPSeg模型识别二维码
def initialize_clipseg():
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    return processor, model


def detect_qr_codes_with_clipseg(image_path, processor, model, prompt="QR code"):
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


# 预处理图像
def preprocess_image(image):
    # 转换为灰度图像
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 应用自适应直方图均衡化（CLAHE）
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # 应用伽马校正
    gamma = 1.2
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    gamma_corrected = cv.LUT(equalized, lookUpTable)

    # 应用高斯模糊
    blurred = cv.GaussianBlur(gamma_corrected, (5, 5), 0)
    # 中值滤波
    # filtered_image = cv.medianBlur(image, 5)
    # 双边滤波
    # filtered_image = cv.bilateralFilter(image, 9, 75, 75)

    # 自适应阈值
    adaptive_thresh = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
    )

    # 光照补偿：使用顶帽变换和黑帽变换
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    top_hat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    black_hat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    compensated = cv.add(cv.subtract(gray, black_hat), top_hat)

    return compensated


# 超分辨率，效果不好
def super_res(image):
    # 使用OpenCV的DNN超分辨率模型
    sr = cv.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("model/ESPCN_x3.pb")
    sr.setModel("espcn", 3)
    result = sr.upsample(image)
    return result


def main():
    # 参数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    input_path = args.input
    output_path = args.output
    save_txt_path = args.save_txt
    input_dir = args.input_dir

    cap = None  # 初始化cap

    # 抽帧参数
    frame_skip = 5  # 每隔5帧处理一次
    frame_count = 0  # 初始化帧计数器

    # 相机准备 #################################################################
    if input_dir:
        if not os.path.isdir(input_dir):
            print("Error: input directory does not exist.")
            return
        process_directory(input_dir, save_txt_path)
    else:
        if input_path:
            if not os.path.isfile(input_path):
                print("Error: input file does not exist.")
                return
            cap = cv.VideoCapture(input_path)
        else:
            cap = cv.VideoCapture(cap_device)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    if cap is None or not cap.isOpened():
        print("Error: No video source available. Exiting.")
        return

    # Detector准备 #############################################################
    qrcode_detector = cv.wechat_qrcode_WeChatQRCode(
        # detect模型 + sr超分模型
        "model/detect.prototxt",
        "model/detect.caffemodel",
        "model/sr.prototxt",
        "model/sr.caffemodel",
    )

    elapsed_time = 0
    seen_qrcodes = set()  # 存储已识别的二维码内容
    current_qrcodes = set()  # 存储当前帧识别的二维码内容

    if input_path and output_path:
        fourcc = cv.VideoWriter_fourcc(*"XVID")
        out = None
        if input_path.endswith((".mp4", ".avi", ".mov")):
            out = cv.VideoWriter(
                output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4)))
            )
        else:
            out = None

    if save_txt_path:
        txt_file = open(save_txt_path, "w")
    else:
        txt_file = None

    while True:
        start_time = time.time()

        # 只在每隔 frame_skip 帧时处理图像
        # if frame_count % frame_skip == 0:

        # 捕捉相机 #############################################################
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)

        # 应用滤波器 ###########################################################
        # filtered_image = preprocess_image(image)
        # super_res_image = super_res(image)

        # 实现检测 #############################################################
        result = qrcode_detector.detectAndDecode(image)
        if not result[0]:  # 如果没有检测到二维码，使用pyzbar再试一次
            result = decode_qrcode_with_pyzbar(image)
            # TODO: 增加pyzxing库识别二维码，可加三次优化四次识别
            # result = pyzxing_decode(image)

        # 每一帧的统计 #########################################################
        current_qrcodes.clear()
        for text in result[0]:
            if text:
                current_qrcodes.add(text)
                if text not in seen_qrcodes:
                    seen_qrcodes.add(text)
                    print(f"New QR Code detected: {text}")
                    # print(f"QR Code detected at: {result[1][0]}")
                    if txt_file:
                        txt_file.write(text + "\n")

        debug_image = draw_tags(
            debug_image, result, elapsed_time, len(seen_qrcodes), len(current_qrcodes)
        )

        # 描画 ################################################################
        # debug_image = draw_tags(debug_image, result, elapsed_time)
        # 当识别到二维码时，打印出二维码的内容
        # if len(result[0]) > 0:
        #     print(result)

        # 存储识别到的二维码内容并避免重复
        # for text in result[0]:
        #     if text and text not in seen_qrcodes:
        #         seen_qrcodes.add(text)
        #         print(f"New QR Code detected: {text}")
        #         # 打印出二维码的位置
        #         print(f"QR Code detected at: {result[1][0]}")

        elapsed_time = time.time() - start_time
        if input_path and out:
            out.write(debug_image)

        # 按键处理(ESC：结束) ##################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        if input_path and not out:
            cv.imwrite(output_path, debug_image)
            break

        # 画面展示 #############################################################
        cv.imshow("QR Code Detector", debug_image)

        # 增加帧计数器
        # frame_count += 1

    cap.release()
    if input_path and out:
        out.release()
    if txt_file:
        txt_file.close()
    cv.destroyAllWindows()


# 每次勾画多个
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


# 实现目录处理
def process_directory(input_dir, save_txt_path):
    qrcode_detector = cv.wechat_qrcode_WeChatQRCode(
        "model/detect.prototxt",
        "model/detect.caffemodel",
        "model/sr.prototxt",
        "model/sr.caffemodel",
    )

    seen_qrcodes = set()

    if save_txt_path:
        txt_file = open(save_txt_path, "w", encoding="utf-8")
    else:
        txt_file = None

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image_path = os.path.join(input_dir, filename)
            image = cv.imread(image_path)
            if image is None:
                continue

            # filtered_image = preprocess_image(image)

            # UnicodeDecodeError: 'utf-8' codec can't decode byte ...
            # result = qrcode_detector.detectAndDecode(image)

            try:
                result = qrcode_detector.detectAndDecode(image)
                if not result[0]:  # 如果没有检测到二维码，使用pyzbar再试一次
                    result = decode_qrcode_with_pyzbar(image)
                # TODO: 增加pyzxing库识别二维码
            except UnicodeDecodeError:
                print("UnicodeDecodeError encountered. Continuing...")
                result = None
                continue

            for text in result[0]:
                if text and text not in seen_qrcodes:
                    seen_qrcodes.add(text)
                    print(f"New QR Code detected in {filename}: {text}")
                    if txt_file:
                        txt_file.write(f"{filename}: {text}\n")

            # 画面展示，保存
            debug_image = draw_tags(image, result, 0, len(seen_qrcodes), len(result[0]))
            output_image_path = os.path.join(input_dir, "annotated_" + filename)
            cv.imwrite(output_image_path, debug_image)

    if txt_file:
        txt_file.write(f"Total QR Codes detected: {len(seen_qrcodes)}\n")
        txt_file.close()

    print(f"Total QR Codes detected: {len(seen_qrcodes)}")


if __name__ == "__main__":
    main()
