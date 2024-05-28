#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import os
import numpy as np
import cv2 as cv
from flask import Flask, request, jsonify
from pyzbar import pyzbar  # 导入pyzbar库
from pyzxing import BarCodeReader  # 导入pyzxing库

app = Flask(__name__)

print(cv.__version__)


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

    return qrcodes, points


# 预处理图像
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


# 超分辨率，效果不好
def super_res(image):
    sr = cv.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("model/ESPCN_x3.pb")
    sr.setModel("espcn", 3)
    result = sr.upsample(image)
    return result


@app.route("/detect_qrcode", methods=["POST"])
def detect_qrcode():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv.imdecode(npimg, cv.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    qrcode_detector = cv.wechat_qrcode_WeChatQRCode(
        "model/detect.prototxt",
        "model/detect.caffemodel",
        "model/sr.prototxt",
        "model/sr.caffemodel",
    )

    result = qrcode_detector.detectAndDecode(image)
    if not result[0]:
        result = decode_qrcode_with_pyzbar(image)

    qrcodes = [
        {"text": text, "points": points} for text, points in zip(result[0], result[1])
    ]
    return jsonify({"qrcodes": qrcodes})


@app.route("/process_directory", methods=["POST"])
def process_directory():
    input_dir = request.form.get("input_dir")
    save_txt_path = request.form.get("save_txt_path")

    if not input_dir or not os.path.isdir(input_dir):
        return jsonify({"error": "Invalid directory"}), 400

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

            try:
                result = qrcode_detector.detectAndDecode(image)
                if not result[0]:
                    result = decode_qrcode_with_pyzbar(image)
            except UnicodeDecodeError:
                print("UnicodeDecodeError encountered. Continuing...")
                continue

            for text in result[0]:
                if text and text not in seen_qrcodes:
                    seen_qrcodes.add(text)
                    if txt_file:
                        txt_file.write(f"{filename}: {text}\n")

            debug_image = draw_tags(image, result, 0, len(seen_qrcodes), len(result[0]))
            output_image_path = os.path.join(input_dir, "annotated_" + filename)
            cv.imwrite(output_image_path, debug_image)

    if txt_file:
        txt_file.write(f"Total QR Codes detected: {len(seen_qrcodes)}\n")
        txt_file.close()

    return jsonify({"total_qrcodes": len(seen_qrcodes)})


def draw_tags(image, qrcode_result, elapsed_time, total_qrcodes, current_qrcodes):
    for i in range(len(qrcode_result[0])):
        text = qrcode_result[0][i]
        corner = qrcode_result[1][i]

        corner_01 = (int(corner[0][0]), int(corner[0][1]))
        corner_02 = (int(corner[1][0]), int(corner[1][1]))
        corner_03 = (int(corner[2][0]), int(corner[2][1]))
        corner_04 = (int(corner[3][0]), int(corner[3][1]))

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
