#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import os

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
        default="output.avi",
    )
    parser.add_argument(
        "--save-txt",
        type=str,
        default="result.txt",
        help="path to save recognized QR code texts",
    )
    parser.add_argument(
        "--input-dir", type=str, help="path to input directory with images"
    )

    args = parser.parse_args()

    return args


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

        # 捕捉相机 #############################################################
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)

        # 实现检测 #############################################################
        result = qrcode_detector.detectAndDecode(image)

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
        txt_file = open(save_txt_path, "w")
    else:
        txt_file = None

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image_path = os.path.join(input_dir, filename)
            image = cv.imread(image_path)
            if image is None:
                continue

            result = qrcode_detector.detectAndDecode(image)

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
        txt_file.close()

    print(f"Total QR Codes detected: {len(seen_qrcodes)}")


if __name__ == "__main__":
    main()
