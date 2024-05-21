#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv

print(cv.__version__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Detector準備 #############################################################
    qrcode_detector = cv.wechat_qrcode_WeChatQRCode(
        "model/detect.prototxt",
        "model/detect.caffemodel",
        "model/sr.prototxt",
        "model/sr.caffemodel",
    )

    elapsed_time = 0
    seen_qrcodes = set()  # 存储已识别的二维码内容

    while True:
        start_time = time.time()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        result = qrcode_detector.detectAndDecode(image)

        # 描画 ################################################################
        debug_image = draw_tags(debug_image, result, elapsed_time)
        # 当识别到二维码时，打印出二维码的内容
        # if len(result[0]) > 0:
        #     print(result)

        # 存储识别到的二维码内容并避免重复
        for text in result[0]:
            if text and text not in seen_qrcodes:
                seen_qrcodes.add(text)
                print(f"New QR Code detected: {text}")
                # 打印出二维码的位置
                print(f"QR Code detected at: {result[1][0]}")

        elapsed_time = time.time() - start_time

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow("QR Code Detector Demo", debug_image)

    cap.release()
    cv.destroyAllWindows()


# def draw_tags(
#     image,
#     qrcode_result,
#     elapsed_time,
# ):
#     if len(qrcode_result[0]) > 0:
#         text = qrcode_result[0][0]
#         corner = qrcode_result[1][0]

#         corner_01 = (int(corner[0][0]), int(corner[0][1]))
#         corner_02 = (int(corner[1][0]), int(corner[1][1]))
#         corner_03 = (int(corner[2][0]), int(corner[2][1]))
#         corner_04 = (int(corner[3][0]), int(corner[3][1]))

#         # 各辺
#         cv.line(
#             image,
#             (corner_01[0], corner_01[1]),
#             (corner_02[0], corner_02[1]),
#             (255, 0, 0),
#             2,
#         )
#         cv.line(
#             image,
#             (corner_02[0], corner_02[1]),
#             (corner_03[0], corner_03[1]),
#             (255, 0, 0),
#             2,
#         )
#         cv.line(
#             image,
#             (corner_03[0], corner_03[1]),
#             (corner_04[0], corner_04[1]),
#             (0, 255, 0),
#             2,
#         )
#         cv.line(
#             image,
#             (corner_04[0], corner_04[1]),
#             (corner_01[0], corner_01[1]),
#             (0, 255, 0),
#             2,
#         )

#         # テキスト
#         cv.putText(
#             image,
#             str(text),
#             (10, 55),
#             cv.FONT_HERSHEY_SIMPLEX,
#             0.75,
#             (0, 255, 0),
#             2,
#             cv.LINE_AA,
#         )

#     # 処理時間
#     cv.putText(
#         image,
#         "Elapsed Time:" + "{:.1f}".format(elapsed_time * 1000) + "ms",
#         (10, 30),
#         cv.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (0, 255, 0),
#         2,
#         cv.LINE_AA,
#     )

#     return image


# 改为一次勾画多个
def draw_tags(image, qrcode_result, elapsed_time):
    for i in range(len(qrcode_result[0])):
        text = qrcode_result[0][i]
        corner = qrcode_result[1][i]

        corner_01 = (int(corner[0][0]), int(corner[0][1]))
        corner_02 = (int(corner[1][0]), int(corner[1][1]))
        corner_03 = (int(corner[2][0]), int(corner[2][1]))
        corner_04 = (int(corner[3][0]), int(corner[3][1]))

        # 各辺
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

        # テキスト
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

    # 処理時間
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

    return image


if __name__ == "__main__":
    main()