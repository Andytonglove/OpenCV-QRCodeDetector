import copy
import cv2 as cv
import os
import gradio as gr


# 定义二维码检测函数
def detect_qr_code(image):
    # 初始化 WeChatQRCode 检测器
    qrcode_detector = cv.wechat_qrcode_WeChatQRCode(
        "model/detect.prototxt",
        "model/detect.caffemodel",
        "model/sr.prototxt",
        "model/sr.caffemodel",
    )

    # 检测并解码二维码
    result = qrcode_detector.detectAndDecode(image)
    seen_qrcodes = set()

    # 创建副本以绘制检测框
    debug_image = copy.deepcopy(image)
    for text in result[0]:
        if text:
            seen_qrcodes.add(text)

    debug_image = draw_tags(debug_image, result, 0, len(seen_qrcodes), len(result[0]))

    return debug_image, list(seen_qrcodes)


# 定义绘制标签的函数
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


# 创建 Gradio 接口
interface = gr.Interface(
    fn=detect_qr_code,
    inputs=gr.Image(type="numpy", label="Upload an image"),
    outputs=[
        gr.Image(type="numpy", label="Annotated Image"),
        gr.Textbox(label="Detected QR Codes"),
    ],
    title="QR Code Detector",
    description="Upload an image to detect QR codes in it using OpenCV and WeChatQRCode.",
)

# 启动 Gradio 接口
interface.launch()
