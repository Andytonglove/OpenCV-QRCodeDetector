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

print(cv.__version__)

"""
调用方法：
python QRCode_WeChatdetectAndDecode.py --input-dir folder --save-txt result.txt
python QRCode_WeChatdetectAndDecode.py --input image.jpg --output output.jpg --save-txt result.txt
python QRCode_WeChatdetectAndDecode.py --input video.mp4 --output output.avi --save-txt result.txt
python QRCode_WeChatdetectAndDecode.py --input image.jpg --clipseg
"""


def get_args():
    """
    Parse command line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
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
    # 参数选择是否进行clipseg识别，该参数只能对图片进行识别，默认为False
    parser.add_argument(
        "--clipseg",
        action="store_true",
        help="Use CLIPSeg model for QR code detection",
    )

    args = parser.parse_args()

    return args


# 通过pyzbar库识别二维码
def decode_qrcode_with_pyzbar(image):
    """
    Decode QR codes in the given image using pyzbar library.

    Args:
        image: The image containing QR codes.

    Returns:
        A tuple containing the decoded QR codes and their corresponding points.

    """
    # 给pyzbar增加一个断言参数，防止警告？ymbols=[pyzbar.ZBarSymbol.QRCODE]
    decoded_objects = pyzbar.decode(image)
    qrcodes = []
    points = []

    for obj in decoded_objects:
        qrcodes.append(obj.data.decode("utf-8"))
        points.append([(int(point.x), int(point.y)) for point in obj.polygon])

    return qrcodes, points


# 集成pyzxing库识别二维码
def pyzxing_decode(image):
    """
    Decodes QR codes from the given image using the pyzxing library.

    Args:
        image: The image containing the QR codes.

    Returns:
        A tuple containing two lists:
        - qrcodes: A list of decoded QR codes as strings.
        - points: A list of points representing the location of each QR code in the image.
    """
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

        # print(f"Filename: {filename}")
        # print(f"Format: {format}")
        # print(f"Type: {type_}")
        # print(f"Raw: {raw}")
        # print(f"Parsed: {parsed}")
        # print(f"Points: {points}")
        # print("\n")

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
    """
    Initializes the CLIPSeg processor and model.

    Returns:
        Tuple: A tuple containing the CLIPSeg processor and model.
    """
    # 中间再引入，可以加快速度
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    return processor, model


def detect_qr_codes_with_clipseg(image_path, processor, model, prompt="QR code"):
    """
    Detects QR codes in an image using the CLIPSeg model.

    Args:
        image_path (str): The path to the image file.
        processor: The CLIP processor used for text and image encoding.
        model: The CLIPSeg model used for QR code detection.
        prompt (str, optional): The prompt used for QR code detection. Defaults to "QR code".

    Returns:
        tuple: A tuple containing the original image and the binary mask of detected QR codes.
    """
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
    """
    Overlay a color mask on an image.

    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The color mask to overlay on the image.
        alpha (float, optional): The transparency of the overlay. Defaults to 0.5.

    Returns:
        numpy.ndarray: The image with the color mask overlay.

    """
    # 将mask转换为彩色图像
    colored_mask = cv.applyColorMap(mask, cv.COLORMAP_JET)

    # 叠加图像和掩码
    overlay = cv.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlay


def crop_to_mask(image, mask):
    """
    Crop the input image based on the provided mask.

    Parameters:
    - image: The input image to be cropped.
    - mask: The mask indicating the region to be cropped.

    Returns:
    - cropped_image: The cropped image.

    """
    # 找到mask的非零像素的边界
    coords = cv.findNonZero(mask)
    x, y, w, h = cv.boundingRect(coords)
    # 裁剪图像
    cropped_image = image[y : y + h, x : x + w]
    return cropped_image


def solo_crop_to_mask(image, mask, padding=10):
    """
    Crop the image based on the provided mask and return a list of cropped images.

    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The mask indicating the regions of interest.
        padding (int, optional): The padding to be added around the bounding box. Defaults to 10.

    Returns:
        list: A list of cropped images.

    """
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


def preprocess_for_qr(image):
    """
    Preprocesses an image for QR code detection.

    Args:
        image: The input image.

    Returns:
        The preprocessed image.

    """
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
    """
    Perform second stage detection on a list of cropped images.

    Args:
        cropped_images (list): A list of cropped images.

    Returns:
        set: A set of detected QR codes from the second stage detection.
    """
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


# 预处理图像
def preprocess_image(image):
    """
    Preprocesses the input image for QR code detection and decoding.

    Args:
        image: The input image to be preprocessed.

    Returns:
        The preprocessed image.

    """
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
    """
    Applies super resolution to the input image using OpenCV's DNN super resolution model.

    Parameters:
    image (numpy.ndarray): The input image to be processed.

    Returns:
    numpy.ndarray: The super resolved image.

    """
    # 使用OpenCV的DNN超分辨率模型
    sr = cv.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("model/ESPCN_x3.pb")
    sr.setModel("espcn", 3)
    result = sr.upsample(image)
    return result


# 主函数
def main():
    """
    Main function that performs QR code detection and decoding.

    This function takes command line arguments, initializes the camera or input file,
    prepares the QR code detector, and starts the detection and decoding process.
    It also handles saving the output video and text file if specified.

    Args:
        None

    Returns:
        None
    """
    # 参数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    input_path = args.input
    output_path = args.output
    save_txt_path = args.save_txt
    input_dir = args.input_dir
    use_clipseg = args.clipseg

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

    elapsed_time = 0  # 初始化时间记录
    seen_qrcodes = set()  # 存储已识别的二维码内容
    current_qrcodes = set()  # 存储当前帧识别的二维码内容
    # clipseg_qrcodes = set()  # 存储CLIPSeg识别的二维码内容

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

    # 初始化CLIPSeg模型
    if use_clipseg:
        print(f"transformers version: {torch.__version__}")
        processor, model = initialize_clipseg()

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
                else:
                    print(f"Duplicate QR Code detected: {text}")

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

        # #####################################################################
        # TODO: 使用CLIPSeg模型识别二维码部分，只对单张图片进行处理
        if use_clipseg and input_path.endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        ):
            # 计算使用opencv识别的二维码数量
            opencv_detect_qrcode = len(result[0])
            cilpseg_detect_qrcode = 0
            print("Using CLIPSeg model for QR code detection...")
            image_np, mask = detect_qr_codes_with_clipseg(input_path, processor, model)

            if mask is not None:
                # 叠加掩码到原图像
                overlay_image = overlay_mask(image_np, mask)
                # 根据掩码裁剪图像
                cropped_image = crop_to_mask(image_np, mask)
                # 根据mask单独裁剪图像
                solo_cropped_images = solo_crop_to_mask(image_np, mask, padding=10)

                # 显示原图、掩码和叠加后的图像，保存 裁剪……
                cv.imwrite("overlay_image.jpg", overlay_image)

                print(f"total detected QR Code Masks: {len(solo_cropped_images)}")
                # 计算显示裁剪后的单个图像
                if len(solo_cropped_images) >= 1:
                    # for i, cropped_image in enumerate(solo_cropped_images):
                    # cv.imshow(f"Cropped Image {i+1}", cropped_image)
                    # cv.imwrite(f"cropped_output_{i+1}.png", cropped_image)
                    # 做第二阶段检测 solo_cropped_images
                    cilpseg_detect_qrcode = (
                        len(solo_cropped_images) - opencv_detect_qrcode
                    )
                    print(
                        f"Maybe ignore QR Codes: {cilpseg_detect_qrcode if cilpseg_detect_qrcode > 0 else 0}"
                    )
                    clipseg_qrcodes = second_stage_detection(solo_cropped_images)
                    for text in clipseg_qrcodes:
                        if text:
                            print(f"QR Code detected in second stage: {text}")
                            # 在seen_qrcodes中查找，如果没有则输出并添加到seen_qrcodes
                            if text not in seen_qrcodes:
                                seen_qrcodes.add(text)
                                if txt_file:
                                    txt_file.write(text + "\n")

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
    print(f"Total QR Codes detected: {len(seen_qrcodes)}")


# 每次勾画多个
def draw_tags(image, qrcode_result, elapsed_time, total_qrcodes, current_qrcodes):
    """
    Draws bounding boxes and text labels on the input image based on the QR code detection results.

    Args:
        image (numpy.ndarray): The input image on which to draw the bounding boxes and text labels.
        qrcode_result (tuple): A tuple containing the QR code detection results. It consists of two lists:
                               the first list contains the text labels of the detected QR codes,
                               and the second list contains the corner coordinates of the bounding boxes.
        elapsed_time (float): The elapsed time for the QR code detection process in seconds.
        total_qrcodes (int): The total number of QR codes detected in the image.
        current_qrcodes (int): The number of QR codes currently being processed.

    Returns:
        numpy.ndarray: The input image with bounding boxes and text labels drawn on it.
    """
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


# ADD: 实现目录处理
def process_directory(input_dir, save_txt_path):
    """
    Process a directory containing images and detect QR codes in each image.

    Args:
        input_dir (str): The path to the directory containing the images.
        save_txt_path (str): The path to save the detected QR codes in a text file.

    Returns:
        None
    """
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
