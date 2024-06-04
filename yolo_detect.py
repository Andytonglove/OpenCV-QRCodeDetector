import cv2
import time
import sys
import numpy as np

import argparse

import QRCode_WeChatdetectAndDecode as decode


def get_args():
    """
    Parse command line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, help="path to input image or video file")
    parser.add_argument(
        "--output",
        type=str,
        help="path to save output image or video file",
        default="output.jpg",  # 默认输出文件名.avi也可
    )

    args = parser.parse_args()

    return args


def build_model(is_cuda):
    """
    Build and configure the YOLO model for QR code detection.

    Required:
    - best.onnx: The YOLO model file, which is a pre-trained model from best.pt.

    Parameters:
    - is_cuda (bool): Flag indicating whether to use CUDA for GPU acceleration.

    Returns:
    - net (cv2.dnn_Net): The configured YOLO model.

    """
    net = cv2.dnn.readNet("model/best.onnx")
    if is_cuda:
        # print("Attempting to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        # print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4
class_list = ["QR"]


def detect(image, net):
    """
    Detects objects in an image using a pre-trained neural network.

    Args:
        image (numpy.ndarray): The input image.
        net: The pre-trained neural network.

    Returns:
        numpy.ndarray: The predictions of the neural network.
    """
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False
    )
    net.setInput(blob)
    preds = net.forward()
    return preds


def load_capture(video_path):
    """
    Loads a video capture object from the specified video path.

    Parameters:
    video_path (str): The path to the video file.

    Returns:
    capture (cv2.VideoCapture): The video capture object.

    """
    capture = cv2.VideoCapture(video_path)
    return capture


def load_classes():
    """
    Load the list of classes from the 'classes.txt' file.

    Returns:
        class_list (list): A list of class names.

    """
    class_list = []
    with open("classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list
    # class_list = load_classes()


def wrap_detection(input_image, output_data):
    """
    Detects objects in an input image using YOLOv5 model.

    Args:
        input_image (numpy.ndarray): The input image.
        output_data (numpy.ndarray): The output data from the YOLOv5 model.

    Returns:
        tuple: A tuple containing the detected class IDs, confidences, and bounding boxes.

    """
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if classes_scores[class_id] > 0.25:

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


def format_yolov5(frame):
    """
    Formats the input frame to have a square shape.

    Args:
        frame: The input frame to be formatted.

    Returns:
        The formatted frame with a square shape.

    """
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


def process_image(image_path, output_path="output.jpg"):
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

    net = build_model(is_cuda)

    frame = cv2.imread(image_path)
    inputImage = format_yolov5(frame)
    outs = detect(inputImage, net)

    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

    qr_regions = []
    regions_result = []
    idx = 0  # QR Region index
    cnt_qrcodes = set()

    for classid, confidence, box in zip(class_ids, confidences, boxes):
        color = colors[int(classid) % len(colors)]
        cv2.rectangle(frame, box, color, 2)
        cv2.rectangle(
            frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1
        )

        # Extract the QR code region and save it to the list
        qr_region = frame[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
        qr_regions.append(qr_region)

        # 使用subprocess调用QRCode_WeChatdetectAndDecode.py
        result = decode.sub_process(qr_region)
        cv2.putText(
            frame,
            class_list[classid] + ": " + str(result[0]) if len(result) > 0 else None,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )
        for text in result[0]:
            if text:
                if text not in cnt_qrcodes:
                    cnt_qrcodes.add(text)
                    print(f"New QR Code detected: {text}")
                    print(f"QR Code detected at: {result[1][0]}")
                else:
                    print(f"Duplicate QR Code detected: {text}")
        regions_result.append(result)
        # print("QR Region " + str(idx + 1) + ":", result)
        # cv2.imshow("QR Region " + str(idx + 1), qr_region)
        idx += 1

    cv2.imshow("output-detect and decode", frame)
    cv2.imwrite(output_path, frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return qr_regions


def process_video(video_path, output_path="output.mp4"):
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

    net = build_model(is_cuda)
    capture = load_capture(video_path)

    # Get video properties
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start = time.time_ns()
    frame_count = 0
    total_frames = 0  # 记录总帧数
    # fps = -1

    # 视频逻辑
    qr_regions = []
    regions_result = []
    idx = 0  # QR Region index
    cnt_qrcodes = set()

    while True:

        _, frame = capture.read()
        if frame is None:
            print("End of stream")
            break

        inputImage = format_yolov5(frame)
        outs = detect(inputImage, net)

        class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

        frame_count += 1
        total_frames += 1

        for classid, confidence, box in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(
                frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1
            )

            # Extract the QR code region and save it to the list
            qr_region = frame[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]

            if qr_region.size == 0 or qr_region is None:
                print(
                    f"Empty QR code region detected at frame {total_frames}. Skipping."
                )
                continue

            qr_regions.append(qr_region)
            # 使用subprocess调用QRCode_WeChatdetectAndDecode.py
            result = decode.sub_process(qr_region)

            cv2.putText(
                frame,
                (
                    class_list[classid] + ": " + str(result[0])
                    if len(result) > 0
                    else None
                ),
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
            )
            for text in result[0]:
                if text:
                    if text not in cnt_qrcodes:
                        cnt_qrcodes.add(text)
                        print(f"New QR Code detected: {text}")
                        print(f"QR Code detected at: {result[1][0]}")
                    else:
                        print(f"Duplicate QR Code detected: {text}")
            regions_result.append(result)
            # print("QR Region " + str(idx + 1) + ":", result)
            # cv2.imshow("QR Region " + str(idx + 1), qr_region)
            idx += 1

        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(
                frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        cv2.imshow("output", frame)

        # Write the frame to the output video file
        out.write(frame)

        if cv2.waitKey(1) > -1:
            print("finished by user")
            break

    print("Total frames: " + str(total_frames))

    out.release()  # Release the VideoWriter
    print("Output video saved to: " + output_path)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_args()
    if args.input:
        if args.input.endswith(".jpg") or args.input.endswith(".png"):
            if args.output:
                process_image(args.input, args.output)
            else:
                process_image(args.input)
        elif args.input.endswith(".mp4") or args.input.endswith(".avi"):
            if args.output:
                process_video(args.input, args.output)
            else:
                process_video(args.input)
        else:
            print("Invalid file type. Supported types: jpg, png, mp4, avi")
    else:
        print("No input file provided.")
