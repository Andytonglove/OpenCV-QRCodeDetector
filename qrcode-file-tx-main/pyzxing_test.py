from pyzxing import BarCodeReader
from pyzbar import pyzbar
import cv2 as cv


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
    # qrcodes中的内容是字节串，需要解码为字符串，但UnicodeDecodeError: 'utf-8' codec can't decode byte 0xef in position 3: invalid continuation byte
    # qrcodes = [
    #     qrcode.decode("utf-8", errors="replace") if qrcode is not None else ""
    #     for qrcode in qrcodes
    # ]
    return qrcodes, points


def decode_qrcode_with_pyzbar(image):
    decoded_objects = pyzbar.decode(image)
    qrcodes = []
    points = []

    for obj in decoded_objects:
        qrcodes.append(obj.data.decode("utf-8"))
        points.append([(int(point.x), int(point.y)) for point in obj.polygon])

    return qrcodes, points


if __name__ == "__main__":
    reader = BarCodeReader()
    results = reader.decode("image.png")
    print(results)

    # 对文件夹中的所有图片进行二维码识别
    # opencv读取图片
    image = cv.imread("image.png")

    qrcodes1, points1 = pyzxing_decode(image)
    print(qrcodes1)
    print(points1)

    qrcodes2, points2 = decode_qrcode_with_pyzbar(image)
    print(qrcodes2)
    print(points2)
