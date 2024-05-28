# QRCodeDetector

This is a QRCode detector and decoder using OpenCV and WeChatCV, used to detect and decode QRCode from image or video from dji.

# Introduction
1. python qr_recognition.py # 效果不错，可以同时识别多个+划线标注打印，相当于检测+定位
2. python sample_WeChatQRCode_detectAndDecode.py # 使用微信库，效果不错，但一次只能识别一个
3. python sample_QRCodeDetector_detectAndDecodeMulti.py # 另外写的，也能一次识别多个
4. python wechatDetectQR_test.py # 使用微信库，一次可以识别多个的，自己写的

5. 终极版本：支持摄像头、图片、文件夹，支持多个二维码识别，支持保存识别结果到图片和txt文件：
    - python QRCode_WeChatdetectAndDecode.py
    - python QRCode_WeChatdetectAndDecode.py --input image.png --output image1_res.jpg --save-txt image1_res.txt
    - python QRCode_WeChatdetectAndDecode.py --input-dir images_test
    - python QRCode_WeChatdetectAndDecode.py --input qrcode-file-tx-main/qrcode_video.mp4 --output qrcode-file-tx-main/qrcode_video_res.mp4 --save-txt qrcode-file-tx-main/result.txt
    - 增加滤波后效果反而不如之前的，增加pyzxing后
    python QRCode_WeChatdetectAndDecode.py --input ./images/QR-00042.jpg --output ./images/annotated-annotated_QR-00042.jpg --save-txt result_pyzxing.txt

# Others
1. huggingface project
2. data-analysis project
3. yolo-tracking project

# Requirement 
* opencv-python>=4.5.1
* opencv-contrib-python>=4.5.1
* numpy
* pyzxing
* pyzbar

# Reference
* [OpenCV Document](https://docs.opencv.org/master/namespaces.html)
* [WeChatCV/opencv_3rdparty]()
* [OpenCV-QRCodeDetector-Sample]()
* [qrcode-file-tx-main]()
 
# License 
[Apache v2 License](LICENSE).
