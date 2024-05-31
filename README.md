# QRCodeDetector

This is a `QRCode detector and decoder` using OpenCV and WeChatCV, used to detect and decode QRCode from image or video from dji.

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


# Requirement 
* opencv-python>=4.5.1
* opencv-contrib-python>=4.5.1
* numpy
* pyzxing
* pyzbar

#### Usage

use cmd to detect QR codes from video or image or convert them. 
make sure you have model and relevant packages installed. 
- `python QRCode_WeChatdetectAndDecode.py` use default camera in your PC, you can use different arguments for need.
- python QRCode_WeChatdetectAndDecode.py --input image.png --output image_res.jpg --save-txt image_res.txt
- python QRCode_WeChatdetectAndDecode.py --input-dir images_test
- python QRCode_WeChatdetectAndDecode.py --input qrcode_video.mp4 --output qrcode_video_res.mp4 --save-txt result.txt
- python segment-clip-test.py --image image1.jpg
- `python detect-mult-test.py` is a sum of the following solutions, including OpenCV and CLIP.

#### Conditions

Under tests, you can detect and decode QRCode in about 20cm with the size of 4cm and the resolution of 200 pixel*.

So, following the exprience forum, you can track and decode QRCode in 5 meters with the following parameters:
```
1. 摄像头参数
焦距：35mm 
传感器尺寸（假设为全画幅传感器，即36mm x 24mm）
光圈：f/5.6
2. 分辨率和DPI
图像分辨率：6144 x 4096像素
DPI：350
```
对于35mm镜头，水平视场角(H_FOV)和垂直视场角(V_FOV)可以通过以下公式计算:
```math
H\_FOV = 2 × arctan ( w / 2f ) \\
V\_FOV = 2 × arctan ( h / 2f )
```
其中，w是传感器宽度(36mm) , h是传感器高度(24mm) , f是焦距(35mm) 。
计算得:

```math
H\_FOV = 2arctan (36/ 2*35) ≈ 53.13° \\
V\_FOV ≈ 37.75°
```

假设二维码在图像中需要至少100 x 100个像素。那么，二维码在图像中的最小尺寸为:
将DPI转换为每米的像素密度：

```math
像素密度=350×1 /0.0254≈13779 pixels/meter
```
二维码的物理尺寸（X_min）为：
```math
X\_min= 100 / 13779 ≈ 0.00726meters ≈ 7.26mm
```
二维码在摄像头前的实际距离可以计算为：`𝐷=𝑓×𝑊/𝑤`
其中：D 是识别距离，f 是焦距，27 mm，W 是二维码的物理宽度，w 是二维码在图像传感器上的宽度，以像素表示，需要转换为 mm。

由此，在无人机拍摄的影像分辨率和参数下：

根据视场角和二维码的物理尺寸，计算最大识别距离（D_max）：
```math
D\_max= X\_min/(2×tan(H\_FOV/2))
```

其中，X_min是二维码的物理尺寸（7.26mm），H_FOV是水平视场角（53.13度）。

计算得：
```math
D\_max =0.00726/(2×tan(53.13/2)) ≈0.00726meters ≈7.75meters
```
最小二维码尺寸：约7.26毫米（边长）
最大识别距离：约7.75×0.6 = 4.65 米（考虑误差校正后减去40%的实际距离）

如果根据我手机拍摄的经验，至少需要至少274 x 270像素，手持手机拍摄20cm内，那么：
最小二维码尺寸：约44.3毫米（边长）
最大识别距离：约10.6米（理论最大值，实际可能折半）


# Others
1. huggingface project
2. data-analysis project
3. yolo-tracking project
4. segment-clip : using clip to segment QR codes from images


# Reference
* [OpenCV Document](https://docs.opencv.org/master/namespaces.html)
* [WeChatCV/opencv_3rdparty]()
* [OpenCV-QRCodeDetector-Sample]()
* [qrcode-file-tx-main]()
 
# License 
[Apache v2 License](LICENSE).
