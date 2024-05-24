import pickle
import gzip
import cv2
import numpy as np
from PIL import Image

# 设置参数
FRAME_RATE = 2  # 帧率为每秒5帧
VIDEO_FILE_NAME = "qrcode_video.mp4"
QRCODE_NUMBER = 28
QR_CODE_NUMBER_PER_ROW = 7
QR_CODE_NUMBER_PER_COL = 4

# 从 pickle 文件加载数据
with gzip.open("qr_cache.pickle", "rb") as f:
    loaded_data = pickle.load(f)

frames_number = loaded_data["frames_number"]
chunks = loaded_data["chunks"]

# 确保 chunks 列表只包含 QR 码数据
chunks = [chunk for chunk in chunks if isinstance(chunk, list)]

# 获取 QR 码图像的尺寸
first_qr_image = None
for chunk in chunks:
    if len(chunk) > 0:
        first_qr_image = chunk[0].convert("RGB")
        break

if first_qr_image is None:
    raise ValueError("No QR code frames found in chunks.")

qr_width, qr_height = first_qr_image.size


# 创建每一页的图像
def create_page_image(frame_idx):
    page_image = Image.new(
        "RGB",
        (qr_width * QR_CODE_NUMBER_PER_ROW, qr_height * QR_CODE_NUMBER_PER_COL),
        "white",
    )
    for i in range(QRCODE_NUMBER):
        chunk_index = i + frame_idx * QRCODE_NUMBER
        if chunk_index < len(chunks) * frames_number:
            chunk_idx = chunk_index // frames_number
            frame_idx_within_chunk = chunk_index % frames_number
            qr_image = chunks[chunk_idx][frame_idx_within_chunk].convert("RGB")
            open_cv_image = cv2.cvtColor(np.array(qr_image), cv2.COLOR_RGB2BGR)
            qr_image = Image.fromarray(open_cv_image)
            x = i % QR_CODE_NUMBER_PER_ROW
            y = i // QR_CODE_NUMBER_PER_ROW
            page_image.paste(qr_image, (x * qr_width, y * qr_height))

    return page_image


# 创建视频写入器
video_writer = cv2.VideoWriter(
    VIDEO_FILE_NAME,
    cv2.VideoWriter_fourcc(*"mp4v"),  # 使用 'mp4v' 编码器生成 mp4 文件
    FRAME_RATE,
    (qr_width * QR_CODE_NUMBER_PER_ROW, qr_height * QR_CODE_NUMBER_PER_COL),
)

# 计算总帧数
total_frames = (len(chunks) * frames_number) // QRCODE_NUMBER + 1

# 写入视频文件
for frame_idx in range(total_frames):
    page_image = create_page_image(frame_idx)
    open_cv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
    video_writer.write(open_cv_image)

# 释放视频写入器
video_writer.release()

print("视频生成完毕！")


# 每面一个 QR 码
# import pickle
# import gzip
# import cv2
# import numpy as np
# from PIL import Image

# # 设置参数
# FRAME_RATE = 5  # 帧率为每秒5帧
# VIDEO_FILE_NAME = "qrcode_video.mp4"

# # 从 pickle 文件加载数据
# with gzip.open("qr_cache.pickle", "rb") as f:
#     loaded_data = pickle.load(f)

# frames_number = loaded_data["frames_number"]
# chunks = loaded_data["chunks"]

# # 确保 chunks 列表只包含 QR 码数据
# chunks = [chunk for chunk in chunks if isinstance(chunk, list)]

# # 检查并获取示例图像的尺寸
# first_frame_found = False
# for chunk in chunks:
#     if len(chunk) > 0:
#         first_qr_image = chunk[0].convert("RGB")
#         first_frame_found = True
#         break

# if not first_frame_found:
#     raise ValueError("No QR code frames found in chunks.")

# # 创建视频写入器
# video_writer = cv2.VideoWriter(
#     VIDEO_FILE_NAME,
#     cv2.VideoWriter_fourcc(*"mp4v"),  # 使用 'mp4v' 编码器生成 mp4 文件
#     FRAME_RATE,
#     first_qr_image.size,
# )

# # 写入视频文件
# for frame_idx in range(frames_number + 1):
#     for chunk in chunks:
#         if frame_idx < len(chunk):
#             img = chunk[frame_idx].convert("RGB")
#             open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#             video_writer.write(open_cv_image)

# # 释放视频写入器
# video_writer.release()

# print("视频生成完毕！")
