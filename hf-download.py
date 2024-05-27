import os

# 配置的hf项目：geochat、clip、chinese-clip、LHRS-Bot
# export HF_ENDPOINT=https://hf-mirror.com
# geochat和clip chinese-clip可跑，LHRS-Bot没试
# geochat跑在服务器上，0000直接换成地址即可，配置时文件夹地址要改成MBZUAI/geochat-7B
# 例如http://0.0.0.0:7860 => http://125.220.157.228:7860/

"""
1. conda activate gc (或者geochat环境)
2. python geochat_demo.py --model-path MBZUAI/geochat-7B/
3. run at http://0.0.0.0:7860, remote at http://125.220.157.228:7860/
"""

# 设置环境变量为镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 下载模型
os.system(
    "huggingface-cli download --resume-download MBZUAI/geochat-7B --local-dir ./geochat-7B"
)
