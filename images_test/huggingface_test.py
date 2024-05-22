# this is my first test file for using huggingface

import warnings

warnings.filterwarnings("ignore")
from transformers import pipeline  # 用人家设计好的流程完成一些简单的任务

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
