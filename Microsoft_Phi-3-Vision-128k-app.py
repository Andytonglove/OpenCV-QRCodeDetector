import subprocess

# Installing flash_attn
subprocess.run(
    "pip install flash-attn --no-build-isolation",
    env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
    shell=True,
)

import gradio as gr
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from transformers import TextIteratorStreamer
import time
from threading import Thread
import torch
import spaces

model_id = "microsoft/Phi-3-vision-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto"
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model.to("cuda:0")

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <img src="https://cdn-thumbnails.huggingface.co/social-thumbnails/models/microsoft/Phi-3-vision-128k-instruct.png" style="width: 80%; max-width: 550px; height: auto; opacity: 0.55;  "> 
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">Microsoft's Phi3 Vision</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Phi-3-Vision is a 4.2B parameter multimodal model that brings together language and vision capabilities.</p>
</div>
"""


@spaces.GPU
def bot_streaming(message, history):
    print(f"message is - {message}")
    print(f"history is - {history}")
    if message["files"]:
        # message["files"][-1] is a Dict or just a string
        if type(message["files"][-1]) == dict:
            image = message["files"][-1]["path"]
        else:
            image = message["files"][-1]
    else:
        # if there's no image uploaded for this turn, look for images in the past turns
        # kept inside tuples, take the last one
        for hist in history:
            if type(hist[0]) == tuple:
                image = hist[0][0]
    try:
        if image is None:
            # Handle the case where image is None
            raise gr.Error(
                "You need to upload an image for Phi3-Vision to work. Close the error and try again with an Image."
            )
    except NameError:
        # Handle the case where 'image' is not defined at all
        raise gr.Error(
            "You need to upload an image for Phi3-Vision to work. Close the error and try again with an Image."
        )

    conversation = []
    flag = False
    for user, assistant in history:
        if assistant is None:
            # pass
            flag = True
            conversation.extend([{"role": "user", "content": ""}])
            continue
        if flag == True:
            conversation[0]["content"] = f"<|image_1|>\n{user}"
            conversation.extend([{"role": "assistant", "content": assistant}])
            flag = False
            continue
        conversation.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        )

    if len(history) == 0:
        conversation.append(
            {"role": "user", "content": f"<|image_1|>\n{message['text']}"}
        )
    else:
        conversation.append({"role": "user", "content": message["text"]})
    print(f"prompt is -\n{conversation}")
    prompt = processor.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image = Image.open(image)
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    streamer = TextIteratorStreamer(
        processor,
        **{
            "skip_special_tokens": True,
            "skip_prompt": True,
            "clean_up_tokenization_spaces": False,
        },
    )
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer


chatbot = gr.Chatbot(scale=1, placeholder=PLACEHOLDER)
chat_input = gr.MultimodalTextbox(
    interactive=True,
    file_types=["image"],
    placeholder="Enter message or upload file...",
    show_label=False,
)
with gr.Blocks(
    fill_height=True,
) as demo:
    gr.ChatInterface(
        fn=bot_streaming,
        title="Phi3 Vision 128K Instruct",
        examples=[
            {"text": "Describe the image in details?", "files": ["./robo.jpg"]},
            {"text": "What does the chart display?", "files": ["./dataviz.png"]},
            {"text": "What is 3?", "files": ["./setofmark1.jpg"]},
            {"text": "Count the number of apples.", "files": ["./setofmark6.png"]},
            {
                "text": "I want to find a seat close to windows, where can I sit?",
                "files": ["./office1.jpg"],
            },
        ],
        description="Try the [Phi3-Vision model](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) from Microsoft. Upload an image and start chatting about it, or simply try one of the examples below. If you won't upload an image, you will receive an error. This is not the official demo.",
        stop_btn="Stop Generation",
        multimodal=True,
        textbox=chat_input,
        chatbot=chatbot,
        cache_examples=False,
        examples_per_page=3,
    )

demo.queue()
demo.launch(debug=True, quiet=True)
