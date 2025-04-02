import gradio as gr
from transformers import AutoProcessor, AutoModelForVision2Seq, TextIteratorStreamer
from transformers.image_utils import load_image
from threading import Thread
import re
import ast
import html
import random
import torch

from PIL import Image, ImageOps
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

def add_random_padding(image, min_percent=0.1, max_percent=0.10):
    image = image.convert("RGB")
    width, height = image.size

    pad_w = int(width * random.uniform(min_percent, max_percent))
    pad_h = int(height * random.uniform(min_percent, max_percent))

    corner_pixel = image.getpixel((0, 0))  # Top-left corner
    return ImageOps.expand(image, border=(pad_w, pad_h, pad_w, pad_h), fill=corner_pixel)

def normalize_values(text, target_max=500):
    def normalize_list(values):
        max_value = max(values) if values else 1
        return [round((v / max_value) * target_max) for v in values]

    def process_match(match):
        num_list = ast.literal_eval(match.group(0))
        normalized = normalize_list(num_list)
        return "".join([f"<loc_{num}>" for num in normalized])

    pattern = r"\[([\d\.\s,]+)\]"
    return re.sub(pattern, process_match, text)

# Load model and processor
MODEL_NAME = "ds4sd/SmolDocling-256M-preview"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to("cuda")

def model_inference(input_dict, history): 
    text = input_dict["text"]
    images = [load_image(image) for image in input_dict["files"]] if input_dict["files"] else []

    if text == "" and not images:
        return "Error: Please provide a query or image(s)."

    if "OCR at text at" in text or "Identify element" in text or "formula" in text:
        text = normalize_values(text)

    resulting_messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        }
    ]

    prompt = processor.apply_chat_template(resulting_messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[images], return_tensors="pt").to('cuda')

    generation_args = {
        "input_ids": inputs.input_ids,
        "pixel_values": inputs.pixel_values,
        "attention_mask": inputs.attention_mask,
        "num_return_sequences": 1,
        "max_new_tokens": 8192,
    }

    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=False)
    generation_args["streamer"] = streamer

    thread = Thread(target=model.generate, kwargs=generation_args)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += html.escape(new_text)
        yield buffer

examples = [[{"text": "Convert this page to docling.", "files": ["example_images/sample1.png"]}],
            [{"text": "Convert this table to OTSL.", "files": ["example_images/sample2.jpg"]}]]

demo = gr.ChatInterface(fn=model_inference, 
                        title="SmolDocling-256M: Ultra-compact VLM for Document Conversion 💫",
                        description="Play with the SmolDocling-256M model. Upload an image and text or try an example.",
                        examples=examples,
                        textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image"], file_count="multiple"),
                        stop_btn="Stop Generation",
                        multimodal=True,
                        cache_examples=False)

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
