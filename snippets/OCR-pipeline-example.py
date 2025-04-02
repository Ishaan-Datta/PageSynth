# extract layout
import layoutparser as lp

image = lp.load_image("slide.png")
model = lp.AutoLayoutModel("lp://PubLayNet/faster_rcnn_R_50_FPN")
layout = model.detect(image)

for block in layout:
    print(block.type, block.coordinates)  # Identify text vs. equation blocks

# recognize text (trOCR)
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1").to("cuda")

image = Image.open("slide_text.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to("cuda")

generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Extracted Text:", text)

# recognize equations (pix2text)
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="latin", det_model_dir="path_to_model")  # Latin-trained model for math
result = ocr.ocr("slide_equation.png")

for res in result:
    print("Equation:", res)  # Should return LaTeX formatted equations

