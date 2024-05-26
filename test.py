from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"

# Check if CUDA is available
if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is not available")
    device = torch.device("cpu")

# Load the model and move the model to the GPU if available
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

image = Image.open('Images\gym.jpg')

# Move the image to the GPU if available
enc_image = model.encode_image(image).to(device)

print(model.answer_question(enc_image, "Describe this image in spanish.", tokenizer))
