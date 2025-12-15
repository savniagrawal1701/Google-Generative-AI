# import requried Libraries
import gradio as gr
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# Load the Diffusion model
print ("Loading the model....")

# Load components
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

# Create pipeline
pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,  # Assuming no safety checker needed for this example
    feature_extractor=None, # Assuming no feature extractor needed for this example
).to("cuda")


print("Model loaded successfully")

# image generation function
def generate_image(prompt):
  """
  Generate image from the text prompt using Diffusion model

  Args:
      prompt (str): Text description of the image to generate

  Return:
      PIL.Image: Generated Image

  """
  try:
      with autocast("cuda"):
          image = pipe(prompt).images[0]
      return image

  except Exception as e:
    print(f"Error generating image:{e}")
    return None

# create Gradio interface
with gr.Blocks(title="Diffision Image Generator") as demo:
     gr.Markdown("# Diffision Image Generator")
     gr.Markdown("Enter a text to generatr an Image.")

     with gr.Row():
      with gr.Column():
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Enter your image descrption here..."
        )

     generate_btn = gr.Button("Generate Image")
     with gr.Column():
        image_output = gr.Image(
            label="Generated Image",
            type="pil"
        )


  # connect the funtion to gradio interface
     generate_btn.click(fn=generate_image,inputs=[prompt_input],outputs=[image_output])

demo.launch(share=True, debug=True)
