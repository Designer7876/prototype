"""
This script uses Gradio to create a web interface for generating comic storyboards from a given story and image.
It leverages various libraries including torch, diffusers, PIL, cv2, and insightface to process images and generate dialogues.
Functions:
- generate_story_prompts(story_text): Generates 4 sequential story prompts from the given story text using Mistral Nemo.
- generate_comic_dialogue(prompt, image_path, story_text): Generates a short scientific dialogue for the person in the image based on the story and prompt using Mistral Nemo.
- embed_text_as_image_novel(image_path, text): Embeds the given text into the image at the specified path.
- process_images_sequentially(image_paths, prompts, story_text): Processes each image to generate dialogue and embed it, returning the paths of the processed images.
- create_comic_storyboard(image_paths, output_path, grid_size, padding, background_color): Creates a comic storyboard from the processed images and saves it to the specified output path.
- generate_images_from_story(image_file, story_text): Generates images based on the story and image, returning the paths of the generated images and the summarized prompts.
- gradio_interface(image, story): Gradio interface function that generates comic storyboards from the given image and story.
Gradio App:
- gr_interface: Sets up the Gradio interface with inputs for an image and a story, and an output for the generated comic storyboard.
- gr_interface.launch(): Launches the Gradio app.
"""
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import subprocess
import os
import textwrap
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

v2 = False
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "ip-adapter-faceid-plus_sd15.bin" if not v2 else "ip-adapter-faceid-plusv2_sd15.bin"
device = "cuda"
story_summary = None
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

# Load IP-Adapter
ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)

# First Mistral call for generating story prompts
def generate_story_prompts(story_text):
    global story_summary
    try:
        # Structured prompt for Mistral to summarize the story into 4 sequential prompts
        structured_prompt = (
            f"{story_text}\n\n"
            "Please summarize and break this story about a scientist flow-wise in exactly 4 concise and coherent sentences. Each sentence should have a maximum of 15 words and must be scientific and in simple words."
            "Do not include any additional text."
        )

        # Call Mistral Nemo to summarize the story into 4 prompts
        command = ["ollama", "run", "mistral-nemo"]
        result = subprocess.run(
            command, input=structured_prompt,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise Exception(f"Error generating story prompts: {result.stderr}")

        output = result.stdout.strip()
        story_summary = output
        sentences = [s.strip() for s in output.replace(';', '.').split('. ') if s.strip()]
        return sentences[:4]

    except Exception as e:
        print(f"An error occurred while generating story prompts: {str(e)}")
        return None

# Second Mistral call for generating dialogue for each image
def generate_comic_dialogue(prompt, image_path, story_text):
    try:
        # Structured prompt for Mistral to generate dialogue based on the image and story
        structured_prompt = (
            f"Story: {story_text}\n\n"
            f"Part: {prompt}\n\n"
            "Generate JUST ONE short scientific dialogue for the person in this image, following the story and part flow. ONLY THE PERSON PRESENT IN IMAGE MUST HAVE DIALOGUE. No other text."
        )

        command = ["ollama", "run", "mistral-nemo", image_path]
        result = subprocess.run(
            command, input=structured_prompt,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise Exception(f"Error generating dialogue: {result.stderr}")

        return result.stdout.strip()

    except Exception as e:
        print(f"An error occurred while generating comic dialogue: {str(e)}")
        return None

# Embed text into image
def embed_text_as_image_novel(image_path, text):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    max_width = image.width - 20
    wrapped_text = textwrap.fill(text, width=40)

    text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    padding = 10
    text_position = (10, image.height - text_height - padding)

    rectangle_bbox = [text_position[0] - 5, text_position[1] - 5, 
                    text_position[0] + text_width + 5, text_position[1] + text_height + 5]
    draw.rectangle(rectangle_bbox, fill="black")
    draw.text(text_position, wrapped_text, font=font, fill="white")

    output_path = "output_" + image_path.split('/')[-1]
    image.save(output_path)
    return output_path

# Process each image to generate dialogue and embed it
def process_images_sequentially(image_paths, prompts, story_text):
    comic_data = []

    for i, image_path in enumerate(image_paths):
        # Call the second Mistral function to generate dialogue for each image
        generated_text = generate_comic_dialogue(prompts[i], image_path, story_text)
        output_image_path = embed_text_as_image_novel(image_path, generated_text)
        comic_data.append(output_image_path)
        print(f"Processed image {i+1}/{len(image_paths)}: Dialogue: {generated_text}")

    return comic_data

# Create comic storyboard from processed images
def create_comic_storyboard(image_paths, output_path, grid_size=(3, 2), padding=10, background_color=(255, 255, 255)):
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    image_width, image_height = images[0].size
    total_width = grid_size[1] * image_width + (grid_size[1] - 1) * padding
    total_height = grid_size[0] * image_height + (grid_size[0] - 1) * padding

    storyboard = Image.new('RGB', (total_width, total_height), color=background_color)

    for index, image in enumerate(images):
        row = index // grid_size[1]
        col = index % grid_size[1]
        x_offset = col * (image_width + padding)
        y_offset = row * (image_height + padding)
        storyboard.paste(image, (x_offset, y_offset))

    storyboard.save(output_path)
    return output_path

# Function to generate images based on the story and image
def generate_images_from_story(image_file, story_text):
    image = cv2.imread(image_file)
    faces = app.get(image)

    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)

    # First, generate 4 story prompts using Mistral
    summarized_prompts = generate_story_prompts(story_text)

    if not summarized_prompts:
        return [None, None, None, None]

    negative_prompt = "multiple faces, multiple hands, deformed fingers, monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
    
    # Generate images based on the prompts
    generated_images_paths = []
    for i, prompt in enumerate(summarized_prompts):
        images = ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            face_image=face_image,
            faceid_embeds=faceid_embeds,
            shortcut=v2,
            s_scale=1.0,
            num_samples=1,
            width=512,
            height=768,
            num_inference_steps=35,
            seed=2023,
            guidance_scale=8
        )
        
        for j, img in enumerate(images):
            img_path = f"generated_image_{i}_{j}.png"
            img.save(img_path)
            generated_images_paths.append(img_path)
    
    return generated_images_paths, summarized_prompts

# Gradio interface function
def gradio_interface(image, story):
    generated_image_paths, prompts = generate_images_from_story(image, story)

    # Process each image with dialogue embedding
    processed_images = process_images_sequentially(generated_image_paths, prompts, story_summary)

    # Create comic storyboard
    storyboard_path = create_comic_storyboard(processed_images, "comic_storyboard.png", grid_size=(2, 2))
    
    return storyboard_path

# Gradio app setup
gr_interface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Image(type="filepath"), gr.Textbox(lines=10, placeholder="Enter your story here...")],
    outputs=gr.Image(label="Generated Comic Storyboard"),
    title="Story-to-Image Comic Generator",
    description="Upload an image and enter a story. The app will generate comic images based on the story, integrate the face from the uploaded image, and create a comic storyboard."
)

# Launch the Gradio app
gr_interface.launch()
