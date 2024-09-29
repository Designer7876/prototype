import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import subprocess
import os

# Initialize models and face analysis
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

# Function to generate text prompts using Mistral
def generate_text_with_mistral(prompt):
    try:
        structured_prompt = (
            f"{prompt}\n\n"
            "Please summarize and break this story about a scientist flow-wise in exactly 4 concise and coherent sentences. Each sentence should have a maximum of 15 words and must be scientific and in simple but scientific words.\n"
            "Each sentence should be structured as a prompt for stable diffusion to generate an image exactly in flow with the story.\n"
            "Do not include any additional text."
        )
        
        command = ["ollama", "run", "mistral-nemo"]
        result = subprocess.run(
            command, input=structured_prompt,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Error generating text: {result.stderr}")
        
        output = result.stdout.strip().replace('\n', ' ')
        sentences = [s.strip() for s in output.replace(';', '.').split('. ') if s.strip()]

        summarized_sentences = sentences[:4]
        return summarized_sentences

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Function to generate images based on the story and image
def generate_images_from_story(image_file, story_text):
    # Load and process the face image
    image = cv2.imread(image_file)
    faces = app.get(image)

    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)

    # Step 1: Generate 4 prompts from the story
    summarized_prompts = generate_text_with_mistral(story_text)
    
    if not summarized_prompts:
        return [None, None, None, None]

    negative_prompt = "multiple faces, multiple hands, deformed fingers, monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
    
    # Step 2: Generate 4 images based on the prompts and save them to file paths
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
        
        # Save the generated images to files and return the file paths
        for j, img in enumerate(images):
            img_path = f"generated_image_{i}_{j}.png"
            img.save(img_path)
            generated_images_paths.append(img_path)
    
    return generated_images_paths

# Gradio interface function
def gradio_interface(image, story):
    generated_image_paths = generate_images_from_story(image, story)
    return generated_image_paths

# Gradio app setup
gr_interface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Image(type="filepath"), gr.Textbox(lines=10, placeholder="Enter your story here...")],
    outputs=[gr.Image(label=f"Generated Image {i+1}") for i in range(4)],
    title="Story-to-Image Generator with Face Integration",
    description="Upload an image and enter a story. The app will generate images based on the story and integrate the face from the uploaded image."
)

# Launch the Gradio app
gr_interface.launch()
