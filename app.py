import gradio as gr
from PIL import Image

def process_image(init_img_with_mask):
    mask = init_img_with_mask["mask"]
    if mask.mode == 'RGBA' and mask.getextrema()[-1] != (255, 255):
        mask = mask.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
    else:
        mask = mask.convert('L')
    mask.save("mask_output.png")
    return init_img_with_mask["image"]

with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion Inpainting Demo")
    with gr.Row():
        with gr.Column():
            init_img_with_mask = gr.Image(label="Image for inpainting with mask", 
                                        type="pil", tool="sketch", image_mode="RGBA")
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter your negative prompt here")
            num_inference_steps = gr.Slider(minimum=1, maximum=100, value=30, label="Number of Inference Steps")
            guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, label="Guidance Scale")
            denoise_strength = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, label="Denoise Strength")
            sampler = gr.Dropdown(choices=["euler_a", "plms", "ddim"], label="Sampler", value="euler_a")
            submit = gr.Button("Generate")
        with gr.Column():
            output = gr.Image(label="Result")
    
    submit.click(fn=process_image, inputs=init_img_with_mask, outputs=output)

if __name__ == "__main__":
    demo.launch()