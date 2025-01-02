import gradio as gr
from PIL import Image
import numpy as np

def process_image(init_img_with_mask):
    image, mask = init_img_with_mask["image"], init_img_with_mask["mask"]
    mask = create_binary_mask(mask)
    mask_path = "mask_output.png"
    mask.save(mask_path)
    return image
copy_image_buttons = []
copy_image_destinations = {}
def create_binary_mask(image, round=True):
    if image.mode == 'RGBA' and image.getextrema()[-1] != (255, 255):
        if round:
            image = image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
        else:
            image = image.split()[-1].convert("L")
    else:
        image = image.convert('L')
    return image
# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion Inpainting Demo")
    
    with gr.Row():
        with gr.Column():
            with gr.TabItem('Inpaint', id='inpaint', elem_id="img2img_inpaint_tab") as tab_inpaint:
                init_img_with_mask = gr.Image(label="Image for inpainting with mask", show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA") 

            submit = gr.Button("Generate")        
        with gr.Column():
            output = gr.Image(label="Result")
    
    submit.click(
        fn=process_image,
        inputs=init_img_with_mask,
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()