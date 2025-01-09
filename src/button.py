import gradio as gr
def create_control_elements():
    expand_direction = gr.Radio(
        label="Direction to expand image", 
        choices=["Left", "Right"],
        value="Left",
    )
    expand_pixels = gr.Number(
        label="Number of pixels to expand",
        minimum=0,
        maximum=1024,
        precision=0,
        value=200,
    )
    prompt = gr.Textbox(
        label="Prompt", 
        placeholder="Enter your prompt here"
    )
    negative_prompt = gr.Textbox(
        label="Negative Prompt",
        placeholder="Enter your negative prompt here",
    )
    num_inference_steps = gr.Slider(
        minimum=1,
        maximum=100,
        value=30,
        step=1,
        label="Number of Inference Steps",
    )
    guidance_scale = gr.Slider(
        minimum=1.0, 
        maximum=20.0, 
        value=7.5, 
        step=0.5,
        label="Guidance Scale"
    )
    denoise_strength = gr.Slider(
        minimum=0.0, 
        maximum=1.0, 
        value=1.0, 
        label="Denoise Strength"
    )
    sampler = gr.Dropdown(
        choices=["euler", "plms", "ddim"],
        label="Sampler",
        value="euler",
    )
    return (expand_direction, expand_pixels, prompt, negative_prompt, 
            num_inference_steps, guidance_scale, denoise_strength, sampler)
