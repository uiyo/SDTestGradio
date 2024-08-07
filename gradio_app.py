from linecache import cache
import gradio as gr
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image, DiffusionPipeline, StableDiffusion3Pipeline
import torch
from diffusers import FluxPipeline
import os

# 定义模型ID
model_id_sd3 = "./models/sd3-diffusers" # "stabilityai/stable-diffusion-3-medium"
model_id_playground_v2_5 = "./models/playground-v2.5-1024px-aesthetic" # playgroundai/playground-v2.5-1024px-aesthetic
model_id_flux_schnell = "black-forest-labs/FLUX.1-schnell" #you can also use `black-forest-labs/FLUX.1-dev`
model_id_flux_dev = "black-forest-labs/FLUX.1-dev" 
cache_dir = "./models/"

if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

# 初始化模型变量
pipe_txt2img = None
pipe_img2img = None
pipe_playground_v2_5 = None
pipeline_txt2img_fluxschnell = None
pipeline_txt2img_fluxdev = None
flag = False

# 定义函数来处理文生图请求
def generate_image_txt2img(prompt, negprompt, model_version):
    global pipe_txt2img, pipe_playground_v2_5, pipeline_txt2img_fluxschnell, pipeline_txt2img_fluxdev, flag
    # 根据选择的模型版本加载模型
    if model_version == "SD3":
        
        if pipe_txt2img is None or pipe_txt2img.model_id != model_id_sd3:
            if flag:
                pipe_playground_v2_5, pipeline_txt2img_fluxschnell, pipeline_txt2img_fluxdev = None, None, None
                flag = False
            pipe_txt2img = StableDiffusion3Pipeline.from_pretrained(model_id_sd3, torch_dtype=torch.float16) # ,cache_dir=cache_dir)
            pipe_txt2img.model_id = model_id_sd3  # 添加一个属性来存储模型ID
            pipe_txt2img = pipe_txt2img.to("cuda")
        pipe = pipe_txt2img
        flag = True
    elif model_version == "Playground V2.5":
        if pipe_playground_v2_5 is None:
            if flag:
                pipe_txt2img, pipeline_txt2img_fluxschnell, pipeline_txt2img_fluxdev = None, None, None
                flag = False
            pipe_playground_v2_5 = DiffusionPipeline.from_pretrained(model_id_playground_v2_5, torch_dtype=torch.float16, variant="fp16") #, cache_dir=cache_dir)
            pipe_playground_v2_5 = pipe_playground_v2_5.to("cuda")
        pipe = pipe_playground_v2_5
        flag = True

    elif model_version == "FluxSchnell":
        if pipeline_txt2img_fluxschnell is None:
            if flag:
                pipe_txt2img, pipe_playground_v2_5, pipeline_txt2img_fluxdev = None, None, None
                flag = False
            pipeline_txt2img_fluxschnell = FluxPipeline.from_pretrained(model_id_flux_schnell, torch_dtype=torch.bfloat16, cache_dir=cache_dir)

            pipeline_txt2img_fluxschnell.to("cuda")
            # pipeline_txt2img_fluxschnell.enable_model_cpu_offload()
        pipe = pipeline_txt2img_fluxschnell
        flag = True
        
    elif model_version == "FluxDev":
        if pipeline_txt2img_fluxdev is None:
            if flag:
                pipe_txt2img, pipe_playground_v2_5, pipeline_txt2img_fluxschnell = None, None, None
                flag = False
            pipeline_txt2img_fluxdev = FluxPipeline.from_pretrained(model_id_flux_dev, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
            pipeline_txt2img_fluxdev.to("cuda")
            # pipeline_txt2img_fluxdev.enable_model_cpu_offload()
        pipe = pipeline_txt2img_fluxdev
        flag = True

    else:
        raise ValueError("Invalid model version")

    # 使用模型生成图像
    if model_version == "FluxSchnell":
        image = pipe(prompt, prompt_2=negprompt, num_inference_steps=4, guidance_scale=0.0, height=768,width=1360,max_sequence_length=256).images[0]
    elif model_version == "FluxDev":
        image = pipe(prompt, prompt_2=negprompt, num_inference_steps=50, guidance_scale=3.5, height=768,width=1360).images[0]
    else:
        image = pipe(prompt, negative_prompt=negprompt, num_inference_steps=28, guidance_scale=7.0).images[0]
    return image

# 定义函数来处理图生图请求
def generate_image_img2img(prompt, image, negprompt, model_version):
    global pipe_img2img, pipe_playground_v2_5
    # 根据选择的模型版本加载模型
    if model_version == "SD3":
        if pipe_img2img is None or pipe_img2img.model_id != model_id_sd3:
            pipe_img2img = AutoPipelineForImage2Image.from_pretrained(model_id_sd3, torch_dtype=torch.float16)
            pipe_img2img.model_id = model_id_sd3  # 添加一个属性来存储模型ID
            pipe_img2img = pipe_img2img.to("cuda")
        pipe = pipe_img2img
    elif model_version == "Playground V2.5":
        if pipe_playground_v2_5 is None:
            pipe_playground_v2_5 = DiffusionPipeline.from_pretrained(model_id_playground_v2_5, torch_dtype=torch.float16, variant="fp16")
            pipe_playground_v2_5 = pipe_playground_v2_5.to("cuda")
        pipe = pipe_playground_v2_5
    else:
        raise ValueError("Invalid model version")

    # 使用模型生成图像
    generated_image = pipe(prompt, negative_prompt=negprompt, image=image, strength=0.95, guidance_scale=7.5).images[0]
    return generated_image

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown('''## PlayGroundV2.5 & Stable Diffusion 3 Medium & Flux Demo''')
    model_version_dropdown = gr.Dropdown(choices=["SD3", "Playground V2.5", "FluxSchnell", "FluxDev"], value="FluxSchnell", label="Model Version")
    
    with gr.Tab("Text-to-Image"):
        with gr.Row():
            with gr.Column():
                txt2img_prompt = gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt")
                txt2img_negprompt = gr.Textbox(lines=2, placeholder="Enter your negative prompt here. If Flux chosen, then enter prompt 2 here...", label="Negative Prompt")
            with gr.Column():
                txt2img_output = gr.Image(type="pil")
        txt2img_button = gr.Button("Generate Image")

    with gr.Tab("Image-to-Image"):
        with gr.Row():
            with gr.Column():
                img2img_prompt = gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt")
                img2img_negprompt = gr.Textbox(lines=2, placeholder="Enter your negative prompt here...", label="Negative Prompt")
                img2img_input = gr.Image(type="pil", label="Initial Image")
            with gr.Column():
                img2img_output = gr.Image(type="pil")
        img2img_button = gr.Button("Generate Image")

    # 定义事件处理函数
    txt2img_button.click(
        fn=generate_image_txt2img,
        inputs=[txt2img_prompt, txt2img_negprompt, model_version_dropdown],
        outputs=txt2img_output
    )

    img2img_button.click(
        fn=generate_image_img2img,
        inputs=[img2img_prompt, img2img_input, img2img_negprompt, model_version_dropdown],
        outputs=img2img_output
    )

# 运行应用
demo.launch(server_name="0.0.0.0", server_port=8188)