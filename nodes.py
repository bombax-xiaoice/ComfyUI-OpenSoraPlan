import os
import os.path
import folder_paths
import sys
comfy_path = os.path.dirname(folder_paths.__file__)
opensoraplan_path = os.path.join(comfy_path, "custom_nodes", "ComfyUI-OpenSoraPlan")
opensoraplan_modelpath = os.path.join(opensoraplan_path, "models")
sys.path.append(opensoraplan_path)
from huggingface_hub import snapshot_download, try_to_load_from_cache, _CACHED_NO_EXIST
import comfy.utils
import random
#import imageio
import torch
from diffusers.schedulers import PNDMScheduler
from huggingface_hub import hf_hub_download
from torchvision.utils import save_image
#from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
#from datetime import datetime
#from typing import List, Union
#import gradio as gr
import numpy as np
#from gradio.components import Textbox, Video, Image
from transformers import MT5EncoderModel, T5EncoderModel, T5Tokenizer, AutoTokenizer
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from transformers import  AutoTokenizer, AutoModelForCausalLM
from PIL import Image #, ImageOps, ImageSequence, ImageFile

from opensora1.models.ae import ae_stride_config, getae_wrapper #, getae
#from opensora1.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora1.models.diffusion.latte.modeling_latte import LatteT2V
from opensora1.models.text_encoder import get_text_enc

from opensora1.sample.pipeline_videogen import VideoGenPipeline
#from opensora1.serve.gradio_utils import block_css, title_markdown, randomize_seed_fn, set_env, examples, DESCRIPTION

from opensora2.models.diffusion.opensora.modeling_opensora import OpenSoraT2V
from opensora2.sample.pipeline_opensora import OpenSoraPipeline
from opensora2.models import CausalVAEModelWrapper as CausalVAEModelWrapper2
from opensora2.models.causalvideovae import ae_stride_config as ae_stride_config_2
from opensora2.models.diffusion.opensora.modeling_inpaint import OpenSoraInpaint
from opensora2.sample.pipeline_inpaint import OpenSoraInpaintPipeline

from opensora3.models.diffusion.opensora_v1_3.modeling_opensora import OpenSoraT2V_v1_3
from opensora3.sample.pipeline_opensora import OpenSoraPipeline as OpenSoraPipeline3
from opensora3.models import CausalVAEModelWrapper as CausalVAEModelWrapper3
from opensora3.models import WFVAEModelWrapper as WFVAEModelWrapper
from opensora3.models.causalvideovae import ae_stride_config as ae_stride_config_3
from opensora3.models.diffusion.opensora_v1_3.modeling_inpaint import OpenSoraInpaint_v1_3
from opensora3.sample.pipeline_inpaint import OpenSoraInpaintPipeline as OpenSoraInpaintPipeline3
from opensora3.utils.sample_utils import get_scheduler

def check_symlink_download(path, link, repoid, sub):
    if not os.path.exists(os.path.join(path, link)):
        download = os.path.join(os.path.dirname(path), os.path.basename(repoid), sub)
        if not os.path.exists(download):
            download = os.path.join(opensoraplan_modelpath, os.path.basename(repoid), sub)
            if not os.path.exists(download):
                download = hf_hub_download(repoid, subfolder=sub, force_download=False, resume_download=True)
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            os.symlink(os.path.join(download, sub), os.path.join(path, link))
        except:
            return download
    return os.path.join(path, link)

def check_download(path, repoid, sub=None):
    if not os.path.exists(path):
        if sub:
            download = hf_hub_download(repoid, subfolder=sub, force_download=False, resume_download=True)
            if download.endswith(sub):
                download = download[:-len(sub)].rstrip(os.path.sep)
        else:
            download = snapshot_download(repoid, force_download=False, resume_download=True)
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            os.symlink(download, path)
        except:
            path = download
    return path

def latent_preview_callback(pbar, device, num_inference_steps):
    try: #progress bar and preview
        import typing, latent_preview
        from comfy import latent_formats
        fakepipe = typing.NewType('FakePipe',typing.Generic)
        setattr(fakepipe, 'load_device', device)
        setattr(fakepipe, 'model', typing.NewType('FakeModel',typing.Generic))
        setattr(fakepipe.model, 'latent_format', latent_formats.SD15())
        latent_callback = latent_preview.prepare_callback(fakepipe, num_inference_steps)
        if latent_callback != None:
            return lambda i,t,x0:latent_callback(i, x0[:,::2] if x0.shape[1]==8 else x0, None, num_inference_steps)            
    except:
        pass
    return lambda i,t,x0:pbar.update_absolute(i+1, num_inference_steps)

class OpenSoraPlanV3LoaderI2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path":("STRING",{"default":os.path.join(opensoraplan_modelpath, "Open-Sora-Plan-v1.3.0")}),
                "ae":(list(ae_stride_config_3.keys()),{"default":"WFVAEModel_D8_4x8x8" if "WFVAEModel_D8_4x8x8" in ae_stride_config_3 else next(iter(ae_stride_config_3))}),
                "text_encoder_name":("STRING",{"default":os.path.join(opensoraplan_modelpath, "mt5-xxl")}),
                "clip_name":("STRING",{"default":os.path.join(opensoraplan_modelpath, "CLIP-ViT-bigG-14-laion2B-39B-b160k")}),
                "version":("STRING",{"default":"any93x640x640_i2v"}),
                "width":("INT",{"default":480,"min":352,"max":640,"step":32}),
                "height":("INT",{"default":480,"min":352,"max":640,"step":32}),
                "num_frames":("INT",{"default":93,"min":1,"max":93,"step":1}),
                "scheduler":(["EulerAncestralDiscrete","DDIM","EulerDiscrete","DDPM","DPMSolverMultistep","DPMSolverSinglestep","PNDM","HeunDiscrete","DEISMultistep","KDPM2AncestralDiscrete","CogVideoX","FlowMatchEulerDiscrete"],{"Default":"EulerAncestralDiscrete"}),
            },
            "hidden": {
                "force_images":("BOOLEAN",{"default":False}),
            },
        }

    RETURN_TYPES = ("OpenSoraPlanModel",)
    FUNCTION = "run"
    CATEGORY = "OpenSoraPlan"

    def run(self,model_path,ae,text_encoder_name,clip_name,version,width,height,num_frames,scheduler,force_images=True):
        if version.startswith('any') and height*width > 236544:
            raise ValueError('product of height and width can not exceed 236544 (roughly 480x480 or 352x640 or 640x352)')
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')
        # Load model:
        model_path = check_download(model_path, "LanguageBind/Open-Sora-Plan-v1.3.0", version)
        transformer_model = OpenSoraInpaint_v1_3.from_pretrained(os.path.join(model_path, version), torch_dtype=torch.bfloat16, cache_dir='cache_dir').to(cpu_device)
        if not ae.startswith("Casual"):
            model_path = check_download(model_path, "LanguageBind/Open-Sora-Plan-v1.3.0", "vae")
            vae = WFVAEModelWrapper(os.path.join(model_path, "vae"), cache_dir='cache_dir').eval()
        else:
            casual_vae = check_symlink_download(model_path, "casual_vae", "LanguageBind/Open-Sora-Plan-v1.2.0", "vae")
            vae = CausalVAEModelWrapper3(casual_vae if casual_vae else os.path.join(model_path, "casual_vae"), cache_dir='cache_dir').eval()
        vae.vae = vae.vae.to(cpu_device, dtype=torch.bfloat16)
        #vae.vae.enable_tiling()

        vae.vae.tile_overlap_factor = 0.25
        vae.vae_scale_factor = ae_stride_config_3[ae]
        #vae.latent_size = (height // ae_stride_config_3[ae][1], width // ae_stride_config_3[ae][2])
        transformer_model.force_images = force_images
        if text_encoder_name:
            text_encoder_name = check_download(text_encoder_name, "google/mt5-xxl")
            tokenizer = T5Tokenizer.from_pretrained(text_encoder_name, cache_dir="cache_dir")
            if "mt5" in text_encoder_name:
                MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
                text_encoder = MT5EncoderModel.from_pretrained(text_encoder_name, cache_dir="cache_dir",torch_dtype=torch.float32).to(cpu_device)
            else:
                T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
                text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, cache_dir="cache_dir",torch_dtype=torch.float32).to(cpu_device)
            text_encoder.eval()
        else:
            tokenizer = None
            text_encoder = None
        if clip_name:
            clip_name = check_download(clip_name, "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
            tokenizer_2 = CLIPTokenizer.from_pretrained(clip_name, cache_dir="cache_dir",torch_dtype=torch.float32)
            CLIPTextModelWithProjection._keys_to_ignore_on_load_unexpected = ["decoder.*"]
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(clip_name, cache_dir="cache_dir",torch_dtype=torch.float32).to(cpu_device)
            text_encoder_2.eval()
        else:
            tokenizer_2 = None
            text_encoder_2 = None
        # set eval mode
        transformer_model.eval()
        vae.eval()
        text_encoder.eval()
        scheduler_object = get_scheduler(type('get_scheduler_args', (object,), {'prediction_type':'v_prediction','rescale_betas_zero_snr':True,'v1_5_scheduler':False,'sample_method':scheduler}))
        videogen_pipeline = OpenSoraInpaintPipeline3(vae=vae,
                                            text_encoder=text_encoder,
                                            tokenizer=tokenizer,
                                            text_encoder_2=text_encoder_2,
                                            tokenizer_2=tokenizer_2,
                                            scheduler=scheduler_object,
                                            transformer=transformer_model) #.to(device=cuda_device)
        return ((videogen_pipeline,transformer_model,str(num_frames)+'x'+str(height)+'x'+str(width) if version.startswith('any') else version),)

class OpenSoraPlanV3SampleI2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("OpenSoraPlanModel",),
                "images":("IMAGE",{"default":None}),
                "imagefiles":("STRING",{"default":""}),
                "prompt":("STRING",{"default":""}),
                "negative_prompt":("STRING",{"default":""}),
                "num_inference_steps":("INT",{"default":50}),
                "guidance_scale":("FLOAT",{"default":10.0}),
                "seed":("INT",{"default":1234}),
                "force_textencoder_cpu":("BOOLEAN",{"default":False}),
                "force_transformer_cpu_offload":("BOOLEAN",{"default":False}),
                "force_vae_cpu_offload":("BOOLEAN",{"default":False}),
                "use_tiling":("BOOLEAN",{"default":False}),
                "tile":("INT",{"default":24}),
                "context":("INT",{"default":16}),
            },
            "hidden": {
                "force_images":("BOOLEAN",{"default":False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "OpenSoraPlan"

    def run(self,model,images,imagefiles,prompt,negative_prompt,num_inference_steps,guidance_scale,seed,force_textencoder_cpu,force_transformer_cpu_offload,force_vae_cpu_offload,use_tiling,tile,context,force_images=False):
        videogen_pipeline,transformer_model,version=model
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')
        videogen_pipeline.to(device=cpu_device)
        if not force_textencoder_cpu and videogen_pipeline.text_encoder!=None:
            videogen_pipeline.text_encoder.to(device=cuda_device, dtype=torch.bfloat16)
        if not force_textencoder_cpu and hasattr(videogen_pipeline, 'text_encoder_2') and videogen_pipeline!=None:
            videogen_pipeline.text_encoder_2.to(device=cuda_device, dtype=torch.bfloat16)
        if not force_transformer_cpu_offload:
            videogen_pipeline.transformer.to(device=cuda_device,dtype=torch.bfloat16)
        else:
            videogen_pipeline.transformer.to(device=cpu_device)
        if not force_vae_cpu_offload:
            videogen_pipeline.vae.to(device=cuda_device,dtype=torch.bfloat16)
        else:
            videogen_pipeline.vae.to(device=cpu_device)
        if use_tiling:
            videogen_pipeline.vae.vae.enable_tiling(use_tiling=use_tiling, tile=tile, context=context)
        else:
            videogen_pipeline.vae.vae.disable_tiling()
        #seed = int(randomize_seed_fn(seed, randomize_seed))
        torch.Generator(device=cuda_device).manual_seed(seed)
        #video_length = transformer_model.config.video_length if not force_images else 1
        height, width = int(version.split('_')[0].rstrip('p').split('x')[1]), int(version.split('_')[0].rstrip('p').split('x')[2 if len(version.split('_')[0].split('x'))>2 else 1])
        num_frames = 1 if force_images else int(version[3:].split('x')[0] if version.startswith('any') else version.split('x')[0])
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        if images != None:
            imagefiles = [None,]*images.shape[0] if not imagefiles else [imagefiles,]+[None,]*(images.shape[0]-1) if isinstance(imagefiles,str) else imagefiles+[None,]*(images.shape[0]-len(imagefiles)) if len(imagefiles) < images.shape[0] else imagefiles[:images.shape[0]]
            for i in range(images.shape[0]):
                if not imagefiles[i]:
                    imagefiles[i] = os.path.join(('/dev/shm/' if os.path.exists('/dev/shm/') else '/tmp/')+ ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)) + '.png')
                im = 255. * images[i,:,:,:].cpu().numpy()
                img = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))
                img.save(imagefiles[i], pnginfo={}, compress_level=4)
        elif isinstance(imagefiles, str):
            imagefiles = imagefiles.split(',')
        
        videos = videogen_pipeline(conditional_pixel_values_path=imagefiles,
                                conditional_pixel_values_indices=[0,] if len(imagefiles)==1 else [0,-1] if len(imagefiles)==2 else list(range(0, num_frames, (num_frames-1)//(len(imagefiles)-1))),
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_frames=num_frames,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                num_samples_per_prompt=1,
                                max_sequence_length=512,
                                height=height,
                                width=width,
                                num_images_per_prompt=1,
                                enable_temporal_attentions=not force_images,
                                mask_feature=True,
                                output_type="latents",
                                callback=latent_preview_callback(pbar, transformer_model.device, num_inference_steps),
                                ).images
        
        videogen_pipeline.to(device=cpu_device)
        videogen_pipeline.text_encoder.to(device=cpu_device)
        videogen_pipeline.transformer.to(device=cpu_device)
        videogen_pipeline.vae.to(device=cpu_device)
        torch.cuda.empty_cache()

        return ({"samples":videos},)

class OpenSoraPlanV3LoaderT2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path":("STRING",{"default":os.path.join(opensoraplan_modelpath, "Open-Sora-Plan-v1.3.0")}),
                "ae":(list(ae_stride_config_3.keys()),{"default":"WFVAEModel_D8_4x8x8"}),
                "text_encoder_name":("STRING",{"default":os.path.join(opensoraplan_modelpath, "mt5-xxl")}),
                "clip_name":("STRING",{"default":os.path.join(opensoraplan_modelpath, "CLIP-ViT-bigG-14-laion2B-39B-b160k")}),
                "version":("STRING",{"default":"any93x640x640"}),
                "height":("INT",{"default":640,"min":352,"max":640,"step":32}),
                "width":("INT",{"default":640,"min":352,"max":640,"step":32}),
                "num_frames":("INT",{"default":93,"min":1,"max":93,"step":1}),
                "scheduler":(["EulerAncestralDiscrete","DDIM","EulerDiscrete","DDPM","DPMSolverMultistep","DPMSolverSinglestep","PNDM","HeunDiscrete","DEISMultistep","KDPM2AncestralDiscrete","CogVideoX","FlowMatchEulerDiscrete"],{"Default":"EulerAncestralDiscrete"}),
            },
            "hidden": {
                "force_images":("BOOLEAN",{"default":False}),
            },
        }

    RETURN_TYPES = ("OpenSoraPlanModel",)
    FUNCTION = "run"
    CATEGORY = "OpenSoraPlan"

    def run(self,model_path,ae,text_encoder_name,clip_name,version,height,width,num_frames,scheduler,force_images=False):
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')
        # Load model:
        model_path = check_download(model_path, "LanguageBind/Open-Sora-Plan-v1.3.0", version)
        transformer_model = OpenSoraT2V_v1_3.from_pretrained(os.path.join(model_path, version), torch_dtype=torch.bfloat16, cache_dir='cache_dir').to(cpu_device)
        if not ae.startswith("Causal"):
            model_path = check_download(model_path, "LanguageBind/Open-Sora-Plan-v1.3.0", "vae")
            vae = WFVAEModelWrapper(os.path.join(model_path, "vae"), cache_dir='cache_dir').eval()
        else:
            casual_vae = check_symlink_download(model_path, "casual_vae", "LanguageBind/Open-Sora-Plan-v1.2.0", "vae")
            vae = CausalVAEModelWrapper3(casual_vae if casual_vae else os.path.join(model_path, "casual_vae"), cache_dir='cache_dir').eval()
        vae.vae = vae.vae.to(cpu_device, dtype=torch.bfloat16)
        #vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = 0.25
        vae.vae_scale_factor = ae_stride_config_3[ae]
        #vae.latent_size = (height // ae_stride_config_3[ae][1], width // ae_stride_config_3[ae][2])
        transformer_model.force_images = force_images
        if text_encoder_name:
            text_encoder_name = check_download(text_encoder_name, "google/mt5-xxl")
            tokenizer = T5Tokenizer.from_pretrained(text_encoder_name, cache_dir="cache_dir")
            if "mt5" in text_encoder_name:
                MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
                text_encoder = MT5EncoderModel.from_pretrained(text_encoder_name, cache_dir="cache_dir",torch_dtype=torch.float32).to(cpu_device)
            else:
                T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
                text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, cache_dir="cache_dir",torch_dtype=torch.float32).to(cpu_device)
            text_encoder.eval()
        else:
            tokenizer = None
            text_encoder = None
        if clip_name:
            clip_name = check_download(clip_name, "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
            tokenizer_2 = CLIPTokenizer.from_pretrained(clip_name, cache_dir="cache_dir",torch_dtype=torch.float32)
            CLIPTextModelWithProjection._keys_to_ignore_on_load_unexpected = ["decoder.*"]
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(clip_name, cache_dir="cache_dir",torch_dtype=torch.float32).to(cpu_device)
            text_encoder_2.eval()
        else:
            tokenizer_2 = None
            text_encoder_2 = None
        # set eval mode
        transformer_model.eval()
        vae.eval()
        scheduler_object = get_scheduler(type('get_scheduler_args', (object,), {'prediction_type':'v_prediction','rescale_betas_zero_snr':True,'v1_5_scheduler':False,'sample_method':scheduler}))
        videogen_pipeline = OpenSoraPipeline3(vae=vae,
                                            text_encoder=text_encoder,
                                            tokenizer=tokenizer,
                                            text_encoder_2=text_encoder_2,
                                            tokenizer_2=tokenizer_2,
                                            scheduler=scheduler_object,
                                            transformer=transformer_model) #.to(device=cuda_device)
        return ((videogen_pipeline,transformer_model,str(num_frames)+'x'+str(height)+'x'+str(width) if version.startswith('any') else version),)

class OpenSoraPlanV2LoaderI2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path":("STRING",{"default":os.path.join(opensoraplan_modelpath, "Open-Sora-Plan-v1.2.0")}),
                "ae":(list(ae_stride_config_2.keys()),{"default":"CausalVAEModel_D8_4x8x8"}),
                "text_encoder_name":("STRING",{"default":os.path.join(opensoraplan_modelpath, "mt5-xxl")}),
                "version":("STRING",{"default":"93x480p_i2v"}),
            },
            "hidden": {
                "force_images":("BOOLEAN",{"default":False}),
            },
        }

    RETURN_TYPES = ("OpenSoraPlanModel",)
    FUNCTION = "run"
    CATEGORY = "OpenSoraPlan/Legacy"

    def run(self,model_path,ae,text_encoder_name,version,force_images=False):
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')

        # Load model:
        model_path = check_download(model_path, "LanguageBind/Open-Sora-Plan-v1.2.0", version)
        transformer_model = OpenSoraInpaint.from_pretrained(os.path.join(model_path, version), torch_dtype=torch.float16, cache_dir='cache_dir').to(cpu_device)
        model_path = check_download(model_path, "LanguageBind/Open-Sora-Plan-v1.2.0", "vae")
        vae = CausalVAEModelWrapper2(os.path.join(model_path, "vae"), cache_dir='cache_dir').eval()
        vae.vae = vae.vae.to(cpu_device, dtype=torch.float16)
        #vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = 0.25
        vae.vae_scale_factor = ae_stride_config_2[ae]
        image_size = int(version.split('_')[0].rstrip('p').split('x')[1])
        latent_size = (image_size // ae_stride_config_2[ae][1], image_size // ae_stride_config_2[ae][2])
        vae.latent_size = latent_size
        transformer_model.force_images = force_images
        text_encoder_name = check_download(text_encoder_name, "google/mt5-xxl")
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_name, cache_dir="cache_dir")
        if "mt5" in text_encoder_name:
            MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
            text_encoder = MT5EncoderModel.from_pretrained(text_encoder_name, cache_dir="cache_dir",torch_dtype=torch.float32).to(cpu_device)
        else:
            T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
            text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, cache_dir="cache_dir",torch_dtype=torch.float32).to(cpu_device)

        # set eval mode
        transformer_model.eval()
        vae.eval()
        text_encoder.eval()
        scheduler = PNDMScheduler()
        videogen_pipeline = OpenSoraInpaintPipeline(vae=vae,
                                            text_encoder=text_encoder,
                                            tokenizer=tokenizer,
                                            scheduler=scheduler,
                                            transformer=transformer_model) #.to(device=cuda_device)
        return ((videogen_pipeline,transformer_model,version),)

class OpenSoraPlanV2SampleI2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("OpenSoraPlanModel",),
                "start_image":("IMAGE",),
                "prompt":("STRING",{"default":""}),
                "negative_prompt":("STRING",{"default":""}),
                "num_inference_steps":("INT",{"default":50}),
                "guidance_scale":("FLOAT",{"default":10.0}),
                "seed":("INT",{"default":1234}),
                "force_textencoder_cpu":("BOOLEAN",{"default":False}),
                "force_transformer_cpu_offload":("BOOLEAN",{"default":False}),
                "use_tiling":("BOOLEAN",{"default":False}),
                "tile":("INT",{"default":32}),
                "context":("INT",{"default":16}),
            },
            "optional":{
                "end_image":("IMAGE",),
            },
            "hidden": {
                "force_images":("BOOLEAN",{"default":False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "OpenSoraPlan/Legacy"

    def run(self,model,start_image,prompt,negative_prompt,num_inference_steps,guidance_scale,seed,force_textencoder_cpu,force_transformer_cpu_offload,use_tiling,tile,context,end_image=None,force_images=False):
        videogen_pipeline,transformer_model,version=model
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')
        videogen_pipeline.to(device=cpu_device)
        if not force_textencoder_cpu:
            videogen_pipeline.text_encoder.to(device=cuda_device, dtype=torch.float16)
        if not force_transformer_cpu_offload:
            videogen_pipeline.transformer.to(device=cuda_device)
        videogen_pipeline.vae.to(device=cpu_device)
        if use_tiling:
            videogen_pipeline.vae.vae.enable_tiling(use_tiling=use_tiling, tile=tile, context=context)
        else:
            videogen_pipeline.vae.vae.disable_tiling()
        #seed = int(randomize_seed_fn(seed, randomize_seed))
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        torch.Generator(device=cuda_device).manual_seed(seed)
        #video_length = transformer_model.config.video_length if not force_images else 1
        height, width = int(version.split('_')[0].rstrip('p').split('x')[1]), int(version.split('_')[0].rstrip('p').split('x')[2 if len(version.split('_')[0].split('x'))>2 else 1])
        num_frames = 1 if force_images else int(version.split('x')[0])
        start_image = start_image.permute(0, 3, 1, 2).contiguous() if start_image!=None else None # b,h,w,c -> b,c,h,w
        end_image = end_image.permute(0, 3, 1, 2).contiguous() if end_image!=None else None
        videos = videogen_pipeline(prompt=prompt,
                                negative_prompt=negative_prompt,
                                conditional_images=[start_image,] if end_image==None else [start_image,end_image],
                                conditional_images_indices=[0,] if end_image==None else [0,-1],
                                num_frames=num_frames,
                                height=height,
                                width=width,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                enable_temporal_attentions=not force_images,
                                num_images_per_prompt=1,
                                mask_feature=True,
                                output_type="latents",
                                callback=latent_preview_callback(pbar, transformer_model.device, num_inference_steps),
                                ).images
        videogen_pipeline.to(device=cpu_device)
        videogen_pipeline.text_encoder.to(device=cpu_device)
        videogen_pipeline.transformer.to(device=cpu_device)
        videogen_pipeline.vae.to(device=cpu_device)
        torch.cuda.empty_cache()

        return ({"samples":videos},)

class OpenSoraPlanV2LoaderT2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path":("STRING",{"default":os.path.join(opensoraplan_modelpath, "Open-Sora-Plan-v1.2.0")}),
                "ae":(list(ae_stride_config_2.keys()),{"default":"CausalVAEModel_D8_4x8x8"}),
                "text_encoder_name":("STRING",{"default":os.path.join(opensoraplan_modelpath, "mt5-xxl")}),
                "version":(["93x720p","93x480p","29x720p","29x480p","1x480p",],{"default":"93x720p"}),
            },
            "hidden": {
                "force_images":("BOOLEAN",{"default":False}),
            },
        }

    RETURN_TYPES = ("OpenSoraPlanModel",)
    FUNCTION = "run"
    CATEGORY = "OpenSoraPlan/Legacy"

    def run(self,model_path,ae,text_encoder_name,version,force_images=False):
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')

        # Load model:
        model_path = check_download(model_path, "LanguageBind/Open-Sora-Plan-v1.2.0", version)
        transformer_model = OpenSoraT2V.from_pretrained(os.path.join(model_path, version), torch_dtype=torch.float16, cache_dir='cache_dir').to(cpu_device)
        model_path = check_download(model_path, "LanguageBind/Open-Sora-Plan-v1.2.0", "vae")
        vae = CausalVAEModelWrapper2(os.path.join(model_path, "vae"), cache_dir='cache_dir').eval()
        vae.vae = vae.vae.to(cpu_device, dtype=torch.float16)
        #vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = 0.25
        vae.vae_scale_factor = ae_stride_config_2[ae]
        image_size = int(version.rstrip('p').split('x')[1])
        latent_size = (image_size // ae_stride_config_2[ae][1], image_size // ae_stride_config_2[ae][2])
        vae.latent_size = latent_size
        transformer_model.force_images = force_images
        text_encoder_name = check_download(text_encoder_name, "google/mt5-xxl")
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_name, cache_dir="cache_dir")
        if "mt5" in text_encoder_name:
            MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
            text_encoder = MT5EncoderModel.from_pretrained(text_encoder_name, cache_dir="cache_dir",torch_dtype=torch.float32).to(cpu_device)
        else:
            T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
            text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, cache_dir="cache_dir",torch_dtype=torch.float32).to(cpu_device)

        # set eval mode
        transformer_model.eval()
        vae.eval()
        text_encoder.eval()
        scheduler = PNDMScheduler()
        videogen_pipeline = OpenSoraPipeline(vae=vae,
                                            text_encoder=text_encoder,
                                            tokenizer=tokenizer,
                                            scheduler=scheduler,
                                            transformer=transformer_model) #.to(device=cuda_device)
        return ((videogen_pipeline,transformer_model,version),)

class OpenSoraPlanV1LoaderT2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path":("STRING",{"default":os.path.join(opensoraplan_modelpath, "Open-Sora-Plan-v1.1.0")}),
                "ae":(list(ae_stride_config.keys()),{"default":"CausalVAEModel_4x8x8"}),
                "text_encoder_name":("STRING",{"default":os.path.join(opensoraplan_modelpath, "t5-v1_1-xxl")}),
                "version":(["221x512x512","65x512x512"],{"default":"221x512x512"}),
            },
            "hidden": {
                "force_images":("BOOLEAN",{"default":False}),
            },
        }

    RETURN_TYPES = ("OpenSoraPlanModel",)
    FUNCTION = "run"
    CATEGORY = "OpenSoraPlan/Legacy"

    def run(self,model_path,ae,text_encoder_name,version,force_images=False):
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')

        # Load model:
        model_path = check_download(model_path, "LanguageBind/Open-Sora-Plan-v1.1.0", version)
        transformer_model = LatteT2V.from_pretrained(model_path, subfolder=version, torch_dtype=torch.float16, cache_dir='cache_dir').to(cpu_device)
        model_path = check_download(model_path, "LanguageBind/Open-Sora-Plan-v1.1.0", "vae")
        vae = getae_wrapper(ae)(model_path, subfolder="vae", cache_dir='cache_dir').to(cpu_device, dtype=torch.float16)
        #vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = 0.25
        vae.vae_scale_factor = ae_stride_config[ae]
        image_size = int(version.split('x')[1])
        #latent_size = (image_size // ae_stride_config[ae][1], image_size // ae_stride_config[ae][2])
        #vae.latent_size = latent_size
        transformer_model.force_images = force_images
        text_encoder_name = check_download(text_encoder_name, "google/t5-v1_1-xxl")
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_name, cache_dir="cache_dir")
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, cache_dir="cache_dir",
                                                    torch_dtype=torch.float32).to(cpu_device)

        # set eval mode
        transformer_model.eval()
        vae.eval()
        text_encoder.eval()
        scheduler = PNDMScheduler()
        videogen_pipeline = VideoGenPipeline(vae=vae,
                                            text_encoder=text_encoder,
                                            tokenizer=tokenizer,
                                            scheduler=scheduler,
                                            transformer=transformer_model) #.to(device=cuda_device)
        return ((videogen_pipeline,transformer_model,version),)

class OpenSoraPlanV0LoaderT2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path":("STRING",{"default":os.path.join(opensoraplan_modelpath, "Open-Sora-Plan-v1.0.0")}),
                "ae":(["CausalVAEModel_4x8x8"],{"default":"CausalVAEModel_4x8x8"}),
                "text_encoder_name":("STRING",{"default":os.path.join(opensoraplan_modelpath, "t5-v1_1-xxl")}),
                "version":(["65x512x512","65x256x256","17x256x256",],{"default":"65x512x512"}),
            },
            "hidden" : {
                "force_images":("BOOLEAN",{"default":False}),
            },
        }

    RETURN_TYPES = ("OpenSoraPlanModel",)
    FUNCTION = "run"
    CATEGORY = "OpenSoraPlan/Legacy"

    def run(self,model_path,ae,text_encoder_name,version,force_images=False):
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')

        # Load model:
        model_path = check_download(model_path, "LanguageBind/Open-Sora-Plan-v1.0.0", version)
        transformer_model = LatteT2V.from_pretrained(model_path, subfolder=version, torch_dtype=torch.float16, cache_dir='cache_dir').to(cpu_device)
        model_path = check_download(model_path, "LanguageBind/Open-Sora-Plan-v1.0.0", "vae")
        vae = getae_wrapper(ae)(model_path, subfolder="vae", cache_dir='cache_dir').to(cpu_device, dtype=torch.float16)
        #vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = 0.25
        vae.vae_scale_factor = ae_stride_config[ae]
        image_size = int(version.split('x')[1])
        #latent_size = (image_size // ae_stride_config[ae][1], image_size // ae_stride_config[ae][2])
        #vae.latent_size = latent_size
        transformer_model.force_images = force_images
        text_encoder_name = check_download(text_encoder_name, "google/t5-v1_1-xxl")
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_name, cache_dir="cache_dir")
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, cache_dir="cache_dir",
                                                    torch_dtype=torch.float32).to(cpu_device)

        # set eval mode
        transformer_model.eval()
        vae.eval()
        text_encoder.eval()
        scheduler = PNDMScheduler()
        videogen_pipeline = VideoGenPipeline(vae=vae,
                                            text_encoder=text_encoder,
                                            tokenizer=tokenizer,
                                            scheduler=scheduler,
                                            transformer=transformer_model) #.to(device=cuda_device)
        return ((videogen_pipeline,transformer_model,version),)

class OpenSoraPlanSampleT2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("OpenSoraPlanModel",),
                "prompt":("STRING",{"default":""}),
                "negative_prompt":("STRING",{"default":""}),
                "num_inference_steps":("INT",{"default":50}),
                "guidance_scale":("FLOAT",{"default":10.0}),
                "seed":("INT",{"default":1234}),
                "force_textencoder_cpu":("BOOLEAN",{"default":False}),
                "force_transformer_cpu_offload":("BOOLEAN",{"default":False}),
            },
            "hidden": {
                "force_images":("BOOLEAN",{"default":False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "OpenSoraPlan"

    def run(self,model,prompt,negative_prompt,num_inference_steps,guidance_scale,seed,force_textencoder_cpu,force_transformer_cpu_offload,force_images=False):
        videogen_pipeline,transformer_model,version=model
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')
        videogen_pipeline.to(device=cpu_device)
        if not force_textencoder_cpu and videogen_pipeline.text_encoder!=None:
            videogen_pipeline.text_encoder.to(device=cuda_device, dtype=torch.bfloat16)
        if not force_transformer_cpu_offload:
            videogen_pipeline.transformer.to(device=cuda_device)
        videogen_pipeline.vae.to(device=cpu_device)
        #seed = int(randomize_seed_fn(seed, randomize_seed))
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        torch.Generator(device=cuda_device).manual_seed(seed)
        #video_length = transformer_model.config.video_length if not force_images else 1
        height, width = int(version.split('_')[0].rstrip('p').split('x')[1]), int(version.split('_')[0].rstrip('p').split('x')[2 if len(version.split('x'))>2 else 1])
        num_frames = 1 if force_images else int(version[3:].split('x')[0] if version.startswith('any') else version.split('x')[0])
        
        videos = videogen_pipeline(POS_PROMPT.format(prompt),
                                negative_prompt=NEG_PROMPT_TEMPLATE.format(negative_prompt) if negative_prompt else NEG_PROMPT,
                                num_frames=num_frames,
                                height=height,
                                width=width,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                enable_temporal_attentions=not force_images,
                                num_images_per_prompt=1,
                                mask_feature=True,
                                output_type="latents",
                                device=torch.device('cuda') if torch.cuda.is_available() and not force_textencoder_cpu else torch.device('cpu'),
                                callback=latent_preview_callback(pbar, transformer_model.device, num_inference_steps),
                                ).images
        
        videogen_pipeline.to(device=cpu_device)
        if videogen_pipeline.text_encoder != None:
            videogen_pipeline.text_encoder.to(device=cpu_device)
        videogen_pipeline.transformer.to(device=cpu_device)
        torch.cuda.empty_cache()

        return ({"samples":videos},)

class OpenSoraPlanDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("OpenSoraPlanModel",),
                "samples": ("LATENT",),
                "use_tiling":("BOOLEAN",{"default":False}),
                "tile":("INT",{"default":32}),
                "context":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "OpenSoraPlan"

    def run(self,model,samples,use_tiling,tile,context):
        videogen_pipeline,transformer_model,version=model
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')
        if use_tiling:
            videogen_pipeline.vae.vae.enable_tiling(use_tiling=use_tiling, tile=tile, context=context)
        else:
            videogen_pipeline.vae.vae.disable_tiling()
        
        olddevice = samples["samples"].device
        olddtype = samples["samples"].dtype
        if torch.cuda.is_available() and next(videogen_pipeline.vae.parameters()) != cuda_device:
            videogen_pipeline.vae.to(device=cuda_device)
        if olddevice != next(videogen_pipeline.vae.parameters()).device or olddtype != next(videogen_pipeline.vae.parameters()).dtype:
            samples["samples"] = samples["samples"].to(device=next(videogen_pipeline.vae.parameters()).device, dtype=next(videogen_pipeline.vae.parameters()).dtype)
            
        with torch.no_grad():
            video = videogen_pipeline.vae.decode(samples["samples"])
            video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().permute(0, 1, 3, 4, 2).contiguous()
            video = video/250.0

        if torch.cuda.is_available() and next(videogen_pipeline.vae.parameters()) != cpu_device:
            videogen_pipeline.vae.to(device=cpu_device)
        if olddevice != samples["samples"].device or olddtype != samples["samples"].dtype:
            samples["samples"] = samples["samples"].to(device=cpu_device)
        
        torch.cuda.empty_cache()

        return video

class OpenSoraPlanPromptRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt":("STRING",{}),
                "model_path":("STRING",{"default":os.path.join(opensoraplan_modelpath, "Open-Sora-Plan-v1.3.0")}),
                "refiner_path":(["prompt_refiner",], {"default":"prompt_refiner"}),
                "enforce_cpu":("BOOLEAN",{"default":False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "refine"
    CATEGORY = "OpenSoraPlan"
    
    def refine(self, prompt, model_path, refiner_path, enforce_cpu):
        model_path = check_download(model_path, "LanguageBind/Open-Sora-Plan-v1.3.0", refiner_path)
        device=torch.device('cuda') if torch.cuda.is_available() and not enforce_cpu else torch.device('cpu')
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, refiner_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(os.path.join(model_path, refiner_path), torch_dtype=torch.float32, trust_remote_code=True).to(device=device).eval()
        TEMPLATE = """
            Refine the sentence: \"{}\" to contain subject description, action, scene description. " \
            "(Optional: camera language, light and shadow, atmosphere) and conceive some additional actions to make the sentence more dynamic. " \
            "Make sure it is a fluent sentence, not nonsense.
            """            
        prompt = TEMPLATE.format(prompt)
        messages = [
                {"role": "system", "content": "You are a caption refiner."},
                {"role": "user", "content": prompt}
        ]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([input_ids], return_tensors="pt").to(device)
        generated_ids = model.generate(model_inputs.input_ids, attention_mask = model_inputs.attention_mask, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        if device != torch.device('cpu'):
            model_inputs = model_inputs.to(device = torch.device('cpu'))
            model = model.to(device = torch.device('cpu'))
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
POS_PROMPT = """
    high quality, high aesthetic, {}
    """

NEG_PROMPT = """
    nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
    """

NEG_PROMPT_TEMPLATE = """
    nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, {}
"""

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenSoraPlan0LoaderT2V":"(Down)Loader_OpenSoraPlanV1.0_T2V",
    "OpenSoraPlan1LoaderT2V":"(Down)Loader_OpenSoraPlanV1.1_T2V",
    "OpenSoraPlan2LoaderT2V":"(Down)Loader_OpenSoraPlanV1.2_T2V",
    "OpenSoraPlan2LoaderI2V":"(Down)Loader_OpenSoraPlanV1.2_I2V",
    "OpenSoraPlan3LoaderT2V":"(Down)Loader_OpenSoraPlanV1.3_T2V",
    "OpenSoraPlan3LoaderI2V":"(Down)Loader_OpenSoraPlanV1.3_I2V",
    "OpenSoraPlanSamplerT2V":"Sampler_OpenSoraPlan_T2V",
    "OpenSoraPlan2SamplerI2V":"Sampler_OpenSoraPlanV1.2_I2V",
    "OpenSoraPlan3SamplerI2V":"Sampler_OpenSoraPlanV1.3_I2V",
    "OpenSoraPlanDecoder":"Decoder_OpenSoraPlan",
    "OpenSoraPlanPromptRefiner":"PromptRefiner_OpenSoraPlan",
}

NODE_CLASS_MAPPINGS = {
    "OpenSoraPlan0LoaderT2V":OpenSoraPlanV0LoaderT2V,
    "OpenSoraPlan1LoaderT2V":OpenSoraPlanV1LoaderT2V,
    "OpenSoraPlan2LoaderT2V":OpenSoraPlanV2LoaderT2V,
    "OpenSoraPlan2LoaderI2V":OpenSoraPlanV2LoaderI2V,
    "OpenSoraPlan3LoaderT2V":OpenSoraPlanV3LoaderT2V,
    "OpenSoraPlan3LoaderI2V":OpenSoraPlanV3LoaderI2V,
    "OpenSoraPlanSamplerT2V":OpenSoraPlanSampleT2V,
    "OpenSoraPlan2SamplerI2V":OpenSoraPlanV2SampleI2V,
    "OpenSoraPlan3SamplerI2V":OpenSoraPlanV3SampleI2V,
    "OpenSoraPlanDecoder":OpenSoraPlanDecoder,
    "OpenSoraPlanPromptRefiner":OpenSoraPlanPromptRefiner,
}