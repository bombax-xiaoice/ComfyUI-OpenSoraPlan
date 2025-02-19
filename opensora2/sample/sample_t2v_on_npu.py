import math
import os
import torch
import argparse
import torchvision
import torch.distributed as dist

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder, Transformer2DModel
from omegaconf import OmegaConf
from torchvision.utils import save_image
from transformers import T5EncoderModel, MT5EncoderModel, T5Tokenizer, AutoTokenizer

import os, sys

from opensora2.adaptor.modules import replace_with_fp32_forwards
from opensora2.adaptor.modules import replace_with_fp32_forwards
from opensora2.models.causalvideovae import ae_stride_config, ae_channel_config, ae_norm, ae_denorm, CausalVAEModelWrapper
from opensora2.models.diffusion.opensora.modeling_opensora import OpenSoraT2V
# from opensora2.models.diffusion.compress3d.modeling_opensora import CompressOpenSoraT2V

from opensora2.models.text_encoder import get_text_enc
from opensora2.utils.utils import save_video_grid

from opensora2.sample.pipeline_opensora import OpenSoraPipeline

import imageio

try:
    import torch_npu
except:
    pass
import time
from opensora2.npu_config import npu_config


def load_t2v_checkpoint(model_path):
    if args.model_3d:
        # transformer_model = UDiTT2V.from_pretrained(model_path, cache_dir=args.cache_dir,
        #                                                 low_cpu_mem_usage=False, device_map=None,
        #                                                 torch_dtype=weight_dtype)
        # if args.agent is not None or args.sparse:
        #     transformer_model = CompressOpenSoraT2V.from_pretrained(model_path, cache_dir=args.cache_dir,
        #                                                     low_cpu_mem_usage=False, device_map=None,
        #                                                     torch_dtype=weight_dtype)
        # else:
        transformer_model = OpenSoraT2V.from_pretrained(model_path, cache_dir=args.cache_dir,
                                                        low_cpu_mem_usage=False, device_map=None,
                                                        torch_dtype=weight_dtype)
    print(transformer_model.config)

    # set eval mode
    transformer_model.eval()
    pipeline = OpenSoraPipeline(vae=vae,
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                scheduler=scheduler,
                                transformer=transformer_model).to(device)

    return pipeline


def get_latest_path():
    # Get the most recent checkpoint
    dirs = os.listdir(args.model_path)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None

    return path


def run_model_and_save_images(pipeline, model_path):
    video_grids = []
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()
        args.text_prompt = [i.strip() for i in text_prompt]

    checkpoint_name = f"{os.path.basename(model_path)}"

    positive_prompt = "(masterpiece), (best quality), (ultra-detailed), {}. emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous"
    negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    prompts_per_rank = len(args.text_prompt) // npu_config.N_NPU_PER_NODE
    for index, prompt in enumerate(args.text_prompt):
        if index < prompts_per_rank * local_rank or index >= prompts_per_rank * (local_rank + 1):
            continue
        print('Processing the ({}) prompt'.format(prompt))
        videos = pipeline(positive_prompt.format(prompt),
                          negative_prompt=negative_prompt,
                          num_frames=args.num_frames,
                          height=args.height,
                          width=args.width,
                          num_inference_steps=args.num_sampling_steps,
                          guidance_scale=args.guidance_scale,
                          num_images_per_prompt=1,
                          mask_feature=True,
                          device=args.device,
                          max_sequence_length=512,
                          ).images
        print(videos.shape)
        try:
            if args.num_frames == 1:
                videos = videos[:, 0].permute(0, 3, 1, 2)  # b t h w c -> b c h w
                save_image(videos / 255.0, os.path.join(args.save_img_path,
                                                        f'{args.model_path.split(os.sep)[-1]}_{args.sample_method}_{index}_{checkpoint_name}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'),
                           nrow=1, normalize=True, value_range=(0, 1))  # t c h w

            else:
                imageio.mimwrite(
                    os.path.join(
                        args.save_img_path,
                        f'{args.model_path.split(os.sep)[-1]}_{args.sample_method}_{index}_{checkpoint_name}__gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'
                    ), videos[0],
                    fps=args.fps, quality=6, codec='libx264',
                    output_params=['-threads', '20'])  # highest quality is 10, lowest is 0
        except:
            print('Error when saving {}'.format(prompt))
        video_grids.append(videos)

    video_grids = torch.cat(video_grids, dim=0).cuda()
    shape = list(video_grids.shape)
    shape[0] *= world_size
    gathered_tensor = torch.zeros(shape, dtype=video_grids.dtype, device=device)
    dist.all_gather_into_tensor(gathered_tensor, video_grids.contiguous())
    video_grids = gathered_tensor.cpu()

    # video_grids = video_grids.repeat(world_size, 1, 1, 1)
    # output = torch.zeros(video_grids.shape, dtype=video_grids.dtype, device=device)
    # dist.all_to_all_single(output, video_grids)
    # video_grids = output.cpu()
    def get_file_name():
        return os.path.join(args.save_img_path,
                            f'{args.model_path.split(os.sep)[-1]}_{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}_{checkpoint_name}.{ext}')

    if args.num_frames == 1:
        save_image(video_grids / 255.0, get_file_name(),
                   nrow=math.ceil(math.sqrt(len(video_grids))), normalize=True, value_range=(0, 1))
    else:
        video_grids = save_video_grid(video_grids)
        imageio.mimwrite(get_file_name(), video_grids, fps=args.fps, quality=6)

    print('save path {}'.format(args.save_img_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--version", type=str, default=None, choices=[None, '65x512x512', '65x256x256', '17x256x256'])
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--ae_path", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--model_3d', action='store_true')
    parser.add_argument('--udit', action='store_true')
    parser.add_argument('--save_memory', action='store_true')
    parser.add_argument("--agent", type=str, default=None, help="agent attention")
    parser.add_argument("--sparse", type=str, default=None, help="sparse attention")
    args = parser.parse_args()
    if args.agent:
        npu_config.agent = npu_config.get_tokens(args.agent)
    if args.sparse:
        npu_config.sparse = npu_config.get_tokens(args.sparse)

    npu_config.print_msg(args)
    npu_config.conv_dtype = torch.bfloat16
    replace_with_fp32_forwards()

    # 初始化分布式环境
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    if npu_config.on_npu:
        torch_npu.npu.set_device(local_rank)
    dist.init_process_group(backend='hccl', init_method='env://', world_size=8, rank=local_rank)

    # torch.manual_seed(args.seed)
    weight_dtype = torch.bfloat16
    device = torch.cuda.current_device()

    # vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir=args.cache_dir)
    vae = CausalVAEModelWrapper(args.ae_path)
    vae.vae = vae.vae.to(device=device, dtype=torch.float32)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
        vae.vae.tile_sample_min_size = 512
        vae.vae.tile_latent_min_size = 64
        vae.vae.tile_sample_min_size_t = 29
        vae.vae.tile_latent_min_size_t = 8
        if args.save_memory:
            vae.vae.tile_sample_min_size = 256
            vae.vae.tile_latent_min_size = 32
            vae.vae.tile_sample_min_size_t = 29
            vae.vae.tile_latent_min_size_t = 8
    vae.vae_scale_factor = ae_stride_config[args.ae]

    text_encoder = MT5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir,
                                                  low_cpu_mem_usage=True, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)

    # set eval mode
    vae.eval()
    text_encoder.eval()

    if args.sample_method == 'DDIM':  #########
        scheduler = DDIMScheduler()
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == 'DDPM':  #############
        scheduler = DDPMScheduler()
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler()
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler()
    elif args.sample_method == 'PNDM':
        scheduler = PNDMScheduler()
    elif args.sample_method == 'HeunDiscrete':  ########
        scheduler = HeunDiscreteScheduler()
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler()
    elif args.sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler()
    elif args.sample_method == 'KDPM2AncestralDiscrete':  #########
        scheduler = KDPM2AncestralDiscreteScheduler()

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path, exist_ok=True)

    if args.num_frames == 1:
        video_length = 1
        ext = 'jpg'
    else:
        ext = 'mp4'

    latest_path = None
    save_img_path = args.save_img_path
    first_in = False
    while True:
        cur_path = get_latest_path()
        if cur_path == latest_path:
            time.sleep(60)
            continue

        if not first_in:
            first_in = True
        else:
            time.sleep(1200)

        latest_path = cur_path

        npu_config.print_msg(f"The latest_path is {latest_path}")
        full_path = f"{args.model_path}/{latest_path}/model_ema"
        # full_path = "/home/opensora/captions/240p_model_ema"
        pipeline = load_t2v_checkpoint(full_path)

        if npu_config.on_npu and npu_config.profiling:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization
            )
            profile_output_path = "/home/image_data/npu_profiling_t2v"
            os.makedirs(profile_output_path, exist_ok=True)

            with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU],
                    with_stack=True,
                    record_shapes=True,
                    profile_memory=True,
                    experimental_config=experimental_config,
                    schedule=torch_npu.profiler.schedule(wait=10000, warmup=0, active=1, repeat=1,
                                                         skip_first=0),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"{profile_output_path}/")
            ) as prof:
                run_model_and_save_images(pipeline, latest_path)
                prof.step()
        else:
            run_model_and_save_images(pipeline, latest_path)
