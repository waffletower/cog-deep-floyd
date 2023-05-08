import os
import sys
from diffusers import (
    DiffusionPipeline,
    IFPipeline,
    IFSuperResolutionPipeline,
)
from diffusers.utils import pt_to_pil
import torch
from cog import BasePredictor, Input, Path
from typing import List
import numpy as np
from tqdm import tqdm
from functools import partialmethod

# disables chatty progress bars
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

MODEL_CACHE = "diffusers-cache"
STAGE1_MODEL = "DeepFloyd/IF-I-XL-v1.0"
STAGE2_MODEL = "DeepFloyd/IF-II-L-v1.0"
STAGE3_MODEL = "stabilityai/stable-diffusion-x4-upscaler"
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')

class Predictor(BasePredictor):
    def setup(self):
        """load pipeline stages into cpu memory and off-load"""
        print("Loading pipeline stages...")

        # stage 1
        self.stage1 = IFPipeline.from_pretrained(STAGE1_MODEL,
                                                 cache_dir=MODEL_CACHE,
                                                 variant="fp16",
                                                 torch_dtype=torch.float16,
                                                 use_auth_token=HUGGINGFACE_TOKEN)
        self.stage1.enable_model_cpu_offload()

        # stage 2
        self.stage2 = IFSuperResolutionPipeline.from_pretrained(STAGE2_MODEL,
                                                                cache_dir=MODEL_CACHE,
                                                                text_encoder=None,
                                                                variant="fp16",
                                                                torch_dtype=torch.float16,
                                                                use_auth_token=HUGGINGFACE_TOKEN)
        self.stage2.enable_model_cpu_offload()

        # stage 3
        self.safety_modules = {}

        # self.safety_modules = {"feature_extractor": self.stage1.feature_extractor,
        #                        "safety_checker": self.stage1.safety_checker,
        #                        "watermarker": self.stage1.watermarker}

        self.stage3 = DiffusionPipeline.from_pretrained(STAGE3_MODEL,
                                                        **(self.safety_modules),
                                                        cache_dir=MODEL_CACHE,
                                                        torch_dtype=torch.float16,
                                                        use_auth_token=HUGGINGFACE_TOKEN)
        self.stage3.enable_model_cpu_offload()
        

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default='A picture of vintage Mickey Mouse with a sad face, in a steel art deco bank vault',
        ),            
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image.",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image.",
            default=1024,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=100
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:

        # text embeds
        prompt_embeds, negative_embeds = self.stage1.encode_prompt(prompt)

        seed = seed or np.random.randint(0, sys.maxsize)
        generator = torch.manual_seed(seed)

        print(f"Using seed: {seed}")
    
        image = self.stage1(prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_embeds,
                            generator=generator,
                            output_type="pt",
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            width=int(width / 16),
                            height=int(height / 16)).images
        
        image = self.stage2(image=image,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_embeds,
                            generator=generator,
                            output_type="pt",
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            width=int(width / 4),
                            height=int(height / 4)).images

        image = self.stage3(prompt=prompt,
                            image=image,
                            generator=generator,
                            noise_level=100).images
        
        output_paths = []
        output_path = f"/tmp/output.png"
        image[0].save(output_path)
        output_paths.append(Path(output_path))
        
        return output_paths
