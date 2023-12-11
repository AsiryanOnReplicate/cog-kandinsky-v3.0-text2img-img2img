# Prediction interface for Cog ⚙️
import os
from typing import List
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, UNet2DConditionModel
import torch
from PIL import Image
import PIL.ImageOps
from transformers import CLIPVisionModelWithProjection
from cog import BasePredictor, Input, Path


MODEL_CACHE = "weights_cache"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        device = torch.device("cuda:0")

        self.text2img_pipe = AutoPipelineForText2Image.from_pretrained(
            "kandinsky-community/kandinsky-3",
            variant="fp16",
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
        ).to(device)
        self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            "kandinsky-community/kandinsky-3",
            variant="fp16",
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
        ).to(device)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="beautiful fairy-tale desert, in the sky a wave of sand merges with the milky way, stars, cosmism, digital art, 8k",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
        ),
        image: Path = Input(
            description="Input image for img2img mode",
            default=None
        ),
        width: int = Input(
            description="Width of output image. Lower the setting if hits memory limits.",
            ge=0,
            le=2048,
            default=1024,
        ),
        height: int = Input(
            description="Height of output image. Lower the setting if hits memory limits.",
            ge=0,
            le=2048,
            default=1024,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", 
            ge=1, 
            le=500,
            default=50
        ),
        strength: float = Input(
            description="Strength/weight", 
            ge=0, 
            le=1, 
            default=0.75
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        if image:
            print("Mode: img2img")
            init_image = Image.open(image).convert('RGB')

            output = self.img2img_pipe(
                image=init_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=num_inference_steps
            ).images[0]
        else:
            print("Mode: text2img")
            output = self.text2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
            ).images[0]

        out_path = Path(f"/tmp/output.png")
        output.save(out_path)
        return out_path
