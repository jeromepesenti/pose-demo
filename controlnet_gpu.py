"""ControlNet module for GPU inference. Imported by server.py when CUDA is available."""
import torch
import threading
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler, AutoencoderTiny
from rtmlib import Body

_pipe = None
_body = None
_lock = threading.Lock()

# OpenPose color scheme for ControlNet conditioning
_COLORS = [
    (255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),
    (85,255,0),(0,255,0),(0,255,85),(0,255,170),(0,255,255),
    (0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),
    (255,0,255),(255,0,170),(255,0,85),
]

_LIMBS = [
    (1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(9,10),
    (1,11),(11,12),(12,13),(1,0),(0,14),(14,16),(0,15),(15,17),
]


def _load_pipeline():
    global _pipe, _body
    with _lock:
        if _pipe is not None:
            return

        print("[ControlNet] Loading models on CUDA...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )

        # LCM-LoRA for fast inference (4 steps)
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        # Tiny VAE for faster decode
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)

        pipe.to("cuda")
        pipe.enable_attention_slicing()

        # Warmup
        from PIL import Image as PILImage
        dummy = PILImage.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        pipe("test", image=dummy, num_inference_steps=1, guidance_scale=1.0,
             width=512, height=512)

        _pipe = pipe
        _body = Body(mode='lightweight', to_openpose=True, backend='onnxruntime')
        print("[ControlNet] Ready.")


def render_openpose(keypoints, scores, w, h, threshold=0.3):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    n = len(keypoints)
    for i, (a, b) in enumerate(_LIMBS):
        if a < n and b < n and scores[a] > threshold and scores[b] > threshold:
            pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
            pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(canvas, pt1, pt2, _COLORS[i % len(_COLORS)], 4)
    for i in range(n):
        if scores[i] > threshold:
            cv2.circle(canvas, (int(keypoints[i][0]), int(keypoints[i][1])), 4,
                       _COLORS[i % len(_COLORS)], -1)
    return canvas


def generate(frame, prompt="person dancing, professional photo, studio lighting, high quality",
             steps=4, width=512, height=512):
    from PIL import Image as PILImage

    _load_pipeline()

    h, w = frame.shape[:2]
    keypoints, scores = _body(frame)
    if keypoints is None or len(keypoints) == 0:
        return frame.copy()

    # Render combined OpenPose image
    pose_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(keypoints)):
        person_pose = render_openpose(keypoints[i], scores[i], w, h)
        pose_img = np.maximum(pose_img, person_pose)

    pose_pil = PILImage.fromarray(pose_img).resize((width, height))

    with torch.inference_mode():
        result = _pipe(
            prompt=prompt,
            negative_prompt="ugly, blurry, low quality, deformed, disfigured",
            image=pose_pil,
            num_inference_steps=steps,
            guidance_scale=1.5,
            width=width,
            height=height,
        ).images[0]

    result = result.resize((w, h), PILImage.LANCZOS)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
