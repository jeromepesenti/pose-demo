"""ControlNet module for GPU inference. Imported by server.py when CUDA is available."""
import torch
import threading
import cv2
import numpy as np
from diffusers import (
    StableDiffusionControlNetPipeline, ControlNetModel,
    LCMScheduler, AutoencoderTiny,
    AnimateDiffControlNetPipeline, MotionAdapter,
)
from rtmlib import Body

_pipe = None
_video_pipe = None
_body = None
_lock = threading.Lock()
_video_lock = threading.Lock()

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

    generator = torch.Generator(device="cuda").manual_seed(42)

    with torch.inference_mode():
        result = _pipe(
            prompt=prompt,
            negative_prompt="ugly, blurry, low quality, deformed, disfigured",
            image=pose_pil,
            num_inference_steps=steps,
            guidance_scale=1.5,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

    result = result.resize((w, h), PILImage.LANCZOS)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


def _load_video_pipeline():
    global _video_pipe, _body
    with _video_lock:
        if _video_pipe is not None:
            return

        print("[AnimateDiff] Loading video pipeline on CUDA...")

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16,
        )
        motion_adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-3",
            torch_dtype=torch.float16,
        )
        pipe = AnimateDiffControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            controlnet=controlnet,
            motion_adapter=motion_adapter,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.fuse_lora()
        pipe.enable_attention_slicing()
        pipe.to("cuda")

        _video_pipe = pipe
        if _body is None:
            _body = Body(mode='lightweight', to_openpose=True, backend='onnxruntime')
        print("[AnimateDiff] Ready.")


# Cache for generated video segments
_video_cache = {
    "frames": [],       # list of BGR output frames
    "start_time": -1,   # start time of cached segment
    "end_time": -1,
    "prompt": "",
}


def generate_video_batch(frames_bgr, prompt="person dancing, professional photo, studio lighting, high quality",
                         steps=4, width=512, height=512):
    """Generate a temporally consistent batch of frames using AnimateDiff + ControlNet."""
    from PIL import Image as PILImage

    _load_video_pipeline()

    if not frames_bgr:
        return []

    h, w = frames_bgr[0].shape[:2]

    # Detect poses and render OpenPose images for each frame
    pose_images = []
    for frame in frames_bgr:
        keypoints, scores = _body(frame)
        pose_img = np.zeros((h, w, 3), dtype=np.uint8)
        if keypoints is not None:
            for i in range(len(keypoints)):
                person_pose = render_openpose(keypoints[i], scores[i], w, h)
                pose_img = np.maximum(pose_img, person_pose)
        pose_pil = PILImage.fromarray(pose_img).resize((width, height))
        pose_images.append(pose_pil)

    generator = torch.Generator(device="cuda").manual_seed(42)

    with torch.inference_mode():
        result = _video_pipe(
            prompt=prompt,
            negative_prompt="ugly, blurry, low quality, deformed, disfigured",
            conditioning_frames=pose_images,
            num_frames=len(pose_images),
            num_inference_steps=steps,
            guidance_scale=1.5,
            width=width,
            height=height,
            generator=generator,
        )

    output_frames = []
    for pil_frame in result.frames[0]:
        resized = pil_frame.resize((w, h), PILImage.LANCZOS)
        output_frames.append(cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR))

    return output_frames


def generate_video_frame(video_path, t, fps, prompt, num_context=8):
    """Generate a single frame using AnimateDiff by processing a batch around time t."""
    import math

    # Check cache
    if (_video_cache["start_time"] <= t <= _video_cache["end_time"]
            and _video_cache["prompt"] == prompt
            and _video_cache["frames"]):
        frame_idx = int((t - _video_cache["start_time"]) * fps)
        frame_idx = min(frame_idx, len(_video_cache["frames"]) - 1)
        return _video_cache["frames"][frame_idx]

    # Read a batch of frames around time t
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    center_frame = int(t * fps)
    start_frame = max(0, center_frame - num_context // 2)
    end_frame = min(total_frames, start_frame + num_context)
    start_frame = max(0, end_frame - num_context)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_bgr = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()

    if not frames_bgr:
        return None

    # Generate batch
    output_frames = generate_video_batch(frames_bgr, prompt=prompt)

    # Cache results
    _video_cache["frames"] = output_frames
    _video_cache["start_time"] = start_frame / fps
    _video_cache["end_time"] = (start_frame + len(output_frames)) / fps
    _video_cache["prompt"] = prompt

    # Return the requested frame
    frame_idx = center_frame - start_frame
    frame_idx = max(0, min(frame_idx, len(output_frames) - 1))
    return output_frames[frame_idx]
