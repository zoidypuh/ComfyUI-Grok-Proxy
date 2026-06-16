from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

import folder_paths
import numpy as np
import torch
from comfy_api.latest import InputImpl
from PIL import Image


DEFAULT_PROXY_BASE_URL = os.environ.get("COMFY_GROK_PROXY_BASE_URL", "").strip() or "http://127.0.0.1:8317/v1"
DEFAULT_API_KEY = os.environ.get("COMFY_GROK_PROXY_API_KEY", "").strip() or "dummy"
MODEL_LIST_TIMEOUT_SECONDS = 2
MODEL_LIST_CACHE_SECONDS = 30

DEFAULT_IMAGE_MODELS = [
    "grok-imagine-image",
    "grok-imagine-image-quality",
]
DEFAULT_VIDEO_MODELS = ["grok-imagine-video", "grok-imagine-video-1.5-preview"]
IMAGE_MODELS = list(DEFAULT_IMAGE_MODELS)
VIDEO_MODELS = list(DEFAULT_VIDEO_MODELS)
_MODEL_LIST_CACHE: dict[str, tuple[float, list[str], list[str]]] = {}
EXCLUDED_IMAGE_MODELS = {"grok-imagine-image-pro"}
IMAGE_ASPECT_RATIOS = [
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "9:16",
    "16:9",
    "9:19.5",
    "19.5:9",
    "9:20",
    "20:9",
    "1:2",
    "2:1",
]
IMAGE_EDIT_ASPECT_RATIOS = ["auto", *IMAGE_ASPECT_RATIOS]
VIDEO_ASPECT_RATIOS = ["auto", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"]
REFERENCE_VIDEO_ASPECT_RATIOS = ["16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"]
DONE_STATUSES = {"done", "completed", "complete", "succeeded", "success", "finished", "ready"}
FAILED_STATUSES = {"failed", "error", "expired", "cancelled", "canceled", "rejected"}


def _join_url(base_url: str, path: str) -> str:
    base = (base_url or DEFAULT_PROXY_BASE_URL).strip().rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def _json_request(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None,
    timeout: int,
    api_key: str,
) -> dict[str, Any]:
    body = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key or DEFAULT_API_KEY}",
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(_join_url(base_url, path), data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Grok media proxy returned HTTP {exc.code} for {path}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Grok media proxy at {base_url}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Grok media proxy returned non-JSON response for {path}: {raw[:500]}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object from Grok media proxy, got {type(data).__name__}")
    return data


def _model_cache_key(base_url: str) -> str:
    return (base_url or DEFAULT_PROXY_BASE_URL).strip().rstrip("/")


def _display_model_id(model_id: str) -> str:
    model_id = (model_id or "").strip()
    if model_id.startswith("xai/"):
        return model_id.removeprefix("xai/")
    return model_id


def _add_model_option(options: list[str], model_id: str) -> None:
    display_id = _display_model_id(model_id)
    if not display_id:
        return
    if display_id not in options:
        options.append(display_id)


def _is_image_model_id(model: str) -> bool:
    display_id = _display_model_id(model)
    return display_id.startswith("grok-imagine-image") and display_id not in EXCLUDED_IMAGE_MODELS


def _is_video_model_id(model: str) -> bool:
    return _display_model_id(model).startswith("grok-imagine-video")


def _model_ids_from_response(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    items = data.get("data")
    if not isinstance(items, list):
        return [], []

    image_models: list[str] = []
    video_models: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            model_id = model_id.strip()
            if _is_image_model_id(model_id):
                _add_model_option(image_models, model_id)
            elif _is_video_model_id(model_id):
                _add_model_option(video_models, model_id)
    return image_models, video_models


def _media_model_options(base_url: str = DEFAULT_PROXY_BASE_URL) -> tuple[list[str], list[str]]:
    cache_key = _model_cache_key(base_url)
    now = time.monotonic()
    cached = _MODEL_LIST_CACHE.get(cache_key)
    if cached and now - cached[0] < MODEL_LIST_CACHE_SECONDS:
        return list(cached[1]), list(cached[2])

    image_models = list(DEFAULT_IMAGE_MODELS)
    video_models = list(DEFAULT_VIDEO_MODELS)
    try:
        data = _json_request("GET", cache_key, "/models", None, MODEL_LIST_TIMEOUT_SECONDS, DEFAULT_API_KEY)
        live_image_models, live_video_models = _model_ids_from_response(data)
        if live_image_models:
            image_models = live_image_models
        if live_video_models:
            video_models = live_video_models
    except RuntimeError:
        pass

    _MODEL_LIST_CACHE[cache_key] = (now, image_models, video_models)
    return list(image_models), list(video_models)


def _image_model_options(base_url: str = DEFAULT_PROXY_BASE_URL) -> list[str]:
    return _media_model_options(base_url)[0]


def _video_model_options(base_url: str = DEFAULT_PROXY_BASE_URL) -> list[str]:
    return _media_model_options(base_url)[1]


def _default_model(options: list[str], preferred: str) -> str:
    preferred_display_id = _display_model_id(preferred)
    if preferred_display_id in options:
        return preferred_display_id
    return options[0]


def _resolve_media_model(model: str, kind: str, base_url: str) -> str:
    image_models, video_models = _media_model_options(base_url)
    options = image_models if kind == "image" else video_models
    display_id = _display_model_id(model)
    if display_id not in options:
        available = ", ".join(options) if options else "none"
        raise ValueError(
            f"{kind.title()} model {model!r} is not exposed by {_model_cache_key(base_url)}/models. "
            f"Available {kind} models: {available}"
        )
    return display_id


def _walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)


def _first_string_key(data: dict[str, Any], keys: set[str]) -> str:
    for obj in _walk_json(data):
        for key in keys:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _request_id_from(data: dict[str, Any]) -> str:
    return _first_string_key(data, {"request_id", "requestId", "id"})


def _status_from(data: dict[str, Any]) -> str:
    return _first_string_key(data, {"status", "state"}).lower()


def _video_url_from(data: dict[str, Any]) -> str:
    direct = _first_string_key(
        data,
        {"video_url", "videoUrl", "download_url", "downloadUrl", "content_url", "contentUrl"},
    )
    if direct:
        return direct
    for obj in _walk_json(data):
        value = obj.get("url")
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return ""


def _validate_prompt(prompt: str) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must not be empty")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "grok"


def _output_path_in_subdir(subdir: str, prefix: str, suffix: str) -> Path:
    output_dir = Path(folder_paths.get_output_directory()) / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{_safe_name(prefix)}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"


def _output_path(prefix: str, suffix: str) -> Path:
    return _output_path_in_subdir("grok_proxy", prefix, suffix)


def _image_output_path(prefix: str, suffix: str = ".png") -> Path:
    return _output_path_in_subdir("grok_proxy_image", prefix, suffix)


def _image_bytes_from_item(item: dict[str, Any], timeout: int) -> bytes:
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    elif item.get("url"):
        request = Request(item["url"], headers={"User-Agent": "ComfyUI-Grok-Proxy/1.0"})
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    raise RuntimeError(f"Image response item had neither b64_json nor url. Keys: {sorted(item.keys())}")


def _image_tensor_from_bytes(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _image_tensor_from_item(item: dict[str, Any], timeout: int) -> torch.Tensor:
    return _image_tensor_from_bytes(_image_bytes_from_item(item, timeout))


def _save_image_bytes(image_bytes: bytes, output_prefix: str, index: int) -> None:
    output_path = _image_output_path(f"{output_prefix}_{index:02d}", ".png")
    Image.open(io.BytesIO(image_bytes)).convert("RGB").save(output_path, format="PNG")


def _tensor_to_png_data_uri(image: torch.Tensor) -> str:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Only one image is supported for this input.")
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected IMAGE tensor shaped [H, W, C] or [1, H, W, C], got {tuple(image.shape)}")

    image_np = (image.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np).convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _image_batch_to_data_uris(images: torch.Tensor, max_images: int) -> list[str]:
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor shaped [N, H, W, C], got {tuple(images.shape)}")
    if images.shape[0] > max_images:
        raise ValueError(f"A maximum of {max_images} input images is supported.")
    return [_tensor_to_png_data_uri(images[i]) for i in range(images.shape[0])]


def _combine_image_tensors(items: list[dict[str, Any]], timeout: int, output_prefix: str | None = None) -> torch.Tensor:
    if not items:
        raise RuntimeError("Grok image response had no data.")
    tensors = []
    for idx, item in enumerate(items, start=1):
        image_bytes = _image_bytes_from_item(item, timeout)
        if output_prefix:
            _save_image_bytes(image_bytes, output_prefix, idx)
        tensors.append(_image_tensor_from_bytes(image_bytes))
    first_shape = tuple(tensors[0].shape)
    for idx, tensor in enumerate(tensors, start=1):
        if tuple(tensor.shape) != first_shape:
            raise RuntimeError(
                f"Generated images have different sizes; item 1={first_shape}, item {idx}={tuple(tensor.shape)}"
            )
    return torch.stack(tensors, dim=0)


def _download_video(video_url: str, output_prefix: str, timeout: int) -> str:
    suffix = Path(video_url.split("?", 1)[0]).suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm", ".mkv"}:
        suffix = ".mp4"
    output_path = _output_path(output_prefix, suffix)
    request = Request(video_url, headers={"User-Agent": "ComfyUI-Grok-Proxy/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response, output_path.open("wb") as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as exc:
        raise RuntimeError(f"Failed to download generated video from {video_url}: {exc}") from exc
    return str(output_path)


def _video_bytes_and_mime(video: Any) -> tuple[bytes, str]:
    source = video.get_stream_source() if hasattr(video, "get_stream_source") else None
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        data = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
        return data, mime
    if isinstance(source, io.BytesIO):
        source.seek(0)
        return source.read(), "video/mp4"
    if hasattr(video, "save_to"):
        buffer = io.BytesIO()
        video.save_to(buffer)
        buffer.seek(0)
        return buffer.read(), "video/mp4"
    raise ValueError("Unsupported VIDEO input; expected a ComfyUI video object.")


def _video_to_data_uri(video: Any, min_duration: float, max_duration: float) -> str:
    if hasattr(video, "get_duration"):
        duration = float(video.get_duration())
        if duration < min_duration or duration > max_duration:
            raise ValueError(f"Video duration must be between {min_duration:g}s and {max_duration:g}s.")
    data, mime = _video_bytes_and_mime(video)
    if len(data) > 50 * 1024 * 1024:
        raise ValueError(f"Video size ({len(data) / 1024 / 1024:.1f}MB) exceeds 50MB limit.")
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def _poll_video(
    base_url: str,
    api_key: str,
    started: dict[str, Any],
    poll_interval_seconds: int,
    max_wait_seconds: int,
    request_timeout_seconds: int,
) -> tuple[str, str]:
    request_id = _request_id_from(started)
    video_url = _video_url_from(started)
    deadline = time.monotonic() + int(max_wait_seconds)
    last_response = started

    while not video_url:
        if not request_id:
            raise RuntimeError(f"Video response did not include request_id or video URL: {started}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Grok video request {request_id} did not finish within {max_wait_seconds}s. "
                f"Last response: {last_response}"
            )
        time.sleep(int(poll_interval_seconds))
        last_response = _json_request(
            "GET",
            base_url,
            f"/videos/{quote(request_id, safe='')}",
            None,
            int(request_timeout_seconds),
            api_key,
        )
        status = _status_from(last_response)
        if status in FAILED_STATUSES:
            raise RuntimeError(f"Grok video request {request_id} ended with status {status}: {last_response}")
        if status in DONE_STATUSES or not status:
            video_url = _video_url_from(last_response)

    return video_url, request_id


def _video_result_from_request(
    base_url: str,
    api_key: str,
    path: str,
    payload: dict[str, Any],
    output_prefix: str,
    poll_interval_seconds: int,
    max_wait_seconds: int,
    request_timeout_seconds: int,
):
    started = _json_request("POST", base_url, path, payload, int(request_timeout_seconds), api_key)
    video_url, _request_id = _poll_video(
        base_url,
        api_key,
        started,
        poll_interval_seconds,
        max_wait_seconds,
        request_timeout_seconds,
    )
    video_path = _download_video(video_url, output_prefix, int(request_timeout_seconds))
    return (InputImpl.VideoFromFile(video_path),)


def _proxy_inputs(default_timeout: int = 60) -> dict[str, Any]:
    return {
        "proxy_base_url": ("STRING", {"default": DEFAULT_PROXY_BASE_URL}),
        "api_key": ("STRING", {"default": DEFAULT_API_KEY}),
        "poll_interval_seconds": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
        "max_wait_seconds": ("INT", {"default": 600, "min": 30, "max": 7200, "step": 10}),
        "request_timeout_seconds": ("INT", {"default": default_timeout, "min": 5, "max": 900, "step": 5}),
    }


class GrokImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        image_models = _image_model_options()
        return {
            "required": {
                "model": (image_models, {"default": _default_model(image_models, "grok-imagine-image")}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "The text prompt used to generate the image"}),
                "aspect_ratio": (IMAGE_ASPECT_RATIOS, {"default": "1:1"}),
                "number_of_images": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "resolution": (["1K", "2K"], {"default": "1K"}),
                "proxy_base_url": ("STRING", {"default": DEFAULT_PROXY_BASE_URL}),
                "api_key": ("STRING", {"default": DEFAULT_API_KEY}),
                "request_timeout_seconds": ("INT", {"default": 240, "min": 30, "max": 900, "step": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/Grok"

    def execute(
        self,
        model: str,
        prompt: str,
        aspect_ratio: str,
        number_of_images: int,
        seed: int,
        resolution: str,
        proxy_base_url: str,
        api_key: str,
        request_timeout_seconds: int,
    ):
        _validate_prompt(prompt)
        payload = {
            "model": _resolve_media_model(model, "image", proxy_base_url),
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "n": int(number_of_images),
            "seed": int(seed),
            "response_format": "b64_json",
            "resolution": resolution.lower(),
        }
        response = _json_request(
            "POST",
            proxy_base_url,
            "/images/generations",
            payload,
            int(request_timeout_seconds),
            api_key,
        )
        return (_combine_image_tensors(response.get("data") or [], int(request_timeout_seconds), "grok_image"),)


class GrokImageEditNode:
    @classmethod
    def INPUT_TYPES(cls):
        image_models = _image_model_options()
        return {
            "required": {
                "model": (image_models, {"default": _default_model(image_models, "grok-imagine-image")}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "tooltip": "The text prompt used to edit the image"}),
                "resolution": (["1K", "2K"], {"default": "1K"}),
                "number_of_images": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "aspect_ratio": (
                    IMAGE_EDIT_ASPECT_RATIOS,
                    {
                        "default": "auto",
                        "tooltip": "Only allowed when multiple images are connected to the image input.",
                    },
                ),
                "proxy_base_url": ("STRING", {"default": DEFAULT_PROXY_BASE_URL}),
                "api_key": ("STRING", {"default": DEFAULT_API_KEY}),
                "request_timeout_seconds": ("INT", {"default": 240, "min": 30, "max": 900, "step": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api node/image/Grok"

    def execute(
        self,
        model: str,
        image: torch.Tensor,
        prompt: str,
        resolution: str,
        number_of_images: int,
        seed: int,
        aspect_ratio: str,
        proxy_base_url: str,
        api_key: str,
        request_timeout_seconds: int,
    ):
        _validate_prompt(prompt)
        image_urls = _image_batch_to_data_uris(image, max_images=3)
        if aspect_ratio != "auto" and len(image_urls) == 1:
            raise ValueError("Custom aspect ratio is only allowed when multiple images are connected to the image input.")
        payload = {
            "model": _resolve_media_model(model, "image", proxy_base_url),
            "images": [{"url": url} for url in image_urls],
            "prompt": prompt,
            "resolution": resolution.lower(),
            "n": int(number_of_images),
            "seed": int(seed),
            "response_format": "b64_json",
            "aspect_ratio": None if aspect_ratio == "auto" else aspect_ratio,
        }
        response = _json_request("POST", proxy_base_url, "/images/edits", payload, int(request_timeout_seconds), api_key)
        return (_combine_image_tensors(response.get("data") or [], int(request_timeout_seconds), "grok_image_edit"),)


class GrokVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        proxy = _proxy_inputs()
        video_models = _video_model_options()
        return {
            "required": {
                "model": (video_models, {"default": _default_model(video_models, "grok-imagine-video")}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "Text description of the desired video."}),
                "resolution": (["480p", "720p"], {"default": "720p"}),
                "aspect_ratio": (VIDEO_ASPECT_RATIOS, {"default": "auto"}),
                "duration": ("INT", {"default": 6, "min": 1, "max": 15, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "proxy_base_url": proxy["proxy_base_url"],
                "api_key": proxy["api_key"],
                "poll_interval_seconds": proxy["poll_interval_seconds"],
                "max_wait_seconds": proxy["max_wait_seconds"],
                "request_timeout_seconds": proxy["request_timeout_seconds"],
            },
            "optional": {
                "image": ("IMAGE",),
                "output_prefix": ("STRING", {"default": "grok_video", "tooltip": "Filename prefix for the downloaded video."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "execute"
    CATEGORY = "api node/video/Grok"

    def execute(
        self,
        model: str,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        proxy_base_url: str,
        api_key: str,
        poll_interval_seconds: int,
        max_wait_seconds: int,
        request_timeout_seconds: int,
        image: torch.Tensor | None = None,
        output_prefix: str = "grok_video",
    ):
        _validate_prompt(prompt)
        payload: dict[str, Any] = {
            "model": _resolve_media_model(model, "video", proxy_base_url),
            "prompt": prompt,
            "resolution": resolution,
            "duration": int(duration),
            "aspect_ratio": None if aspect_ratio == "auto" else aspect_ratio,
            "seed": int(seed),
        }
        if image is not None:
            payload["image"] = {"url": _tensor_to_png_data_uri(image)}
        return _video_result_from_request(
            proxy_base_url,
            api_key,
            "/videos/generations",
            payload,
            output_prefix or "grok_video",
            poll_interval_seconds,
            max_wait_seconds,
            request_timeout_seconds,
        )


class GrokVideoEditNode:
    @classmethod
    def INPUT_TYPES(cls):
        proxy = _proxy_inputs()
        video_models = _video_model_options()
        return {
            "required": {
                "model": (video_models, {"default": _default_model(video_models, "grok-imagine-video")}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "Text description of the desired edit."}),
                "video": ("VIDEO", {"tooltip": "Maximum supported duration is 8.7 seconds and 50MB file size."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "proxy_base_url": proxy["proxy_base_url"],
                "api_key": proxy["api_key"],
                "poll_interval_seconds": proxy["poll_interval_seconds"],
                "max_wait_seconds": proxy["max_wait_seconds"],
                "request_timeout_seconds": proxy["request_timeout_seconds"],
            }
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "execute"
    CATEGORY = "api node/video/Grok"

    def execute(
        self,
        model: str,
        prompt: str,
        video: Any,
        seed: int,
        proxy_base_url: str,
        api_key: str,
        poll_interval_seconds: int,
        max_wait_seconds: int,
        request_timeout_seconds: int,
    ):
        _validate_prompt(prompt)
        payload = {
            "model": _resolve_media_model(model, "video", proxy_base_url),
            "prompt": prompt,
            "video": {"url": _video_to_data_uri(video, min_duration=1, max_duration=8.7)},
            "seed": int(seed),
        }
        return _video_result_from_request(
            proxy_base_url,
            api_key,
            "/videos/edits",
            payload,
            "grok_video_edit",
            poll_interval_seconds,
            max_wait_seconds,
            request_timeout_seconds,
        )


class GrokVideoExtendNode:
    @classmethod
    def INPUT_TYPES(cls):
        proxy = _proxy_inputs()
        video_models = _video_model_options()
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "tooltip": "Text description of what should happen next."}),
                "video": ("VIDEO", {"tooltip": "Source video to extend. MP4 format, 2-15 seconds."}),
                "model": (video_models, {"default": _default_model(video_models, "grok-imagine-video")}),
                "duration": ("INT", {"default": 8, "min": 2, "max": 10, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "proxy_base_url": proxy["proxy_base_url"],
                "api_key": proxy["api_key"],
                "poll_interval_seconds": proxy["poll_interval_seconds"],
                "max_wait_seconds": proxy["max_wait_seconds"],
                "request_timeout_seconds": proxy["request_timeout_seconds"],
            }
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "execute"
    CATEGORY = "api node/video/Grok"

    def execute(
        self,
        prompt: str,
        video: Any,
        model: str,
        duration: int,
        seed: int,
        proxy_base_url: str,
        api_key: str,
        poll_interval_seconds: int,
        max_wait_seconds: int,
        request_timeout_seconds: int,
    ):
        del seed
        _validate_prompt(prompt)
        payload = {
            "prompt": prompt,
            "video": {"url": _video_to_data_uri(video, min_duration=2, max_duration=15)},
            "duration": int(duration),
            "model": _resolve_media_model(model, "video", proxy_base_url),
        }
        return _video_result_from_request(
            proxy_base_url,
            api_key,
            "/videos/extensions",
            payload,
            "grok_video_extend",
            poll_interval_seconds,
            max_wait_seconds,
            request_timeout_seconds,
        )


class GrokVideoReferenceNode:
    @classmethod
    def INPUT_TYPES(cls):
        proxy = _proxy_inputs()
        video_models = _video_model_options()
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "tooltip": "Text description of the desired video."}),
                "model": (video_models, {"default": _default_model(video_models, "grok-imagine-video")}),
                "resolution": (["480p", "720p"], {"default": "720p"}),
                "aspect_ratio": (REFERENCE_VIDEO_ASPECT_RATIOS, {"default": "16:9"}),
                "duration": ("INT", {"default": 6, "min": 2, "max": 10, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "proxy_base_url": proxy["proxy_base_url"],
                "api_key": proxy["api_key"],
                "poll_interval_seconds": proxy["poll_interval_seconds"],
                "max_wait_seconds": proxy["max_wait_seconds"],
                "request_timeout_seconds": proxy["request_timeout_seconds"],
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
                "reference_image_5": ("IMAGE",),
                "reference_image_6": ("IMAGE",),
                "reference_image_7": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "execute"
    CATEGORY = "api node/video/Grok"

    def execute(
        self,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        proxy_base_url: str,
        api_key: str,
        poll_interval_seconds: int,
        max_wait_seconds: int,
        request_timeout_seconds: int,
        reference_image_1: torch.Tensor | None = None,
        reference_image_2: torch.Tensor | None = None,
        reference_image_3: torch.Tensor | None = None,
        reference_image_4: torch.Tensor | None = None,
        reference_image_5: torch.Tensor | None = None,
        reference_image_6: torch.Tensor | None = None,
        reference_image_7: torch.Tensor | None = None,
    ):
        _validate_prompt(prompt)
        reference_images = []
        for image in (
            reference_image_1,
            reference_image_2,
            reference_image_3,
            reference_image_4,
            reference_image_5,
            reference_image_6,
            reference_image_7,
        ):
            if image is not None:
                reference_images.append({"url": _tensor_to_png_data_uri(image)})
        if not reference_images:
            raise ValueError("At least one reference image is required.")
        payload = {
            "model": _resolve_media_model(model, "video", proxy_base_url),
            "reference_images": reference_images,
            "prompt": prompt,
            "resolution": resolution,
            "duration": int(duration),
            "aspect_ratio": aspect_ratio,
            "seed": int(seed),
        }
        return _video_result_from_request(
            proxy_base_url,
            api_key,
            "/videos/generations",
            payload,
            "grok_reference_video",
            poll_interval_seconds,
            max_wait_seconds,
            request_timeout_seconds,
        )


class GrokProxyVideoGenerate:
    """Compatibility shim for workflows created with the earlier proxy-video node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic slow-motion shot"}),
                "proxy_base_url": ("STRING", {"default": DEFAULT_PROXY_BASE_URL}),
                "api_key": ("STRING", {"default": DEFAULT_API_KEY}),
                "model": ("STRING", {"default": "grok-imagine-video"}),
                "duration": ("INT", {"default": 10, "min": 1, "max": 15, "step": 1}),
                "aspect_ratio": (VIDEO_ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (["720p", "480p"], {"default": "720p"}),
                "output_prefix": ("STRING", {"default": "grok_video"}),
                "poll_interval_seconds": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
                "max_wait_seconds": ("INT", {"default": 600, "min": 30, "max": 7200, "step": 10}),
                "request_timeout_seconds": ("INT", {"default": 60, "min": 5, "max": 600, "step": 5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "image_url": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "reference_images_json": ("STRING", {"multiline": True, "default": "[]"}),
                "extra_json": ("STRING", {"multiline": True, "default": "{}"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "video_url", "request_id")
    FUNCTION = "generate"
    CATEGORY = "Grok/Compatibility"

    def generate(
        self,
        prompt: str,
        proxy_base_url: str,
        api_key: str,
        model: str,
        duration: int,
        aspect_ratio: str,
        resolution: str,
        output_prefix: str,
        poll_interval_seconds: int,
        max_wait_seconds: int,
        request_timeout_seconds: int,
        seed: int,
        image_url: str = "",
        image: torch.Tensor | None = None,
        reference_images_json: str = "[]",
        extra_json: str = "{}",
    ):
        _validate_prompt(prompt)
        payload: dict[str, Any] = {
            "model": _resolve_media_model(model.strip() or "grok-imagine-video", "video", proxy_base_url),
            "prompt": prompt,
            "duration": int(duration),
            "aspect_ratio": None if aspect_ratio == "auto" else aspect_ratio,
            "resolution": resolution,
            "seed": int(seed),
        }
        image_url = image_url.strip()
        if image is not None and image_url:
            raise ValueError("Use either the image input or image_url, not both.")
        if image is not None:
            payload["image"] = {"url": _tensor_to_png_data_uri(image)}
        if image_url:
            payload["image"] = {"url": image_url}

        try:
            reference_images = json.loads(reference_images_json or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError(f"reference_images_json must be valid JSON: {exc}") from exc
        if reference_images:
            if not isinstance(reference_images, list):
                raise ValueError("reference_images_json must be a JSON list")
            if image is not None or image_url:
                raise ValueError("reference_images_json cannot be combined with image or image_url on Grok video.")
            payload["reference_images"] = reference_images

        try:
            extras = json.loads(extra_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"extra_json must be valid JSON: {exc}") from exc
        if not isinstance(extras, dict):
            raise ValueError("extra_json must be a JSON object")
        payload.update(extras)
        if "model" in payload:
            payload["model"] = _resolve_media_model(str(payload["model"]), "video", proxy_base_url)

        started = _json_request("POST", proxy_base_url, "/videos/generations", payload, int(request_timeout_seconds), api_key)
        video_url, request_id = _poll_video(
            proxy_base_url,
            api_key,
            started,
            poll_interval_seconds,
            max_wait_seconds,
            request_timeout_seconds,
        )
        video_path = _download_video(video_url, output_prefix, int(request_timeout_seconds))
        return (video_path, video_url, request_id)


NODE_CLASS_MAPPINGS = {
    "GrokImageNode": GrokImageNode,
    "GrokImageEditNode": GrokImageEditNode,
    "GrokVideoNode": GrokVideoNode,
    "GrokVideoReferenceNode": GrokVideoReferenceNode,
    "GrokVideoEditNode": GrokVideoEditNode,
    "GrokVideoExtendNode": GrokVideoExtendNode,
    "GrokProxyVideoGenerate": GrokProxyVideoGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokImageNode": "Grok Image",
    "GrokImageEditNode": "Grok Image Edit",
    "GrokVideoNode": "Grok Video",
    "GrokVideoReferenceNode": "Grok Reference-to-Video",
    "GrokVideoEditNode": "Grok Video Edit",
    "GrokVideoExtendNode": "Grok Video Extend",
    "GrokProxyVideoGenerate": "Grok Proxy Video Generate (Compatibility)",
}


def _install_partner_key_overrides() -> None:
    try:
        import nodes as comfy_nodes
    except Exception:
        return

    for name, node_cls in NODE_CLASS_MAPPINGS.items():
        if name == "GrokProxyVideoGenerate":
            continue
        comfy_nodes.NODE_CLASS_MAPPINGS[name] = node_cls
        comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS[name] = NODE_DISPLAY_NAME_MAPPINGS[name]
        node_cls.RELATIVE_PYTHON_MODULE = "custom_nodes.ComfyUI-Grok-Proxy"


_install_partner_key_overrides()
