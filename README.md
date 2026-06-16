# ComfyUI Grok Proxy Nodes

ComfyUI custom nodes for Grok image and video models through a local
OpenAI-compatible media proxy.

This extension is proxy-neutral. Use any local proxy that exposes compatible
Grok media routes; for example, set the node's `proxy_base_url` to your proxy's
`/v1` endpoint.

## Install

From your ComfyUI checkout:

```bash
cd custom_nodes
git clone https://github.com/zoidypuh/ComfyUI-Grok-Proxy.git
```

Then restart ComfyUI.

## Proxy Settings

Each node has `proxy_base_url` and `api_key` inputs. Defaults can also be set
with environment variables:

```bash
export COMFY_GROK_PROXY_BASE_URL=http://127.0.0.1:8317/v1
export COMFY_GROK_PROXY_API_KEY=dummy
```

The default URL is `http://127.0.0.1:8317/v1`, and the default API key is
`dummy`. Change either value in the node UI if your proxy uses a different port
or token.

## Nodes

- `Grok Image`
- `Grok Image Edit`
- `Grok Video`
- `Grok Video Edit`
- `Grok Video Extend`
- `Grok Reference-to-Video`
- `Grok Proxy Video Generate (Compatibility)`

## Models

Model dropdowns show normal, non-prefixed Grok model names.

The default image models are:

- `grok-imagine-image`
- `grok-imagine-image-quality`

The default video models are:

- `grok-imagine-video`
- `grok-imagine-video-1.5-preview`

When `/models` is available, the node refreshes the dropdowns from the proxy,
strips any `xai/` prefix for display, and filters out
`grok-imagine-image-pro`.

## Routes

The proxy should expose OpenAI-style JSON routes:

- `GET /models`
- `POST /images/generations`
- `POST /images/edits`
- `POST /videos/generations`
- `POST /videos/edits`
- `POST /videos/extensions`
- `GET /videos/{request_id}`

Generated videos are downloaded into ComfyUI's `output/grok_proxy/` directory.
Generated images are also saved into `output/grok_proxy_image/` and returned as
ComfyUI `IMAGE` tensors.

