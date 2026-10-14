# `image_qa` — Vision-model Q&A over images

## Parameters
| Param | Type | Default | Description |
|---|---|---|---|
| `file_path` | string | (required) | Image file path (relative to workspace) or HTTP/HTTPS URL. |
| `question` | string | (required) | Natural-language question about the image. |

## How it works
1. **URL support**: if `file_path` is a URL, downloads image to workspace cache first.
2. **Extension validation**: only `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`. Non-image files rejected with hints.
3. **Image reading**: MCP `read_image_for_qa` → base64 Data URI.
4. **MIME validation**: confirms `image/jpeg`, `image/png`, `image/gif`, or `image/webp`.
5. **VLM call**: sends image + question to agent's `vlm_config` model.
6. **Auto-summary**: if response exceeds 8,000 chars, compresses it.

## LLM involvement
- Uses agent's `vlm_config` (separate from `llm_config`). Typically GPT-4o or GLM-4-Vision.
- Single-round: one VLM call per invocation.

## Limitations
- **Images only.** Cannot process PDF, DOCX, videos, or non-image formats.
- **No multi-image support.** One image per call.
- **VLM quality varies.** Small text, complex diagrams, or scanned documents may not be accurately interpreted.
- **Workspace paths only** — remote URLs are rejected; save the image into the workspace first.
- **Supported formats only**: JPEG, PNG, GIF, WebP. BMP and TIFF NOT supported.

## Synergies and overlaps
| Pair | Relationship |
|---|---|
| `list_file_tree` → `image_qa` | Locate image artefacts in the workspace, then analyse them. |
| `image_qa` vs `text_qa` on PDF | PDFs are NOT images. `image_qa` rejects them. Use `text_qa`/`read_file`. For scanned PDFs: `read_file` extracts images to `<pdf_stem>_images/`, then `image_qa` on each. |
| `image_qa` vs `read_file` | `read_file` cannot interpret image content. `image_qa` is the only way to understand images. |

## Common failure patterns
- Agent calls `image_qa` on PDF → rejected. Should use `text_qa`. **Fix**: tool description already redirects; add skill if persists.
- Agent asks vague questions about complex images → unhelpful answers. **Fix**: skill enforcing specific, targeted questions.
