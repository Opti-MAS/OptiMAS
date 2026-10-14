### `image_qa` — Vision-model Q&A on images

**Parameters:** `file_path` (required — local path or HTTP URL), `question` (required).

- Supported formats: JPEG, PNG, GIF, WebP only.
- If given a URL, downloads the image to workspace first.
- Rejects non-image files (PDF, DOCX, etc.) with clear error messages and alternative tool suggestions.
- Single-round VLM call.

**Best practices:**
- Ask specific, targeted questions (e.g., "What text appears in the header?") rather than "Describe this image."
- For PDF files, use `text_qa` or `read_file` — PDFs are not images.
- For scanned PDFs: `read_file` extracts embedded images to `<pdf_stem>_images/`; use `image_qa` on each extracted image.
