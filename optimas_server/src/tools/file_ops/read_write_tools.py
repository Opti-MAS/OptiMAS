"""
File read and write operations tools.

Provides:
- read_file: Read file content with line range support (text + rich formats)
- write_file: Write or append content to file
- write_file_lines: Write file content as array of lines
- read_file_for_qa: Read file content in chunks for Q&A (64k chars per chunk)
- read_image_for_qa: Read image file and return base64 Data URI
"""
import asyncio
import base64
import logging
import aiofiles
from pathlib import Path
from typing import Annotated, List, Optional

from mcp import McpError
from mcp.types import ErrorData
from pydantic import Field

from src.tools.common_utils.path_utils import resolve_workspace_path
from src.utils.error_codes import INVALID_PARAMS, INTERNAL_ERROR
from src.tools.file_ops.rich_file_reader import is_rich_format, read_rich_file

logger = logging.getLogger(__name__)


async def _repair_docker_owned_path_if_possible(path: Path) -> None:
    # Local import avoids coupling normal read paths to edit tool internals.
    from src.tools.file_ops.edit_tools import _try_repair_docker_owned_path

    await _try_repair_docker_owned_path(path)


async def _validate_and_read_text_file(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> tuple[Path, list[str], int]:
    """
    Validate file path and read text file content asynchronously, optionally within a line range.
    Uses stream reading for large files to avoid blocking the event loop.
    
    Args:
        file_path: File path relative to workspace or absolute path
        start_line: Optional start line number (1-based, inclusive). If None, read from beginning.
        end_line: Optional end line number (1-based, inclusive). If None, read to end.
        
    Returns:
        Tuple of (full_path, lines, total_lines) where:
        - full_path: Resolved file path
        - lines: List of file lines (within range if specified)
        - total_lines: Total number of lines in the file
        
    Raises:
        McpError: If file doesn't exist, is a directory, or can't be decoded as UTF-8
    """
    # Validate file path
    full_path = await resolve_workspace_path(file_path)
    
    if not full_path.exists():
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message=f"File does not exist: {file_path}"
            )
        )
    
    if full_path.is_dir():
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message=f"Path is a directory, not a file: {file_path}"
            )
        )
    
    # Read file content asynchronously (UTF-8 only)
    try:
        # If line range is specified, use stream reading to only read needed lines
        if start_line is not None or end_line is not None:
            lines = []
            total_lines = 0
            current_line = 0
            start_idx = max(0, (start_line - 1) if start_line is not None else 0)
            end_idx = end_line if end_line is not None else None
            
            async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                async for line in f:
                    total_lines += 1
                    current_line += 1
                    
                    # Only collect lines within the specified range
                    if current_line > start_idx:
                        if end_idx is None:
                            # No end_line specified, read to end
                            lines.append(line)
                        elif current_line <= end_idx:
                            # Within range, collect the line
                            lines.append(line)
                        else:
                            # Beyond end_line, stop reading (optimization)
                            break
        else:
            # Read entire file asynchronously
            async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                lines = await f.readlines()
            total_lines = len(lines)
            
    except UnicodeDecodeError as e:
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"Failed to decode file as UTF-8: {e}. The file may be binary or use an unsupported encoding."
            )
        )
    
    return full_path, lines, total_lines


def _format_lines_with_numbers(lines: list[str], start_line_idx: int) -> list[str]:
    """
    Format lines with right-aligned line numbers: ``    1 | content``.

    The ``<spaces><number> | `` prefix is clearly separated from file content
    by the `` | `` delimiter so downstream consumers (including LLMs) can
    distinguish the line-number gutter from the actual text.
    
    Args:
        lines: List of lines to format
        start_line_idx: Starting line index (0-based) for line numbering
        
    Returns:
        List of formatted lines with line numbers
    """
    last_line_num = start_line_idx + len(lines)
    width = max(len(str(last_line_num)), 4)
    formatted_lines = []
    for i, line in enumerate(lines):
        line_num = start_line_idx + i + 1
        content = line.rstrip('\n\r')
        formatted_lines.append(f"{line_num:>{width}} | {content}")
    return formatted_lines


async def _read_rich_file_as_text(
    file_path: str,
    full_path: Path,
    *,
    include_line_numbers: bool,
    start_line: int = 1,
    end_line: Optional[int] = None,
) -> dict:
    """Extract text from a rich-format file and return it like a plain text read.

    Runs the (synchronous) extractor in a thread-pool executor so the event
    loop is not blocked by potentially heavy PDF / DOCX parsing.
    """
    loop = asyncio.get_running_loop()

    try:
        text, meta = await loop.run_in_executor(
            None, lambda: read_rich_file(full_path, extract_images=True)
        )
    except Exception as exc:
        logger.warning("Rich file extraction failed for %s: %s", file_path, exc)
        return {
            "success": False,
            "data": None,
            "error": f"Failed to extract text from {full_path.suffix} file: {exc}",
        }

    if meta.get("error"):
        return {"success": False, "data": None, "error": text}

    all_lines = text.split("\n")
    total_lines = len(all_lines)

    if total_lines == 0:
        return {
            "success": True,
            "data": f"Total lines in extracted text: 0 (format: {meta.get('format', '?')})\n\n<file_content></file_content>",
        }

    start_idx = max(0, start_line - 1)
    end_idx = end_line if end_line is not None else total_lines
    selected_lines = all_lines[start_idx:end_idx]

    if include_line_numbers:
        formatted_lines = _format_lines_with_numbers(
            [l + "\n" for l in selected_lines], start_idx
        )
    else:
        formatted_lines = list(selected_lines)

    result_text = "\n".join(formatted_lines)

    fmt_label = meta.get("format", full_path.suffix.lstrip("."))
    total_lines_header = f"Total lines in extracted text: {total_lines} (format: {fmt_label})\n\n"

    max_char_len = 40000
    if len(result_text) > max_char_len:
        truncated_lines = []
        char_count = 0
        for line in formatted_lines:
            line_with_newline = line + "\n"
            if char_count + len(line_with_newline) > max_char_len:
                break
            truncated_lines.append(line)
            char_count += len(line_with_newline)

        displayed_start_line = start_line
        displayed_end_line = start_line + len(truncated_lines) - 1
        content_text = "\n".join(truncated_lines)
        truncation_header = (
            f"(truncated and line {displayed_start_line}-{displayed_end_line} displayed, below are extracted content):\n"
            f"---\n"
        )
        result_text = (
            total_lines_header
            + truncation_header
            + f"<file_content>{content_text}</file_content>"
        )
    else:
        result_text = total_lines_header + f"<file_content>{result_text}</file_content>"

    return {"success": True, "data": result_text}


def _register_read_write_tools(mcp):
    """Register read and write file tools to FastMCP instance (internal)."""

    @mcp.tool
    async def read_file(
        file_path: Annotated[str, Field(description="File path relative to the session workspace root, or absolute starting with '/'")],
        include_line_numbers: Annotated[bool, Field(description="If True, prefix each line with a gutter like '   1 | content' (the number and ' | ' are NOT file content). If False, return raw content without line numbers to save tokens.")] = False,
        start_line: Annotated[int, Field(description="Start line number (inclusive, 1-based).", ge=1)] = 1,
        end_line: Annotated[int, Field(description="End line number (inclusive, 1-based). If not provided, read until the end of file")] = None,
    ) -> dict:
        """
        Read file content within line range. Supports plain text and rich formats (PDF, DOCX, PPTX, XLSX, CSV, HTML).
        For rich formats, text is extracted automatically; start_line/end_line apply to the extracted text lines.
        File content will be wrapped in <file_content></file_content> tags. Only content within these tags is the actual file content. 
        If include_line_numbers is True, each line is prefixed with a right-aligned gutter: "   1 | content". The " <number> | " prefix is NOT part of the actual file content — strip it before using the text in file_edit or patch_file.
        """
        try:
            full_path = await resolve_workspace_path(file_path)
            if not full_path.exists():
                return {"success": False, "data": None, "error": f"File does not exist: {file_path}"}
            if full_path.is_dir():
                return {"success": False, "data": None, "error": f"Path is a directory, not a file: {file_path}"}

            if start_line < 1:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="start_line must be >= 1")
                )
            if end_line is not None and end_line < start_line:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="end_line must be >= start_line")
                )

            # --- Rich format branch (PDF, DOCX, PPTX, XLSX, CSV, HTML) ---
            if is_rich_format(full_path):
                return await _read_rich_file_as_text(
                    file_path, full_path,
                    include_line_numbers=include_line_numbers,
                    start_line=start_line,
                    end_line=end_line,
                )

            # --- Plain text branch (original behaviour) ---
            full_path, selected_lines, total_lines = await _validate_and_read_text_file(
                file_path,
                start_line=start_line,
                end_line=end_line,
            )

            if total_lines == 0:
                return {
                    "success": True,
                    "data": "Total lines in file: 0\n\n<file_content></file_content>",
                }

            start_idx = max(0, start_line - 1)

            if include_line_numbers:
                formatted_lines = _format_lines_with_numbers(selected_lines, start_idx)
            else:
                formatted_lines = [line.rstrip('\n\r') for line in selected_lines]

            result_text = '\n'.join(formatted_lines)

            total_lines_header = f"Total lines in file: {total_lines}\n\n"

            max_char_len = 40000
            if len(result_text) > max_char_len:
                truncated_lines = []
                char_count = 0
                for line in formatted_lines:
                    line_with_newline = line + '\n'
                    if char_count + len(line_with_newline) > max_char_len:
                        break
                    truncated_lines.append(line)
                    char_count += len(line_with_newline)

                displayed_start_line = start_line
                displayed_end_line = start_line + len(truncated_lines) - 1
                content_text = '\n'.join(truncated_lines)
                truncation_header = (
                    f"(truncated and line {displayed_start_line}-{displayed_end_line} displayed, below are file content):\n"
                    f"---\n"
                )
                result_text = (
                    total_lines_header
                    + truncation_header
                    + f"<file_content>{content_text}</file_content>"
                )
            else:
                result_text = total_lines_header + f"<file_content>{result_text}</file_content>"

            return {"success": True, "data": result_text}

        except McpError:
            raise
        except Exception as e:
            raise McpError(
                ErrorData(code=INTERNAL_ERROR, message=f"File read failed: {str(e)}")
            )

    @mcp.tool
    async def write_file(
        file_path: Annotated[str, Field(description="File path relative to the session workspace root, or absolute starting with '/'")],
        content: Annotated[str, Field(description="Content to write to the file")],
        append: Annotated[bool, Field(description="If True, append to the file; if False, overwrite the file.")] = False,
    ) -> dict:
        """
        Write or append content to file.
        """
        try:
            # Resolve client path to workspace
            full_path = await resolve_workspace_path(file_path)
            
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write or append content to file
            encoding = "utf-8"
            if append and full_path.exists():
                # Append mode: add content to existing file
                with full_path.open("a", encoding=encoding) as f:
                    f.write(content)
            else:
                # Overwrite mode: create new file or overwrite existing file
                full_path.write_text(content, encoding=encoding)
            
            # Count lines in written content
            line_count = content.count('\n') + (1 if content and not content.endswith('\n') else 0)

            # Generate message based on append mode
            if append:
                message = f"Successfully appended {len(content)} characters ({line_count} lines) to {file_path}"
            else:
                message = f"Successfully wrote {len(content)} characters ({line_count} lines) to {file_path}"

            return {
                "success": True,
                "data": message,
            }

        except McpError:
            raise
        except PermissionError:
            await _repair_docker_owned_path_if_possible(full_path)
            try:
                if append and full_path.exists():
                    with full_path.open("a", encoding="utf-8") as f:
                        f.write(content)
                else:
                    full_path.write_text(content, encoding="utf-8")
            except Exception as retry_error:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Permission denied: {retry_error}"
                    )
                )
            line_count = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
            message = (
                f"Successfully appended {len(content)} characters ({line_count} lines) to {file_path}"
                if append else
                f"Successfully wrote {len(content)} characters ({line_count} lines) to {file_path}"
            )
            return {"success": True, "data": message}
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"File write failed: {str(e)}"
                )
            )

    @mcp.tool
    async def write_file_lines(
        file_path: Annotated[str, Field(description="File path relative to the session workspace root, or absolute starting with '/'")],
        content_lines: Annotated[List[str], Field(description="File content as an array of logical lines. Do NOT include line endings in each string.")],
    ) -> dict:
        """
        Write file content using a list of logical lines instead of a single large string.
        
        Each element of content_lines represents ONE logical line of code/text.
        DO NOT add '\\n' or other line endings inside each string.
        """
        try:
            if content_lines is None:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message="content_lines parameter is required"
                    )
                )
            if not isinstance(content_lines, list):
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message="content_lines must be a list of strings"
                    )
                )

            # Ensure all items are strings
            normalized_lines: List[str] = []
            for idx, line in enumerate(content_lines):
                if not isinstance(line, str):
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS,
                            message=f"content_lines[{idx}] is not a string (got {type(line).__name__})"
                        )
                    )
                normalized_lines.append(line)

            # Join lines with '\n' and ensure final newline if file is non-empty
            if normalized_lines:
                content = "\n".join(normalized_lines)
                if not content.endswith("\n"):
                    content = content + "\n"
            else:
                content = ""

            # Resolve client path to workspace
            full_path = await resolve_workspace_path(file_path)

            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content to file (overwrite)
            encoding = "utf-8"
            try:
                full_path.write_text(content, encoding=encoding)
            except PermissionError:
                await _repair_docker_owned_path_if_possible(full_path)
                full_path.write_text(content, encoding=encoding)

            # Count lines in written content
            line_count = len(normalized_lines)

            return {
                "success": True,
                "data": f"Successfully wrote {len(content)} characters ({line_count} lines) to {file_path}",
            }
        except McpError:
            raise
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"write_file_lines failed: {str(e)}"
                )
            )

    @mcp.tool
    async def read_file_for_qa(
        file_path: Annotated[str, Field(description="File path relative to the session workspace root, or absolute starting with '/'")],
        chunk_index: Annotated[int, Field(description="Chunk index (1-based, first chunk is 1)", ge=1)],
        start_line: Annotated[int, Field(description="Start line number (1-based) for the chunk. If provided, directly read from this line. If not provided, calculate chunk boundaries first.")] = None,
        end_line: Annotated[int, Field(description="End line number (1-based, exclusive) for the chunk. If provided with start_line, directly read this range.")] = None,
    ) -> dict:
        """
        Read file content in chunks for Q&A purposes.
        
        Each chunk contains approximately 96k characters, maintaining line integrity.
        First call: calculates chunk boundaries and returns them. Subsequent calls: use start_line/end_line for direct access.
        
        Returns:
        - content: The text content of the requested chunk
        - progress: Progress string (e.g., "1/5 chunks")
        - chunk_size: Character count of current chunk
        - chunk_boundaries: List of start line numbers (1-based) for each chunk (only on first call or when not provided)
        """
        try:
            # If start_line and end_line are provided, directly read the range asynchronously
            # Note: When using cached boundaries, we still need to validate chunk_index
            # to ensure it's within valid range. However, if boundaries are provided,
            # we trust the client and skip validation for performance.
            if start_line is not None and end_line is not None:
                full_path, chunk_lines, _ = await _validate_and_read_text_file(
                    file_path, 
                    start_line=start_line, 
                    end_line=end_line
                )
                # Join lines: each line from readlines() includes trailing '\n', so join preserves newlines between lines
                content = ''.join(chunk_lines)
                # Remove trailing newline if present (to avoid extra blank line in template, but preserve newlines between lines)
                if content.endswith('\n'):
                    content = content.rstrip('\n')
                
                # Try to get total_chunks from progress if available, otherwise use chunk_index as placeholder
                # When using cached boundaries, we don't have total_chunks info, so we use a placeholder
                return {
                    "success": True,
                    "data": {
                        "content": content,
                        "progress": f"{chunk_index} chunks",  # Placeholder when using cached boundaries
                        "chunk_size": len(content),
                    },
                }
            
            # Otherwise, calculate chunk boundaries (first call or when boundaries not provided)
            # Read entire file asynchronously to calculate boundaries
            full_path, lines, total_lines = await _validate_and_read_text_file(file_path)
            
            # Handle empty file
            if len(lines) == 0:
                return {
                    "success": True,
                    "data": {
                        "content": "",
                        "progress": "1/1 chunks",
                        "chunk_size": 0,
                        "chunk_boundaries": [1],
                    },
                }
            
            # Calculate chunk boundaries (maintaining line integrity)
            CHUNK_SIZE = 96 * 1024  # 96k characters
            chunk_boundaries = [0]  # Start of first chunk (line index 0)
            current_chunk_size = 0
            
            for i, line in enumerate(lines):
                line_size = len(line)
                if current_chunk_size + line_size > CHUNK_SIZE and current_chunk_size > 0:
                    chunk_boundaries.append(i)
                    current_chunk_size = line_size
                else:
                    current_chunk_size += line_size
            
            chunk_boundaries.append(len(lines))
            total_chunks = len(chunk_boundaries) - 1
            
            # Validate chunk_index
            if chunk_index < 1 or chunk_index > total_chunks:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"chunk_index {chunk_index} is out of range. Valid range: 1 to {total_chunks}"
                    )
                )
            
            # Extract the requested chunk
            chunk_idx = chunk_index - 1
            start_line_idx = chunk_boundaries[chunk_idx]
            end_line_idx = chunk_boundaries[chunk_idx + 1]
            
            chunk_lines = lines[start_line_idx:end_line_idx]
            # Join lines: each line from readlines() includes trailing '\n', so join preserves newlines between lines
            content = ''.join(chunk_lines)
            # Remove trailing newline if present (to avoid extra blank line in template, but preserve newlines between lines)
            if content.endswith('\n'):
                content = content.rstrip('\n')
            
            chunk_size = len(content)
            progress = f"{chunk_index}/{total_chunks} chunks"
            
            # Convert boundaries to 1-based line numbers for client
            chunk_boundaries_1based = [b + 1 for b in chunk_boundaries]

            return {
                "success": True,
                "data": {
                    "content": content,
                    "progress": progress,
                    "chunk_size": chunk_size,
                    "chunk_boundaries": chunk_boundaries_1based,  # Return boundaries for client to cache
                },
            }

        except McpError:
            raise
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"read_file_for_qa failed: {str(e)}"
                )
            )

    @mcp.tool
    async def read_image_for_qa(
        file_path: Annotated[str, Field(description="Image file path relative to the session workspace root, or absolute starting with '/'")],
    ) -> dict:
        """
        Read image file and return as base64 Data URI.
        
        Supported formats: JPEG, PNG, GIF, WebP.
        Returns the image in Data URI format suitable for embedding in Q&A contexts.
        
        Returns:
        - data_uri: Base64-encoded image in Data URI format (data:image/{mime_type};base64,{base64_string})
        - file_path: The file path
        - mime_type: Detected MIME type
        - size_bytes: File size in bytes
        """
        try:
            # Validate file path
            full_path = await resolve_workspace_path(file_path)
            
            if not full_path.exists():
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"Image file does not exist: {file_path}"
                    )
                )
            
            if full_path.is_dir():
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"Path is a directory, not a file: {file_path}"
                    )
                )

            # Determine MIME type from file extension
            image_ext = full_path.suffix.lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_types.get(image_ext, 'image/jpeg')

            # Read image file as binary asynchronously
            try:
                async with aiofiles.open(full_path, 'rb') as image_file:
                    image_data = await image_file.read()
                    size_bytes = len(image_data)
            except Exception as e:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Failed to read image file: {e}"
                    )
                )

            # Encode to base64
            base64_string = base64.b64encode(image_data).decode('utf-8')
            
            # Build Data URI
            data_uri = f"data:{mime_type};base64,{base64_string}"

            return {
                "success": True,
                "data": {
                    "data_uri": data_uri,
                    "file_path": file_path,
                    "mime_type": mime_type,
                    "size_bytes": size_bytes,
                },
            }

        except McpError:
            raise
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"read_image_for_qa failed: {str(e)}"
                )
            )

