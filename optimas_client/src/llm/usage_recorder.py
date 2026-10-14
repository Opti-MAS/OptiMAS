"""API usage recorder — simple yet robust implementation."""
import os
import json
import secrets
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class APIUsageRecord:
    """API usage record"""
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    success: bool
    error_message: str
    timestamp: str
    base_url: Optional[str]
    inputs: List[Dict[str, Any]]  # Full input messages
    outputs: List[Dict[str, Any]]  # Full output responses
    metadata: Dict[str, Any]  # Additional metadata (finish_reason, model_used, etc.)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, handling non-serializable fields."""
        def _sanitize(value: Any, depth: int = 0) -> Any:
            """Recursively sanitize values to ensure JSON serializability."""
            if depth > 6:  # Prevent infinite recursion
                return str(value)
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            if isinstance(value, dict):
                return {str(k): _sanitize(v, depth + 1) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_sanitize(v, depth + 1) for v in value]
            try:
                json.dumps(value)
                return value
            except (TypeError, ValueError):
                return str(value)
        
        record_dict = asdict(self)
        return _sanitize(record_dict)


class APIUsageRecorder:
    """API usage recorder — simple yet robust implementation."""
    
    def __init__(self, storage_path: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the recorder.
        
        Args:
            storage_path: Storage directory; defaults to ``logs/api_usage`` under the project root.
            model_name: Model name, used for generating the log file name.
        """
        if storage_path:
            storage_dir = Path(storage_path).expanduser().resolve()
        else:
            proj_root = Path(__file__).resolve().parents[3]
            storage_dir = proj_root / "logs" / "api_usage"
        
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_path = storage_dir
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        hex_id_16 = secrets.token_hex(8)
        self.hex_id_16 = hex_id_16
        
        if model_name:
            safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
            safe_model_name = safe_model_name.replace("*", "_").replace("?", "_").replace('"', "_")
            safe_model_name = safe_model_name.replace("<", "_").replace(">", "_").replace("|", "_")
            if len(safe_model_name) > 50:
                safe_model_name = safe_model_name[:50]
            self.file_name = f"api_usage_{current_time}_{safe_model_name}_{hex_id_16}.jsonl"
        else:
            self.file_name = f"api_usage_{current_time}_{hex_id_16}.jsonl"
        
        self.file_path = storage_dir / self.file_name
        
        self._lock = threading.Lock()
        self._file_handle = None
        self._flushed_on_exit = False
        
    
    def _safe_flush(self):
        """Flush and close the file handle, catching all exceptions."""
        if self._flushed_on_exit:
            return
        try:
            if self._file_handle is not None:
                try:
                    self._file_handle.flush()
                    os.fsync(self._file_handle.fileno())
                    self._file_handle.close()
                except Exception:
                    pass
                finally:
                    self._file_handle = None
            self._flushed_on_exit = True
        except Exception:
            pass
    
    def record(self, record: APIUsageRecord) -> None:
        """
        Record a single API call (writes immediately and syncs to disk).
        
        Args:
            record: The API usage record to persist.
        """
        with self._lock:
            try:
                record_dict = record.to_dict()
                line = json.dumps(record_dict, ensure_ascii=False) + "\n"
                
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    f.write(line)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except (OSError, AttributeError):
                        pass
                        
            except Exception:
                try:
                    fallback_path = self.file_path.with_name(
                        self.file_path.stem + "_fallback" + self.file_path.suffix
                    )
                    with open(fallback_path, 'a', encoding='utf-8') as f:
                        record_dict = record.to_dict()
                        line = json.dumps(record_dict, ensure_ascii=False) + "\n"
                        f.write(line)
                        f.flush()
                except Exception:
                    pass
    
    def flush(self) -> None:
        """Explicitly flush pending data."""
        self._safe_flush()
    
    def __del__(self):
        """Last-resort flush on garbage collection."""
        try:
            if not self._flushed_on_exit:
                self._safe_flush()
        except Exception:
            pass
