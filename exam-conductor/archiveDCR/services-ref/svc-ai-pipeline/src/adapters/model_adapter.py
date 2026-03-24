"""ONNX model loading and inference adapter.

Provides a ModelRegistry that tracks loaded models, their versions,
and exposes inference callables consumed by the domain layer.

In mock mode, returns canned results without ONNX runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Canned mock responses for tests and mock mode.
_MOCK_HWR_RESULT: dict[str, Any] = {
    "chars": list("Hello"),
    "confidences": [0.95, 0.92, 0.88, 0.91, 0.87],
}

_MOCK_CLASSIFIER_FEATURES: dict[str, Any] = {
    "has_math_symbols": False,
    "has_drawn_shapes": False,
    "text_line_count": 3,
    "symbol_ratio": 0.02,
    "stroke_density": 0.1,
}


class ModelInfo:
    """Metadata for a loaded model."""

    def __init__(self, name: str, version: str, path: str) -> None:
        self.name = name
        self.version = version
        self.path = path
        self.session: Any = None  # onnxruntime.InferenceSession when loaded


class ModelRegistry:
    """Manages ONNX model loading and provides inference callables."""

    def __init__(self, model_dir: str, mock: bool = False) -> None:
        self._model_dir = Path(model_dir)
        self._mock = mock
        self._models: dict[str, ModelInfo] = {}
        self._version: str = "mock-v1" if mock else "unknown"

    def load_all(self) -> None:
        """Scan MODEL_DIR for .onnx files and load them."""
        if self._mock:
            self._models["hwr"] = ModelInfo("hwr", "mock-v1", "mock")
            self._models["classifier"] = ModelInfo(
                "classifier", "mock-v1", "mock",
            )
            self._version = "mock-v1"
            logger.info("Mock mode: registered canned models")
            return

        if not self._model_dir.exists():
            logger.warning("Model directory %s does not exist", self._model_dir)
            return

        for onnx_file in self._model_dir.glob("*.onnx"):
            name = onnx_file.stem
            version = self._read_version(onnx_file)
            info = ModelInfo(name, version, str(onnx_file))
            self._load_onnx_session(info)
            self._models[name] = info
            logger.info("Loaded model %s version=%s", name, version)

        if self._models:
            self._version = next(iter(self._models.values())).version

    def _read_version(self, path: Path) -> str:
        """Read model version from a sidecar .version file if present."""
        version_file = path.with_suffix(".version")
        if version_file.exists():
            return version_file.read_text().strip()
        return path.stem

    def _load_onnx_session(self, info: ModelInfo) -> None:
        """Load an ONNX InferenceSession. Imported lazily."""
        try:
            import onnxruntime as ort
            info.session = ort.InferenceSession(info.path)
        except Exception:
            logger.exception("Failed to load ONNX model %s", info.name)

    def current_version(self) -> str:
        """Return the version string of the currently loaded models."""
        return self._version

    def list_models(self) -> list[dict[str, str]]:
        """Return metadata for all loaded models."""
        return [
            {"name": m.name, "version": m.version}
            for m in self._models.values()
        ]

    def get_inference_fn(self, model_name: str) -> Callable:
        """Return a callable that runs inference on the named model.

        For tests / mock mode: returns canned results.
        For real mode: wraps the ONNX session.
        """
        if self._mock or model_name not in self._models:
            return _mock_inference_fn(model_name)

        info = self._models[model_name]
        if info.session is None:
            return _mock_inference_fn(model_name)

        def _run(input_data: bytes) -> Any:
            import numpy as np
            # Real inference — adapt input shape to model requirements
            arr = np.frombuffer(input_data, dtype=np.uint8)
            input_name = info.session.get_inputs()[0].name
            result = info.session.run(None, {input_name: arr})
            return result[0]

        return _run

    def run_inference(
        self, model_name: str, input_data: bytes,
    ) -> Any:
        """Convenience: run inference directly."""
        fn = self.get_inference_fn(model_name)
        return fn(input_data)


def _mock_inference_fn(model_name: str) -> Callable:
    """Return a canned inference function for mock/test mode."""
    if model_name == "hwr":
        def _hwr_mock(_input: bytes) -> dict:
            return dict(_MOCK_HWR_RESULT)
        return _hwr_mock

    if model_name == "classifier":
        def _cls_mock(_input: bytes) -> dict:
            return dict(_MOCK_CLASSIFIER_FEATURES)
        return _cls_mock

    def _noop(_input: bytes) -> dict:
        return {}
    return _noop
