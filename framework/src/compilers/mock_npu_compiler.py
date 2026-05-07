from pathlib import Path
from typing import Any, Dict

from .base import Compiler, CompileResult
from core.model_spec import Model_Spec


class MockNpuCompiler(Compiler):
    """
    SDK 없이 NPU plugin pipeline을 검증하기 위한 compiler.

    실제 벤더 compiler adapter는 이 클래스처럼 Compiler 계약만 만족하면
    target registry에 연결될 수 있다.
    """

    def __init__(self, **compile_options):
        super().__init__(**compile_options)
        self.vendor = self.compile_options.get("vendor", "MockNPU")
        self.artifact_format = self.compile_options.get("artifact_format", "mockbin")

    def get_artifact_name(self, model_spec: Model_Spec) -> str:
        return f"{model_spec.name}_mock_npu.{self.artifact_format}"

    def compile(self, model_spec: Model_Spec, output_dir: str) -> CompileResult:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        artifact_path = output_path / self.get_artifact_name(model_spec)
        cache_hit = artifact_path.exists()

        if not cache_hit:
            source_path = model_spec.model_paths.get("onnx", "")
            artifact_path.write_text(
                "\n".join([
                    "mock_npu_artifact=1",
                    f"model={model_spec.name}",
                    f"source={source_path}",
                    f"vendor={self.vendor}",
                ]),
                encoding="utf-8",
            )

        metadata: Dict[str, Any] = {
            "compiler_name": "mock_npu",
            "compiler_version": "0.1",
            "vendor": self.vendor,
            "artifact_format": self.artifact_format,
            "cache_hit": cache_hit,
            "compile_options": self.compile_options,
        }
        return CompileResult(artifact_path=str(artifact_path), metadata=metadata)
