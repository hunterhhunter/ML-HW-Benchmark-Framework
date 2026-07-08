import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from .base import Compiler, CompileResult
from core.model_spec import Model_Spec


class DeepXCompiler(Compiler):
    """DX-COM CLI adapter that compiles ONNX models into DEEPX .dxnn artifacts."""

    _ALLOWED_OPTIONS = {
        "config_path",
        "dxcom_bin",
        "opt_level",
        "aggressive_partitioning",
        "gen_log",
        "float64_calibration",
        "compile_input_nodes",
        "compile_output_nodes",
    }

    def __init__(self, **compile_options):
        super().__init__(**compile_options)
        unknown = sorted(set(self.compile_options) - self._ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(f"Unsupported DeepX compiler option(s): {unknown}")

        config_path = self.compile_options.get("config_path")
        if not config_path:
            raise ValueError("DeepX compiler requires compile option 'config_path=/path/to/config.json'.")

        self.config_path = Path(str(config_path)).expanduser()
        if not self.config_path.exists():
            raise FileNotFoundError(f"DeepX compiler config_path does not exist: {self.config_path}")
        if not self.config_path.is_file():
            raise ValueError(f"DeepX compiler config_path must be a file: {self.config_path}")

        self.dxcom_bin = str(self.compile_options.get("dxcom_bin", "dxcom"))

    def get_artifact_name(self, model_spec: Model_Spec) -> str:
        cache_key = self._cache_key(model_spec)
        return f"{model_spec.name}_deepx_{cache_key[:12]}.dxnn"

    def compile(self, model_spec: Model_Spec, output_dir: str) -> CompileResult:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        source_model = self._source_onnx_path(model_spec)
        expected_artifact = output_path / self.get_artifact_name(model_spec)
        cache_key = self._cache_key(model_spec)
        work_dir = output_path / f"{model_spec.name}_deepx_{cache_key[:12]}_work"

        if expected_artifact.exists():
            return CompileResult(
                artifact_path=str(expected_artifact),
                metadata=self._metadata(
                    cache_hit=True,
                    work_dir=work_dir,
                    command=None,
                    compiler_version=None,
                ),
            )

        dxcom = self._resolve_dxcom()
        command = self._build_command(dxcom, source_model, work_dir)
        compiler_version = self._compiler_version(dxcom)

        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            message = [
                "DX-COM compilation failed.",
                f"command: {' '.join(command)}",
            ]
            if exc.stdout:
                message.append(f"stdout:\n{exc.stdout}")
            if exc.stderr:
                message.append(f"stderr:\n{exc.stderr}")
            raise RuntimeError("\n".join(message)) from exc

        generated_artifact = self._discover_generated_artifact(work_dir)
        if generated_artifact != expected_artifact:
            shutil.copy2(generated_artifact, expected_artifact)

        return CompileResult(
            artifact_path=str(expected_artifact),
            metadata=self._metadata(
                cache_hit=False,
                work_dir=work_dir,
                command=command,
                compiler_version=compiler_version,
            ),
        )

    def _source_onnx_path(self, model_spec: Model_Spec) -> Path:
        source = model_spec.model_paths.get("onnx")
        if not source:
            raise ValueError("DeepX compiler requires an ONNX model path in Model_Spec.model_paths['onnx'].")
        source_path = Path(source).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"ONNX model file does not exist: {source_path}")
        if not source_path.is_file():
            raise ValueError(f"ONNX model path must be a file: {source_path}")
        return source_path

    def _resolve_dxcom(self) -> str:
        has_path_separator = os.path.sep in self.dxcom_bin or (
            os.path.altsep is not None and os.path.altsep in self.dxcom_bin
        )
        if has_path_separator:
            candidate = Path(self.dxcom_bin).expanduser()
            if candidate.exists() and candidate.is_file():
                return str(candidate)
            raise FileNotFoundError(f"DX-COM executable not found: {candidate}")

        resolved = shutil.which(self.dxcom_bin)
        if resolved:
            return resolved
        raise FileNotFoundError(
            f"DX-COM executable '{self.dxcom_bin}' was not found. "
            "Install the DX-COM wheel and ensure dxcom is on PATH, or pass dxcom_bin=/path/to/dxcom."
        )

    def _build_command(self, dxcom: str, source_model: Path, work_dir: Path) -> list[str]:
        command = [
            dxcom,
            "-m",
            str(source_model),
            "-c",
            str(self.config_path),
            "-o",
            str(work_dir),
        ]

        opt_level = self.compile_options.get("opt_level")
        if opt_level not in (None, ""):
            opt_level_int = int(opt_level)
            if opt_level_int not in (0, 1):
                raise ValueError("DeepX opt_level must be 0 or 1.")
            command.extend(["--opt_level", str(opt_level_int)])

        if self._bool_option("aggressive_partitioning"):
            command.append("--aggressive_partitioning")
        if self._bool_option("gen_log"):
            command.append("--gen_log")
        if self._bool_option("float64_calibration"):
            command.append("--float64_calibration")

        input_nodes = self._node_option("compile_input_nodes")
        if input_nodes:
            command.extend(["--compile_input_nodes", input_nodes])
        output_nodes = self._node_option("compile_output_nodes")
        if output_nodes:
            command.extend(["--compile_output_nodes", output_nodes])

        return command

    def _discover_generated_artifact(self, work_dir: Path) -> Path:
        artifacts = sorted(path for path in work_dir.rglob("*.dxnn") if path.is_file())
        if not artifacts:
            raise RuntimeError(f"DX-COM completed but produced no .dxnn artifact in {work_dir}.")
        if len(artifacts) > 1:
            artifact_list = ", ".join(str(path) for path in artifacts)
            raise RuntimeError(f"DX-COM produced multiple .dxnn artifacts; cannot choose one: {artifact_list}")
        return artifacts[0]

    def _compiler_version(self, dxcom: str) -> str | None:
        try:
            result = subprocess.run(
                [dxcom, "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return None
        output = (result.stdout or result.stderr).strip()
        return output.splitlines()[0] if output else None

    def _metadata(
        self,
        *,
        cache_hit: bool,
        work_dir: Path,
        command: list[str] | None,
        compiler_version: str | None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "compiler_name": "deepx",
            "compiler_version": compiler_version,
            "vendor": "DEEPX",
            "artifact_format": "dxnn",
            "cache_hit": cache_hit,
            "config_path": str(self.config_path),
            "output_dir": str(work_dir),
            "compile_options": self._effective_options(),
        }
        if command is not None:
            metadata["compiler_command"] = command
        return metadata

    def _effective_options(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.compile_options.items()
            if key != "dxcom_bin"
        }

    def _cache_key(self, model_spec: Model_Spec) -> str:
        source_model = self._source_onnx_path(model_spec)
        digest = hashlib.sha256()
        digest.update(str(source_model.resolve()).encode("utf-8"))
        digest.update(self._file_digest(source_model).encode("ascii"))
        digest.update(str(self.config_path.resolve()).encode("utf-8"))
        digest.update(self._file_digest(self.config_path).encode("ascii"))
        for key, value in sorted(self._effective_options().items()):
            digest.update(str(key).encode("utf-8"))
            digest.update(str(value).encode("utf-8"))
        return digest.hexdigest()

    def _file_digest(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _bool_option(self, key: str) -> bool:
        value = self.compile_options.get(key)
        if value in (None, ""):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
        return bool(value)

    def _node_option(self, key: str) -> str:
        value = self.compile_options.get(key)
        if value in (None, ""):
            return ""
        if isinstance(value, (list, tuple)):
            return ",".join(str(item).strip() for item in value if str(item).strip())
        return str(value)
