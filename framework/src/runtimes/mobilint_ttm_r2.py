"""TTM-R2 semantic-to-artifact tensor boundary for Mobilint ARIES."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ttm_r1.mobilint_aries import quantize_core_input, restore_artifact_output


MOBILINT_TTM_R2_ADAPTER_ID = "ttm-r2-aries-v1"
_ARTIFACT_INPUT_SHAPE = (1, 8, 64)
_ARTIFACT_OUTPUT_SHAPE = (1, 1, 96)


class MobilintTTMR2Adapter:
    """Adapt the fixed FP32 semantic ABI to one validated ARIES MXQ ABI."""

    def __init__(self) -> None:
        self._model = None
        self._input_scale = None
        self._artifact_input_shape: tuple[int, ...] | None = None
        self._artifact_output_shape: tuple[int, ...] | None = None
        self._input_scale_mode: str | None = None
        self._input_zero_point: int | None = None
        self.reset_measurement()

    def bind(self, model: Any) -> None:
        input_shapes = self._call_metadata(model, "get_model_input_shape")
        output_shapes = self._call_metadata(model, "get_model_output_shape")
        if not isinstance(input_shapes, (list, tuple)) or len(input_shapes) != 1:
            raise ValueError("TTM-R2 ARIES artifact must expose exactly one input")
        if not isinstance(output_shapes, (list, tuple)) or len(output_shapes) != 1:
            raise ValueError("TTM-R2 ARIES artifact must expose exactly one output")
        input_shape = tuple(input_shapes[0])
        output_shape = tuple(output_shapes[0])
        if input_shape != _ARTIFACT_INPUT_SHAPE:
            raise ValueError(
                "TTM-R2 ARIES artifact input shape must be [1,8,64], got "
                f"{input_shape}"
            )
        if output_shape != _ARTIFACT_OUTPUT_SHAPE:
            raise ValueError(
                "TTM-R2 ARIES artifact output shape must be [1,1,96], got "
                f"{output_shape}"
            )

        raw_dtype = self._call_metadata(model, "get_model_input_data_type")
        if isinstance(raw_dtype, (list, tuple)):
            if len(raw_dtype) != 1:
                raise ValueError(
                    "TTM-R2 ARIES artifact must expose one input dtype"
                )
            raw_dtype = raw_dtype[0]
        dtype_name = getattr(raw_dtype, "name", raw_dtype)
        try:
            normalized_dtype = np.dtype(dtype_name).name
        except (TypeError, ValueError):
            try:
                normalized_dtype = np.dtype(str(dtype_name).lower()).name
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "TTM-R2 ARIES artifact input dtype is invalid: "
                    f"{raw_dtype!r}"
                ) from exc
        if normalized_dtype != "int8":
            raise ValueError(
                "TTM-R2 ARIES artifact input dtype must be int8, got "
                f"{normalized_dtype}"
            )

        scales = self._call_metadata(model, "get_input_scale")
        if not isinstance(scales, (list, tuple)) or len(scales) != 1:
            raise ValueError(
                "TTM-R2 ARIES artifact must expose exactly one input scale"
            )
        scale = scales[0]
        zero_point = getattr(scale, "zero_point", None)
        if bool(getattr(scale, "is_asymmetric", False)):
            raise ValueError("TTM-R2 ARIES input scale must be symmetric")
        if zero_point != 0:
            raise ValueError(
                "TTM-R2 ARIES input scale must use zero point 0"
            )
        if bool(getattr(scale, "is_uniform", False)):
            scale_values = np.asarray([getattr(scale, "scale", None)], dtype=np.float64)
            scale_mode = "uniform"
        else:
            scale_values = np.asarray(
                getattr(scale, "scale_list", ()), dtype=np.float64
            )
            if scale_values.shape != (64,):
                raise ValueError(
                    "TTM-R2 ARIES non-uniform input scale must have 64 values"
                )
            scale_mode = "per_last_axis"
        if not bool(np.isfinite(scale_values).all()) or bool(
            (scale_values <= 0.0).any()
        ):
            raise ValueError("TTM-R2 ARIES input scale values must be finite and positive")

        self._model = model
        self._input_scale = scale
        self._artifact_input_shape = input_shape
        self._artifact_output_shape = output_shape
        self._input_scale_mode = scale_mode
        self._input_zero_point = int(zero_point)
        self.reset_measurement()

    def run(
        self,
        model: Any,
        inputs: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if self._model is None or self._input_scale is None:
            raise RuntimeError("TTM-R2 ARIES adapter is not bound")
        if model is not self._model:
            raise RuntimeError("TTM-R2 ARIES adapter is bound to another model")
        if set(inputs) != {"past_values"}:
            raise ValueError(
                "TTM-R2 ARIES semantic inputs must contain only 'past_values'"
            )
        semantic = np.asarray(inputs["past_values"])
        if semantic.dtype != np.float32 or semantic.shape != (1, 512, 1):
            raise ValueError(
                "TTM-R2 ARIES semantic input must be FP32 [1,512,1]"
            )
        if not bool(np.isfinite(semantic).all()):
            raise ValueError("TTM-R2 ARIES semantic input must be finite")

        quantized, saturated = quantize_core_input(
            semantic,
            self._artifact_input_shape,
            self._input_scale,
        )
        outputs = model.infer_to_float([quantized])
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise RuntimeError(
                "TTM-R2 ARIES infer_to_float must return exactly one output"
            )
        forecast = restore_artifact_output(
            outputs[0], self._artifact_output_shape
        )
        self._saturation_elements += saturated
        self._quantized_elements += int(quantized.size)
        return {"forecast": np.ascontiguousarray(forecast, dtype=np.float32)}

    def reset_measurement(self) -> None:
        self._saturation_elements = 0
        self._quantized_elements = 0

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "tensor_boundary_adapter_id": MOBILINT_TTM_R2_ADAPTER_ID,
            "mobilint_artifact_input_shape": self._artifact_input_shape,
            "mobilint_artifact_output_shape": self._artifact_output_shape,
            "mobilint_quantization_status": (
                "saturated" if self._saturation_elements else "unsaturated"
            ),
            "mobilint_saturation_elements": self._saturation_elements,
            "mobilint_saturation_total": self._quantized_elements,
            "mobilint_input_scale_mode": self._input_scale_mode,
            "mobilint_input_zero_point": self._input_zero_point,
        }

    def dispose(self) -> None:
        self._model = None
        self._input_scale = None
        self._artifact_input_shape = None
        self._artifact_output_shape = None
        self._input_scale_mode = None
        self._input_zero_point = None
        self.reset_measurement()

    @staticmethod
    def _call_metadata(model: Any, name: str) -> Any:
        getter = getattr(model, name, None)
        if not callable(getter):
            raise ValueError(
                f"TTM-R2 ARIES artifact requires callable {name}()"
            )
        try:
            return getter()
        except BaseException as exc:
            raise ValueError(
                f"TTM-R2 ARIES artifact metadata call {name}() failed"
            ) from exc


def create_mobilint_ttm_r2_adapter(
    adapter_id: Any,
) -> MobilintTTMR2Adapter | None:
    if adapter_id is None:
        return None
    if adapter_id != MOBILINT_TTM_R2_ADAPTER_ID:
        raise ValueError(
            "tensor_boundary_adapter_id must be "
            f"{MOBILINT_TTM_R2_ADAPTER_ID!r}, got {adapter_id!r}"
        )
    return MobilintTTMR2Adapter()
