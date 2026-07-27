from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import compile_furiosa_resnet50 as tool


class _FakeTensor:
    def __init__(self, value, *, device="cpu"):
        self._value = np.asarray(value, dtype=np.float32)
        self.device = device

    @property
    def shape(self):
        return self._value.shape

    def to(self, device):
        return _FakeTensor(self._value.copy(), device=str(device))

    def detach(self):
        return self

    def cpu(self):
        return _FakeTensor(self._value.copy(), device="cpu")

    def float(self):
        return self

    def numpy(self):
        return self._value.copy()


class _FakeModel:
    def __init__(self, state):
        self._state = state
        self.state_id = state.model_state_id
        self.device = "cpu"

    def __deepcopy__(self, memo):
        del memo
        self._state.deepcopied_state_ids.append(self.state_id)
        return _FakeModel(self._state)

    def eval(self):
        return self

    def to(self, device):
        self.device = str(device)
        return self

    def __call__(self, input_tensor):
        self._state.model_calls.append((self.device, input_tensor.device))
        if self.device == "cpu":
            return _FakeTensor(self._state.cpu_logits)
        return _FakeTensor(self._state.npu_logits, device=self.device)


class _FakeTorch:
    float32 = "float32"

    def __init__(self, state):
        self._state = state

    def manual_seed(self, seed):
        self._state.seeds.append(seed)

    def randn(self, *shape, dtype):
        self._state.randn_calls.append((shape, dtype))
        return _FakeTensor(np.zeros(shape, dtype=np.float32))

    def inference_mode(self):
        return nullcontext()

    def device(self, value):
        return value

    def compile(self, model, **kwargs):
        self._state.compile_calls.append({"model": model, **kwargs})
        if self._state.compile_error is not None:
            raise self._state.compile_error

        def compiled(input_tensor):
            self._state.compiled_calls += 1
            return model(input_tensor)

        return compiled


class _FakeBackend:
    def __init__(self, state):
        self._state = state

    def with_config(self, compiler_config, **kwargs):
        call = {"compiler_config": compiler_config, **kwargs}
        self._state.backend_calls.append(call)
        return "strict-furiosa-backend"


def _fake_dependencies():
    cpu_logits = np.zeros((1, 1000), dtype=np.float32)
    cpu_logits[0, 7] = 3.0
    state = SimpleNamespace(
        imagenet_v2_weights=object(),
        model_state_id=object(),
        cpu_logits=cpu_logits,
        npu_logits=cpu_logits.copy(),
        seeds=[],
        randn_calls=[],
        resnet_calls=[],
        model_calls=[],
        deepcopied_state_ids=[],
        backend_calls=[],
        compile_calls=[],
        compiled_calls=0,
        compile_error=None,
    )

    def resnet50(*, weights):
        state.resnet_calls.append(weights)
        return _FakeModel(state)

    class CompilerConfig:
        def __init__(self, *, tactic_hint):
            self.tactic_hint = tactic_hint

    class TacticHintConfig:
        Default = "default-tactic"

    state.torch = _FakeTorch(state)
    state.furiosa_torch = SimpleNamespace(backend=_FakeBackend(state))
    state.CompilerConfig = CompilerConfig
    state.TacticHintConfig = TacticHintConfig
    state.resnet50 = resnet50
    return state


def test_parser_defaults_to_imagenet_v2_on_furiosa_zero():
    args = tool.build_parser().parse_args([])

    assert args.device == "furiosa:0"
    assert args.seed == 0
    assert args.torch_home is None


def test_compile_check_uses_imagenet_v2_and_strict_backend():
    dependencies = _fake_dependencies()

    result = tool.run_compile_check(
        tool.CompileCheckConfig(),
        dependencies=dependencies,
        timer=iter((10.0, 14.0, 20.0, 20.5)).__next__,
        emit=lambda message: None,
    )

    assert dependencies.resnet_calls == [dependencies.imagenet_v2_weights]
    assert dependencies.seeds == [0]
    assert dependencies.randn_calls == [(((1, 3, 224, 224)), "float32")]
    assert dependencies.deepcopied_state_ids == [dependencies.model_state_id]
    assert dependencies.backend_calls[0]["eager_fallback"] is False
    assert dependencies.backend_calls[0]["compiler_config"].tactic_hint == (
        "default-tactic"
    )
    assert dependencies.compile_calls[0]["backend"] == "strict-furiosa-backend"
    assert dependencies.compile_calls[0]["fullgraph"] is True
    assert dependencies.compile_calls[0]["dynamic"] is False
    assert dependencies.compiled_calls == 2
    assert result.first_call_seconds == 4.0
    assert result.warm_call_seconds == 0.5
    assert result.cpu_top1 == result.npu_top1 == 7
    assert result.output_shape == (1, 1000)
    assert result.max_abs_diff == 0.0


def test_torch_home_is_an_optional_path():
    args = tool.build_parser().parse_args(["--torch-home", "/tmp/torch-cache"])

    assert args.torch_home == Path("/tmp/torch-cache")


def test_compile_check_rejects_top1_mismatch():
    dependencies = _fake_dependencies()
    dependencies.npu_logits[0, 7] = 0.0
    dependencies.npu_logits[0, 9] = 4.0

    with pytest.raises(RuntimeError, match="Top-1 mismatch"):
        tool.run_compile_check(
            tool.CompileCheckConfig(),
            dependencies=dependencies,
            emit=lambda message: None,
        )


def test_compile_check_propagates_compiler_failure():
    dependencies = _fake_dependencies()
    dependencies.compile_error = RuntimeError("compiler panic")

    with pytest.raises(RuntimeError, match="compiler panic"):
        tool.run_compile_check(
            tool.CompileCheckConfig(),
            dependencies=dependencies,
            emit=lambda message: None,
        )


@pytest.mark.parametrize(
    "bad_shape",
    [
        (1, 999),
        (1000,),
        (2, 1000),
    ],
)
def test_compile_check_rejects_static_output_shape_mismatch(bad_shape):
    dependencies = _fake_dependencies()
    dependencies.npu_logits = np.zeros(bad_shape, dtype=np.float32)

    with pytest.raises(RuntimeError, match="output shape mismatch"):
        tool.run_compile_check(
            tool.CompileCheckConfig(),
            dependencies=dependencies,
            emit=lambda message: None,
        )


def test_compile_check_rejects_non_finite_output():
    dependencies = _fake_dependencies()
    dependencies.npu_logits[0, 5] = np.nan

    with pytest.raises(RuntimeError, match="non-finite"):
        tool.run_compile_check(
            tool.CompileCheckConfig(),
            dependencies=dependencies,
            emit=lambda message: None,
        )


def test_compile_check_sets_torch_home(monkeypatch, tmp_path):
    dependencies = _fake_dependencies()
    cache_path = tmp_path / "torch-cache"
    monkeypatch.delenv("TORCH_HOME", raising=False)

    tool.run_compile_check(
        tool.CompileCheckConfig(torch_home=cache_path),
        dependencies=dependencies,
        emit=lambda message: None,
    )

    assert tool.os.environ["TORCH_HOME"] == str(cache_path.resolve())
