"""
하드웨어 모니터링 registry.

벤더별 Collector를 lazy import로 등록하고 TargetSpec의 monitor_names에 따라
HWMonitor에 조립한다.
"""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, Optional

from .base import Collector, HWMonitor
from core.registry import normalize_registry_key, register_entry_keys


@dataclass(frozen=True)
class CollectorEntry:
    name: str
    module: str
    class_name: str
    aliases: tuple[str, ...] = ()
    default_options: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def load(self) -> type[Collector]:
        module = import_module(self.module)
        return getattr(module, self.class_name)


_COLLECTOR_REGISTRY: Dict[str, CollectorEntry] = {}


def register_collector(entry: CollectorEntry) -> None:
    register_entry_keys(_COLLECTOR_REGISTRY, entry, (entry.name, *entry.aliases), "collector")


def get_collector_entry(name: str) -> CollectorEntry:
    key = normalize_registry_key(name, "collector")
    entry = _COLLECTOR_REGISTRY.get(key)
    if entry is None:
        supported = sorted(_COLLECTOR_REGISTRY.keys())
        raise ValueError(f"지원하지 않는 collector입니다: {name}. 지원 목록: {supported}")
    return entry


def list_collectors() -> list[dict]:
    seen: set[str] = set()
    result = []
    for entry in _COLLECTOR_REGISTRY.values():
        if entry.name in seen:
            continue
        seen.add(entry.name)
        result.append({
            "name": entry.name,
            "aliases": list(entry.aliases),
            "description": entry.description,
        })
    return result


def create_collector(name: str, **options) -> Collector:
    entry = get_collector_entry(name)

    merged_options = {**entry.default_options, **options}
    try:
        collector_cls = entry.load()
    except Exception as exc:
        raise RuntimeError(f"모니터 플러그인 '{name}' 로드 실패: {exc}") from exc
    return collector_cls(**merged_options)


def create_hw_monitor(
    interval: float = 0.2,
    gpu_index: int = 0,
    device: str = "cpu",
    collector_names: Optional[list[str]] = None,
    collector_options: Optional[dict[str, dict[str, Any]]] = None,
) -> Optional[HWMonitor]:
    """
    TargetSpec 기반 또는 legacy device 기반으로 HWMonitor를 생성한다.

    Args:
        collector_names: 지정되면 해당 collector들을 registry에서 생성한다.
                         None이면 기존 cpu/cuda 자동 선택 동작을 사용한다.
    """
    monitor = HWMonitor(interval=interval)
    collector_options = collector_options or {}

    if collector_names is None:
        collector_names = ["system"]
        if device.startswith("cuda"):
            collector_names.insert(0, "nvidia")

    for name in collector_names:
        options = dict(collector_options.get(name, {}))
        if name == "nvidia" and "gpu_index" not in options:
            options["gpu_index"] = gpu_index
        try:
            collector = create_collector(name, **options)
            if collector.is_available():
                if hasattr(collector, "init_nvml"):
                    collector.init_nvml()
                monitor.add_collector(collector)
                print(f"[HWMonitor] collector enabled: {name}")
            else:
                print(f"[HWMonitor] collector unavailable: {name}")
        except ImportError:
            print(f"[HWMonitor] collector dependency missing: {name}")
        except Exception as exc:
            print(f"[HWMonitor] collector skipped ({name}): {exc}")

    return monitor


register_collector(CollectorEntry(
    name="system",
    module="monitors.system_collector",
    class_name="SystemCollector",
    description="CPU/RAM process and system metrics",
))

register_collector(CollectorEntry(
    name="nvidia",
    module="monitors.nvidia_collector",
    class_name="NvidiaCollector",
    aliases=("gpu",),
    description="NVIDIA NVML GPU metrics",
))

register_collector(CollectorEntry(
    name="mock_npu",
    module="monitors.mock_npu_collector",
    class_name="MockNpuCollector",
    aliases=("vendor_mock_npu",),
    description="SDK-free NPU metrics used to validate monitor wiring",
))

register_collector(CollectorEntry(
    name="hailo",
    module="monitors.hailo_collector",
    class_name="HailoCollector",
    aliases=("hailo8", "hailort"),
    description="HailoRT temperature and power telemetry collector",
))

register_collector(CollectorEntry(
    name="deepx",
    module="monitors.deepx_collector",
    class_name="DeepXCollector",
    aliases=("dxrt", "deepx_npu"),
    description="DEEPX DX-RT DeviceStatus telemetry collector",
))


__all__ = [
    "Collector",
    "HWMonitor",
    "CollectorEntry",
    "register_collector",
    "get_collector_entry",
    "list_collectors",
    "create_collector",
    "create_hw_monitor",
]
