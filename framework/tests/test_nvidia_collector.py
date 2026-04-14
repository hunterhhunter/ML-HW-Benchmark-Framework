"""NvidiaCollector 단위 테스트. pynvml을 mock하여 GPU 없이 테스트."""

from unittest.mock import patch, MagicMock
import types

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# pynvml이 설치되지 않은 환경에서도 테스트 가능하도록 mock 모듈 생성
@pytest.fixture
def mock_pynvml():
    """pynvml 모듈을 mock으로 대체."""
    mock_module = MagicMock()
    mock_module.NVMLError = type('NVMLError', (Exception,), {})
    mock_module.NVML_TEMPERATURE_GPU = 0
    mock_module.NVML_CLOCK_SM = 0
    mock_module.NVML_CLOCK_MEM = 1
    return mock_module


@pytest.fixture
def nvidia_collector(mock_pynvml):
    """NvidiaCollector 인스턴스를 mock pynvml과 함께 생성."""
    with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
        # 모듈을 다시 임포트하여 mock이 적용되도록
        if 'monitors.nvidia_collector' in sys.modules:
            del sys.modules['monitors.nvidia_collector']
        from monitors.nvidia_collector import NvidiaCollector
        collector = NvidiaCollector(gpu_index=0)
        collector._pynvml = mock_pynvml
        return collector, mock_pynvml


class TestNvidiaCollectorAvailability:
    def test_is_available_with_gpu(self, nvidia_collector):
        collector, mock_pynvml = nvidia_collector
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlShutdown.return_value = None

        with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
            if 'monitors.nvidia_collector' in sys.modules:
                del sys.modules['monitors.nvidia_collector']
            from monitors.nvidia_collector import NvidiaCollector
            c = NvidiaCollector()
            assert c.is_available() is True

    def test_is_available_without_gpu(self, nvidia_collector):
        collector, mock_pynvml = nvidia_collector
        mock_pynvml.nvmlInit.side_effect = Exception("No NVIDIA GPU")

        with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
            if 'monitors.nvidia_collector' in sys.modules:
                del sys.modules['monitors.nvidia_collector']
            from monitors.nvidia_collector import NvidiaCollector
            c = NvidiaCollector()
            assert c.is_available() is False


class TestNvidiaCollectorCollect:
    def test_collect_all_metrics(self, nvidia_collector):
        collector, mock_pynvml = nvidia_collector

        # Mock NVML responses
        util_mock = MagicMock()
        util_mock.gpu = 75
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util_mock

        mem_mock = MagicMock()
        mem_mock.used = 2 * 1024 ** 2  # 2 MB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_mock

        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000  # 150W in mW
        mock_pynvml.nvmlDeviceGetClockInfo.side_effect = [1500, 5000]  # SM, Mem

        handle = MagicMock()
        collector._handle = handle

        with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
            result = collector.collect()

        assert result["hw_gpu_util"] == 75.0
        assert result["hw_gpu_mem_used_mb"] == 2.0
        assert result["hw_gpu_temp_c"] == 65.0
        assert result["hw_gpu_power_w"] == 150.0
        assert result["hw_gpu_clock_sm_mhz"] == 1500.0
        assert result["hw_gpu_clock_mem_mhz"] == 5000.0

    def test_collect_partial_failure(self, nvidia_collector):
        collector, mock_pynvml = nvidia_collector

        # utilization 성공, 나머지 실패
        util_mock = MagicMock()
        util_mock.gpu = 50
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util_mock

        error = mock_pynvml.NVMLError("Not supported")
        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = error
        mock_pynvml.nvmlDeviceGetTemperature.side_effect = error
        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = error
        mock_pynvml.nvmlDeviceGetClockInfo.side_effect = error

        handle = MagicMock()
        collector._handle = handle

        with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
            result = collector.collect()

        assert result["hw_gpu_util"] == 50.0
        assert result["hw_gpu_mem_used_mb"] is None
        assert result["hw_gpu_temp_c"] is None
        assert result["hw_gpu_power_w"] is None
        assert result["hw_gpu_clock_sm_mhz"] is None
        assert result["hw_gpu_clock_mem_mhz"] is None

    def test_collect_all_failure(self, nvidia_collector):
        collector, mock_pynvml = nvidia_collector

        error = mock_pynvml.NVMLError("All failed")
        mock_pynvml.nvmlDeviceGetUtilizationRates.side_effect = error
        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = error
        mock_pynvml.nvmlDeviceGetTemperature.side_effect = error
        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = error
        mock_pynvml.nvmlDeviceGetClockInfo.side_effect = error

        handle = MagicMock()
        collector._handle = handle

        with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
            result = collector.collect()

        # 모든 값이 None
        assert all(v is None for v in result.values())


class TestNvidiaCollectorStartStop:
    def test_start_calls_nvml_init(self, nvidia_collector):
        collector, mock_pynvml = nvidia_collector
        mock_pynvml.nvmlInit.return_value = None
        handle_mock = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = handle_mock

        with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
            collector.start()

        mock_pynvml.nvmlInit.assert_called_once()
        mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)

    def test_stop_calls_nvml_shutdown(self, nvidia_collector):
        collector, mock_pynvml = nvidia_collector

        with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
            collector.stop()

        mock_pynvml.nvmlShutdown.assert_called_once()
