"""SystemCollector 단위 테스트."""

from unittest.mock import patch, MagicMock

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monitors.system_collector import SystemCollector


class TestSystemCollector:
    def test_collect_cpu_ram(self):
        collector = SystemCollector()

        with patch('monitors.system_collector.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 45.0
            mock_mem = MagicMock()
            mock_mem.used = 4 * 1024 ** 2  # 4 MB
            mock_mem.total = 16 * 1024 ** 2  # 16 MB
            mock_psutil.virtual_memory.return_value = mock_mem

            result = collector.collect()

        assert result["hw_cpu_util"] == 45.0
        assert result["hw_ram_used_mb"] == 4.0
        assert result["hw_ram_total_mb"] == 16.0

    def test_start_initializes_cpu_percent(self):
        collector = SystemCollector()

        with patch('monitors.system_collector.psutil') as mock_psutil:
            collector.start()
            mock_psutil.cpu_percent.assert_called_once_with(interval=None)

    def test_stop_is_noop(self):
        collector = SystemCollector()
        # stop should not raise
        collector.stop()
