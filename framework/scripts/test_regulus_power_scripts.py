"""Behavior checks for the script-only REGULUS power measurement path."""

import csv
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from unittest import TestCase, main


SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from regulus_power_link import PowerServer, RemotePowerSession, analyze_power_run
from run_regulus_power import execute_framework_case


class RegulusPowerScriptTests(TestCase):
  def test_server_cli_builds_ina_worker_from_explicit_connection_settings(self):
    from serve_regulus_power import build_parser, make_server

    with tempfile.TemporaryDirectory() as directory:
      arguments = build_parser().parse_args([
          "--bind", "192.0.2.10", "--allow-client", "192.0.2.11",
          "--output-root", directory, "--bus", "7", "--address", "0x40",
      ])
      server = make_server(arguments)
      self.assertEqual(server.host, "192.0.2.10")
      self.assertEqual(server.allowed_client_ip, "192.0.2.11")
      self.assertEqual(server.sample_fn.bus, 7)
      self.assertEqual(server.sample_fn.address, 0x40)

  def test_overcurrent_read_does_not_start_measurement(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      server = PowerServer(
          host="127.0.0.1", port=0, output_root=root,
          sample_interval_sec=0.01,
          sample_fn=lambda: (12.0, 3.1, 37.2),
      )
      server.start()
      try:
        session = RemotePowerSession(
            host="127.0.0.1", port=server.port,
            measurement_id="overcurrent-run", baseline_sec=0.05,
            output_dir=root / "local",
        )
        with self.assertRaisesRegex(RuntimeError, "rejected READY"):
          session.start(expected_samples=5)
        session.close()
      finally:
        server.stop()
      self.assertTrue((root / "overcurrent-run" / "failure.txt").is_file())

  def test_unexpected_bus_voltage_does_not_start_measurement(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      server = PowerServer(
          host="127.0.0.1", port=0, output_root=root,
          sample_interval_sec=0.01,
          sample_fn=lambda: (3.96, 0.25, 0.99),
      )
      server.start()
      try:
        session = RemotePowerSession(
            host="127.0.0.1", port=server.port,
            measurement_id="low-bus-run", baseline_sec=0.05,
            output_dir=root / "local",
        )
        with self.assertRaisesRegex(RuntimeError, "rejected READY"):
          session.start(expected_samples=5)
        session.close()
      finally:
        server.stop()
      self.assertTrue((root / "low-bus-run" / "failure.txt").is_file())

  def test_tcp_session_records_power_and_phase_on_jetson(self):
    with tempfile.TemporaryDirectory() as directory:
      self._tcp_session_records_power_and_phase_on_jetson(Path(directory))

  def _tcp_session_records_power_and_phase_on_jetson(self, tmp_path):
    server = PowerServer(
        host="127.0.0.1",
        port=0,
        output_root=tmp_path / "jetson",
        sample_interval_sec=0.01,
        sample_fn=lambda: (12.0, 0.1, 1.2),
    )
    server.start()
    try:
        session = RemotePowerSession(
            host="127.0.0.1",
            port=server.port,
            measurement_id="test-run-1",
            baseline_sec=0.08,
            output_dir=tmp_path / "regulus",
        )
        session.start(model="resnet50", mode="e2e", expected_samples=5)
        time.sleep(0.08)
        session.stop()
        artifact = session.finalize(completed_samples=5, run_status="valid")
    finally:
        server.stop()

    self.assertEqual(artifact["status"], "complete")
    self.assertEqual(artifact["completed_samples"], 5)
    self.assertEqual(artifact["power_scope"], "regulus_dc_input")
    self.assertTrue(Path(artifact["sensor_path"]).is_file())
    events = [json.loads(line) for line in Path(artifact["events_path"]).read_text().splitlines()]
    names = [event["event"] for event in events]
    self.assertEqual(names, ["baseline_start", "baseline_end", "inference_start", "inference_end"])
    self.assertTrue(all(events[i]["monotonic_ns"] < events[i + 1]["monotonic_ns"] for i in range(3)))
    with Path(artifact["sensor_path"]).open(newline="") as handle:
        readings = list(csv.DictReader(handle))
    self.assertGreaterEqual(len(readings), 5)
    self.assertTrue(all(abs(float(reading["power_W"]) - 1.2) < 1e-6 for reading in readings))


  def test_analysis_integrates_exact_marker_window(self):
    with tempfile.TemporaryDirectory() as directory:
      self._analysis_integrates_exact_marker_window(Path(directory))

  def _analysis_integrates_exact_marker_window(self, tmp_path):
    sensor = tmp_path / "sensor.csv"
    sensor.write_text(
        "monotonic_ns,bus_V,current_A,power_W\n"
        "0,12,0.083333,1\n"
        "1000000000,12,0.083333,1\n"
        "2000000000,12,0.083333,1\n"
        "3000000000,12,0.25,3\n"
        "4000000000,12,0.25,3\n"
        "5000000000,12,0.25,3\n"
    )
    events = tmp_path / "events.jsonl"
    events.write_text("".join(
        json.dumps({"event": name, "monotonic_ns": timestamp}) + "\n"
        for name, timestamp in (
            ("baseline_start", 0),
            ("baseline_end", 2000000000),
            ("inference_start", 3000000000),
            ("inference_end", 5000000000),
        )
    ))

    result = analyze_power_run(sensor, events, completed_samples=4, max_sample_gap_sec=1.1)
    self.assertAlmostEqual(result["idle_median_w"], 1)
    self.assertAlmostEqual(result["inference_mean_w"], 3)
    self.assertAlmostEqual(result["incremental_energy_j"], 4)
    self.assertAlmostEqual(result["incremental_j_per_sample"], 1)


  def test_analysis_rejects_gap_in_inference_samples(self):
    with tempfile.TemporaryDirectory() as directory:
      self._analysis_rejects_gap_in_inference_samples(Path(directory))

  def _analysis_rejects_gap_in_inference_samples(self, tmp_path):
    sensor = tmp_path / "sensor.csv"
    sensor.write_text(
        "monotonic_ns,bus_V,current_A,power_W\n"
        "0,12,0.1,1.2\n"
        "10000000,12,0.1,1.2\n"
        "20000000,12,0.1,1.2\n"
        "30000000,12,0.2,2.4\n"
        "1030000000,12,0.2,2.4\n"
    )
    events = tmp_path / "events.jsonl"
    events.write_text("".join(
        json.dumps({"event": name, "monotonic_ns": timestamp}) + "\n"
        for name, timestamp in (
            ("baseline_start", 0),
            ("baseline_end", 20000000),
            ("inference_start", 30000000),
            ("inference_end", 1030000000),
        )
    ))

    with self.assertRaisesRegex(ValueError, "sample gap"):
        analyze_power_run(sensor, events, completed_samples=4, max_sample_gap_sec=0.2)

  def test_script_wrapper_runs_existing_main_and_links_result(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      server = PowerServer(
          host="127.0.0.1", port=0, output_root=root / "jetson",
          sample_interval_sec=0.01,
          sample_fn=lambda: (12.0, 0.1, 1.2),
      )
      server.start()
      class FakeMain:
        _create_requested_hw_monitor = None

        def main(self):
          monitor = self._create_requested_hw_monitor(
              SimpleNamespace(), SimpleNamespace(target_id="mobilint-regulus")
          )
          monitor.record_after_load_vram()
          monitor.start()
          time.sleep(0.08)
          monitor.stop()
          result_path = Path(sys.argv[sys.argv.index("--results-path") + 1])
          result_path.write_text(
              "run_id,inference_mode,Total Samples\n"
              "framework-run-1,e2e,5\n"
          )
          return 0

      try:
        artifact = execute_framework_case(
            case={"name": "resnet50-e2e", "model": "resnet50", "mode": "e2e",
                  "expected_samples": 5,
                  "args": ["--model", "resnet50", "--target", "mobilint-regulus",
                           "--max-steps", "5"]},
            main_module=FakeMain(), case_dir=root / "case",
            jetson_host="127.0.0.1", jetson_port=server.port,
            baseline_sec=0.08,
        )
      finally:
        server.stop()
      self.assertEqual(artifact["framework_run_id"], "framework-run-1")
      self.assertEqual(artifact["status"], "complete")
      self.assertEqual(artifact["completed_samples"], 5)
      self.assertTrue((root / "case" / "power-summary.json").is_file())

  def test_script_wrapper_uses_legacy_monitors_factory_without_source_edits(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      server = PowerServer(
          host="127.0.0.1", port=0, output_root=root / "jetson",
          sample_interval_sec=0.01, sample_fn=lambda: (12.0, 0.1, 1.2),
      )
      server.start()
      old_factory = lambda **kwargs: None
      fake_monitors = SimpleNamespace(create_hw_monitor=old_factory)

      class LegacyMain:
        def main(self):
          from monitors import create_hw_monitor
          self.assert_monitor_arg()
          monitor = create_hw_monitor(interval=0.2, device="cpu", collector_names=[])
          monitor.start()
          time.sleep(0.08)
          monitor.stop()
          result_path = Path(sys.argv[sys.argv.index("--results-path") + 1])
          result_path.write_text("run_id,Total Samples\nlegacy-run,5\n")
          return 0

        def assert_monitor_arg(self):
          assert "--monitor" in sys.argv

      try:
        with patch.dict(sys.modules, {"monitors": fake_monitors}):
          artifact = execute_framework_case(
              case={"name": "legacy-e2e", "model": "resnet50", "mode": "e2e",
                    "expected_samples": 5,
                    "args": ["--model", "resnet50", "--target", "mobilint-regulus",
                             "--max-steps", "5"]},
              main_module=LegacyMain(), case_dir=root / "case",
              jetson_host="127.0.0.1", jetson_port=server.port,
              baseline_sec=0.08,
          )
          self.assertIs(fake_monitors.create_hw_monitor, old_factory)
      finally:
        server.stop()
      self.assertEqual(artifact["framework_run_id"], "legacy-run")

  def test_short_framework_run_is_preserved_as_partial(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      server = PowerServer(
          host="127.0.0.1", port=0, output_root=root / "jetson",
          sample_interval_sec=0.01, sample_fn=lambda: (12.0, 0.1, 1.2),
      )
      server.start()

      class ShortMain:
        _create_requested_hw_monitor = None

        def main(self):
          monitor = self._create_requested_hw_monitor(
              SimpleNamespace(), SimpleNamespace(target_id="mobilint-regulus")
          )
          monitor.start()
          time.sleep(0.08)
          monitor.stop()
          Path(sys.argv[sys.argv.index("--results-path") + 1]).write_text(
              "run_id,Total Samples\nshort-run,4\n"
          )
          return 0

      try:
        with self.assertRaisesRegex(RuntimeError, "partial"):
          execute_framework_case(
              case={"name": "short-e2e", "model": "resnet50", "mode": "e2e",
                    "expected_samples": 5,
                    "args": ["--model", "resnet50", "--target", "mobilint-regulus",
                             "--max-steps", "5"]},
              main_module=ShortMain(), case_dir=root / "case",
              jetson_host="127.0.0.1", jetson_port=server.port,
              baseline_sec=0.08,
          )
      finally:
        server.stop()
      artifact = json.loads((root / "case" / "power-summary.json").read_text())
      self.assertEqual(artifact["status"], "partial")
      self.assertEqual(artifact["completed_samples"], 4)


if __name__ == "__main__":
    main()
