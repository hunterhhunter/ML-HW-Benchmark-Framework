"""
ResultStore 유닛 테스트

CSV 파일 기반 벤치마크 결과 저장/조회/삭제 로직을 검증합니다.
임시 디렉토리를 사용하므로 실제 결과 파일에 영향을 주지 않습니다.

실행 방법:
    python -m pytest tests/test_result_store.py -v
"""

import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path

from core.result_store import (
    delete_result,
    get_result,
    load_results,
    reserve_run_artifacts,
    save_result,
)


@pytest.fixture
def tmp_csv(tmp_path):
    """각 테스트마다 독립된 임시 CSV 경로를 제공합니다."""
    return tmp_path / "test_results.csv"


# ------------------------------------------------------------------
# save_result 테스트
# ------------------------------------------------------------------

class TestSaveResult:
    def test_external_server_accepts_reserved_artifact_transaction(self, tmp_csv):
        reservation = reserve_run_artifacts(
            results_path=tmp_csv,
            run_id="server001",
        )

        run_id = save_result(
            metrics={"server_successful_requests": 2},
            model_name="llama",
            task="NLP_GENERATION",
            backend="furiosa_llm_server",
            device="npu:0",
            batch_size=1,
            warmup_runs=0,
            results_path=tmp_csv,
            run_id=reservation.run_id,
            inference_mode="external_server",
            async_run_status="valid",
            reservation=reservation,
        )

        assert run_id == "server001"
        assert load_results(results_path=tmp_csv)[0]["inference_mode"] == (
            "external_server"
        )

    def test_save_creates_csv_with_header(self, tmp_csv):
        """첫 번째 저장 시 CSV 파일과 헤더가 생성된다."""
        metrics = {"Top-1 Accuracy": 75.42, "Average Latency (ms)": 12.34}
        run_id = save_result(
            metrics=metrics,
            model_name="resnet50",
            task="IMAGE_CLASSIFICATION",
            backend="onnxruntime",
            device="cuda",
            batch_size=1,
            warmup_runs=2,
            results_path=tmp_csv,
        )

        assert tmp_csv.exists()
        assert len(run_id) == 8

        with open(tmp_csv, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["model_name"] == "resnet50"
        assert rows[0]["task"] == "IMAGE_CLASSIFICATION"
        assert rows[0]["Top-1 Accuracy"] == "75.42"

    def test_save_appends_multiple_results(self, tmp_csv):
        """여러 결과를 저장하면 행이 누적된다."""
        for i in range(3):
            save_result(
                metrics={"accuracy": 90.0 + i},
                model_name=f"model_{i}",
                task="NLP_CLASSIFICATION",
                backend="onnxruntime",
                device="cpu",
                batch_size=1,
                warmup_runs=2,
                results_path=tmp_csv,
            )

        rows = load_results(results_path=tmp_csv)
        assert len(rows) == 3

    def test_save_auto_extends_columns(self, tmp_csv):
        """서로 다른 메트릭을 가진 결과 저장 시 컬럼이 자동 확장된다."""
        save_result(
            metrics={"Top-1 Accuracy": 75.0},
            model_name="resnet50",
            task="IMAGE_CLASSIFICATION",
            backend="onnxruntime",
            device="cpu",
            batch_size=1,
            warmup_runs=2,
            results_path=tmp_csv,
        )
        save_result(
            metrics={"MAE": 0.123, "MSE": 0.234},
            model_name="patchtst",
            task="TIME_SERIES_FORECASTING",
            backend="onnxruntime",
            device="cpu",
            batch_size=1,
            warmup_runs=2,
            results_path=tmp_csv,
        )

        with open(tmp_csv, "r") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)

        # 두 종류의 메트릭 컬럼이 모두 존재해야 함
        assert "Top-1 Accuracy" in headers
        assert "MAE" in headers
        assert "MSE" in headers

        # 첫 행은 MAE/MSE가 비어 있어야 함
        assert rows[0]["MAE"] == ""
        # 두 번째 행은 Top-1 Accuracy가 비어 있어야 함
        assert rows[1]["Top-1 Accuracy"] == ""
        assert rows[1]["MAE"] == "0.123"

    def test_save_extends_columns_for_vllm_timing_sources(self, tmp_csv):
        """vLLM measured/estimated timing 컬럼이 함께 누적 저장된다."""
        save_result(
            metrics={
                "Average Latency (ms)": 80.0,
                "Avg TTFT estimate (ms)": 20.0,
                "Avg TPOT estimate (ms)": 20.0,
            },
            model_name="llama",
            task="NLP_GENERATION",
            backend="vllm",
            device="cuda",
            batch_size=1,
            warmup_runs=2,
            results_path=tmp_csv,
        )
        save_result(
            metrics={
                "Average Latency (ms)": 75.0,
                "Avg TTFT (ms)": 12.0,
                "Avg TPOT (ms)": 8.0,
            },
            model_name="llama",
            task="NLP_GENERATION",
            backend="vllm",
            device="cuda",
            batch_size=1,
            warmup_runs=2,
            results_path=tmp_csv,
        )

        with open(tmp_csv, "r") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)

        assert "Avg TTFT estimate (ms)" in headers
        assert "Avg TPOT estimate (ms)" in headers
        assert "Avg TTFT (ms)" in headers
        assert "Avg TPOT (ms)" in headers
        assert rows[0]["Avg TTFT (ms)"] == ""
        assert rows[0]["Avg TTFT estimate (ms)"] == "20.0"
        assert rows[1]["Avg TTFT estimate (ms)"] == ""
        assert rows[1]["Avg TTFT (ms)"] == "12.0"

    def test_save_max_steps_optional(self, tmp_csv):
        """max_steps가 None이면 빈 문자열로 저장된다."""
        save_result(
            metrics={"accuracy": 90.0},
            model_name="bert",
            task="NLP_CLASSIFICATION",
            backend="onnxruntime",
            device="cpu",
            batch_size=1,
            warmup_runs=2,
            max_steps=None,
            results_path=tmp_csv,
        )

        rows = load_results(results_path=tmp_csv)
        assert rows[0]["max_steps"] == ""

    def test_save_max_steps_value(self, tmp_csv):
        """max_steps가 지정되면 해당 값으로 저장된다."""
        save_result(
            metrics={"accuracy": 90.0},
            model_name="bert",
            task="NLP_CLASSIFICATION",
            backend="onnxruntime",
            device="cpu",
            batch_size=1,
            warmup_runs=2,
            max_steps=100,
            results_path=tmp_csv,
        )

        rows = load_results(results_path=tmp_csv)
        assert rows[0]["max_steps"] == "100"

    def test_save_target_metadata(self, tmp_csv):
        """target/runtime/compiler 메타데이터가 CSV 메타 컬럼으로 저장된다."""
        save_result(
            metrics={"accuracy": 90.0},
            model_name="bert",
            task="NLP_CLASSIFICATION",
            backend="mock_npu",
            device="npu0",
            batch_size=1,
            warmup_runs=2,
            target_id="vendor_mock_npu",
            accelerator_vendor="MockNPU",
            accelerator_name="Mock NPU PCIe Adapter",
            runtime_name="mock_npu",
            compiler_name="mock_npu",
            artifact_format="mockbin",
            results_path=tmp_csv,
        )

        rows = load_results(results_path=tmp_csv)
        assert rows[0]["target_id"] == "vendor_mock_npu"
        assert rows[0]["accelerator_vendor"] == "MockNPU"
        assert rows[0]["compiler_name"] == "mock_npu"

    def test_save_mobilint_decoder_metadata_as_result_metadata(self, tmp_csv):
        save_result(
            metrics={"mAP": 0.75},
            model_name="yolov5m",
            task="OBJECT_DETECTION",
            backend="mobilint",
            device="0",
            batch_size=1,
            warmup_runs=0,
            mobilint_vision_profile_id="mobilint-yolov5m-default",
            mobilint_yolo_confidence_threshold=0.2,
            mobilint_yolo_iou_threshold=0.4,
            mobilint_yolo_max_nms_candidates=123,
            mobilint_yolo_max_detections=7,
            mobilint_yolo_max_class_offset=4096.0,
            results_path=tmp_csv,
        )

        row = load_results(results_path=tmp_csv)[0]
        assert row["mAP"] == "0.75"
        assert row["mobilint_vision_profile_id"] == "mobilint-yolov5m-default"
        assert row["mobilint_yolo_confidence_threshold"] == "0.2"
        assert row["mobilint_yolo_iou_threshold"] == "0.4"
        assert row["mobilint_yolo_max_nms_candidates"] == "123"
        assert row["mobilint_yolo_max_detections"] == "7"
        assert row["mobilint_yolo_max_class_offset"] == "4096.0"

    def test_save_mobilint_static_artifact_profile(self, tmp_csv):
        save_result(
            metrics={"accuracy": 0.8},
            model_name="bert-base-uncased",
            task="NLP_CLASSIFICATION",
            backend="mobilint",
            device="0",
            batch_size=1,
            warmup_runs=0,
            mobilint_artifact_profile_id=(
                "mobilint-bert-base-uncased-tensor-v1"
            ),
            results_path=tmp_csv,
        )

        row = load_results(results_path=tmp_csv)[0]
        assert row["mobilint_artifact_profile_id"] == (
            "mobilint-bert-base-uncased-tensor-v1"
        )


# ------------------------------------------------------------------
# load_results 테스트
# ------------------------------------------------------------------

class TestLoadResults:
    def test_load_empty_file(self, tmp_csv):
        """CSV 파일이 없으면 빈 리스트를 반환한다."""
        rows = load_results(results_path=tmp_csv)
        assert rows == []

    def test_load_filter_by_model_name(self, tmp_csv):
        """model_name으로 필터링한다."""
        save_result(metrics={"acc": 90}, model_name="resnet50", task="IMG", backend="onnx", device="cpu", batch_size=1, warmup_runs=2, results_path=tmp_csv)
        save_result(metrics={"acc": 85}, model_name="bert", task="NLP", backend="onnx", device="cpu", batch_size=1, warmup_runs=2, results_path=tmp_csv)
        save_result(metrics={"acc": 92}, model_name="resnet50", task="IMG", backend="onnx", device="cuda", batch_size=1, warmup_runs=2, results_path=tmp_csv)

        rows = load_results(results_path=tmp_csv, model_name="resnet50")
        assert len(rows) == 2
        assert all(r["model_name"] == "resnet50" for r in rows)

    def test_load_filter_by_task(self, tmp_csv):
        """task로 필터링한다."""
        save_result(metrics={"acc": 90}, model_name="resnet50", task="IMAGE_CLASSIFICATION", backend="onnx", device="cpu", batch_size=1, warmup_runs=2, results_path=tmp_csv)
        save_result(metrics={"acc": 85}, model_name="bert", task="NLP_CLASSIFICATION", backend="onnx", device="cpu", batch_size=1, warmup_runs=2, results_path=tmp_csv)

        rows = load_results(results_path=tmp_csv, task="NLP_CLASSIFICATION")
        assert len(rows) == 1
        assert rows[0]["model_name"] == "bert"

    def test_load_filter_by_backend(self, tmp_csv):
        """backend로 필터링한다."""
        save_result(metrics={"acc": 90}, model_name="m1", task="T", backend="onnxruntime", device="cpu", batch_size=1, warmup_runs=2, results_path=tmp_csv)
        save_result(metrics={"acc": 85}, model_name="m2", task="T", backend="vllm", device="cuda", batch_size=1, warmup_runs=2, results_path=tmp_csv)

        rows = load_results(results_path=tmp_csv, backend="vllm")
        assert len(rows) == 1
        assert rows[0]["model_name"] == "m2"

    def test_load_limit(self, tmp_csv):
        """limit으로 반환 건수를 제한한다."""
        for i in range(5):
            save_result(metrics={"acc": i}, model_name=f"m{i}", task="T", backend="onnx", device="cpu", batch_size=1, warmup_runs=2, results_path=tmp_csv)

        rows = load_results(results_path=tmp_csv, limit=3)
        assert len(rows) == 3

    def test_load_returns_newest_first(self, tmp_csv):
        """최신 결과가 먼저 반환된다 (역순)."""
        save_result(metrics={"acc": 1}, model_name="first", task="T", backend="onnx", device="cpu", batch_size=1, warmup_runs=2, results_path=tmp_csv)
        save_result(metrics={"acc": 2}, model_name="second", task="T", backend="onnx", device="cpu", batch_size=1, warmup_runs=2, results_path=tmp_csv)

        rows = load_results(results_path=tmp_csv)
        assert rows[0]["model_name"] == "second"
        assert rows[1]["model_name"] == "first"


# ------------------------------------------------------------------
# get_result 테스트
# ------------------------------------------------------------------

class TestGetResult:
    def test_get_existing_result(self, tmp_csv):
        """존재하는 run_id로 결과를 조회한다."""
        run_id = save_result(
            metrics={"accuracy": 95.0},
            model_name="bert",
            task="NLP_CLASSIFICATION",
            backend="onnxruntime",
            device="cpu",
            batch_size=1,
            warmup_runs=2,
            results_path=tmp_csv,
        )

        result = get_result(run_id, results_path=tmp_csv)
        assert result is not None
        assert result["run_id"] == run_id
        assert result["model_name"] == "bert"

    def test_get_nonexistent_result(self, tmp_csv):
        """존재하지 않는 run_id는 None을 반환한다."""
        result = get_result("nonexistent", results_path=tmp_csv)
        assert result is None

    def test_get_from_nonexistent_file(self, tmp_csv):
        """파일이 없으면 None을 반환한다."""
        result = get_result("any_id", results_path=tmp_csv)
        assert result is None


# ------------------------------------------------------------------
# delete_result 테스트
# ------------------------------------------------------------------

class TestDeleteResult:
    def test_delete_existing_result(self, tmp_csv):
        """존재하는 run_id를 삭제하면 True를 반환하고 행이 제거된다."""
        run_id = save_result(
            metrics={"accuracy": 95.0},
            model_name="bert",
            task="NLP",
            backend="onnx",
            device="cpu",
            batch_size=1,
            warmup_runs=2,
            results_path=tmp_csv,
        )

        assert delete_result(run_id, results_path=tmp_csv) is True
        assert get_result(run_id, results_path=tmp_csv) is None

    def test_delete_nonexistent_result(self, tmp_csv):
        """존재하지 않는 run_id 삭제 시 False를 반환한다."""
        save_result(metrics={"acc": 90}, model_name="m1", task="T", backend="o", device="cpu", batch_size=1, warmup_runs=2, results_path=tmp_csv)
        assert delete_result("nonexistent", results_path=tmp_csv) is False

    def test_delete_preserves_other_rows(self, tmp_csv):
        """삭제 시 다른 행은 보존된다."""
        id1 = save_result(metrics={"acc": 90}, model_name="m1", task="T", backend="o", device="cpu", batch_size=1, warmup_runs=2, results_path=tmp_csv)
        id2 = save_result(metrics={"acc": 85}, model_name="m2", task="T", backend="o", device="cpu", batch_size=1, warmup_runs=2, results_path=tmp_csv)

        delete_result(id1, results_path=tmp_csv)

        assert get_result(id1, results_path=tmp_csv) is None
        assert get_result(id2, results_path=tmp_csv) is not None

    def test_delete_from_nonexistent_file(self, tmp_csv):
        """파일이 없으면 False를 반환한다."""
        assert delete_result("any_id", results_path=tmp_csv) is False
