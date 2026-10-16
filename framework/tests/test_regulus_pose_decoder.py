import numpy as np
import pytest

from core.model_spec import Model_Spec, Task
from dataloader.regulus_vision_profiles import REGULUS_VG_E2E_YOLOV8S_POSE_RAW_V1 as PROFILE
from decoders import create_decoder
from decoders.regulus_yolov8_pose import RegulusYoloV8PoseRawHeadDecoder
from decoders.pose_estimation import YoloV8PoseDecoder


def _outputs():
    heads = {}
    for size in (80, 40, 20):
        heads[f"k{size}"] = np.zeros((1, size, size, 51), dtype=np.float32)
        heads[f"d{size}"] = np.zeros((1, size, size, 64), dtype=np.float32)
        heads[f"c{size}"] = np.full((1, size, size, 1), -20, dtype=np.float32)
    heads["c80"][0, 0, 0, 0] = 10
    return heads


def _spec():
    return Model_Spec(name="yolov8s-pose", task=Task.POSE_ESTIMATION,
                      input_shapes={"image": (1, 640, 640, 3)},
                      input_dtype={"image": "uint8"},
                      output_shapes={"head": (1, 80, 80, 51)},
                      model_paths={"mobilint": "model.mxq"})


def test_regulus_pose_decoder_decodes_nine_heads_to_coco_payload():
    result = RegulusYoloV8PoseRawHeadDecoder(PROFILE).decode(_outputs())
    assert result["detections"].shape == (1, 7)
    assert result["keypoints"].shape == (1, 17, 3)
    assert result["detections"][0, 2] == pytest.approx(0.9999546, abs=1e-5)
    assert result["detections"][0, 3:].tolist() == pytest.approx([-56, -56, 64, 64])
    assert result["keypoints"][0, 0].tolist() == pytest.approx([0, 0, 0.5])


def test_regulus_pose_decoder_accepts_head_permutation():
    heads = _outputs()
    reversed_heads = dict(reversed(list(heads.items())))
    first = RegulusYoloV8PoseRawHeadDecoder(PROFILE).decode(heads)
    second = RegulusYoloV8PoseRawHeadDecoder(PROFILE).decode(reversed_heads)
    np.testing.assert_array_equal(first["detections"], second["detections"])
    np.testing.assert_array_equal(first["keypoints"], second["keypoints"])


def test_regulus_pose_decoder_rejects_duplicate_missing_nonfinite_heads():
    decoder = RegulusYoloV8PoseRawHeadDecoder(PROFILE)
    cases = []
    missing = _outputs(); missing.pop("k20"); cases.append(missing)
    duplicate = _outputs(); duplicate.pop("k20"); duplicate["also_k80"] = duplicate["k80"].copy(); cases.append(duplicate)
    nonfinite = _outputs(); nonfinite["d80"][0, 0, 0, 0] = np.nan; cases.append(nonfinite)
    wrong_layout = _outputs(); wrong_layout["d80"] = wrong_layout["d80"].transpose(0, 3, 1, 2); cases.append(wrong_layout)
    for outputs in cases:
        with pytest.raises(ValueError):
            decoder.decode(outputs)


def test_regulus_pose_factory_does_not_route_aries_or_unknown_abi():
    spec = _spec()
    selected = create_decoder(spec, backend="qbruntime", target_id="mobilint-regulus",
                              regulus_pose_profile=PROFILE)
    assert isinstance(selected, RegulusYoloV8PoseRawHeadDecoder)
    assert isinstance(create_decoder(spec, backend="qbruntime", target_id="mobilint-aries"), YoloV8PoseDecoder)
    with pytest.raises(ValueError):
        create_decoder(spec, backend="qbruntime", target_id="mobilint-regulus",
                       regulus_pose_profile=object())
