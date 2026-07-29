import numpy as np
import pytest

from decoders.pose_estimation import YoloV8PoseDecoder


def _pose_prediction(batch_size=1):
    prediction = np.zeros((batch_size, 56, 2), dtype=np.float32)
    prediction[:, :5, 0] = [4, 4, 4, 4, 0.9]
    prediction[:, 5:, 0] = np.tile([4, 3, 0.8], 17)
    prediction[:, :5, 1] = [4.1, 4.1, 4, 4, 0.7]
    prediction[:, 5:, 1] = np.tile([1, 1, 0.2], 17)
    return prediction


def test_pose_decoder_keeps_keypoints_aligned_after_nms():
    result = YoloV8PoseDecoder(iou_threshold=0.5).decode(
        {"pose": _pose_prediction()}
    )

    assert result["detections"].shape == (1, 7)
    assert result["detections"][0, :3].tolist() == pytest.approx(
        [0, 0, 0.9]
    )
    assert result["keypoints"].shape == (1, 17, 3)
    assert result["keypoints"].dtype == np.float32
    np.testing.assert_allclose(result["keypoints"][0, 0], [4, 3, 0.8])


def test_pose_decoder_accepts_bnc_and_preserves_batch_indices():
    prediction = _pose_prediction(batch_size=2).transpose(0, 2, 1)

    result = YoloV8PoseDecoder(iou_threshold=0.5).decode(
        {"pose": prediction}
    )

    assert result["detections"][:, 0].tolist() == [0.0, 1.0]
    assert result["keypoints"].shape == (2, 17, 3)


def test_pose_decoder_returns_typed_empty_arrays():
    prediction = _pose_prediction()
    prediction[:, 4, :] = 0.0

    result = YoloV8PoseDecoder().decode({"pose": prediction})

    assert result["detections"].shape == (0, 7)
    assert result["detections"].dtype == np.float32
    assert result["keypoints"].shape == (0, 17, 3)
    assert result["keypoints"].dtype == np.float32


def test_pose_decoder_respects_max_detections():
    prediction = _pose_prediction()
    prediction[0, :4, 1] = [12, 12, 2, 2]

    result = YoloV8PoseDecoder(max_detections=1).decode(
        {"pose": prediction}
    )

    assert result["detections"].shape == (1, 7)
    np.testing.assert_allclose(result["keypoints"][0, 0], [4, 3, 0.8])


def test_pose_decoder_rejects_wrong_feature_count():
    with pytest.raises(ValueError, match="pose prediction"):
        YoloV8PoseDecoder().decode(
            {"pose": np.zeros((1, 55, 10), dtype=np.float32)}
        )


def test_pose_decoder_rejects_non_finite_prediction():
    prediction = _pose_prediction()
    prediction[0, 5, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        YoloV8PoseDecoder().decode({"pose": prediction})
