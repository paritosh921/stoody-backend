from __future__ import annotations

import io

import pytest
from PIL import Image


def _module():
    from api.v1._exampen_imports import load_exampen

    return load_exampen("pcr.services.orientation_views")


def _png(width: int = 120, height: int = 200) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="PNG")
    return output.getvalue()


def test_upright_signal_keeps_only_original_view():
    module = _module()
    original = _png()
    views = module.build_orientation_views(
        original,
        physical_page_number=3,
        detector=lambda image, line_ratio: (False, {"method": "test", "reason": "weak"}),
    )

    assert len(views) == 1
    view = views[0]
    assert view.is_original is True
    assert view.rotation_degrees_clockwise == 0
    assert view.alternate_of == view.view_id
    assert view.image_bytes == original
    assert view.coordinate_frame["invertible"] is True
    assert view.coordinate_frame["original_width_px"] == 120
    assert view.coordinate_frame["original_height_px"] == 200


def test_sideways_signal_exposes_alternates_of_same_physical_page():
    module = _module()
    views = module.build_orientation_views(
        _png(120, 200),
        physical_page_number=4,
        detector=lambda image, line_ratio: (True, {"method": "test", "sideways": True}),
    )

    assert [view.rotation_degrees_clockwise for view in views] == [0, 90, 270]
    assert len({view.physical_page_number for view in views}) == 1
    assert len({view.alternate_of for view in views}) == 1
    assert all(view.alternate_of == views[0].view_id for view in views)
    assert views[1].width_px == 200 and views[1].height_px == 120
    assert views[2].width_px == 200 and views[2].height_px == 120
    assert views[1].coordinate_frame["original_width_px"] == 120
    assert views[1].coordinate_frame["view_rotation_degrees_clockwise"] == 90
    assert all(view.orientation_evidence["sideways"] is True for view in views)


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_point_transform_is_invertible(rotation: int):
    module = _module()
    original = (137.0, 821.0)
    view = module.original_point_to_view(*original, rotation_degrees_clockwise=rotation)
    round_trip = module.view_point_to_original(*view, rotation_degrees_clockwise=rotation)

    assert round_trip[0] == pytest.approx(original[0])
    assert round_trip[1] == pytest.approx(original[1])


def test_region_transform_returns_original_frame_metadata():
    module = _module()
    region = {"region_id": "q1-r1", "x_start": 100, "y_start": 200, "x_end": 800, "y_end": 900}
    transformed = module.view_region_to_original(region, rotation_degrees_clockwise=90)

    assert transformed["region_id"] == "q1-r1"
    assert transformed["coordinate_space"] == "normalized_1000"
    assert transformed["source_rotation_degrees_clockwise"] == 90
    assert transformed["coordinate_transform"]["invertible"] is True
    assert transformed["x_start"] == 200
    assert transformed["y_start"] == 200
    assert transformed["x_end"] == 900
    assert transformed["y_end"] == 900


def test_detector_failure_fails_closed_without_alternate_views():
    module = _module()

    def broken_detector(image, line_ratio):
        raise RuntimeError("detector unavailable")

    views = module.build_orientation_views(
        _png(), physical_page_number=1, detector=broken_detector
    )
    assert len(views) == 1
    assert views[0].orientation_evidence["reason"] == "detector_error"


def test_sideways_invalid_image_fails_closed_instead_of_fake_rotated_bytes():
    module = _module()
    with pytest.raises(module.OrientationViewError):
        module.build_orientation_views(
            b"not-an-image",
            physical_page_number=1,
            width_px=100,
            height_px=100,
            detector=lambda image, line_ratio: (True, {"method": "test"}),
        )
