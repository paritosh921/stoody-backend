from pathlib import Path

from core.upload_security.coverage import find_upload_routes, uncovered_upload_routes


def test_current_uploadfile_routes_are_covered_by_policy_map():
    backend_root = Path(__file__).resolve().parents[1]
    discovered = find_upload_routes(backend_root)
    uncovered = uncovered_upload_routes(discovered)

    assert uncovered == []


def test_known_upload_routes_are_discovered():
    backend_root = Path(__file__).resolve().parents[1]
    paths = {(route.method, route.path_template) for route in find_upload_routes(backend_root)}

    assert ("POST", "/api/v1/debugger/upload") in paths
    assert ("POST", "/api/debugger/upload") in paths
    assert ("POST", "/api/v1/stoody-book/sessions/{session_id}/pdfs") in paths
    assert ("POST", "/api/v1/pdf/questions") in paths
    assert ("POST", "/api/v1/ingest/camera/{exam_id}/{student_id}/{page_num}") in paths
