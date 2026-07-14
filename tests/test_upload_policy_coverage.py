from pathlib import Path
import ast

from core.upload_security.coverage import find_upload_routes, uncovered_upload_routes


CENTRAL_UPLOAD_GATEWAY_CALLS = {
    "secure_upload",
    "secure_upload_many",
    "upload_message_attachments",
    "_store_exam_template_file",
    "_secure_student_copy_pages",
    "_validate_bulk_upload_file",
    "parse_upload_file",
    "_read_clean_timetable_upload",
}


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


def test_uploadfile_routes_call_central_upload_gateway_or_wrapper():
    backend_root = Path(__file__).resolve().parents[1]
    routes = find_upload_routes(backend_root)
    missing_gateway = []

    for route in routes:
        source_path = backend_root / route.module_path
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == route.function_name
        )
        call_names = {
            _call_name(node.func)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
        }
        if call_names.isdisjoint(CENTRAL_UPLOAD_GATEWAY_CALLS):
            missing_gateway.append((route.method, route.path_template, route.module_path, route.function_name))

    assert missing_gateway == []


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""
