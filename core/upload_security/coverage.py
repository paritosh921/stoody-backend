"""Static coverage guard for FastAPI UploadFile routes."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .routes import UPLOAD_ROUTE_POLICY_MAP


@dataclass(frozen=True)
class DiscoveredUploadRoute:
    module_path: str
    function_name: str
    method: str
    path_template: str


ROUTE_PREFIXES_BY_MODULE: dict[str, tuple[str, ...]] = {
    "auth_async.py": ("/api/v1/auth", "/auth"),
    "admin_async.py": ("/api/v1/admin",),
    "superadmin_async.py": ("/api/v1/superadmin",),
    "debugger_async.py": ("/api/v1/debugger", "/api/debugger"),
    "stoody_book_async.py": ("/api/v1/stoody-book",),
    "pdf_async.py": ("/api/v1/pdf",),
    "exam_tally_async.py": ("/api/v1",),
    "images_async.py": ("/api/v1/images",),
    "settings_async.py": ("/api/v1/admin",),
    "teaching_materials_async.py": ("",),
    "desktop_diagnostics_async.py": ("/api/v1",),
    "desktop_bug_reports_async.py": ("/api/v1",),
    "camera_upload_async.py": ("/api/v1/ingest/camera",),
    "student_bulk_upload.py": ("/api/v1/admin",),
    "tutor_bulk_upload.py": ("/api/v1/tutor",),
    "timetable_bulk_upload.py": ("/api/v1/admin",),
}

UPLOAD_ROUTE_EXEMPTIONS: tuple[dict[str, str], ...] = ()


def find_upload_routes(backend_root: Path) -> list[DiscoveredUploadRoute]:
    api_root = backend_root / "api" / "v1"
    routes: list[DiscoveredUploadRoute] = []
    for path in sorted(api_root.glob("*.py")):
        prefixes = ROUTE_PREFIXES_BY_MODULE.get(path.name)
        if prefixes is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        router_prefix = _router_prefix(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _has_upload_file_param(node):
                continue
            route_decorators = _route_decorators(node)
            for method, route_path in route_decorators:
                for prefix in prefixes:
                    full_path = _join_paths(prefix, router_prefix, route_path)
                    routes.append(
                        DiscoveredUploadRoute(
                            module_path=str(path.relative_to(backend_root)),
                            function_name=node.name,
                            method=method,
                            path_template=full_path,
                        )
                    )
    return routes


def uncovered_upload_routes(routes: list[DiscoveredUploadRoute]) -> list[DiscoveredUploadRoute]:
    covered = {(entry.method, entry.path_template) for entry in UPLOAD_ROUTE_POLICY_MAP}
    exemptions = {(item["method"].upper(), item["path_template"]) for item in UPLOAD_ROUTE_EXEMPTIONS}
    return [
        route
        for route in routes
        if (route.method, route.path_template) not in covered
        and (route.method, route.path_template) not in exemptions
    ]


def _router_prefix(tree: ast.AST) -> str:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "router" for target in node.targets):
            continue
        value = node.value
        if isinstance(value, ast.Call) and _call_name(value.func) == "APIRouter":
            for keyword in value.keywords:
                if keyword.arg == "prefix" and isinstance(keyword.value, ast.Constant):
                    return str(keyword.value.value)
    return ""


def _has_upload_file_param(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    args = list(node.args.args) + list(node.args.kwonlyargs)
    defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults) + list(node.args.kw_defaults)
    for arg, default in zip(args, defaults):
        if not _annotation_contains_upload_file(arg.annotation):
            continue
        if _default_uses_file(default):
            return True
    return False


def _annotation_contains_upload_file(annotation: ast.AST | None) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id == "UploadFile"
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "UploadFile"
    if isinstance(annotation, ast.Subscript):
        return _annotation_contains_upload_file(annotation.value) or _annotation_contains_upload_file(annotation.slice)
    if isinstance(annotation, ast.Tuple):
        return any(_annotation_contains_upload_file(element) for element in annotation.elts)
    if isinstance(annotation, ast.BinOp):
        return _annotation_contains_upload_file(annotation.left) or _annotation_contains_upload_file(annotation.right)
    return False


def _default_uses_file(default: ast.AST | None) -> bool:
    if default is None:
        return False
    if isinstance(default, ast.Call):
        return _call_name(default.func) == "File"
    return False


def _route_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, str]]:
    routes: list[tuple[str, str]] = []
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if not isinstance(func, ast.Attribute):
            continue
        method = func.attr.upper()
        if method not in {"POST", "PUT", "PATCH"}:
            continue
        if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
            continue
        routes.append((method, str(decorator.args[0].value)))
    return routes


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _join_paths(*parts: str) -> str:
    clean_parts = [part.strip("/") for part in parts if part and part != "/"]
    return "/" + "/".join(clean_parts)
