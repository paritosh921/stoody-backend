"""Authenticated downloads for clean released upload objects."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager
from utils.s3_storage import download_file as s3_download_file


router = APIRouter()
DownloadAuthorizer = Callable[[Dict[str, Any], Dict[str, Any]], bool]


@router.get("/uploads/{upload_id}/download")
async def download_clean_upload(
    upload_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    verdict = await db.mongo_find_one("upload_security_verdicts", {"upload_id": upload_id})
    if not verdict:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")
    if verdict.get("status") != "clean":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not available")
    _authorize_download(verdict, current_user)

    storage_path = verdict.get("released_storage_path")
    if not storage_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload file not found")

    if str(storage_path).startswith("s3://"):
        data = await s3_download_file(storage_path)
    else:
        path = Path(str(storage_path))
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload file not found")
        async with aiofiles.open(path, "rb") as handle:
            data = await handle.read()

    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload file not found")

    filename = _safe_download_filename(verdict.get("original_filename") or "download.bin")
    content_type = verdict.get("declared_content_type") or "application/octet-stream"
    return Response(
        content=data,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


def _authorize_download(verdict: Dict[str, Any], current_user: Dict[str, Any]) -> None:
    tenant_db = verdict.get("tenant_db")
    current_db = current_user.get("db_name") or current_user.get("tenant_db")
    if tenant_db and current_db and tenant_db != current_db:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Upload not authorized")

    metadata = verdict.get("purpose_metadata") or {}
    purpose = metadata.get("purpose")
    authorizer = _DOWNLOAD_AUTHORIZERS.get(str(purpose or ""))
    if authorizer is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Upload not authorized")
    if not authorizer(verdict, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Upload not authorized")


def _owner_metadata_authorizer(verdict: Dict[str, Any], current_user: Dict[str, Any]) -> bool:
    metadata = verdict.get("purpose_metadata") or {}
    user_ids = {
        str(value)
        for value in (
            current_user.get("user_id"),
            current_user.get("admin_id"),
            current_user.get("tutor_id"),
            current_user.get("student_id"),
            current_user.get("_id"),
        )
        if value
    }
    owner_values = {
        str(value)
        for value in (
            metadata.get("created_by"),
            metadata.get("admin_id"),
            metadata.get("tutor_id"),
            metadata.get("student_id"),
            verdict.get("user_id"),
        )
        if value
    }
    return bool(owner_values) and not user_ids.isdisjoint(owner_values)


_DOWNLOAD_AUTHORIZERS: dict[str, DownloadAuthorizer] = {
    "support_message_attachment": _owner_metadata_authorizer,
    "teaching_material": _owner_metadata_authorizer,
    "stoody_book_pdf": _owner_metadata_authorizer,
    "desktop_diagnostics_zip": _owner_metadata_authorizer,
    "desktop_bug_image": _owner_metadata_authorizer,
    "debugger_document": _owner_metadata_authorizer,
    "school_logo": _owner_metadata_authorizer,
    "generic_image_upload": _owner_metadata_authorizer,
}


def _safe_download_filename(filename: str) -> str:
    name = os.path.basename(str(filename).replace("\\", "/")).replace('"', "")
    return name or "download.bin"
