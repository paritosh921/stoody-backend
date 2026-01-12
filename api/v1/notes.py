"""
Notes Management API

This module handles saving, loading, and organizing canvas notes and folders.
Notes are stored in a JSON file for persistence.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notes", tags=["notes"])

# Storage path for notes
NOTES_DIR = Path(__file__).resolve().parents[2] / "data" / "notes"
NOTES_FILE = NOTES_DIR / "notes.json"
FOLDERS_FILE = NOTES_DIR / "folders.json"


def ensure_notes_dir():
    """Ensure the notes directory exists."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)


class StrokePoint(BaseModel):
    x: float
    y: float
    pressure: Optional[float] = None


class Stroke(BaseModel):
    id: Optional[str] = None
    points: List[StrokePoint]
    color: Optional[str] = None
    strokeWidth: Optional[float] = None
    timestamp: Optional[float] = None


class CanvasData(BaseModel):
    strokes: List[Stroke] = Field(default_factory=list)
    background: str = "plain"


class NoteFolder(BaseModel):
    id: str
    name: str
    color: Optional[str] = "#2563eb"
    createdAt: datetime = Field(default_factory=datetime.utcnow)


class SavedNote(BaseModel):
    id: str
    title: str
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)
    folderId: Optional[str] = None
    canvasData: CanvasData = Field(default_factory=CanvasData)


class CreateFolderRequest(BaseModel):
    name: str
    color: Optional[str] = None


class UpdateFolderRequest(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None


class CreateNoteRequest(BaseModel):
    title: str
    folderId: Optional[str] = None
    canvasData: Optional[CanvasData] = None


class UpdateNoteRequest(BaseModel):
    title: Optional[str] = None
    folderId: Optional[str] = None
    canvasData: Optional[CanvasData] = None


class NotesResponse(BaseModel):
    notes: List[SavedNote]
    folders: List[NoteFolder]


def get_app_state():
    from main_async import app
    return app.state


def load_folders() -> List[Dict[str, Any]]:
    """Load folders from storage."""
    ensure_notes_dir()
    if FOLDERS_FILE.exists():
        with open(FOLDERS_FILE, "r") as f:
            return json.load(f)
    return []


def save_folders(folders: List[Dict[str, Any]]) -> None:
    """Save folders to storage."""
    ensure_notes_dir()
    with open(FOLDERS_FILE, "w") as f:
        json.dump(folders, f, indent=2, default=str)


def load_notes() -> List[Dict[str, Any]]:
    """Load notes from storage."""
    ensure_notes_dir()
    if NOTES_FILE.exists():
        with open(NOTES_FILE, "r") as f:
            return json.load(f)
    return []


def save_notes(notes: List[Dict[str, Any]]) -> None:
    """Save notes to storage."""
    ensure_notes_dir()
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f, indent=2, default=str)


@router.get("", response_model=NotesResponse)
async def get_all_notes(token: Optional[str] = None, state=Depends(get_app_state)):
    """Get all notes and folders."""
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

    folders_data = load_folders()
    notes_data = load_notes()

    folders = [
        NoteFolder(
            id=f["id"],
            name=f["name"],
            color=f.get("color", "#2563eb"),
            createdAt=datetime.fromisoformat(f["createdAt"]) if isinstance(f.get("createdAt"), str) else f.get("createdAt", datetime.utcnow()),
        )
        for f in folders_data
    ]

    notes = [
        SavedNote(
            id=n["id"],
            title=n["title"],
            createdAt=datetime.fromisoformat(n["createdAt"]) if isinstance(n.get("createdAt"), str) else n.get("createdAt", datetime.utcnow()),
            updatedAt=datetime.fromisoformat(n["updatedAt"]) if isinstance(n.get("updatedAt"), str) else n.get("updatedAt", datetime.utcnow()),
            folderId=n.get("folderId"),
            canvasData=CanvasData(**n.get("canvasData", {})),
        )
        for n in notes_data
    ]

    return NotesResponse(notes=notes, folders=folders)


# Folder endpoints
@router.post("/folders", response_model=NoteFolder)
async def create_folder(payload: CreateFolderRequest, token: Optional[str] = None, state=Depends(get_app_state)):
    """Create a new folder."""
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

    folders = load_folders()

    # Generate random color if not provided
    colors = ["#2563eb", "#16a34a", "#ea580c", "#9333ea", "#0d9488", "#dc2626", "#7c3aed"]
    color = payload.color or colors[len(folders) % len(colors)]

    new_folder = {
        "id": f"f-{uuid4().hex[:8]}",
        "name": payload.name,
        "color": color,
        "createdAt": datetime.utcnow().isoformat(),
    }

    folders.append(new_folder)
    save_folders(folders)

    logger.info(f"Created folder: {new_folder['name']}")

    return NoteFolder(
        id=new_folder["id"],
        name=new_folder["name"],
        color=new_folder["color"],
        createdAt=datetime.fromisoformat(new_folder["createdAt"]),
    )


@router.put("/folders/{folder_id}", response_model=NoteFolder)
async def update_folder(folder_id: str, payload: UpdateFolderRequest, token: Optional[str] = None, state=Depends(get_app_state)):
    """Update a folder."""
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

    folders = load_folders()

    for folder in folders:
        if folder["id"] == folder_id:
            if payload.name is not None:
                folder["name"] = payload.name
            if payload.color is not None:
                folder["color"] = payload.color
            save_folders(folders)
            return NoteFolder(
                id=folder["id"],
                name=folder["name"],
                color=folder.get("color", "#2563eb"),
                createdAt=datetime.fromisoformat(folder["createdAt"]) if isinstance(folder.get("createdAt"), str) else datetime.utcnow(),
            )

    raise HTTPException(status_code=404, detail="Folder not found")


@router.delete("/folders/{folder_id}")
async def delete_folder(folder_id: str, token: Optional[str] = None, state=Depends(get_app_state)):
    """Delete a folder. Notes in the folder will become unorganized."""
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

    folders = load_folders()
    original_count = len(folders)
    folders = [f for f in folders if f["id"] != folder_id]

    if len(folders) == original_count:
        raise HTTPException(status_code=404, detail="Folder not found")

    save_folders(folders)

    # Move notes from deleted folder to unorganized
    notes = load_notes()
    for note in notes:
        if note.get("folderId") == folder_id:
            note["folderId"] = None
    save_notes(notes)

    logger.info(f"Deleted folder: {folder_id}")
    return {"status": "ok"}


# Note endpoints
@router.post("", response_model=SavedNote)
async def create_note(payload: CreateNoteRequest, token: Optional[str] = None, state=Depends(get_app_state)):
    """Create a new note."""
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

    notes = load_notes()
    now = datetime.utcnow()

    new_note = {
        "id": f"n-{uuid4().hex[:8]}",
        "title": payload.title,
        "createdAt": now.isoformat(),
        "updatedAt": now.isoformat(),
        "folderId": payload.folderId,
        "canvasData": payload.canvasData.model_dump() if payload.canvasData else {"strokes": [], "background": "plain"},
    }

    notes.insert(0, new_note)  # Add to beginning
    save_notes(notes)

    logger.info(f"Created note: {new_note['title']}")

    return SavedNote(
        id=new_note["id"],
        title=new_note["title"],
        createdAt=now,
        updatedAt=now,
        folderId=new_note["folderId"],
        canvasData=CanvasData(**new_note["canvasData"]),
    )


@router.get("/{note_id}", response_model=SavedNote)
async def get_note(note_id: str, token: Optional[str] = None, state=Depends(get_app_state)):
    """Get a single note by ID."""
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

    notes = load_notes()

    for note in notes:
        if note["id"] == note_id:
            return SavedNote(
                id=note["id"],
                title=note["title"],
                createdAt=datetime.fromisoformat(note["createdAt"]) if isinstance(note.get("createdAt"), str) else datetime.utcnow(),
                updatedAt=datetime.fromisoformat(note["updatedAt"]) if isinstance(note.get("updatedAt"), str) else datetime.utcnow(),
                folderId=note.get("folderId"),
                canvasData=CanvasData(**note.get("canvasData", {})),
            )

    raise HTTPException(status_code=404, detail="Note not found")


@router.put("/{note_id}", response_model=SavedNote)
async def update_note(note_id: str, payload: UpdateNoteRequest, token: Optional[str] = None, state=Depends(get_app_state)):
    """Update a note."""
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

    notes = load_notes()

    for note in notes:
        if note["id"] == note_id:
            if payload.title is not None:
                note["title"] = payload.title
            if payload.folderId is not None:
                note["folderId"] = payload.folderId if payload.folderId != "" else None
            if payload.canvasData is not None:
                note["canvasData"] = payload.canvasData.model_dump()
            note["updatedAt"] = datetime.utcnow().isoformat()

            save_notes(notes)

            return SavedNote(
                id=note["id"],
                title=note["title"],
                createdAt=datetime.fromisoformat(note["createdAt"]) if isinstance(note.get("createdAt"), str) else datetime.utcnow(),
                updatedAt=datetime.fromisoformat(note["updatedAt"]) if isinstance(note.get("updatedAt"), str) else datetime.utcnow(),
                folderId=note.get("folderId"),
                canvasData=CanvasData(**note.get("canvasData", {})),
            )

    raise HTTPException(status_code=404, detail="Note not found")


@router.delete("/{note_id}")
async def delete_note(note_id: str, token: Optional[str] = None, state=Depends(get_app_state)):
    """Delete a note."""
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

    notes = load_notes()
    original_count = len(notes)
    notes = [n for n in notes if n["id"] != note_id]

    if len(notes) == original_count:
        raise HTTPException(status_code=404, detail="Note not found")

    save_notes(notes)

    logger.info(f"Deleted note: {note_id}")
    return {"status": "ok"}
