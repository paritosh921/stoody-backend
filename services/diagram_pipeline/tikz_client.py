"""
Kimi + TikZ Diagram Generation Client

Generates exam-style diagrams by:
1. Asking Kimi to produce TikZ/LaTeX source
2. Compiling with a local TeX engine
3. Converting PDF output to PNG

This avoids Gemini image-generation API costs.
"""

import asyncio
import base64
import logging
import os
import re
import glob
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

from .kimi_client import KimiClient
from .models import DiagramImage

logger = logging.getLogger(__name__)

IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "images"))
DIAGRAMS_SUBDIR = "diagrams"


class TikzImageClient:
    """Generate black-and-white diagram PNGs via Kimi-authored TikZ."""

    SYSTEM_PROMPT = """You are an expert technical illustrator using TikZ.
Return ONLY compilable LaTeX code.

REQUIRED OUTPUT:
- A COMPLETE standalone .tex document.
- Must include \\documentclass[tikz,border=2pt]{standalone}
- Must include \\begin{document} ... \\end{document}
- Must include exactly one \\begin{tikzpicture} ... \\end{tikzpicture}

STYLE RULES:
- Black lines on white background.
- No color fills, no gradients, no decorative art.
- Exam-quality clarity for labels and numeric values.
- Keep code deterministic and concise.
- Use only standard TikZ libraries likely available by default.

IMPORTANT:
- Do not include markdown fences.
- Do not include explanations.
- Output only LaTeX source."""

    REPAIR_PROMPT = """You are fixing a broken TikZ LaTeX file.
Return ONLY a corrected standalone LaTeX document.
No markdown, no explanations."""

    def __init__(self):
        self.kimi = KimiClient()
        self.api_key = self.kimi.api_key
        self.model = os.getenv("DIAGRAM_IMAGE_MODEL", "kimi-k2.5-tikz")
        self.api_url = "local-tikz-renderer"
        self.images_dir = IMAGES_DIR / DIAGRAMS_SUBDIR
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.is_windows = os.name == "nt"

        self.latex_engine = self._resolve_override(os.getenv("TIKZ_LATEX_ENGINE")) or self._pick_latex_engine()
        self.pdf_renderer = self._resolve_override(os.getenv("TIKZ_PDF_RENDERER")) or self._pick_pdf_renderer()

        if not self.latex_engine:
            logger.warning("No LaTeX engine found (pdflatex/xelatex/lualatex/tectonic)")
        if not self.pdf_renderer:
            logger.warning("No PDF-to-PNG renderer found (pdftoppm/magick/convert)")
        if self.latex_engine or self.pdf_renderer:
            logger.info(
                f"[TIKZ] Using LaTeX engine={self.latex_engine or 'None'} "
                f"renderer={self.pdf_renderer or 'None'}"
            )
        if not self.api_key:
            logger.warning("KIMI_API_KEY not set - TikZ diagram generation will not work")

    def _get_safe_temp_base(self) -> Optional[str]:
        """Get a temp directory path without spaces for pdflatex compatibility on Windows."""
        if not self.is_windows:
            return None  # Use default temp on Linux/Mac
        
        # Try paths without spaces in order of preference
        candidates = [
            os.getenv("TIKZ_TEMP_DIR"),  # User override
            "C:\\Temp",
            "D:\\Temp",
            os.path.join(os.environ.get("SystemDrive", "C:"), "Temp"),
        ]
        
        for candidate in candidates:
            if not candidate:
                continue
            # Skip if path contains spaces
            if " " in candidate:
                continue
            try:
                path = Path(candidate)
                path.mkdir(parents=True, exist_ok=True)
                if path.exists() and os.access(str(path), os.W_OK):
                    return str(path)
            except Exception:
                continue
        
        # Fallback: try to get short path name of default temp
        try:
            import ctypes
            default_temp = tempfile.gettempdir()
            buf = ctypes.create_unicode_buffer(512)
            ctypes.windll.kernel32.GetShortPathNameW(default_temp, buf, 512)
            short_path = buf.value
            if short_path and " " not in short_path:
                return short_path
        except Exception:
            pass
        
        return None

    def _resolve_override(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None

        candidates = [item.strip() for item in value.split(";") if item.strip()]
        for override in candidates:
            resolved = self._resolve_override_candidate(override)
            if resolved:
                return resolved
        return None

    def _resolve_override_candidate(self, override: str) -> Optional[str]:
        if self.is_windows:
            override = override.strip().strip('"')

            mnt_match = re.match(r"^/mnt/([A-Za-z])/(.+)", override)
            if mnt_match:
                drive = mnt_match.group(1).upper()
                tail = mnt_match.group(2).replace("/", "\\")
                win_path = f"{drive}:\\{tail}"
                if Path(win_path).exists():
                    return win_path

            if Path(override).exists():
                return override

            resolved = shutil.which(override)
            if resolved:
                return resolved

            win_match = re.match(r"^[A-Za-z]:[\\/]", override)
            if win_match and Path(override).exists():
                return override

            return None

        win_match = re.match(r"^[A-Za-z]:[\\/]", override)
        if win_match:
            drive = override[0].lower()
            tail = override[2:].replace("\\", "/").lstrip("/")
            wsl_path = f"/mnt/{drive}/{tail}"
            if Path(wsl_path).exists():
                return wsl_path
            return None

        if Path(override).exists():
            return override

        resolved = shutil.which(override)
        if resolved:
            return resolved

        return None

    def _find_windows_binary(self, patterns: list) -> Optional[str]:
        for pattern in patterns:
            for match in glob.glob(pattern, recursive=True):
                if match and os.path.isfile(match):
                    return match
        return None

    def _pick_latex_engine(self) -> Optional[str]:
        for cmd in ("pdflatex", "xelatex", "lualatex", "tectonic"):
            resolved = shutil.which(cmd)
            if resolved:
                return resolved

        if self.is_windows:
            windows_patterns = [
                r"C:\Program Files\MiKTeX*\miktex\bin\**\pdflatex.exe",
                r"C:\Users\*\AppData\Local\Programs\MiKTeX*\miktex\bin\**\pdflatex.exe",
                r"C:\texlive\*\bin\win32\pdflatex.exe",
                r"C:\texlive\*\bin\win64\pdflatex.exe",
            ]
            found = self._find_windows_binary(windows_patterns)
            if found:
                return found
        else:
            windows_patterns = [
                "/mnt/c/Program Files/MiKTeX*/miktex/bin/*/pdflatex.exe",
                "/mnt/c/Program Files/MiKTeX*/miktex/bin/pdflatex.exe",
                "/mnt/c/Users/*/AppData/Local/Programs/MiKTeX*/miktex/bin/*/pdflatex.exe",
                "/mnt/c/texlive/*/bin/win32/pdflatex.exe",
                "/mnt/c/texlive/*/bin/win64/pdflatex.exe",
            ]
            found = self._find_windows_binary(windows_patterns)
            if found:
                return found

        # WSL/Linux fallbacks if PATH is not populated correctly.
        for cmd in ("/usr/bin/pdflatex", "/usr/bin/xelatex", "/usr/bin/lualatex", "/usr/bin/tectonic"):
            if Path(cmd).exists():
                return cmd

        return None

    def _is_imagemagick_binary(self, cmd: str) -> bool:
        """Validate that `magick/convert` is ImageMagick, not Windows built-in convert.exe."""
        resolved = shutil.which(cmd)
        if not resolved and Path(cmd).exists():
            resolved = cmd
        if not resolved:
            return False

        resolved_norm = resolved.replace("\\", "/").lower()
        if cmd == "convert" and "windows/system32/convert" in resolved_norm:
            return False

        try:
            check = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            output = f"{check.stdout}\n{check.stderr}"
            return "imagemagick" in output.lower()
        except Exception:
            return False

    def _pick_pdf_renderer(self) -> Optional[str]:
        resolved = shutil.which("pdftoppm")
        if resolved:
            return resolved

        if self.is_windows:
            poppler_patterns = [
                r"C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler*\poppler-*\Library\bin\pdftoppm.exe",
                r"C:\Program Files\poppler*\Library\bin\pdftoppm.exe",
            ]
            poppler_path = self._find_windows_binary(poppler_patterns)
            if poppler_path:
                return poppler_path
        else:
            poppler_patterns = [
                "/mnt/c/Users/*/AppData/Local/Microsoft/WinGet/Packages/oschwartz10612.Poppler*/poppler-*/Library/bin/pdftoppm.exe",
                "/mnt/c/Program Files/poppler*/Library/bin/pdftoppm.exe",
            ]
            poppler_path = self._find_windows_binary(poppler_patterns)
            if poppler_path:
                return poppler_path

        # WSL/Linux fallbacks if PATH is not populated correctly.
        for cmd in ("/usr/bin/pdftoppm", "/usr/local/bin/pdftoppm"):
            if Path(cmd).exists():
                return cmd

        if self._is_imagemagick_binary("magick"):
            return "magick"

        if self.is_windows:
            magick_patterns = [
                r"C:\Program Files\ImageMagick-*\magick.exe",
                r"C:\Program Files\ImageMagick*\magick.exe",
                r"C:\Program Files\Imagemagick*\magick.exe",
            ]
            magick_path = self._find_windows_binary(magick_patterns)
            if magick_path and self._is_imagemagick_binary(magick_path):
                return magick_path
        else:
            magick_patterns = [
                "/mnt/c/Program Files/ImageMagick-*/magick.exe",
                "/mnt/c/Program Files/ImageMagick*/magick.exe",
                "/mnt/c/Program Files/Imagemagick*/magick.exe",
            ]
            magick_path = self._find_windows_binary(magick_patterns)
            if magick_path and self._is_imagemagick_binary(magick_path):
                return magick_path

        if self._is_imagemagick_binary("convert"):
            return "convert"

        return None

    def _resolution_to_dpi(self, resolution: str) -> int:
        value = (resolution or "2K").upper()
        if value == "1K":
            return 160
        if value == "4K":
            return 320
        return 220

    def _strip_markdown_fences(self, text: str) -> str:
        cleaned = (text or "").strip()
        fence_match = re.search(r"```(?:latex)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()
        return cleaned

    def _wrap_tikz_document(self, tikz_body: str) -> str:
        return (
            "\\documentclass[tikz,border=2pt]{standalone}\n"
            "\\usepackage{tikz}\n"
            "\\usetikzlibrary{arrows.meta,angles,quotes,calc,positioning}\n"
            "\\begin{document}\n"
            f"{tikz_body.strip()}\n"
            "\\end{document}\n"
        )

    def _extract_latex_document(self, raw_text: str) -> str:
        text = self._strip_markdown_fences(raw_text)

        doc_match = re.search(
            r"(\\documentclass[\s\S]*\\end\{document\})",
            text,
            re.IGNORECASE,
        )
        if doc_match:
            return doc_match.group(1).strip()

        tikz_match = re.search(
            r"(\\begin\{tikzpicture\}[\s\S]*\\end\{tikzpicture\})",
            text,
            re.IGNORECASE,
        )
        if tikz_match:
            return self._wrap_tikz_document(tikz_match.group(1))

        # Fallback placeholder if response is malformed.
        fallback_tikz = (
            "\\begin{tikzpicture}[line width=0.9pt]\n"
            "\\draw (0,0) rectangle (8,4);\n"
            "\\node at (4,2) {Diagram};\n"
            "\\end{tikzpicture}"
        )
        return self._wrap_tikz_document(fallback_tikz)

    async def _run_command(self, cmd: list, cwd: Path) -> Tuple[int, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            output = (stdout or b"").decode("utf-8", errors="ignore") + "\n" + (stderr or b"").decode("utf-8", errors="ignore")
            return process.returncode, output.strip()
        except NotImplementedError:
            logger.warning("[TIKZ] Async subprocess not supported; falling back to sync execution")
        except Exception as e:
            logger.warning(f"[TIKZ] Async subprocess failed; falling back to sync execution: {e!r}")

        return await asyncio.to_thread(self._run_command_sync, cmd, cwd)

    def _run_command_sync(self, cmd: list, cwd: Path) -> Tuple[int, str]:
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                check=False,
            )
            output = (completed.stdout or "") + "\n" + (completed.stderr or "")
            return completed.returncode, output.strip()
        except Exception as e:
            return 1, f"Command failed: {e!r}"

    async def _compile_latex(self, tex_path: Path, temp_dir: Path) -> Tuple[bool, str]:
        if self.latex_engine == "tectonic":
            cmd = [
                "tectonic",
                str(tex_path),
                "--outdir",
                str(temp_dir),
                "--keep-logs",
                "--keep-intermediates",
            ]
        else:
            cmd = [
                self.latex_engine or "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(temp_dir),
                str(tex_path),
            ]

        code, log_text = await self._run_command(cmd, cwd=temp_dir)
        return code == 0, log_text

    async def _convert_pdf_to_png(self, pdf_path: Path, png_path: Path, dpi: int, temp_dir: Path) -> Tuple[bool, str]:
        renderer = self.pdf_renderer or ""
        renderer_name = Path(renderer).name.lower()

        if renderer_name in {"pdftoppm", "pdftoppm.exe"} or renderer == "pdftoppm":
            prefix = png_path.with_suffix("")
            cmd = [
                renderer or "pdftoppm",
                "-singlefile",
                "-png",
                "-r",
                str(dpi),
                str(pdf_path),
                str(prefix),
            ]
            code, output = await self._run_command(cmd, cwd=temp_dir)
            rendered = prefix.with_suffix(".png")
            if code == 0 and rendered.exists():
                if rendered != png_path:
                    rendered.replace(png_path)
                return True, output
            return False, output

        if renderer_name in {"magick", "magick.exe", "convert", "convert.exe"} or renderer in {"magick", "convert"}:
            cmd = [
                renderer or "magick",
                "-density",
                str(dpi),
                str(pdf_path),
                "-background",
                "white",
                "-alpha",
                "remove",
                "-alpha",
                "off",
                str(png_path),
            ]
            code, output = await self._run_command(cmd, cwd=temp_dir)
            return code == 0 and png_path.exists(), output

        return False, "No supported PDF renderer available"

    async def _generate_latex_document(self, instructions: str) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Create a precise exam-style diagram for the following specification.\n\n"
                    f"{instructions}\n\n"
                    "Return only standalone LaTeX source."
                ),
            },
        ]
        raw_response = await self.kimi._call_api(messages, max_tokens=2600)
        return self._extract_latex_document(raw_response)

    async def _repair_latex_document(self, current_doc: str, error_log: str, instructions: str) -> str:
        error_excerpt = (error_log or "")[-3000:]
        messages = [
            {"role": "system", "content": self.REPAIR_PROMPT},
            {
                "role": "user",
                "content": (
                    "Diagram intent:\n"
                    f"{instructions}\n\n"
                    "Current LaTeX:\n"
                    f"{current_doc}\n\n"
                    "Compiler/conversion errors:\n"
                    f"{error_excerpt}\n\n"
                    "Fix and return corrected standalone LaTeX only."
                ),
            },
        ]
        repaired = await self.kimi._call_api(messages, max_tokens=3000)
        return self._extract_latex_document(repaired)

    def _get_image_dimensions(self, image_bytes: bytes) -> Tuple[Optional[int], Optional[int]]:
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(image_bytes))
            return img.size
        except Exception as e:
            logger.warning(f"Could not determine diagram dimensions: {e}")
            return (None, None)

    async def generate_diagram(
        self,
        instructions: str,
        question_id: str,
        instructions_version: int = 1,
        aspect_ratio: str = "1:1",
        resolution: str = "2K",
    ) -> DiagramImage:
        """Generate a diagram image from instructions via Kimi -> TikZ -> PNG."""
        if not self.api_key:
            raise ValueError("KIMI_API_KEY not configured")
        if not self.latex_engine:
            raise RuntimeError(
                "No TeX engine found. Install MiKTeX/TeX Live (pdflatex) or tectonic to render TikZ diagrams."
            )
        if not self.pdf_renderer:
            raise RuntimeError(
                "No PDF-to-PNG renderer found. Install poppler (pdftoppm) or ImageMagick (magick)."
            )

        render_attempts = int(os.getenv("TIKZ_MAX_RENDER_ATTEMPTS", "3"))
        render_attempts = max(1, min(5, render_attempts))
        dpi = self._resolution_to_dpi(resolution)

        image_id = str(uuid.uuid4())[:8]
        filename = f"diagram_{question_id}_{instructions_version}_{image_id}.png"
        final_png_path = self.images_dir / filename
        relative_path = f"{DIAGRAMS_SUBDIR}/{filename}"

        latex_doc = await self._generate_latex_document(instructions)
        logger.info(
            f"[TIKZ] Rendering diagram for question={question_id}, "
            f"engine={self.latex_engine}, renderer={self.pdf_renderer}, dpi={dpi}, aspect_ratio={aspect_ratio}"
        )

        last_error = ""
        safe_temp_base = self._get_safe_temp_base()
        if safe_temp_base:
            logger.info(f"[TIKZ] Using safe temp directory: {safe_temp_base}")
        with tempfile.TemporaryDirectory(prefix="tikz_render_", dir=safe_temp_base) as tmp:
            temp_dir = Path(tmp)
            logger.debug(f"[TIKZ] Temp directory created: {temp_dir}")
            tex_path = temp_dir / "diagram.tex"
            pdf_path = temp_dir / "diagram.pdf"
            png_path = temp_dir / "diagram.png"

            for attempt in range(1, render_attempts + 1):
                tex_path.write_text(latex_doc, encoding="utf-8")

                compiled, compile_log = await self._compile_latex(tex_path, temp_dir)
                if not compiled or not pdf_path.exists():
                    last_error = f"LaTeX compile failed (attempt {attempt}): {compile_log[-1800:]}"
                    logger.warning(f"[TIKZ] {last_error}")
                else:
                    converted, convert_log = await self._convert_pdf_to_png(pdf_path, png_path, dpi, temp_dir)
                    if converted and png_path.exists():
                        image_bytes = png_path.read_bytes()
                        final_png_path.write_bytes(image_bytes)

                        if os.getenv("TIKZ_SAVE_LATEX_SOURCE", "true").lower() not in {"0", "false", "no"}:
                            tex_out = final_png_path.with_suffix(".tex")
                            tex_out.write_text(latex_doc, encoding="utf-8")

                        width, height = self._get_image_dimensions(image_bytes)
                        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                        return DiagramImage(
                            id=image_id,
                            path=relative_path,
                            filename=filename,
                            format="png",
                            width=width,
                            height=height,
                            size_bytes=len(image_bytes),
                            instructions_version=instructions_version,
                            base64_data=image_b64,
                        )

                    last_error = f"PDF->PNG conversion failed (attempt {attempt}): {convert_log[-1800:]}"
                    logger.warning(f"[TIKZ] {last_error}")

                if attempt < render_attempts:
                    try:
                        latex_doc = await self._repair_latex_document(latex_doc, last_error, instructions)
                    except Exception as repair_error:
                        logger.warning(f"[TIKZ] Failed to repair LaTeX on attempt {attempt}: {repair_error}")

        raise Exception(last_error or "TikZ rendering failed")

    async def describe_diagram(
        self,
        image_b64: str,
        question_text: str,
        instructions: str,
    ) -> str:
        """No vision model used in TikZ mode; keep interface compatibility."""
        return ""

    async def get_image_base64(self, image_path: str) -> Optional[str]:
        full_path = IMAGES_DIR / image_path
        if not full_path.exists():
            full_path = IMAGES_DIR.parent / image_path
        if not full_path.exists():
            return None
        return base64.b64encode(full_path.read_bytes()).decode("utf-8")

    def delete_image(self, image_path: str) -> bool:
        full_path = IMAGES_DIR / image_path
        if not full_path.exists():
            full_path = IMAGES_DIR.parent / image_path
        if full_path.exists():
            full_path.unlink()
            return True
        return False
