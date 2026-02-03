"""
Paper Assembly Service - PDF Generation for Question Papers

This service handles:
1. Organizing questions into structured paper sections
2. Generating question paper PDFs
3. Generating answer key PDFs
4. Generating detailed marking scheme PDFs
"""

import io
import logging
import os
import base64
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .models.config import PaperConfig, QuestionGenerationConfig
from .models.question import GeneratedQuestion, QuestionOption, MarkingStep
from .models.paper import GeneratedPaper, PaperSection, PaperStatus


logger = logging.getLogger(__name__)

# Singleton instance
_paper_assembly_service: Optional["PaperAssemblyService"] = None


# ============================================================================
# PDF Styles Configuration
# ============================================================================

def get_pdf_styles() -> Dict[str, ParagraphStyle]:
    """Get custom PDF styles for question papers."""
    base_styles = getSampleStyleSheet()
    
    styles = {
        # Header styles
        "SchoolName": ParagraphStyle(
            "SchoolName",
            parent=base_styles["Heading1"],
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=6,
            textColor=colors.darkblue,
        ),
        "ExamTitle": ParagraphStyle(
            "ExamTitle",
            parent=base_styles["Heading2"],
            fontSize=14,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=colors.black,
        ),
        "SubjectGrade": ParagraphStyle(
            "SubjectGrade",
            parent=base_styles["Normal"],
            fontSize=12,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "MetaInfo": ParagraphStyle(
            "MetaInfo",
            parent=base_styles["Normal"],
            fontSize=10,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=colors.gray,
        ),
        
        # Section styles
        "SectionHeader": ParagraphStyle(
            "SectionHeader",
            parent=base_styles["Heading2"],
            fontSize=12,
            alignment=TA_LEFT,
            spaceBefore=18,
            spaceAfter=6,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=4,
        ),
        "SectionInstructions": ParagraphStyle(
            "SectionInstructions",
            parent=base_styles["Normal"],
            fontSize=10,
            alignment=TA_LEFT,
            spaceAfter=12,
            textColor=colors.gray,
            fontName="Helvetica-Oblique",
        ),
        
        # Question styles
        "QuestionNumber": ParagraphStyle(
            "QuestionNumber",
            parent=base_styles["Normal"],
            fontSize=11,
            alignment=TA_LEFT,
            fontName="Helvetica-Bold",
        ),
        "QuestionText": ParagraphStyle(
            "QuestionText",
            parent=base_styles["Normal"],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            leftIndent=24,
        ),
        "OptionText": ParagraphStyle(
            "OptionText",
            parent=base_styles["Normal"],
            fontSize=10,
            alignment=TA_LEFT,
            leftIndent=36,
            spaceAfter=3,
        ),
        "MarksText": ParagraphStyle(
            "MarksText",
            parent=base_styles["Normal"],
            fontSize=9,
            alignment=TA_RIGHT,
            textColor=colors.gray,
        ),
        
        # Answer key styles
        "AnswerHeader": ParagraphStyle(
            "AnswerHeader",
            parent=base_styles["Heading3"],
            fontSize=11,
            alignment=TA_LEFT,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.darkgreen,
        ),
        "AnswerText": ParagraphStyle(
            "AnswerText",
            parent=base_styles["Normal"],
            fontSize=10,
            alignment=TA_JUSTIFY,
            leftIndent=24,
            spaceAfter=6,
        ),
        "SolutionText": ParagraphStyle(
            "SolutionText",
            parent=base_styles["Normal"],
            fontSize=10,
            alignment=TA_JUSTIFY,
            leftIndent=24,
            spaceAfter=12,
            textColor=colors.darkblue,
        ),
        
        # Marking scheme styles
        "MarkingStep": ParagraphStyle(
            "MarkingStep",
            parent=base_styles["Normal"],
            fontSize=10,
            alignment=TA_LEFT,
            leftIndent=36,
            spaceAfter=3,
        ),
        "MarkingMarks": ParagraphStyle(
            "MarkingMarks",
            parent=base_styles["Normal"],
            fontSize=10,
            alignment=TA_RIGHT,
            textColor=colors.darkgreen,
        ),
        
        # Instructions
        "Instructions": ParagraphStyle(
            "Instructions",
            parent=base_styles["Normal"],
            fontSize=10,
            alignment=TA_LEFT,
            leftIndent=12,
            spaceAfter=3,
        ),
        "InstructionsBold": ParagraphStyle(
            "InstructionsBold",
            parent=base_styles["Normal"],
            fontSize=11,
            alignment=TA_LEFT,
            fontName="Helvetica-Bold",
            spaceBefore=12,
            spaceAfter=6,
        ),
        
        # Footer
        "Footer": ParagraphStyle(
            "Footer",
            parent=base_styles["Normal"],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.gray,
        ),
        "PageNumber": ParagraphStyle(
            "PageNumber",
            parent=base_styles["Normal"],
            fontSize=9,
            alignment=TA_CENTER,
        ),
    }
    
    return styles


class PaperAssemblyError(Exception):
    """Exception for paper assembly errors."""
    pass


class PaperAssemblyService:
    """
    Service for assembling questions into formatted papers and generating PDFs.
    
    Features:
    - Organize questions into sections by type
    - Generate professional question paper PDFs
    - Generate answer key PDFs
    - Generate detailed marking scheme PDFs
    - Support for diagrams and mathematical formulas
    """
    
    def __init__(self):
        self._initialized = False
        self._styles = None
        self._page_size = A4
        self._margins = {
            "left": 20 * mm,
            "right": 20 * mm,
            "top": 25 * mm,
            "bottom": 25 * mm,
        }
        self._s3_service = None
    
    async def initialize(self) -> bool:
        """Initialize the service."""
        if self._initialized:
            return True
        
        try:
            self._styles = get_pdf_styles()
            
            # Try to import S3 service for storing PDFs
            try:
                from utils.s3_storage import upload_file_to_s3, is_s3_enabled
                if is_s3_enabled():
                    self._s3_service = {"upload": upload_file_to_s3}
            except ImportError:
                logger.warning("S3 storage not available, PDFs will be returned as bytes only")
            
            self._initialized = True
            logger.info("PaperAssemblyService initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PaperAssemblyService: {e}")
            raise PaperAssemblyError(f"Initialization failed: {e}")
    
    async def assemble_paper(
        self,
        paper: GeneratedPaper,
        generate_pdfs: bool = True,
        upload_to_s3: bool = True,
    ) -> GeneratedPaper:
        """
        Assemble a complete paper with PDFs.
        
        Args:
            paper: GeneratedPaper with sections and questions
            generate_pdfs: Whether to generate PDF files
            upload_to_s3: Whether to upload PDFs to S3
            
        Returns:
            Updated GeneratedPaper with PDF URLs
        """
        await self.initialize()
        
        try:
            if generate_pdfs:
                # Generate question paper PDF
                question_paper_bytes = await self.generate_question_paper_pdf(paper)
                
                # Generate answer key PDF
                answer_key_bytes = await self.generate_answer_key_pdf(paper)
                
                # Generate marking scheme PDF
                marking_scheme_bytes = await self.generate_marking_scheme_pdf(paper)
                
                if upload_to_s3 and self._s3_service:
                    # Upload to S3
                    paper.question_paper_url = await self._upload_pdf(
                        question_paper_bytes,
                        paper.paper_id,
                        "question_paper",
                        paper.tenant_id,
                    )
                    paper.answer_key_url = await self._upload_pdf(
                        answer_key_bytes,
                        paper.paper_id,
                        "answer_key",
                        paper.tenant_id,
                    )
                    paper.marking_scheme_url = await self._upload_pdf(
                        marking_scheme_bytes,
                        paper.paper_id,
                        "marking_scheme",
                        paper.tenant_id,
                    )
                
                paper.status = PaperStatus.COMPLETED
                paper.completed_at = datetime.utcnow()
            
            return paper
            
        except Exception as e:
            paper.status = PaperStatus.FAILED
            paper.error_message = str(e)
            logger.error(f"Paper assembly failed: {e}", exc_info=True)
            raise PaperAssemblyError(f"Assembly failed: {e}")
    
    async def generate_question_paper_pdf(
        self,
        paper: GeneratedPaper,
    ) -> bytes:
        """
        Generate question paper PDF.
        
        Args:
            paper: GeneratedPaper with sections and questions
            
        Returns:
            PDF as bytes
        """
        await self.initialize()
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self._page_size,
            leftMargin=self._margins["left"],
            rightMargin=self._margins["right"],
            topMargin=self._margins["top"],
            bottomMargin=self._margins["bottom"],
        )
        
        story = []
        
        # Add header
        story.extend(self._build_paper_header(paper))
        
        # Add general instructions
        story.extend(self._build_instructions(paper))
        
        # Add sections with questions
        question_number = 1
        for section in paper.sections:
            section_elements, question_number = self._build_section_questions(
                section, question_number, include_answers=False
            )
            story.extend(section_elements)
        
        # Add footer with "All the Best!"
        story.append(Spacer(1, 24))
        story.append(Paragraph(
            "*** All the Best! ***",
            self._styles["MetaInfo"]
        ))
        
        # Build PDF
        doc.build(story, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)
        
        buffer.seek(0)
        return buffer.getvalue()
    
    async def generate_answer_key_pdf(
        self,
        paper: GeneratedPaper,
    ) -> bytes:
        """
        Generate answer key PDF.
        
        Args:
            paper: GeneratedPaper with sections and questions
            
        Returns:
            PDF as bytes
        """
        await self.initialize()
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self._page_size,
            leftMargin=self._margins["left"],
            rightMargin=self._margins["right"],
            topMargin=self._margins["top"],
            bottomMargin=self._margins["bottom"],
        )
        
        story = []
        
        # Add header
        story.extend(self._build_answer_key_header(paper))
        
        # Add sections with answers
        question_number = 1
        for section in paper.sections:
            section_elements, question_number = self._build_section_answers(
                section, question_number
            )
            story.extend(section_elements)
        
        # Build PDF
        doc.build(story, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)
        
        buffer.seek(0)
        return buffer.getvalue()
    
    async def generate_marking_scheme_pdf(
        self,
        paper: GeneratedPaper,
    ) -> bytes:
        """
        Generate detailed marking scheme PDF.
        
        Args:
            paper: GeneratedPaper with sections and questions
            
        Returns:
            PDF as bytes
        """
        await self.initialize()
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self._page_size,
            leftMargin=self._margins["left"],
            rightMargin=self._margins["right"],
            topMargin=self._margins["top"],
            bottomMargin=self._margins["bottom"],
        )
        
        story = []
        
        # Add header
        story.extend(self._build_marking_scheme_header(paper))
        
        # Add sections with marking schemes
        question_number = 1
        for section in paper.sections:
            section_elements, question_number = self._build_section_marking_scheme(
                section, question_number
            )
            story.extend(section_elements)
        
        # Build PDF
        doc.build(story, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def _build_paper_header(self, paper: GeneratedPaper) -> List:
        """Build the paper header elements."""
        elements = []
        config = paper.paper_config
        
        # School name (if provided)
        if config and config.school_name:
            elements.append(Paragraph(
                config.school_name,
                self._styles["SchoolName"]
            ))
        
        # Exam title
        title = paper.title
        if config and config.exam_name:
            title = f"{config.exam_name}: {title}"
        elements.append(Paragraph(title, self._styles["ExamTitle"]))
        
        # Subject and grade
        elements.append(Paragraph(
            f"Subject: {paper.subject} | Class: {paper.grade}",
            self._styles["SubjectGrade"]
        ))
        
        # Meta info (date, duration, marks)
        meta_parts = []
        if config and config.exam_date:
            meta_parts.append(f"Date: {config.exam_date}")
        meta_parts.append(f"Duration: {paper.duration_minutes} minutes")
        meta_parts.append(f"Maximum Marks: {paper.total_marks}")
        
        elements.append(Paragraph(
            " | ".join(meta_parts),
            self._styles["MetaInfo"]
        ))
        
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_answer_key_header(self, paper: GeneratedPaper) -> List:
        """Build the answer key header."""
        elements = []
        config = paper.paper_config
        
        if config and config.school_name:
            elements.append(Paragraph(
                config.school_name,
                self._styles["SchoolName"]
            ))
        
        elements.append(Paragraph(
            f"ANSWER KEY: {paper.title}",
            self._styles["ExamTitle"]
        ))
        
        elements.append(Paragraph(
            f"Subject: {paper.subject} | Class: {paper.grade}",
            self._styles["SubjectGrade"]
        ))
        
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_marking_scheme_header(self, paper: GeneratedPaper) -> List:
        """Build the marking scheme header."""
        elements = []
        config = paper.paper_config
        
        if config and config.school_name:
            elements.append(Paragraph(
                config.school_name,
                self._styles["SchoolName"]
            ))
        
        elements.append(Paragraph(
            f"MARKING SCHEME: {paper.title}",
            self._styles["ExamTitle"]
        ))
        
        elements.append(Paragraph(
            f"Subject: {paper.subject} | Class: {paper.grade} | Total Marks: {paper.total_marks}",
            self._styles["SubjectGrade"]
        ))
        
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_instructions(self, paper: GeneratedPaper) -> List:
        """Build general instructions section."""
        elements = []
        config = paper.paper_config
        
        elements.append(Paragraph(
            "General Instructions:",
            self._styles["InstructionsBold"]
        ))
        
        instructions = []
        if config and config.general_instructions:
            instructions = config.general_instructions
        else:
            instructions = [
                "All questions are compulsory.",
                "Write your answers clearly and legibly.",
                "Show all working for numerical problems.",
                "Diagrams should be drawn with pencil.",
                f"This paper consists of {paper.total_questions} questions.",
            ]
        
        for i, instruction in enumerate(instructions, 1):
            elements.append(Paragraph(
                f"{i}. {instruction}",
                self._styles["Instructions"]
            ))
        
        elements.append(Spacer(1, 18))
        
        return elements
    
    def _build_section_questions(
        self,
        section: PaperSection,
        start_number: int,
        include_answers: bool = False,
    ) -> Tuple[List, int]:
        """Build section with questions."""
        elements = []
        question_number = start_number
        
        # Section header
        elements.append(Paragraph(
            section.name,
            self._styles["SectionHeader"]
        ))
        
        # Section instructions
        if section.instructions:
            elements.append(Paragraph(
                section.instructions,
                self._styles["SectionInstructions"]
            ))
        
        # Questions
        for question in section.questions:
            q_elements = self._build_question(
                question, question_number, include_answers
            )
            elements.extend(q_elements)
            question_number += 1
        
        return elements, question_number
    
    def _fetch_and_embed_image(
        self,
        image_url: str,
        max_width: float = 4 * inch,
        max_height: float = 2.5 * inch,
    ) -> Optional[Image]:
        """
        Fetch image from URL or file path and create ReportLab Image element.

        Handles:
        - Local file paths (uploads/..., /absolute/path)
        - HTTP/HTTPS URLs
        - Base64 data URIs
        - S3 paths

        Args:
            image_url: URL, file path, or base64 data URI
            max_width: Maximum width for the image
            max_height: Maximum height for the image

        Returns:
            ReportLab Image object or None if fetch fails
        """
        try:
            image_data = None

            # Case 1: Base64 data URI
            if image_url.startswith('data:image'):
                logger.info(f"Loading base64 image")
                base64_data = image_url.split(',')[1]
                image_data = io.BytesIO(base64.b64decode(base64_data))

            # Case 2: HTTP/HTTPS URL
            elif image_url.startswith('http://') or image_url.startswith('https://'):
                logger.info(f"Fetching image from URL: {image_url[:100]}...")
                try:
                    response = requests.get(image_url, timeout=15)
                    if response.status_code == 200:
                        image_data = io.BytesIO(response.content)
                    else:
                        logger.warning(f"Failed to fetch image: HTTP {response.status_code}")
                except requests.RequestException as e:
                    logger.warning(f"Failed to fetch image from URL: {e}")

            # Case 3: Local file path
            else:
                # Handle relative and absolute paths
                if image_url.startswith('/') or os.path.isabs(image_url):
                    file_path = image_url
                elif image_url.startswith('uploads/'):
                    file_path = os.path.join(os.getcwd(), image_url)
                else:
                    # Try relative to current working directory
                    file_path = os.path.join(os.getcwd(), image_url)

                # Also check if it's an S3-style path stored locally
                if not os.path.exists(file_path) and 'uploads/' in image_url:
                    # Extract the path after uploads/
                    alt_path = os.path.join(os.getcwd(), 'uploads', image_url.split('uploads/')[-1])
                    if os.path.exists(alt_path):
                        file_path = alt_path

                if os.path.exists(file_path):
                    logger.info(f"Loading local image: {file_path}")
                    with open(file_path, 'rb') as f:
                        image_data = io.BytesIO(f.read())
                else:
                    logger.warning(f"Image file not found: {file_path}")

            # Create Image element if we have data
            if image_data:
                img = Image(image_data)

                # Get original dimensions
                orig_width = img.imageWidth
                orig_height = img.imageHeight

                if orig_width > 0 and orig_height > 0:
                    # Calculate aspect ratio
                    aspect = orig_width / orig_height

                    # Scale to fit within max bounds while preserving aspect ratio
                    new_width = orig_width
                    new_height = orig_height

                    if new_width > max_width:
                        new_width = max_width
                        new_height = max_width / aspect

                    if new_height > max_height:
                        new_height = max_height
                        new_width = max_height * aspect

                    img.drawWidth = new_width
                    img.drawHeight = new_height

                logger.info(f"Successfully loaded image ({img.drawWidth}x{img.drawHeight})")
                return img

        except Exception as e:
            logger.error(f"Failed to embed image {image_url[:100]}: {e}")

        return None

    def _build_question(
        self,
        question: GeneratedQuestion,
        number: int,
        include_answer: bool = False,
    ) -> List:
        """Build a single question element."""
        elements = []
        
        # Question number and text
        marks_text = f"[{question.marks} mark{'s' if question.marks > 1 else ''}]"
        
        # Create a table for question number, text, and marks
        question_text = self._escape_html(question.question_text)
        
        question_table_data = [[
            Paragraph(f"Q{number}.", self._styles["QuestionNumber"]),
            Paragraph(question_text, self._styles["QuestionText"]),
            Paragraph(marks_text, self._styles["MarksText"]),
        ]]
        
        question_table = Table(
            question_table_data,
            colWidths=[30, 400, 50]
        )
        question_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        
        elements.append(question_table)
        
        # Add diagram if present
        if question.has_diagram and question.diagram_url:
            elements.append(Spacer(1, 6))
            img_element = self._fetch_and_embed_image(question.diagram_url)
            if img_element:
                elements.append(img_element)
            else:
                # Fallback: show URL if image fetch fails
                elements.append(Paragraph(
                    f"[Image unavailable: {question.diagram_url}]",
                    self._styles["OptionText"]
                ))

        # Handle OCR-extracted figures (from uploaded PDFs)
        figure_refs = getattr(question, 'figure_refs', None) or question.metadata.get('figure_refs', []) if hasattr(question, 'metadata') else []
        if figure_refs:
            for fig_ref in figure_refs:
                elements.append(Spacer(1, 4))
                img_element = self._fetch_and_embed_image(fig_ref)
                if img_element:
                    elements.append(img_element)
        
        # Add options for MCQ
        if question.options:
            elements.append(Spacer(1, 6))
            for option in question.options:
                option_text = f"({option.label}) {self._escape_html(option.content)}"
                if include_answer and option.is_correct:
                    option_text = f"<b>{option_text} ✓</b>"
                elements.append(Paragraph(option_text, self._styles["OptionText"]))
        
        # Add answer if requested
        if include_answer and question.correct_answer:
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(
                f"<b>Answer:</b> {question.correct_answer}",
                self._styles["AnswerText"]
            ))
        
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_section_answers(
        self,
        section: PaperSection,
        start_number: int,
    ) -> Tuple[List, int]:
        """Build section with answers."""
        elements = []
        question_number = start_number
        
        # Section header
        elements.append(Paragraph(
            section.name,
            self._styles["SectionHeader"]
        ))
        
        # Answers for each question
        for question in section.questions:
            elements.extend(self._build_answer(question, question_number))
            question_number += 1
        
        return elements, question_number
    
    def _build_answer(
        self,
        question: GeneratedQuestion,
        number: int,
    ) -> List:
        """Build answer element for a question."""
        elements = []
        
        # Question reference
        q_text = question.question_text[:100] + "..." if len(question.question_text) > 100 else question.question_text
        elements.append(Paragraph(
            f"<b>Q{number}.</b> {self._escape_html(q_text)} [{question.marks} marks]",
            self._styles["AnswerHeader"]
        ))
        
        # Correct answer
        if question.correct_answer:
            elements.append(Paragraph(
                f"<b>Answer:</b> {question.correct_answer}",
                self._styles["AnswerText"]
            ))
        
        # Solution
        if question.solution:
            elements.append(Paragraph(
                f"<b>Solution:</b>",
                self._styles["AnswerText"]
            ))
            elements.append(Paragraph(
                self._escape_html(question.solution),
                self._styles["SolutionText"]
            ))
        
        # Solution steps
        if question.solution_steps:
            for i, step in enumerate(question.solution_steps, 1):
                elements.append(Paragraph(
                    f"Step {i}: {self._escape_html(step)}",
                    self._styles["MarkingStep"]
                ))
        
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_section_marking_scheme(
        self,
        section: PaperSection,
        start_number: int,
    ) -> Tuple[List, int]:
        """Build section with marking scheme."""
        elements = []
        question_number = start_number
        
        # Section header
        elements.append(Paragraph(
            section.name,
            self._styles["SectionHeader"]
        ))
        
        # Marking scheme for each question
        for question in section.questions:
            elements.extend(self._build_marking_scheme_question(question, question_number))
            question_number += 1
        
        return elements, question_number
    
    def _build_marking_scheme_question(
        self,
        question: GeneratedQuestion,
        number: int,
    ) -> List:
        """Build marking scheme for a question."""
        elements = []
        
        # Question reference
        q_text = question.question_text[:80] + "..." if len(question.question_text) > 80 else question.question_text
        elements.append(Paragraph(
            f"<b>Q{number}.</b> {self._escape_html(q_text)} <b>[Total: {question.marks} marks]</b>",
            self._styles["AnswerHeader"]
        ))
        
        # Marking steps
        if question.marking_scheme:
            # Create a table for marking steps
            table_data = [["Step", "Criteria", "Marks"]]
            
            for i, step in enumerate(question.marking_scheme, 1):
                criteria = step.criteria or step.step
                table_data.append([
                    str(i),
                    self._escape_html(criteria),
                    f"{step.marks}"
                ])
            
            # Add total row
            total_marks = sum(step.marks for step in question.marking_scheme)
            table_data.append(["", "<b>Total</b>", f"<b>{total_marks}</b>"])
            
            marking_table = Table(
                [[Paragraph(str(cell), self._styles["MarkingStep"]) for cell in row] for row in table_data],
                colWidths=[40, 350, 60]
            )
            marking_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (2, 0), (2, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, -1), (-1, -1), colors.lightyellow),
            ]))
            
            elements.append(marking_table)
        else:
            # Simple marking if no detailed scheme
            elements.append(Paragraph(
                f"Award {question.marks} mark(s) for correct answer.",
                self._styles["MarkingStep"]
            ))
            
            if question.solution:
                elements.append(Paragraph(
                    f"Expected answer: {self._escape_html(question.solution[:200])}",
                    self._styles["MarkingStep"]
                ))
        
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters for ReportLab."""
        if not text:
            return ""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
    
    def _add_page_number(self, canvas, doc):
        """Add page number to PDF."""
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(
            self._page_size[0] / 2,
            15 * mm,
            text
        )
        canvas.restoreState()
    
    async def _upload_pdf(
        self,
        pdf_bytes: bytes,
        paper_id: str,
        pdf_type: str,
        tenant_id: str,
    ) -> Optional[str]:
        """Upload PDF to S3 and return URL."""
        if not self._s3_service:
            return None
        
        try:
            filename = f"papers/{tenant_id}/{paper_id}_{pdf_type}.pdf"
            url = await self._s3_service["upload"](
                pdf_bytes,
                filename,
                content_type="application/pdf"
            )
            return url
        except Exception as e:
            logger.error(f"Failed to upload PDF to S3: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return {
            "service": "PaperAssemblyService",
            "initialized": self._initialized,
            "s3_available": self._s3_service is not None,
        }


# ============================================================================
# Module-level functions
# ============================================================================

async def get_paper_assembly_service() -> PaperAssemblyService:
    """Get the singleton PaperAssemblyService instance."""
    global _paper_assembly_service
    
    if _paper_assembly_service is None:
        _paper_assembly_service = PaperAssemblyService()
        await _paper_assembly_service.initialize()
    
    return _paper_assembly_service


def get_paper_assembly_service_sync() -> PaperAssemblyService:
    """Get the service without initialization (for dependency injection)."""
    global _paper_assembly_service
    
    if _paper_assembly_service is None:
        _paper_assembly_service = PaperAssemblyService()
    
    return _paper_assembly_service
