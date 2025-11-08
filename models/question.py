from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

@dataclass
class QuestionImage:
    """Data model for question images"""
    id: str
    filename: str
    path: str  # Relative path to stored image file
    description: str
    type: str
    base64Data: Optional[str] = None  # Only used for transfer, not stored
    bbox: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ChromaDB storage"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuestionImage':
        """Create instance from dictionary"""
        return cls(**data)

@dataclass
class QuestionOption:
    """Data model for enhanced question options"""
    id: str
    type: str  # 'text' | 'image'
    content: str  # For text: the text content, For image: base64 data or path
    label: Optional[str] = None  # Option label (A, B, C, D, etc.)
    description: Optional[str] = None  # Description for image options

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuestionOption':
        """Create instance from dictionary"""
        return cls(**data)

@dataclass
class Question:
    """Data model for questions"""
    id: str
    text: str
    subject: str
    difficulty: str  # 'easy' | 'medium' | 'hard'
    extractedAt: str  # ISO format datetime
    pdfSource: str
    images: List[QuestionImage]
    options: Optional[List[str]] = None  # Legacy support
    enhancedOptions: Optional[List[QuestionOption]] = None  # New enhanced options
    correctAnswer: Optional[str] = None
    document_type: Optional[str] = None  # Document type: "Practice Sets", "Test Series", "Chapter Notes"
    document_id: Optional[str] = None  # Document identifier
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ChromaDB storage"""
        data = asdict(self)
        # Convert image objects to dictionaries
        data['images'] = [img.to_dict() for img in self.images]
        # Convert enhanced option objects to dictionaries
        if self.enhancedOptions:
            data['enhancedOptions'] = [opt.to_dict() for opt in self.enhancedOptions]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        """Create instance from dictionary"""
        # Convert image dictionaries to objects
        if 'images' in data and data['images']:
            data['images'] = [QuestionImage.from_dict(img) for img in data['images']]
        else:
            data['images'] = []

        # Convert enhanced option dictionaries to objects
        if 'enhancedOptions' in data and data['enhancedOptions']:
            data['enhancedOptions'] = [QuestionOption.from_dict(opt) for opt in data['enhancedOptions']]
        elif 'enhancedOptions' not in data:
            data['enhancedOptions'] = None

        return cls(**data)

    def to_chromadb_format(self) -> tuple:
        """Convert to ChromaDB format (document, metadata, id)"""
        # The main text content for ChromaDB embedding
        document = f"{self.text} {self.subject}"
        if self.options:
            document += " " + " ".join(self.options)

        # Include enhanced options in the document for better search
        if self.enhancedOptions:
            for opt in self.enhancedOptions:
                if opt.type == 'text':
                    document += " " + opt.content
                if opt.description:
                    document += " " + opt.description

        # Extract document_type from metadata if not set directly
        doc_type = self.document_type
        if not doc_type and self.metadata:
            doc_type = self.metadata.get('document_type')

        # Metadata for ChromaDB filtering and retrieval
        metadata = {
            "subject": self.subject,
            "difficulty": self.difficulty,
            "extractedAt": self.extractedAt,
            "pdfSource": self.pdfSource,
            "hasImages": len(self.images) > 0,
            "imageCount": len(self.images),
            "optionCount": len(self.options) if self.options else 0,
            "enhancedOptionCount": len(self.enhancedOptions) if self.enhancedOptions else 0,
            "correctAnswer": self.correctAnswer or "",
            # CRITICAL: Add document_type as top-level metadata for filtering
            "document_type": doc_type or "Chapter Notes",  # Default to Chapter Notes for backward compatibility
            # Store serialized data for full reconstruction
            "fullData": json.dumps(self.to_dict())
        }

        return document, metadata, self.id

    @classmethod
    def from_chromadb_result(cls, document: str, metadata: Dict[str, Any], id: str) -> 'Question':
        """Create instance from ChromaDB result"""
        # Reconstruct from stored full data
        if 'fullData' in metadata:
            full_data = json.loads(metadata['fullData'])
            return cls.from_dict(full_data)
        
        # Fallback construction from available metadata
        return cls(
            id=id,
            text=document.split(metadata.get('subject', ''))[0].strip(),
            subject=metadata.get('subject', ''),
            difficulty=metadata.get('difficulty', 'medium'),
            extractedAt=metadata.get('extractedAt', datetime.now().isoformat()),
            pdfSource=metadata.get('pdfSource', ''),
            images=[],
            options=None,
            correctAnswer=metadata.get('correctAnswer', ''),
            metadata=metadata
        )