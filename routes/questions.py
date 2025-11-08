from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
from flask import Blueprint, request, jsonify
import logging

from services.question_service import question_service

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
class Question:
    """Data model for questions"""
    id: str
    text: str
    subject: str
    difficulty: str  # 'easy' | 'medium' | 'hard'
    extractedAt: str  # ISO format datetime
    pdfSource: str
    images: List[QuestionImage]
    options: Optional[List[str]] = None
    correctAnswer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ChromaDB storage"""
        data = asdict(self)
        # Convert image objects to dictionaries
        data['images'] = [img.to_dict() for img in self.images]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        """Create instance from dictionary"""
        # Convert image dictionaries to objects
        if 'images' in data and data['images']:
            data['images'] = [QuestionImage.from_dict(img) for img in data['images']]
        else:
            data['images'] = []
        return cls(**data)

    def to_chromadb_format(self) -> tuple:
        """Convert to ChromaDB format (document, metadata, id)"""
        # The main text content for ChromaDB embedding
        document = f"{self.text} {self.subject}"
        if self.options:
            document += " " + " ".join(self.options)

        # Metadata for ChromaDB filtering and retrieval
        metadata = {
            "subject": self.subject,
            "difficulty": self.difficulty,
            "extractedAt": self.extractedAt,
            "pdfSource": self.pdfSource,
            "hasImages": len(self.images) > 0,
            "imageCount": len(self.images),
            "optionCount": len(self.options) if self.options else 0,
            "correctAnswer": self.correctAnswer or "",
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

# Create blueprint for API endpoints
questions_bp = Blueprint('questions', __name__, url_prefix='/api/questions')

@questions_bp.route('/save', methods=['POST'])
def save_question():
    """Save a single question to ChromaDB"""
    try:
        question_data = request.get_json()

        if not question_data:
            return jsonify({
                'success': False,
                'error': 'No question data provided'
            }), 400

        # Validate required fields
        required_fields = ['id', 'text', 'subject', 'difficulty']
        for field in required_fields:
            if field not in question_data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        success, question_id, error = question_service.save_question(question_data)

        if success:
            return jsonify({
                'success': True,
                'question_id': question_id,
                'message': 'Question saved successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': error or 'Failed to save question'
            }), 500

    except Exception as e:
        logging.error(f"Error in save_question: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@questions_bp.route('/batch-save', methods=['POST'])
def save_questions_batch():
    """Save multiple questions to ChromaDB"""
    try:
        questions_data = request.get_json()

        if not questions_data or not isinstance(questions_data, list):
            return jsonify({
                'success': False,
                'error': 'No questions data provided or invalid format'
            }), 400

        if len(questions_data) == 0:
            return jsonify({
                'success': False,
                'error': 'Empty questions list'
            }), 400

        success_count, total_count, error = question_service.save_questions_batch(questions_data)

        return jsonify({
            'success': success_count > 0,
            'success_count': success_count,
            'total_count': total_count,
            'message': f'Saved {success_count} out of {total_count} questions',
            'error': error
        })

    except Exception as e:
        logging.error(f"Error in save_questions_batch: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@questions_bp.route('/<question_id>', methods=['GET'])
def get_question(question_id: str):
    """Get a specific question by ID"""
    try:
        include_images = request.args.get('include_images', 'true').lower() == 'true'

        question_data = question_service.get_question(question_id, include_images)

        if question_data:
            return jsonify({
                'success': True,
                'question': question_data
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Question not found'
            }), 404

    except Exception as e:
        logging.error(f"Error in get_question: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@questions_bp.route('/', methods=['GET'])
def get_questions():
    """Get questions with optional filtering"""
    try:
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        subject = request.args.get('subject')
        difficulty = request.args.get('difficulty')
        include_images = request.args.get('include_images', 'true').lower() == 'true'

        questions_data = question_service.get_questions(
            limit=limit,
            subject=subject,
            difficulty=difficulty,
            include_images=include_images
        )

        return jsonify({
            'success': True,
            'questions': questions_data,
            'count': len(questions_data)
        })

    except Exception as e:
        logging.error(f"Error in get_questions: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@questions_bp.route('/search', methods=['GET'])
def search_questions():
    """Search questions with optional filters"""
    try:
        query = request.args.get('query')
        subject = request.args.get('subject')
        difficulty = request.args.get('difficulty')
        has_images = request.args.get('has_images')
        limit = int(request.args.get('limit', 50))
        include_images = request.args.get('include_images', 'false').lower() == 'true'

        # Convert has_images to boolean if provided
        if has_images is not None:
            has_images = has_images.lower() == 'true'

        questions = question_service.search_questions(
            query=query,
            subject=subject,
            difficulty=difficulty,
            has_images=has_images,
            limit=min(limit, 100),  # Cap at 100 for performance
            include_images=include_images
        )

        return jsonify({
            'success': True,
            'questions': questions,
            'count': len(questions)
        })

    except Exception as e:
        logging.error(f"Error in search_questions: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@questions_bp.route('/<question_id>', methods=['PUT'])
def update_question(question_id: str):
    """Update an existing question"""
    try:
        updates = request.get_json()

        if not updates:
            return jsonify({
                'success': False,
                'error': 'No update data provided'
            }), 400

        success, error = question_service.update_question(question_id, updates)

        if success:
            return jsonify({
                'success': True,
                'message': 'Question updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': error or 'Failed to update question'
            }), 500 if error != 'Question not found' else 404

    except Exception as e:
        logging.error(f"Error in update_question: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@questions_bp.route('/<question_id>', methods=['DELETE'])
def delete_question(question_id: str):
    """Delete a question and its associated images"""
    try:
        success, error = question_service.delete_question(question_id)

        if success:
            return jsonify({
                'success': True,
                'message': 'Question deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': error or 'Failed to delete question'
            }), 500

    except Exception as e:
        logging.error(f"Error in delete_question: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@questions_bp.route('/stats', methods=['GET'])
def get_statistics():
    """Get questions collection statistics"""
    try:
        stats = question_service.get_statistics()

        return jsonify({
            'success': True,
            'statistics': stats
        })

    except Exception as e:
        logging.error(f"Error in get_statistics: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@questions_bp.route('/export', methods=['GET'])
def export_questions():
    """Export all questions with images"""
    try:
        include_images = request.args.get('include_images', 'false').lower() == 'true'
        subject = request.args.get('subject')
        difficulty = request.args.get('difficulty')

        questions = question_service.search_questions(
            subject=subject,
            difficulty=difficulty,
            limit=1000,  # Large limit for export
            include_images=include_images
        )

        return jsonify({
            'success': True,
            'questions': questions,
            'count': len(questions),
            'exported_at': question_service.chromadb_client.collection.get()['metadatas'][0].get('exportedAt') if questions else None
        })

    except Exception as e:
        logging.error(f"Error in export_questions: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
