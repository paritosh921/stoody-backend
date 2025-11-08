import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from models.question import Question, QuestionImage, QuestionOption
from models.chromadb_client import get_chromadb_client
from .image_service import image_service

class QuestionService:
    """Service for managing question operations with ChromaDB and image storage"""

    def __init__(self, admin_id: str = None):
        self.admin_id = admin_id
        self.chromadb_client = get_chromadb_client(admin_id)
    
    def _process_images(self, images_data: Optional[List[Dict[str, Any]]]) -> List[QuestionImage]:
        processed_images: List[QuestionImage] = []
        for image_data in images_data or []:
            if isinstance(image_data, QuestionImage):
                question_image = image_data
            elif isinstance(image_data, dict):
                question_image = QuestionImage.from_dict(image_data)
            else:
                continue

            filename = question_image.filename or f"image_{question_image.id}.png"

            if question_image.base64Data:
                success, file_path, error = image_service.save_base64_image(
                    question_image.base64Data,
                    filename
                )

                if success and file_path:
                    question_image.path = file_path
                    question_image.base64Data = None
                else:
                    logging.warning(f"Failed to save image {filename}: {error}")

            processed_images.append(question_image)

        return processed_images

    def _process_enhanced_options(self, options_data: Optional[List[Dict[str, Any]]]) -> Optional[List[QuestionOption]]:
        if not options_data:
            return None

        processed_options: List[QuestionOption] = []
        for opt_data in options_data:
            if isinstance(opt_data, QuestionOption):
                processed_options.append(opt_data)
                continue
            if not isinstance(opt_data, dict):
                continue

            option_id = str(opt_data.get('id') or f"option_{int(datetime.now().timestamp() * 1000)}")
            option_type = opt_data.get('type', 'text')
            content = opt_data.get('content', '')

            if option_type == 'image' and isinstance(content, str) and content:
                if content.startswith('data:image'):
                    label = opt_data.get('label', 'img')
                    filename = f"option_{option_id}_{label}.png"
                    success, file_path, error = image_service.save_base64_image(content, filename)
                    if success and file_path:
                        content = file_path
                    else:
                        logging.warning(f"Failed to save enhanced option image {filename}: {error}")

            processed_options.append(QuestionOption(
                id=option_id,
                type=option_type,
                content=content or '',
                label=opt_data.get('label'),
                description=opt_data.get('description')
            ))

        return processed_options

    def save_question(self, question_data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Save a question with its images to ChromaDB and file system

        Returns:
            (success, question_id, error_message)
        """
        try:
            processed_images = self._process_images(question_data.get('images'))
            enhanced_options = self._process_enhanced_options(question_data.get('enhancedOptions'))

            # Extract document_type from question_data or metadata
            document_type = question_data.get('document_type')
            if not document_type and question_data.get('metadata'):
                document_type = question_data.get('metadata', {}).get('document_type')

            question = Question(
                id=question_data['id'],
                text=question_data['text'],
                subject=question_data['subject'],
                difficulty=question_data['difficulty'],
                extractedAt=question_data.get('extractedAt', datetime.now().isoformat()),
                pdfSource=question_data.get('pdfSource', ''),
                images=processed_images,
                options=question_data.get('options', []),
                enhancedOptions=enhanced_options if enhanced_options else None,
                correctAnswer=question_data.get('correctAnswer', ''),
                document_type=document_type,  # Preserve document_type
                metadata=question_data.get('metadata', {})
            )

            success = self.chromadb_client.save_question(question)

            if success:
                logging.info(f"Successfully saved question {question.id} with document_type={document_type}")
                return True, question.id, None
            else:
                return False, None, "Failed to save question to database"

        except Exception as e:
            logging.error(f"Failed to save question: {str(e)}")
            return False, None, f"Failed to save question: {str(e)}"

    def save_questions_batch(self, questions_data: List[Dict[str, Any]]) -> Tuple[int, int, Optional[str]]:
        """
        Save multiple questions with their images

        Returns:
            (success_count, total_count, error_message)
        """
        try:
            processed_questions = []

            for question_data in questions_data:
                processed_images = self._process_images(question_data.get('images'))
                enhanced_options = self._process_enhanced_options(question_data.get('enhancedOptions'))

                # Extract document_type from question_data or metadata
                document_type = question_data.get('document_type')
                if not document_type and question_data.get('metadata'):
                    document_type = question_data.get('metadata', {}).get('document_type')

                question = Question(
                    id=question_data['id'],
                    text=question_data['text'],
                    subject=question_data['subject'],
                    difficulty=question_data['difficulty'],
                    extractedAt=question_data.get('extractedAt', datetime.now().isoformat()),
                    pdfSource=question_data.get('pdfSource', ''),
                    images=processed_images,
                    options=question_data.get('options', []),
                    enhancedOptions=enhanced_options if enhanced_options else None,
                    correctAnswer=question_data.get('correctAnswer', ''),
                    document_type=document_type,  # Preserve document_type
                    metadata=question_data.get('metadata', {})
                )

                processed_questions.append(question)

            success_count, total_count = self.chromadb_client.save_questions_batch(processed_questions)

            logging.info(f"Batch save completed: {success_count}/{total_count} questions saved")
            return success_count, total_count, None

        except Exception as e:
            logging.error(f"Failed to save questions batch: {str(e)}")
            return 0, len(questions_data), f"Failed to save questions: {str(e)}"

    def get_question(self, question_id: str, include_images: bool = True) -> Optional[Dict[str, Any]]:
        """Get a question by ID with optional image data"""
        try:
            # Try admin-specific collection first
            question = self.chromadb_client.get_question(question_id)
            # Fallback: legacy shared collection (pre-migration)
            if not question and self.admin_id:
                from models.chromadb_client import get_chromadb_client as _get
                legacy_client = _get(None)
                question = legacy_client.get_question(question_id)
            if not question:
                return None
            
            question_dict = question.to_dict()
            
            # Load image data if requested
            if include_images and question.images:
                for i, image in enumerate(question_dict['images']):
                    if image['path']:
                        # Get base64 data for display
                        base64_data = image_service.get_image_base64(image['path'])
                        if base64_data:
                            question_dict['images'][i]['base64Data'] = base64_data

            # Load enhanced option images if requested
            if include_images and question_dict.get('enhancedOptions'):
                for i, option in enumerate(question_dict['enhancedOptions']):
                    if option.get('type') == 'image' and option.get('content'):
                        # If content is a file path, load the base64 data
                        if not option['content'].startswith('data:image'):
                            base64_data = image_service.get_image_base64(option['content'])
                            if base64_data:
                                question_dict['enhancedOptions'][i]['content'] = base64_data
                                logging.info(f"Loaded enhanced option image: {option['content']}")

            return question_dict
            
        except Exception as e:
            logging.error(f"Failed to get question {question_id}: {str(e)}")
            return None
    
    def search_questions(
        self, 
        query: str = None, 
        subject: str = None, 
        difficulty: str = None,
        has_images: bool = None,
        document_type: str = None,
        limit: int = 50,
        include_images: bool = False
    ) -> List[Dict[str, Any]]:
        """Search questions with optional filters including document_type for mode-specific retrieval"""
        try:
            # Search admin-specific collection first
            questions = self.chromadb_client.search_questions(
                query=query,
                subject=subject,
                difficulty=difficulty,
                has_images=has_images,
                document_type=document_type,
                limit=limit
            )
            # Fallback: if empty and admin_id exists, read from legacy shared collection
            if not questions and self.admin_id:
                from models.chromadb_client import get_chromadb_client as _get
                legacy_client = _get(None)
                questions = legacy_client.search_questions(
                    query=query,
                    subject=subject,
                    difficulty=difficulty,
                    has_images=has_images,
                    document_type=document_type,
                    limit=limit
                )
            
            results = []
            for question in questions:
                question_dict = question.to_dict()
                
                # Load image data if requested
                if include_images and question.images:
                    for i, image in enumerate(question_dict['images']):
                        if image['path']:
                            base64_data = image_service.get_image_base64(image['path'])
                            if base64_data:
                                question_dict['images'][i]['base64Data'] = base64_data

                # Load enhanced option images if requested
                if include_images and question_dict.get('enhancedOptions'):
                    for i, option in enumerate(question_dict['enhancedOptions']):
                        if option.get('type') == 'image' and option.get('content'):
                            # If content is a file path, load the base64 data
                            if not option['content'].startswith('data:image'):
                                base64_data = image_service.get_image_base64(option['content'])
                                if base64_data:
                                    question_dict['enhancedOptions'][i]['content'] = base64_data

                results.append(question_dict)
            
            return results
            
        except Exception as e:
            logging.error(f"Failed to search questions: {str(e)}")
            return []
    
    def update_question(self, question_id: str, updates: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Update an existing question"""
        try:
            # Get existing question
            existing_question = self.chromadb_client.get_question(question_id)
            if not existing_question:
                return False, "Question not found"

            # Update fields
            for key, value in updates.items():
                if hasattr(existing_question, key):
                    if key == 'images' and isinstance(value, list):
                        processed_images = self._process_images(value)
                        setattr(existing_question, key, processed_images)
                    elif key == 'enhancedOptions':
                        if value is None:
                            setattr(existing_question, key, None)
                        elif isinstance(value, list):
                            processed_options = self._process_enhanced_options(value)
                            setattr(existing_question, key, processed_options if processed_options else None)
                        else:
                            setattr(existing_question, key, value)
                    else:
                        setattr(existing_question, key, value)

            # Save updated question
            success = self.chromadb_client.save_question(existing_question)

            if success:
                return True, None
            else:
                return False, "Failed to update question in database"

        except Exception as e:
            logging.error(f"Failed to update question {question_id}: {str(e)}")
            return False, f"Failed to update question: {str(e)}"

    def delete_question(self, question_id: str) -> Tuple[bool, Optional[str]]:
        """Delete a question and its associated images"""
        try:
            # Get question to delete associated images
            question = self.chromadb_client.get_question(question_id)
            if question:
                # Delete associated image files
                for image in question.images:
                    if image.path:
                        image_service.delete_image(image.path)
            
            # Delete from ChromaDB
            success = self.chromadb_client.delete_question(question_id)
            
            if success:
                return True, None
            else:
                return False, "Failed to delete question from database"
                
        except Exception as e:
            logging.error(f"Failed to delete question {question_id}: {str(e)}")
            return False, f"Failed to delete question: {str(e)}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get questions collection statistics"""
        try:
            stats = self.chromadb_client.get_collection_stats()
            return stats
            
        except Exception as e:
            logging.error(f"Failed to get statistics: {str(e)}")
            return {}

# DO NOT create a global instance here - each request should create its own instance with the correct admin_id
# question_service = QuestionService()  # REMOVED to prevent ChromaDB conflicts