import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from models.mcq_solution import MCQSolution
from models.mcq_solutions_client import get_mcq_solutions_client
from models.chromadb_client import get_chromadb_client
from .openai_service import get_openai_service

class MCQService:
    """Service for managing MCQ checking and solution storage"""
    
    def __init__(self):
        self.mcq_solutions_client = get_mcq_solutions_client()
        self.questions_client = get_chromadb_client()
    
    def check_mcq_answer(self, question_id: str, selected_answer: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Check MCQ answer and return result with solution

        Returns:
            (success, result_data, error_message)
        """
        try:
            # First, get the question to validate it exists (with images)
            from services.question_service import question_service
            question_dict = question_service.get_question(question_id, include_images=True)

            if not question_dict:
                return False, {}, "Question not found"

            # Check if we have a stored solution
            stored_solution = self.mcq_solutions_client.get_solution(question_id)

            if stored_solution:
                # Use stored solution
                is_correct = selected_answer.strip().upper() == stored_solution.correct_answer.strip().upper()

                result = {
                    "question_id": question_id,
                    "selected_answer": selected_answer,
                    "correct_answer": stored_solution.correct_answer,
                    "is_correct": is_correct,
                    "explanation": stored_solution.explanation,
                    "solution_source": "database",
                    "confidence_score": stored_solution.confidence_score or 1.0
                }

                logging.info(f"MCQ answer checked using stored solution for question {question_id}")
                return True, result, None

            else:
                # Generate solution using LLM with full question data (including images)
                success, llm_result, error = self._generate_solution_with_llm(question_dict, selected_answer)

                if not success:
                    return False, {}, error

                # Store the generated solution for future use
                solution = MCQSolution(
                    id=f"sol_{question_id}",
                    question_id=question_id,
                    correct_answer=llm_result["correct_answer"],
                    explanation=llm_result["explanation"],
                    generated_by="llm",
                    generated_at=datetime.now().isoformat(),
                    llm_model=llm_result.get("model_used", "gpt-3.5-turbo"),
                    confidence_score=llm_result.get("confidence_score", 0.8),
                    validated=False
                )

                # Save the solution
                save_success = self.mcq_solutions_client.save_solution(solution)
                if save_success:
                    logging.info(f"Generated and saved new MCQ solution for question {question_id}")
                else:
                    logging.warning(f"Failed to save generated MCQ solution for question {question_id}")

                result = {
                    "question_id": question_id,
                    "selected_answer": selected_answer,
                    "correct_answer": llm_result["correct_answer"],
                    "is_correct": llm_result["is_correct"],
                    "explanation": llm_result["explanation"],
                    "solution_source": "llm_generated",
                    "confidence_score": llm_result.get("confidence_score", 0.8)
                }

                return True, result, None

        except Exception as e:
            logging.error(f"Failed to check MCQ answer for question {question_id}: {str(e)}")
            return False, {}, f"Failed to check MCQ answer: {str(e)}"
    
    def _generate_solution_with_llm(self, question_dict: Dict[str, Any], selected_answer: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Generate solution using LLM with full question data including images"""
        try:
            # Get OpenAI service
            openai_service = get_openai_service()

            # Format the question for LLM
            question_text = question_dict.get('text', '')
            options = question_dict.get('options', [])
            enhanced_options = question_dict.get('enhancedOptions', [])
            images = question_dict.get('images', [])

            # Create prompt for LLM with image information and enhanced options
            prompt = self._create_mcq_prompt_with_images(question_text, options, enhanced_options, images, selected_answer)

            # Prepare messages for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert tutor who can analyze multiple choice questions, including those with diagrams and images. When provided with image descriptions, use them to understand the complete context of the question."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # Get response from OpenAI using the correct method
            response = openai_service.chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=1200  # Increased for more detailed explanations with images
            )

            if not response.get('success'):
                return False, {}, response.get('error', 'Unknown OpenAI error')

            # Parse the response to extract answer and explanation
            result = self._parse_llm_response(response['response'], selected_answer, options, enhanced_options)

            return True, result, None

        except Exception as e:
            logging.error(f"Failed to generate solution with LLM: {str(e)}")
            return False, {}, f"Failed to generate solution: {str(e)}"
    
    def _create_mcq_prompt(self, question_text: str, options: list, selected_answer: str) -> str:
        """Create prompt for LLM to solve MCQ"""
        options_text = ""
        if options:
            for i, option in enumerate(options):
                # Clean option text from prefixes like "(A)", "(B)" etc.
                clean_option = option
                if option.strip().startswith(f"({chr(65+i)})"):
                    clean_option = option.replace(f"({chr(65+i)})", "").strip()
                elif option.strip().startswith(f"{chr(65+i)}."):
                    clean_option = option.replace(f"{chr(65+i)}.", "").strip()
                elif option.strip().startswith(f"{chr(65+i)}"):
                    clean_option = option.replace(f"{chr(65+i)}", "").strip()
                
                options_text += f"{chr(65+i)}. {clean_option}\n"
        
        prompt = f"""You are an expert tutor. Please analyze this multiple choice question and provide the correct answer with a detailed explanation.

Question: {question_text}

Options:
{options_text}

Student's selected answer: {selected_answer}

Please respond in this exact JSON format:
{{
    "correct_answer": "A",
    "is_correct": true,
    "explanation": "Detailed explanation of why this is the correct answer and the reasoning behind it",
    "confidence_score": 0.85
}}

Important:
- correct_answer should be the letter (A, B, C, D) of the correct option
- is_correct should be true if the student's answer matches the correct answer, false otherwise
- explanation should be clear, educational, and help the student understand the concept
- confidence_score should be between 0.0 and 1.0
- Respond ONLY with valid JSON, no additional text"""
        
        return prompt

    def _create_mcq_prompt_with_images(self, question_text: str, options: list, enhanced_options: list, images: list, selected_answer: str) -> str:
        """Create enhanced prompt for LLM to solve MCQ with image context and enhanced options"""
        options_text = ""

        # Handle enhanced options first (if available)
        if enhanced_options:
            for i, option in enumerate(enhanced_options):
                option_label = chr(65 + i)  # A, B, C, D...

                if option.get('type') == 'text':
                    # Text option
                    clean_option = option.get('content', '')
                    options_text += f"{option_label}. {clean_option}\n"
                elif option.get('type') == 'image':
                    # Image option
                    option_desc = option.get('description', 'Image option')
                    options_text += f"{option_label}. [IMAGE OPTION] {option_desc}\n"
                    # Note: In future, we could include the actual image content here

        # Fallback to legacy options if no enhanced options
        elif options:
            for i, option in enumerate(options):
                # Clean option text from prefixes like "(A)", "(B)" etc.
                clean_option = option
                if option.strip().startswith(f"({chr(65+i)})"):
                    clean_option = option.replace(f"({chr(65+i)})", "").strip()
                elif option.strip().startswith(f"{chr(65+i)}."):
                    clean_option = option.replace(f"{chr(65+i)}.", "").strip()
                elif option.strip().startswith(f"{chr(65+i)}"):
                    clean_option = option.replace(f"{chr(65+i)}", "").strip()

                options_text += f"{chr(65+i)}. {clean_option}\n"

        # Add image context if available
        image_context = ""
        if images:
            image_context = "\n\nImage Information:\n"
            for i, image in enumerate(images):
                image_desc = image.get('description', f'Image {i+1} associated with the question')
                image_context += f"- Image {i+1}: {image_desc}\n"

                # Add image metadata if available
                if image.get('type'):
                    image_context += f"  Type: {image.get('type')}\n"

                # Note: Base64 data is available but not included in prompt for now
                # In future, we could send the actual image to GPT-4 Vision
                if image.get('base64Data'):
                    image_context += f"  (Visual content available - diagram/figure showing relevant information)\n"

            image_context += "\nPlease consider the visual information from the images when solving this question.\n"

        prompt = f"""You are an expert tutor analyzing a multiple choice question. Please provide the correct answer with a detailed explanation.

Question: {question_text}

Options:
{options_text}
{image_context}
Student's selected answer: {selected_answer}

Please analyze this question carefully, considering:
1. The question text and any mathematical concepts involved
2. The visual information provided in any images/diagrams
3. The relationship between the question and the answer options
4. The underlying physics/chemistry/mathematical principles

Respond in this exact JSON format:
{{
    "correct_answer": "A",
    "is_correct": true,
    "explanation": "Detailed step-by-step explanation of the solution, referencing any visual elements from images when relevant. Explain the physics/chemistry/math concepts clearly.",
    "confidence_score": 0.85
}}

Important:
- correct_answer should be the letter (A, B, C, D) of the correct option
- is_correct should be true if the student's answer matches the correct answer, false otherwise
- explanation should be comprehensive, educational, and reference visual elements when applicable
- confidence_score should be between 0.0 and 1.0
- Respond ONLY with valid JSON, no additional text"""

        return prompt

    def _parse_llm_response(self, response: str, selected_answer: str, options: list, enhanced_options: list = None) -> Dict[str, Any]:
        """Parse LLM response to extract structured data"""
        try:
            import json
            
            # Try to parse as JSON first
            try:
                parsed = json.loads(response)
                if all(key in parsed for key in ["correct_answer", "explanation"]):
                    # Check if student's answer is correct
                    is_correct = selected_answer.strip().upper() == parsed["correct_answer"].strip().upper()
                    
                    return {
                        "correct_answer": parsed["correct_answer"],
                        "is_correct": is_correct,
                        "explanation": parsed["explanation"],
                        "confidence_score": parsed.get("confidence_score", 0.8),
                        "model_used": "gpt-3.5-turbo"
                    }
            except json.JSONDecodeError:
                pass
            
            # Fallback: try to extract information from text response
            lines = response.strip().split('\n')
            correct_answer = None
            explanation = response
            
            # Look for patterns like "Answer: A" or "Correct answer: B"
            for line in lines:
                line_lower = line.lower().strip()
                if 'answer' in line_lower and any(letter in line_lower for letter in ['a', 'b', 'c', 'd']):
                    for letter in ['a', 'b', 'c', 'd']:
                        if letter in line_lower:
                            correct_answer = letter.upper()
                            break
                    if correct_answer:
                        break
            
            # If we couldn't find a clear answer, default to A
            if not correct_answer:
                correct_answer = "A"
                explanation += "\n\n(Note: Could not clearly determine correct answer from response)"
            
            is_correct = selected_answer.strip().upper() == correct_answer
            
            return {
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "explanation": explanation,
                "confidence_score": 0.6,  # Lower confidence for parsed response
                "model_used": "gpt-3.5-turbo"
            }
            
        except Exception as e:
            logging.error(f"Failed to parse LLM response: {str(e)}")
            # Return a safe fallback
            return {
                "correct_answer": "A",
                "is_correct": False,
                "explanation": f"Unable to parse response properly. Original response: {response}",
                "confidence_score": 0.3,
                "model_used": "gpt-3.5-turbo"
            }
    
    def get_solution(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Get stored solution for a question"""
        try:
            solution = self.mcq_solutions_client.get_solution(question_id)
            if solution:
                return solution.to_dict()
            return None
        except Exception as e:
            logging.error(f"Failed to get solution for question {question_id}: {str(e)}")
            return None
    
    def save_solution(self, solution_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Save a solution manually"""
        try:
            solution = MCQSolution.from_dict(solution_data)
            success = self.mcq_solutions_client.save_solution(solution)
            
            if success:
                return True, None
            else:
                return False, "Failed to save solution to database"
                
        except Exception as e:
            logging.error(f"Failed to save solution: {str(e)}")
            return False, f"Failed to save solution: {str(e)}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MCQ solutions statistics"""
        try:
            stats = self.mcq_solutions_client.get_collection_stats()
            return stats
        except Exception as e:
            logging.error(f"Failed to get MCQ statistics: {str(e)}")
            return {}

# Global instance
mcq_service = MCQService()
