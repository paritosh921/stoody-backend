import os
import requests
import logging
from typing import Dict, Optional
import base64
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

class MistralOCRService:
    def __init__(self):
        self.api_key = os.getenv('MISTRAL_API_KEY')
        self.model = os.getenv('MISTRAL_OCR_MODEL', 'mistral-ocr-latest')
        self.base_url = "https://api.mistral.ai/v1"

        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")

        logger.info(f"Mistral OCR service initialized with model: {self.model}")

    def extract_text_from_canvas(self, canvas_data: str) -> Dict:
        """
        Extract academic content from canvas image using Mistral OCR

        Args:
            canvas_data: Base64 encoded image data (data:image/png;base64,...)

        Returns:
            Dictionary with extracted text and success status
        """
        try:
            # Prepare the request for Mistral OCR
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            # Enhanced messages for comprehensive academic content extraction
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """🎓 COMPREHENSIVE ACADEMIC CONTENT EXTRACTION

Please analyze this academic content and extract ALL text, equations, and annotations you see. This could include:

**MATHEMATICAL CONTENT:**
- Equations, integrals, derivatives, limits, summations
- Algebraic expressions, geometric formulas
- Variables, constants, mathematical symbols (∫, √, π, Σ, etc.)

**SCIENTIFIC CONTENT:**
- Chemical formulas and reactions (H₂O, CH₄, etc.)
- Physical equations and constants (F=ma, E=mc², etc.)
- Units and measurements (kg, m/s², mol, etc.)

**BIOLOGICAL CONTENT:**
- Process descriptions, anatomical labels
- Classification terms, scientific names

**HANDWRITTEN TEXT:**
- Problem statements, explanations, steps
- Labels, annotations, notes
- Variable definitions and explanations

**DIAGRAM ELEMENTS:**
- Labels on diagrams, arrows with text
- Axis labels, scale markers
- Component names and values

Please provide a comprehensive extraction that captures ALL readable content, maintaining original formatting and mathematical notation as much as possible."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": canvas_data
                            }
                        }
                    ]
                }
            ]

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 2000,  # Increased for comprehensive content
                "temperature": 0.1  # Low temperature for consistent OCR results
            }

            logger.info("Sending canvas to Mistral OCR for comprehensive academic content extraction")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=45  # Increased timeout for complex content
            )

            if response.status_code == 200:
                result = response.json()
                extracted_text = result['choices'][0]['message']['content'].strip()

                logger.info(f"Mistral OCR extraction successful: {len(extracted_text)} characters extracted")

                return {
                    'success': True,
                    'extracted_text': extracted_text,
                    'model': self.model,
                    'usage': result.get('usage', {}),
                    'extraction_type': 'comprehensive_academic'
                }
            else:
                error_msg = f"Mistral OCR API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'extracted_text': '',
                    'extraction_type': 'failed'
                }

        except requests.exceptions.Timeout:
            error_msg = "Mistral OCR request timeout - content may be too complex"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'extracted_text': '',
                'extraction_type': 'timeout'
            }
        except requests.exceptions.RequestException as e:
            error_msg = f"Mistral OCR network error: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'extracted_text': '',
                'extraction_type': 'network_error'
            }
        except Exception as e:
            error_msg = f"Mistral OCR service error: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'extracted_text': '',
                'extraction_type': 'service_error'
            }

# Global service instance
_mistral_ocr_service = None

def get_mistral_ocr_service() -> MistralOCRService:
    """Get or create Mistral OCR service instance"""
    global _mistral_ocr_service
    if _mistral_ocr_service is None:
        _mistral_ocr_service = MistralOCRService()
    return _mistral_ocr_service