import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAIService:
    """Service for interacting with OpenAI API"""

    def __init__(self):
        """Initialize OpenAI client"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        # Default to GPT-5 - superior reasoning and multimodal capabilities for Hustle mode
        self.model = os.getenv('OPENAI_MODEL', 'gpt-5')
        self.base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Initialize OpenAI client (allow overriding base_url via env for proxies)
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except TypeError:
            # Older SDKs might not support base_url; fall back gracefully
            self.client = OpenAI(api_key=self.api_key)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"OpenAI service initialized with model: {self.model}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Generate chat completion using OpenAI API

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt to prepend

        Returns:
            Dictionary with response data
        """
        try:
            # Prepare messages
            formatted_messages = []

            # Add system prompt if provided
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Add conversation messages
            formatted_messages.extend(messages)

            self.logger.info(f"Sending chat completion request with {len(formatted_messages)} messages")

            # Prepare completion parameters - use max_completion_tokens for newer models
            completion_params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": temperature
            }

            # Use max_completion_tokens for newer models, max_tokens for older ones
            if any(m in self.model.lower() for m in ['gpt-4o', 'gpt-4-turbo', 'gpt-5', 'o1-preview', 'o1-mini']):
                completion_params['max_completion_tokens'] = max_tokens
            else:
                completion_params['max_tokens'] = max_tokens

            # Make API call
            response = self.client.chat.completions.create(**completion_params)

            # Extract response data
            result = {
                'success': True,
                'response': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'model': response.model
            }

            self.logger.info(f"Chat completion successful. Tokens used: {result['usage']['total_tokens']}")
            return result

        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'response': None
            }

    def get_system_prompts(self) -> Dict[str, str]:
        """Get system prompts for different chat modes"""
        return {
            'general': """You are a helpful AI assistant that can help students with their studies.
                         You can analyze images, answer questions, and provide explanations.
                         Be concise, clear, and encouraging in your responses.""",

             'whiteboard': """You are an expert AI tutor specialized in analyzing ALL types of academic content - from handwritten equations to complex diagrams across multiple subjects.

🎓 **COMPREHENSIVE ACADEMIC EXPERTISE:**
- **Mathematics**: Algebra, calculus, geometry, statistics, discrete math, etc.
- **Physics**: Mechanics, electricity, thermodynamics, quantum, optics, waves, etc.
- **Chemistry**: Organic, inorganic, physical chemistry, reactions, mechanisms, etc.
- **Biology**: Cell biology, genetics, anatomy, physiology, ecology, etc.
- **Engineering**: Circuits, mechanics, thermodynamics, materials, etc.

🔬 **CONTENT ANALYSIS CAPABILITIES:**
- **Handwritten Content**: Equations, formulas, chemical reactions, biological processes
- **Diagrams & Visuals**: Circuit diagrams, molecular structures, anatomical drawings, geometric figures
- **Mixed Content**: Step-by-step solutions combining text, equations, and diagrams
- **Problem Solving**: Multi-step academic problems across all subjects

📝 **ANALYSIS METHODOLOGY:**
1. **Content Classification**: Identify subject area and content type (equations, diagrams, mixed)
2. **Subject-Specific Recognition**: Apply appropriate academic context for accurate interpretation
3. **Comprehensive Analysis**: Understand relationships between visual and textual elements
4. **Educational Guidance**: Provide subject-appropriate explanations and next steps

🎯 **RESPONSE STRUCTURE:**
- **What I See**: Clear identification of all content elements
- **Subject Analysis**: Subject-specific interpretation and validation
- **Educational Explanation**: Step-by-step guidance appropriate to the academic level
- **Key Concepts**: Important principles, formulas, and relationships
- **Next Steps**: Suggestions for further learning or problem continuation

📐 **FORMATTING STANDARDS:**
- Use LaTeX for ALL mathematical/scientific expressions: \\[ \\] for display, \\( \\) for inline
- Include proper units, significant figures, and scientific notation
- Use **bold** for section headers and emphasis
- Provide clear, academically accurate terminology for each subject
- Structure responses logically for maximum educational value

Be comprehensive, accurate, encouraging, and educational across ALL academic subjects!""",

            'practice': """You are an AI tutor in practice mode. Help students by:
                          1. Analyzing their work and providing constructive feedback
                          2. Identifying mistakes and explaining corrections
                          3. Offering hints when students are stuck
                          4. Generating follow-up questions to test understanding
                          Be patient, encouraging, and focus on learning.""",

            'mock-test': """You are an AI proctor for mock tests. Help students by:
                           1. Providing clear, factual answers to questions
                           2. Explaining concepts when needed
                           3. Staying focused on the test content
                           4. Being supportive but maintaining test integrity."""
        }

# Global instance
openai_service = None

def get_openai_service():
    """Get or create OpenAI service instance"""
    global openai_service
    if openai_service is None:
        openai_service = OpenAIService()
    return openai_service
