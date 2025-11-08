from flask import Blueprint, request, jsonify
import logging
from services.openai_service import get_openai_service
# from services.mistral_ocr_service import get_mistral_ocr_service  # Temporarily disabled due to API issues

# Create chat blueprint
chat_bp = Blueprint('chat', __name__, url_prefix='/api')

# Configure logging
logger = logging.getLogger(__name__)

@chat_bp.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat requests and return AI responses

    Expected JSON payload:
    {
        "message": "User message",
        "sessionId": "unique_session_id",
        "userId": "user_identifier",
        "mode": "general|whiteboard|practice|mock-test",
        "conversationHistory": [
            {"role": "user|assistant", "content": "message", "timestamp": "ISO_string"}
        ],
        "canvasData": "base64_image_data (optional)",
        "subject": "subject_name (optional)"
    }
    """
    try:
        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400

        # Extract required fields
        message = data.get('message', '').strip()
        if not message:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400

        # Extract optional fields
        session_id = data.get('sessionId', 'default')
        user_id = data.get('userId', 'anonymous')
        mode = data.get('mode', 'general')
        conversation_history = data.get('conversationHistory', [])
        canvas_data = data.get('canvasData')
        subject = data.get('subject', 'general')

        logger.info(f"Chat request - User: {user_id}, Session: {session_id}, Mode: {mode}")

        # Get OpenAI service
        openai_service = get_openai_service()
        system_prompts = openai_service.get_system_prompts()

        # Select system prompt based on mode
        system_prompt = system_prompts.get(mode, system_prompts['general'])

        # Prepare messages for OpenAI
        messages = []

        # Add recent conversation history (limit to last 10 messages)
        if conversation_history:
            recent_history = conversation_history[-10:]
            for msg in recent_history:
                if msg.get('role') in ['user', 'assistant'] and msg.get('content'):
                    messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })

        # Add current message with comprehensive OCR + Vision hybrid analysis
        if canvas_data:
            # STAGE 1: Attempt OCR extraction for text/equations
            logger.info("Initiating comprehensive academic content analysis...")

            # STABLE VISION-ONLY ANALYSIS (OCR temporarily disabled due to API issues)
            logger.info("Using comprehensive vision-only academic analysis")

            enhanced_message = f"""{message}

🔍 **MULTI-PASS ITERATIVE ANALYSIS SYSTEM**

Perform a systematic multi-pass analysis of this complex academic content. Each pass focuses on different aspects to ensure complete understanding:

## **PASS 1: OVERALL CONTENT SURVEY**
First, scan the entire image and provide:
- **Subject Identification**: What academic subject is this? (Chemistry, Physics, Math, Biology, etc.)
- **Content Type**: Equations, diagrams, mixed problem-solving, step-by-step derivation
- **Complexity Level**: Simple, intermediate, or complex multi-component content
- **Spatial Layout**: How is information distributed across the canvas?

## **PASS 2: DETAILED ELEMENT IDENTIFICATION**
Now examine each individual component separately:

**STAGE 1: DETAILED CONTENT IDENTIFICATION & TRANSCRIPTION**

*For CHEMISTRY:*
- **Starting Materials**: Identify all reactant molecules (left side of arrows)
- **Products**: Identify all product molecules (right side of arrows)
- **Molecular Structures**: Benzene rings (hexagons), functional groups, carbon chains
- **Reaction Conditions**: Catalysts, temperature, pressure, solvents
- **Balancing Coefficients**: Numbers in front of molecular formulas

*For MATHEMATICS:*
- **Individual Equations**: Each mathematical expression separately
- **Variables and Constants**: All symbols and their meanings
- **Operations**: Addition, integration, differentiation, etc.
- **Geometric Elements**: Shapes, angles, measurements

*For PHYSICS:*
- **Circuit Components**: Resistors, capacitors, voltage sources, current paths
- **Force Vectors**: Magnitude and direction of all forces
- **Field Lines**: Electric, magnetic, gravitational field representations
- **Measurement Values**: All numerical values with units

*For BIOLOGY:*
- **Cellular Components**: Organelles, membranes, molecular structures
- **Process Steps**: Sequential stages in biological pathways
- **Labels and Annotations**: All text labels and arrows

## **PASS 3: SPATIAL RELATIONSHIP ANALYSIS**
Now connect the individual elements you identified:

*CONNECTION PATTERNS:*
- **Flow Direction**: What is the logical flow? (Left→Right, Top→Bottom, Circular)
- **Arrows and Lines**: What do arrows connect? Where do lines lead?
- **Groupings**: Which elements belong together? Which are separate processes?
- **Sequential Steps**: Is this a multi-step process? What's the order?

*SPECIFIC RELATIONSHIP ANALYSIS:*
- **Chemistry**: How do reactants transform into products? What are intermediate steps?
- **Mathematics**: How do equations build on each other? What's the derivation flow?
- **Physics**: How do components interact? What are the cause-effect relationships?
- **Biology**: What's the pathway flow? How do processes connect?

## **PASS 4: INTEGRATION & VERIFICATION**
Combine all information for complete understanding:

*COMPLETE PICTURE CONSTRUCTION:*
- **Overall Process**: What is the complete reaction/equation/process from start to finish?
- **Missing Elements**: Are there any gaps in the transcription or understanding?
- **Consistency Check**: Do all parts make sense together? Are there contradictions?
- **Scientific Validity**: Does the complete process follow known scientific principles?

*VERIFICATION QUESTIONS:*
- Does the complete chemical equation balance properly?
- Do the mathematical steps follow logically?
- Are the physics relationships consistent with known laws?
- Does the biological pathway make physiological sense?

## **PASS 5: COMPREHENSIVE EDUCATIONAL RESPONSE**

*FINAL INTEGRATED ANALYSIS:*
- **What I See (Complete)**: Full transcription of the entire content with all connections
- **Subject-Specific Interpretation**: Apply appropriate academic context and principles
- **Process Analysis**: Explain the complete mechanism/derivation/pathway step-by-step
- **Accuracy Assessment**: Identify any errors or missing components
- **Educational Guidance**: Provide learning insights and next steps

**CRITICAL SUCCESS FACTORS:**
- **Complete Integration**: Don't analyze parts in isolation - show how everything connects
- **Spatial Awareness**: Pay attention to the physical layout and relationships
- **Sequential Logic**: Follow the logical flow from start to finish
- **Verification**: Double-check that your complete analysis makes scientific/mathematical sense
- **Educational Value**: Provide insights that help students understand the complete process

**FORMATTING REQUIREMENTS:**
- Use LaTeX notation for mathematical expressions: \\[ \\] for display, \\( \\) for inline
- Include proper units, significant figures, and scientific notation
- Use **bold** for section headers and emphasis
- Structure your response to show the complete, integrated understanding

**ULTIMATE GOAL**: Provide a complete, accurate, and educationally valuable analysis that captures the ENTIRE content and its relationships!"""

            messages.append({
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': enhanced_message},
                    {'type': 'image_url', 'image_url': {'url': canvas_data}}
                ]
            })
        else:
            messages.append({
                'role': 'user',
                'content': message
            })

        # Get response from OpenAI
        response = openai_service.chat_completion(
            messages=messages,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000
        )

        if not response['success']:
            logger.error(f"OpenAI API error: {response['error']}")
            return jsonify({
                'success': False,
                'error': 'Failed to generate response',
                'details': response['error']
            }), 500

        # Return successful response
        return jsonify({
            'success': True,
            'data': {
                'response': response['response'],
                'usage': response['usage'],
                'model': response['model'],
                'sessionId': session_id,
                'mode': mode
            }
        })

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@chat_bp.route('/chat/health', methods=['GET'])
def chat_health():
    """Health check for chat service"""
    try:
        # Test OpenAI service initialization
        openai_service = get_openai_service()

        return jsonify({
            'success': True,
            'message': 'Chat service is healthy',
            'model': openai_service.model,
            'service': 'openai'
        })

    except Exception as e:
        logger.error(f"Chat health check failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Chat service unhealthy',
            'details': str(e)
        }), 500

@chat_bp.route('/chat/models', methods=['GET'])
def get_available_models():
    """Get information about available AI models"""
    try:
        openai_service = get_openai_service()

        return jsonify({
            'success': True,
            'data': {
                'current_model': openai_service.model,
                'provider': 'openai',
                'system_prompts': list(openai_service.get_system_prompts().keys())
            }
        })

    except Exception as e:
        logger.error(f"Models endpoint error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get model information',
            'details': str(e)
        }), 500