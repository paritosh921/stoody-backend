"""
Tool-Based Diagram Generator

This module integrates the DiagramToolkit with OpenAI's function calling
to allow LLMs to dynamically generate diagrams by calling drawing tools.

Enhanced with plan-aware generation for better accuracy and JEE/NEET compliance.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime

from .diagram_toolkit import DiagramBuilder, get_diagram_tools, get_diagram_system_prompt
from .academic_standards import AcademicStandards

if TYPE_CHECKING:
    from .diagram_planner import DiagramPlan

logger = logging.getLogger(__name__)


class ToolBasedDiagramGenerator:
    """
    Generates diagrams using LLM tool calling.

    The LLM is given drawing tools and constructs diagrams by
    calling these tools in sequence. Supports regeneration with
    feedback from verification.
    """

    def __init__(self, openai_service=None, kimi_service=None, use_kimi: bool = True):
        """
        Initialize the generator.

        Args:
            openai_service: The AsyncOpenAIService instance for LLM calls (fallback)
            kimi_service: The KimiService instance for LLM calls (preferred)
            use_kimi: Whether to use Kimi for diagram generation (default: True)
        """
        self._openai_service = openai_service
        self._kimi_service = kimi_service
        self._use_kimi = use_kimi and kimi_service is not None
        self._max_tool_calls = 30  # Safety limit on tool calls

        if self._use_kimi:
            logger.info("ToolBasedDiagramGenerator initialized with Kimi K2.5")
        else:
            logger.info("ToolBasedDiagramGenerator initialized with OpenAI")
    
    async def generate_diagram(
        self,
        question_text: str,
        subject: str,
        diagram_description: Optional[str] = None,
        additional_context: Optional[str] = None,
        feedback: Optional[str] = None,
        plan: Optional["DiagramPlan"] = None,
    ) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """
        Generate a diagram for a question using LLM tool calling.
        
        Args:
            question_text: The question that needs a diagram
            subject: Subject (physics, maths, chemistry, biology)
            diagram_description: Optional description of what the diagram should show
            additional_context: Additional context for the LLM
            feedback: Optional feedback from previous failed attempt for regeneration
            plan: Optional pre-validated DiagramPlan for plan-aware generation
            
        Returns:
            Tuple of (image_bytes, metadata_dict)
            Returns (None, error_dict) if generation fails
        """
        from .diagram_toolkit import DiagramBuilder, get_diagram_tools, get_diagram_system_prompt
        
        builder = DiagramBuilder()
        
        # Build the prompt for the LLM
        if feedback:
            user_prompt = self._build_regeneration_prompt(
                question_text, subject, diagram_description, feedback, plan
            )
        elif plan:
            user_prompt = self._build_prompt_with_plan(
                question_text, subject, plan, additional_context
            )
        else:
            user_prompt = self._build_prompt(
                question_text, subject, diagram_description, additional_context
            )
        
        messages = [
            {"role": "system", "content": get_diagram_system_prompt()},
            {"role": "user", "content": user_prompt}
        ]
        
        tools = get_diagram_tools()
        tool_call_count = 0
        is_regeneration = feedback is not None
        
        try:
            while tool_call_count < self._max_tool_calls:
                # Call LLM with tools (Kimi or OpenAI)
                # Note: Kimi K2.5 ONLY supports temperature=0.6 (enforced in kimi_service)
                openai_temperature = 0.2 if not is_regeneration else 0.3

                if self._use_kimi and self._kimi_service:
                    response = await self._kimi_service.chat_completion_with_tools(
                        messages=messages,
                        tools=tools,
                        temperature=0.6,  # Kimi K2.5 only supports 0.6
                        max_tokens=2000,
                        tool_choice="auto"
                    )
                else:
                    response = await self._openai_service.chat_completion_with_tools(
                        messages=messages,
                        tools=tools,
                        model="gpt-4o",
                        temperature=openai_temperature,
                        max_tokens=2000,
                        tool_choice="auto"
                    )
                
                if not response.get("success"):
                    logger.error(f"LLM call failed: {response.get('error')}")
                    return None, {"error": response.get("error"), "tool_calls": tool_call_count}
                
                # Check if response has tool calls
                assistant_message = response.get("message", {})
                tool_calls = assistant_message.get("tool_calls", [])
                
                if not tool_calls:
                    # No more tool calls - LLM is done
                    logger.info(f"Diagram generation complete after {tool_call_count} tool calls")
                    break
                
                # Add assistant message to conversation
                messages.append(assistant_message)
                
                # Execute each tool call
                tool_results = []
                for tool_call in tool_calls:
                    tool_call_count += 1
                    function_name = tool_call.get("function", {}).get("name", "")
                    arguments_str = tool_call.get("function", {}).get("arguments", "{}")
                    
                    try:
                        arguments = json.loads(arguments_str)
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    logger.debug(f"Executing tool: {function_name}({arguments})")
                    
                    # Execute the tool
                    result = builder.execute_tool(function_name, arguments)
                    
                    tool_results.append({
                        "tool_call_id": tool_call.get("id"),
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(result)
                    })
                    
                    # Check if this was the finalize call
                    if function_name == "finalize_diagram":
                        logger.info(f"Diagram finalized after {tool_call_count} tool calls")
                        break
                
                # Add tool results to conversation
                messages.extend(tool_results)
                
                # Check if we just finalized
                if builder.is_finalized:
                    break
            
            # Get the image
            if builder.fig is not None:
                if not builder.is_finalized:
                    builder._tool_finalize_diagram()
                
                image_bytes = builder.get_image_bytes(format="png", dpi=150)
                
                metadata = {
                    "tool_calls": tool_call_count,
                    "elements_drawn": len(builder._elements),
                    "title": builder.title,
                    "generated_at": datetime.utcnow().isoformat(),
                    "is_regeneration": is_regeneration,
                    "used_plan": plan is not None,
                    "plan_complexity": plan.complexity_level if plan else None
                }
                
                return image_bytes, metadata
            else:
                return None, {"error": "No diagram was created", "tool_calls": tool_call_count}
                
        except Exception as e:
            logger.error(f"Diagram generation error: {e}", exc_info=True)
            return None, {"error": str(e), "tool_calls": tool_call_count}
        finally:
            builder.reset()
    
    def _build_prompt(
        self,
        question_text: str,
        subject: str,
        diagram_description: Optional[str],
        additional_context: Optional[str]
    ) -> str:
        """Build the user prompt for initial diagram generation."""
        
        prompt_parts = [
            f"Create a MINIMAL, CLEAN diagram for this {subject.upper()} competitive exam question.",
            "",
            "⚠️ CRITICAL: This is for JEE/NEET/CBSE exams. The diagram must be SIMPLE like exam papers.",
            "",
            "QUESTION:",
            question_text,
        ]
        
        if diagram_description:
            prompt_parts.extend([
                "",
                f"DIAGRAM SHOULD SHOW: {diagram_description}"
            ])
        
        if additional_context:
            prompt_parts.extend([
                "",
                f"CONTEXT: {additional_context}"
            ])
        
        # Subject-specific guidance
        question_lower = question_text.lower()
        
        # Detect diagram type
        is_circuit = any(word in question_lower for word in [
            'circuit', 'resistor', 'capacitor', 'battery', 'series', 'parallel',
            'ohm', 'ammeter', 'voltmeter', 'inductor', 'lamp', 'bulb', 'switch'
        ])
        
        is_force = any(word in question_lower for word in [
            'force', 'newton', 'tension', 'friction', 'weight', 'normal',
            'push', 'pull', 'applied force', 'free body'
        ])
        
        is_number_line = any(word in question_lower for word in [
            'number line', 'inequality', 'greater than', 'less than',
            'x >', 'x <', 'x ≥', 'x ≤', 'x>=', 'x<='
        ])
        
        is_geometry = any(word in question_lower for word in [
            'triangle', 'angle', 'rectangle', 'square', 'circle', 'polygon',
            'perpendicular', 'parallel lines', 'vertex', 'diagonal'
        ])
        
        # CRITICAL RULES - emphasized
        prompt_parts.extend([
            "",
            "=" * 60,
            "⛔ ABSOLUTE RULES - MUST FOLLOW ⛔",
            "=" * 60,
            "",
            "1. DO NOT SOLVE THE QUESTION",
            "   - No calculations, no derived values, no answers",
            "   - Only show what is GIVEN in the question",
            "",
            "2. KEEP IT MINIMAL",
            "   - Only essential elements needed to understand the setup",
            "   - No decorations, no extra details",
            "   - White/light background, black lines",
            "",
            "3. LABEL ONLY GIVEN VALUES",
            "   - If question says 'mass = 5 kg', show 'm = 5 kg'",
            "   - DO NOT add calculated values like 'F = ma = 50 N'",
            "   - DO NOT show formulas or solutions",
            "",
            "4. SIMPLE TITLE (2-3 words max)",
            "   - 'Force Diagram', 'Circuit', 'Number Line'",
            "   - NOT 'Force Diagram showing calculation of...'",
            "",
            "5. NO QUESTION TEXT IN DIAGRAM",
            "   - The diagram is separate from the question",
        ])
        
        if is_circuit:
            prompt_parts.extend([
                "",
                "=== CIRCUIT ===",
                "- Use draw_circuit tool directly",
                "- Show components with GIVEN values only",
                "- R₁ = 10Ω (if given), NOT 'R_eq = 5Ω' (calculated)",
                "",
                "Call draw_circuit (no create_canvas needed)."
            ])
        elif is_force:
            prompt_parts.extend([
                "",
                "=== FORCE DIAGRAM ===",
                "- Simple box/circle for object",
                "- Arrows for forces with GIVEN magnitudes only",
                "- Show: Weight, Normal, Friction, Applied (as mentioned)",
                "- DO NOT calculate net force or acceleration",
                "",
                "Start with create_canvas(title='Force Diagram')"
            ])
        elif is_number_line:
            prompt_parts.extend([
                "",
                "=== NUMBER LINE ===",
                "- Horizontal line with arrow at right",
                "- Mark scale points",
                "- Show the inequality region (shaded/bold)",
                "- DO NOT solve for x, just show the given inequality",
                "",
                "Start with create_canvas(title='Number Line')"
            ])
        elif is_geometry:
            prompt_parts.extend([
                "",
                "=== GEOMETRY ===",
                "- Draw shape with correct proportions",
                "- Label vertices (A, B, C)",
                "- Show GIVEN angles/sides only",
                "- DO NOT calculate unknown angles/sides",
                "",
                "Start with create_canvas(title='Geometry')"
            ])
        else:
            prompt_parts.extend([
                "",
                "Start with create_canvas(title='<2-3 word title>')."
            ])
        
        prompt_parts.extend([
            "",
            "End with finalize_diagram().",
            "",
            "Think like an exam paper designer: SIMPLE, CLEAN, MINIMAL."
        ])
        
        return "\n".join(prompt_parts)

    def _build_prompt_with_plan(
        self,
        question_text: str,
        subject: str,
        plan: "DiagramPlan",
        additional_context: Optional[str]
    ) -> str:
        """Build prompt using a pre-validated diagram plan for more accurate generation."""
        
        prompt_parts = [
            f"Create a MINIMAL exam-style diagram for this {subject.upper()} question.",
            "",
            "⚠️ THIS IS FOR JEE/NEET/CBSE - Keep it SIMPLE like real exam papers!",
            "",
            "=" * 60,
            "QUESTION:",
            "=" * 60,
            question_text,
        ]
        
        if additional_context:
            prompt_parts.extend([
                "",
                f"CONTEXT: {additional_context}"
            ])
        
        # Add the pre-validated plan (simplified)
        prompt_parts.extend([
            "",
            "=" * 60,
            "WHAT TO DRAW (from validated plan):",
            "=" * 60,
            "",
            f"Type: {plan.diagram_type}",
        ])
        
        # Only show GIVEN values - emphasize this
        if plan.extracted_values:
            prompt_parts.extend([
                "",
                "VALUES FROM QUESTION (show ONLY these, no calculations):",
            ])
            for key, value in plan.extracted_values.items():
                prompt_parts.append(f"  • {key}: {value}")
        
        # Objects to draw (simplified)
        if plan.objects:
            prompt_parts.extend([
                "",
                f"ELEMENTS TO DRAW ({len(plan.objects)} items):",
            ])
            for obj in plan.objects:
                prompt_parts.append(f"  • {obj.name} ({obj.type})")
        
        # Labels (simplified)
        if plan.labels:
            prompt_parts.extend([
                "",
                f"LABELS ({len(plan.labels)} items):",
            ])
            for label in plan.labels:
                prompt_parts.append(f"  • {label.text}")
        
        # CRITICAL RULES - very prominent
        prompt_parts.extend([
            "",
            "=" * 60,
            "⛔ ABSOLUTE RULES ⛔",
            "=" * 60,
            "",
            "1. DO NOT SOLVE THE QUESTION",
            "   ❌ No calculations",
            "   ❌ No derived values (like F=ma results)",
            "   ❌ No answers or solutions",
            "",
            "2. SHOW ONLY GIVEN INFORMATION",
            "   ✓ Values explicitly stated in question",
            "   ✓ Setup/arrangement described",
            "   ❌ NOT calculated results",
            "",
            "3. KEEP IT MINIMAL",
            "   ✓ Simple shapes",
            "   ✓ Clean lines",
            "   ✓ Readable labels (fontsize 14+)",
            "   ❌ No decorations",
            "   ❌ No formulas",
            "",
            "4. SHORT TITLE (2-3 words)",
            "   ✓ 'Force Diagram'",
            "   ✓ 'Circuit'", 
            "   ❌ 'Diagram showing the calculation of net force'",
            "",
        ])
        
        # Diagram-type specific
        diagram_type = plan.diagram_type.lower()
        if "circuit" in diagram_type:
            prompt_parts.append("Use draw_circuit tool directly.")
        else:
            title = plan.diagram_type.replace('_', ' ').title()
            if len(title) > 20:
                title = title.split()[0]  # Just first word
            prompt_parts.append(f"Start with create_canvas(title='{title}').")
        
        prompt_parts.extend([
            "End with finalize_diagram().",
            "",
            "Think: What would this look like in a JEE/NEET paper? THAT simple."
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_regeneration_prompt(
        self,
        question_text: str,
        subject: str,
        diagram_description: Optional[str],
        feedback: str,
        plan: Optional["DiagramPlan"] = None
    ) -> str:
        """Build the prompt for regenerating a diagram with feedback.
        
        ENHANCED: More specific guidance for fixing issues identified by verifier.
        """
        
        prompt_parts = [
            "⚠️ DIAGRAM REJECTED - TRY AGAIN ⚠️",
            "",
            "Your previous diagram had issues. READ THE FEEDBACK CAREFULLY and fix them.",
            "",
            "=" * 50,
            "QUESTION:",
            "=" * 50,
            question_text,
        ]
        
        if diagram_description:
            prompt_parts.extend([
                "",
                f"SHOULD SHOW: {diagram_description}"
            ])
        
        # Include plan values if available
        if plan and plan.extracted_values:
            prompt_parts.extend([
                "",
                "✓ GIVEN VALUES (MUST include all of these):",
            ])
            for key, value in plan.extracted_values.items():
                prompt_parts.append(f"  • {key}: {value}")
        
        prompt_parts.extend([
            "",
            "=" * 50,
            "⛔ PROBLEMS WITH YOUR PREVIOUS DIAGRAM ⛔",
            "=" * 50,
            feedback,
            "",
            "=" * 50,
            "HOW TO FIX",
            "=" * 50,
            "",
            "1. CORRECTNESS (Most Important)",
            "   - Check all values match the question EXACTLY",
            "   - Verify force directions are physically correct",
            "   - Don't show calculated values, only GIVEN ones",
            "",
            "2. VISUAL CLARITY (Fix these common issues)",
            "   - Labels overlapping shapes? → Use label_offset parameter",
            "   - Text too small? → Use fontsize=16",
            "   - Elements too crowded? → Spread them out more",
            "   - Labels overlapping each other? → Adjust positions",
            "",
            "3. LABEL POSITIONING",
            "   - For arrows: place label 0.3-0.5 units beyond the arrow head",
            "   - For shapes: place label OUTSIDE, not on top",
            "   - Use ha='center' and va='bottom' for text below shapes",
            "   - Use ha='center' and va='top' for text above shapes",
            "",
            "4. SPACING",
            "   - Main object should be at center of canvas (~5, 4)",
            "   - Leave 1+ unit gap between elements",
            "   - Call set_view() to ensure everything fits",
            "",
            "5. SIMPLICITY",
            "   - If diagram is too complex, remove less important elements",
            "   - A clear diagram with 3 labels beats a cluttered one with 10",
            "",
            "DRAWING SEQUENCE:",
            "1. create_canvas(title='Short Title')",
            "2. Draw main shapes first (rectangle, circle, polygon)",
            "3. Add arrows/lines",
            "4. Add labels with draw_text (fontsize 14-16)",
            "5. set_view() to frame everything",
            "6. finalize_diagram()",
            "",
            "For CIRCUITS: Use draw_circuit tool (better quality)",
        ])
        
        return "\n".join(prompt_parts)


async def generate_diagram_with_tools(
    openai_service,
    question_text: str,
    subject: str,
    diagram_description: Optional[str] = None,
) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """
    Convenience function to generate a diagram using tool calling.
    
    Args:
        openai_service: The OpenAI service instance
        question_text: The question needing a diagram
        subject: Subject area
        diagram_description: Optional description of desired diagram
        
    Returns:
        Tuple of (image_bytes, metadata)
    """
    generator = ToolBasedDiagramGenerator(openai_service)
    return await generator.generate_diagram(
        question_text=question_text,
        subject=subject,
        diagram_description=diagram_description
    )
