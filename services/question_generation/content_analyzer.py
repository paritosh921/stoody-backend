"""
Content Analyzer Service

Analyzes educational content to extract key concepts, topics, and 
learning objectives for intelligent question generation.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ConceptType(str, Enum):
    """Types of educational concepts."""
    DEFINITION = "definition"
    FORMULA = "formula"
    THEOREM = "theorem"
    PROCESS = "process"
    EXAMPLE = "example"
    APPLICATION = "application"
    COMPARISON = "comparison"
    DIAGRAM_CONCEPT = "diagram_concept"  # Concepts that need visual representation


@dataclass
class ExtractedConcept:
    """A concept extracted from educational content."""
    concept_id: str
    name: str
    type: ConceptType
    description: str
    related_text: str  # The actual text chunk related to this concept
    difficulty_estimate: str = "medium"  # easy, medium, hard
    requires_diagram: bool = False
    diagram_type_hint: Optional[str] = None  # Suggested diagram type
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "related_text": self.related_text,
            "difficulty_estimate": self.difficulty_estimate,
            "requires_diagram": self.requires_diagram,
            "diagram_type_hint": self.diagram_type_hint,
            "keywords": self.keywords,
        }


@dataclass
class ContentAnalysis:
    """Result of content analysis."""
    subject: str
    grade: str
    chapter: Optional[str]
    total_concepts: int
    concepts: List[ExtractedConcept]
    main_topics: List[str]
    difficulty_distribution: Dict[str, int]  # {easy: 5, medium: 10, hard: 3}
    diagram_opportunities: int  # Number of concepts suitable for diagrams
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "grade": self.grade,
            "chapter": self.chapter,
            "total_concepts": self.total_concepts,
            "concepts": [c.to_dict() for c in self.concepts],
            "main_topics": self.main_topics,
            "difficulty_distribution": self.difficulty_distribution,
            "diagram_opportunities": self.diagram_opportunities,
        }


# Prompt for concept extraction
CONCEPT_EXTRACTION_PROMPT = """You are an expert educational content analyzer. Analyze the following {subject} content for {grade} level students.

CONTENT TO ANALYZE:
{content}

TASK: Extract all key concepts that could be tested in an exam. For each concept, identify:
1. What type of concept it is (definition, formula, theorem, process, example, application, comparison)
2. Whether it GENUINELY needs a diagram (BE CONSERVATIVE - most concepts do NOT need diagrams)
3. If it needs a diagram, what type would be appropriate
4. Estimated difficulty level for testing this concept

⚠️ IMPORTANT: BE VERY CONSERVATIVE about requires_diagram!
Most concepts do NOT need diagrams. Only mark requires_diagram: true when:
- The concept is inherently visual (geometry, graphs, circuits, anatomy)
- A diagram would significantly help students understand the question
- The question type specifically requires visual representation

DO NOT mark requires_diagram: true for:
- Definitions
- Conceptual explanations  
- Word problems (unless specifically about graphing)
- Historical or theoretical questions
- Formula derivations

SUBJECT-SPECIFIC GUIDANCE (when diagrams ARE appropriate):
{diagram_guidance}

OUTPUT FORMAT (JSON):
{{
    "main_topics": ["Topic 1", "Topic 2"],
    "concepts": [
        {{
            "name": "Name of the concept",
            "type": "definition|formula|theorem|process|example|application|comparison|diagram_concept",
            "description": "Brief description of what this concept is about",
            "related_text": "The relevant text excerpt from the content",
            "difficulty_estimate": "easy|medium|hard",
            "requires_diagram": true/false,
            "diagram_type_hint": "coordinate_graph|geometry_construction|circuit|etc or null",
            "keywords": ["keyword1", "keyword2"]
        }}
    ]
}}

Extract ALL testable concepts. Be thorough but focused on educational value.
Remember: requires_diagram should be FALSE for most concepts!"""


class ContentAnalyzerService:
    """
    Service for analyzing educational content and extracting concepts.
    """
    
    # Diagram guidance by subject - BE CONSERVATIVE about requiring diagrams
    DIAGRAM_GUIDANCE = {
        "maths": """
For Mathematics, diagrams are ONLY needed for:
✓ Graphing specific equations with given values → coordinate_graph
✓ Geometry problems with specific shapes/measurements → geometry_construction  
✓ Inequalities that need visual representation → number_line
✓ Set operations with specific sets → venn_diagram
✓ Data representation with specific values → bar_chart, pie_chart

⚠️ DO NOT require diagrams for:
✗ Definitions (e.g., "Define linear equation")
✗ Solving algebraic equations without graphing
✗ Word problems unless specifically about graphs
✗ Theorem statements without specific examples
✗ Conceptual explanations""",
        
        "physics": """
For Physics, diagrams are ONLY needed for:
✓ Circuit problems with specific component values → series_circuit, parallel_circuit
✓ Optics problems with given distances → ray_diagram_lens, ray_diagram_mirror
✓ Force problems with specific mass/force values → free_body_diagram
✓ Inclined plane with specific angle and mass → inclined_plane
✓ Projectile with specific velocity/angle → projectile_motion
✓ Waves with specific wavelength/amplitude → wave_diagram

⚠️ DO NOT require diagrams for:
✗ Definitions (e.g., "Define Newton's first law")
✗ Conceptual questions (e.g., "Explain inertia")
✗ Historical questions (e.g., "Describe Galileo's experiment")
✗ Derivations of formulas
✗ Theory explanations without numerical data""",
        
        "chemistry": """
For Chemistry, diagrams are ONLY needed for:
✓ Specific molecular structures → molecule_2d
✓ Lab setup descriptions → lab_setup_titration, lab_setup_distillation
✓ Electron configurations of specific elements → orbital_diagram

⚠️ DO NOT require diagrams for:
✗ Definitions of terms
✗ Balancing equations (text is sufficient)
✗ Conceptual questions about reactions
✗ Periodic table trends explanations""",
        
        "biology": """
For Biology, diagrams are ONLY needed for:
✓ Anatomy questions asking to label parts → human_heart, nephron, etc.
✓ Cell structure identification → plant_cell, animal_cell
✓ Process visualization (mitosis, meiosis, DNA replication)

⚠️ DO NOT require diagrams for:
✗ Definitions of biological terms
✗ Function explanations (e.g., "What is the function of mitochondria?")
✗ Comparison questions (text comparison is sufficient)
✗ Process descriptions without visualization need"""
    }
    
    def __init__(self, openai_service=None, kimi_service=None, use_kimi: bool = False):
        """
        Initialize the content analyzer.
        
        Args:
            openai_service: The OpenAI service for LLM calls (fallback)
            kimi_service: The Kimi service for LLM calls (preferred if use_kimi=True)
            use_kimi: Whether to use Kimi instead of OpenAI
        """
        self._openai_service = openai_service
        self._kimi_service = kimi_service
        self._use_kimi = use_kimi and kimi_service is not None
        
        # Kimi-specific settings
        if self._use_kimi:
            self._model = "kimi-k2.5"
            self._temperature = 0.6  # Recommended for kimi-k2.5 instant mode
            logger.info("ContentAnalyzer using Kimi model")
        else:
            self._model = "gpt-5-mini"
            self._temperature = 1
            logger.info("ContentAnalyzer using OpenAI model")  # gpt-5-nano only supports temperature=1
    
    async def analyze_content(
        self,
        content: str,
        subject: str,
        grade: str,
        chapter: Optional[str] = None,
    ) -> ContentAnalysis:
        """
        Analyze educational content to extract concepts.
        
        Args:
            content: The text content to analyze
            subject: Subject name (maths, physics, chemistry, biology)
            grade: Grade level
            chapter: Optional chapter name
            
        Returns:
            ContentAnalysis with extracted concepts
        """
        logger.info(f"Analyzing content for {subject} {grade}")
        logger.info(f"Content length: {len(content)} characters")
        logger.info(f"Using {'Kimi' if self._use_kimi else 'OpenAI'} for analysis")
        
        # Get subject-specific diagram guidance
        diagram_guidance = self.DIAGRAM_GUIDANCE.get(
            subject.lower(), 
            "Consider what diagrams would help visualize the concepts."
        )
        
        # Prepare prompt
        prompt = CONCEPT_EXTRACTION_PROMPT.format(
            subject=subject,
            grade=grade,
            content=content[:12000],  # Limit content size
            diagram_guidance=diagram_guidance,
        )
        
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Using model: {self._model}, temperature: {self._temperature}")
        
        # Call LLM (Kimi or OpenAI)
        if self._use_kimi and self._kimi_service:
            response = await self._kimi_service.chat_completion_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=8000,  # kimi-k2.5 has 256k context
                model=self._model,
            )
        else:
            response = await self._openai_service.chat_completion_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=8000,  # kimi-k2.5 has 256k context
                model=self._model,
            )
        
        if not response.get("success"):
            logger.error(f"LLM call failed: {response}")
            raise ValueError(f"Content analysis failed: {response.get('error')}")
        
        # Log the actual response
        response_text = response.get("response", "")
        logger.info(f"LLM response length: {len(response_text)} characters")
        if len(response_text) < 100:
            logger.warning(f"LLM response seems short: '{response_text}'")
        
        # Parse response
        concepts = self._parse_analysis_response(
            response_text,
            subject,
            grade,
            chapter,
        )
        
        return concepts
    
    def _parse_analysis_response(
        self,
        response_text: str,
        subject: str,
        grade: str,
        chapter: Optional[str],
    ) -> ContentAnalysis:
        """Parse LLM response into ContentAnalysis."""
        # Log the response for debugging
        if not response_text or not response_text.strip():
            logger.error("LLM returned empty response for content analysis")
            logger.error(f"Response was: '{response_text}'")
            return self._empty_analysis(subject, grade, chapter)
        
        logger.debug(f"Parsing analysis response: {response_text[:500]}...")
        
        try:
            # Extract JSON from response
            json_str = self._extract_json(response_text)
            
            if not json_str or not json_str.strip():
                logger.error(f"Could not extract JSON from response: {response_text[:500]}")
                return self._empty_analysis(subject, grade, chapter)
            
            data = json.loads(json_str)
            
            concepts = []
            difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
            diagram_count = 0
            
            for i, c in enumerate(data.get("concepts", [])):
                concept_type = c.get("type", "definition")
                try:
                    ctype = ConceptType(concept_type)
                except ValueError:
                    ctype = ConceptType.DEFINITION
                
                concept = ExtractedConcept(
                    concept_id=f"concept_{i+1}",
                    name=c.get("name", f"Concept {i+1}"),
                    type=ctype,
                    description=c.get("description", ""),
                    related_text=c.get("related_text", ""),
                    difficulty_estimate=c.get("difficulty_estimate", "medium"),
                    requires_diagram=c.get("requires_diagram", False),
                    diagram_type_hint=c.get("diagram_type_hint"),
                    keywords=c.get("keywords", []),
                )
                
                concepts.append(concept)
                
                # Count difficulty
                diff = concept.difficulty_estimate
                if diff in difficulty_counts:
                    difficulty_counts[diff] += 1
                
                # Count diagram opportunities
                if concept.requires_diagram:
                    diagram_count += 1
            
            logger.info(f"Successfully parsed {len(concepts)} concepts from content")
            
            return ContentAnalysis(
                subject=subject,
                grade=grade,
                chapter=chapter,
                total_concepts=len(concepts),
                concepts=concepts,
                main_topics=data.get("main_topics", []),
                difficulty_distribution=difficulty_counts,
                diagram_opportunities=diagram_count,
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis response: {e}")
            logger.error(f"Response text was: {response_text[:1000]}")
            return self._empty_analysis(subject, grade, chapter)
    
    def _empty_analysis(self, subject: str, grade: str, chapter: Optional[str]) -> ContentAnalysis:
        """Return empty analysis when parsing fails."""
        return ContentAnalysis(
            subject=subject,
            grade=grade,
            chapter=chapter,
            total_concepts=0,
            concepts=[],
            main_topics=[],
            difficulty_distribution={"easy": 0, "medium": 0, "hard": 0},
            diagram_opportunities=0,
        )
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text."""
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        
        if "{" in text:
            start = text.find("{")
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
        
        return text
