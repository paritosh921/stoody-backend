import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import json

from services.diagram_engine.verified_diagram_service import VerifiedDiagramService
from services.diagram_engine.diagram_planner import DiagramPlanner, DiagramPlan

class TestVerifiedDiagramFlow(unittest.IsolatedAsyncioTestCase):
    
    async def asyncSetUp(self):
        self.mock_openai = AsyncMock()
        self.service = VerifiedDiagramService(
            openai_service=self.mock_openai,
            use_planning=True,
            use_enhanced_verification=True,
            use_specialized_renderers=True
        )
        
        # Mock the internal components
        self.service._planner = MagicMock(spec=DiagramPlanner)
        self.service._planner.create_plan = AsyncMock()
        self.service._planner.update_plan = AsyncMock()
        self.service._planner.validate_plan = MagicMock()
        
        self.service._generator = AsyncMock()
        self.service._verifier = MagicMock()
        self.service._verifier.verify_diagram_enhanced = AsyncMock()
        self.service._verifier.verify_diagram = AsyncMock()
        self.service._verifier.build_targeted_feedback = MagicMock()
        
        self.service._diagram_engine = MagicMock() # Sync mock for engine structure
        
        # Mock renderer retrieval
        self.mock_renderer = MagicMock()
        self.mock_renderer.render = AsyncMock()
        self.service._diagram_engine.get_renderer.return_value = self.mock_renderer
        self.mock_renderer.supports_type.return_value = True
        self.mock_renderer.render.return_value = MagicMock(image_data=b"fake_image", width=100, height=100)

    async def test_uses_provided_spec(self):
        """Test that the service uses the provided diagram_spec instead of planning from scratch."""
        
        # Setup
        diagram_spec = {
            "diagram_type": "series_circuit",
            "subject": "physics",
            "objects": [{"name": "R1", "type": "resistor"}],
            "labels": [],
            "extracted_values": {}
        }
        
        # Valid plan returned by validation
        valid_plan = DiagramPlan(**diagram_spec)
        self.service._planner.validate_plan.return_value = MagicMock(is_valid=True, issues=[])
        
        # Execution
        result = await self.service.generate_verified_diagram(
            question_text="Draw a circuit",
            subject="physics",
            diagram_spec=diagram_spec,
            skip_verification=True
        )
        
        # Verification
        # Should NOT call create_plan because spec was provided
        self.service._planner.create_plan.assert_not_called()
        
        # Should have validated the provided plan
        self.service._planner.validate_plan.assert_called()
        
    async def test_feedback_loop_updates_plan(self):
        """Test that feedback triggers update_plan and uses the new plan."""
        
        # Initial Plan
        initial_plan = DiagramPlan(
            diagram_type="series_circuit",
            subject="physics",
            objects=[{"name": "R1", "type": "resistor", "properties": {"value": "5"}}],
            labels=[]
        )
        
        # Setup mocks
        # Phase 1: Create plan
        self.service._planner.create_plan.return_value = (initial_plan, [])
        self.service._planner.validate_plan.return_value = MagicMock(is_valid=True, issues=[])
        
        # Phase 2: Render (Mock engine)
        self.service._diagram_engine.get_renderer.return_value = self.mock_renderer
        
        # Phase 3: Verify - First attempt fails
        bad_verification = MagicMock(
            is_acceptable=False, 
            status="rejected", 
            issues=[MagicMock(severity="critical", category=MagicMock(value="conceptual"), what="Wrong value", fix="Change to 10")],
            conceptual_score=0.4,
            labeling_score=0.8,
            visual_score=0.9
        )
        bad_verification.get_composite_score.return_value = 0.5
        
        # Second attempt succeeds
        good_verification = MagicMock(
            is_acceptable=True, 
            status="approved", 
            issues=[],
            conceptual_score=0.9,
            labeling_score=0.9,
            visual_score=0.9,
            get_composite_score=lambda: 0.95
        )
        
        self.service._verifier.verify_diagram_enhanced.side_effect = [bad_verification, good_verification]
        self.service._verifier.build_targeted_feedback.return_value = "Change resistor value to 10"
        
        # Phase 4: Correct - Update Plan
        updated_plan = DiagramPlan(
            diagram_type="series_circuit",
            subject="physics",
            objects=[{"name": "R1", "type": "resistor", "properties": {"value": "10"}}],
            labels=[]
        )
        self.service._planner.update_plan.return_value = (updated_plan, [])
        
        # Execution
        result = await self.service.generate_verified_diagram(
            question_text="Circuit with 10 ohm resistor",
            subject="physics"
        )
        
        # Assertions
        # 1. create_plan called initially
        self.service._planner.create_plan.assert_called_once()
        
        # 2. update_plan called with correct feedback
        self.service._planner.update_plan.assert_called_once()
        args = self.service._planner.update_plan.call_args
        self.assertEqual(args[0][0], initial_plan) # First arg is old plan
        self.assertEqual(args[0][1], "Change resistor value to 10") # Second arg is feedback
        
        # 3. Success result
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)
        
    async def test_mapping_to_specialized_renderer(self):
        """Test that the correct diagram type triggers the specialized renderer."""
        
        plan = DiagramPlan(
            diagram_type="molecule_2d", # This should map to specialized
            subject="chemistry", 
            objects=[], 
            labels=[]
        )
        self.service._planner.create_plan.return_value = (plan, [])
        
        self.service._verifier.verify_diagram_enhanced.return_value = MagicMock(
            is_acceptable=True,
            status="approved",
            conceptual_score=1.0,
            labeling_score=1.0,
            visual_score=1.0,
            alignment_score=1.0,
            academic_score=1.0,
            issues=[]
        )
        self.service._verifier.verify_diagram_enhanced.return_value.get_composite_score.return_value = 1.0
        
        # Execution
        await self.service.generate_verified_diagram("Draw a molecule", "chemistry")
        
        # Assertions
        # Should call _generate_with_engine (implied by accessing _diagram_engine)
        # We can check if _diagram_engine.get_renderer was called with correct subject
        # Note: In the real code it calls get_renderer(DiagramSubject(subject.lower()))
        # Since I mocked the engine, verified_diagram_service logic runs.
        
        # Verify supports_type checked "molecule_2d"
        self.mock_renderer.supports_type.assert_called_with("molecule_2d")
        
        # Verify render called
        self.mock_renderer.render.assert_called()

if __name__ == "__main__":
    unittest.main()
