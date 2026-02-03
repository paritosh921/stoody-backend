"""
Diagram Toolkit - Tool-based diagram generation for LLMs.

This module provides drawing tools that can be called by an LLM to dynamically
construct diagrams. Instead of hardcoded diagram types, the LLM uses these
primitives to draw any diagram it needs.

Available Tools:
- create_canvas: Initialize a new diagram
- draw_circle: Draw a circle/disc/sphere
- draw_rectangle: Draw a rectangle/box
- draw_line: Draw a line
- draw_arrow: Draw an arrow (line with arrowhead)
- draw_polygon: Draw a polygon
- draw_arc: Draw an arc
- draw_text: Add text label
- draw_angle_arc: Draw an angle indicator
- set_axis_limits: Set the view bounds
- finalize: Save and return the diagram

The LLM calls these tools in sequence to build up a diagram.
"""

import io
import base64
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrowPatch, Arc, Wedge
from matplotlib.lines import Line2D
import numpy as np

# Try to import schemdraw for circuit diagrams
try:
    import schemdraw
    import schemdraw.elements as elm
    HAS_SCHEMDRAW = True
except ImportError:
    HAS_SCHEMDRAW = False

logger = logging.getLogger(__name__)


# Tool definitions for the LLM (OpenAI function calling format)
DIAGRAM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_canvas",
            "description": "Initialize a new diagram canvas. Call this FIRST before any drawing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "width": {"type": "number", "description": "Canvas width in inches (default: 10)"},
                    "height": {"type": "number", "description": "Canvas height in inches (default: 8)"},
                    "title": {"type": "string", "description": "Diagram title"},
                    "background_color": {"type": "string", "description": "Background color (default: white)"}
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draw_circle",
            "description": "Draw a circle. Use for: spheres, balls, cylinders (side view), discs, wheels, pulleys, atoms, cells.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate of center"},
                    "y": {"type": "number", "description": "Y coordinate of center"},
                    "radius": {"type": "number", "description": "Radius of circle"},
                    "fill_color": {"type": "string", "description": "Fill color (e.g., 'lightblue', '#87CEEB')"},
                    "edge_color": {"type": "string", "description": "Edge/border color (default: 'black')"},
                    "line_width": {"type": "number", "description": "Border width (default: 2)"},
                    "label": {"type": "string", "description": "Text label to show at center"},
                    "label_offset": {"type": "array", "items": {"type": "number"}, "description": "[dx, dy] offset for label from center"}
                },
                "required": ["x", "y", "radius"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draw_rectangle",
            "description": "Draw a rectangle. Use for: blocks, boxes, resistors, capacitors, walls, surfaces.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate of bottom-left corner (or center if centered=true)"},
                    "y": {"type": "number", "description": "Y coordinate of bottom-left corner (or center if centered=true)"},
                    "width": {"type": "number", "description": "Width of rectangle"},
                    "height": {"type": "number", "description": "Height of rectangle"},
                    "angle": {"type": "number", "description": "Rotation angle in degrees (default: 0)"},
                    "centered": {"type": "boolean", "description": "If true, x,y is the center (default: false)"},
                    "fill_color": {"type": "string", "description": "Fill color"},
                    "edge_color": {"type": "string", "description": "Edge color (default: 'black')"},
                    "line_width": {"type": "number", "description": "Border width (default: 2)"},
                    "label": {"type": "string", "description": "Text label to show at center"},
                    "label_offset": {"type": "array", "items": {"type": "number"}, "description": "[dx, dy] offset for label from center"}
                },
                "required": ["x", "y", "width", "height"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "draw_line",
            "description": "Draw a straight line. Use for: surfaces, ground, axes, connections, rays.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {"type": "number", "description": "X coordinate of start point"},
                    "y1": {"type": "number", "description": "Y coordinate of start point"},
                    "x2": {"type": "number", "description": "X coordinate of end point"},
                    "y2": {"type": "number", "description": "Y coordinate of end point"},
                    "color": {"type": "string", "description": "Line color (default: 'black')"},
                    "line_width": {"type": "number", "description": "Line width (default: 2)"},
                    "line_style": {"type": "string", "description": "Style: 'solid', 'dashed', 'dotted' (default: 'solid')"},
                    "label": {"type": "string", "description": "Label for the line"},
                    "label_position": {"type": "string", "description": "Where to place label: 'middle', 'start', 'end' (default: 'middle')"}
                },
                "required": ["x1", "y1", "x2", "y2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draw_arrow",
            "description": "Draw an arrow (line with arrowhead). Use for: force vectors, velocity, displacement, field lines, flow direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {"type": "number", "description": "X coordinate of arrow start (tail)"},
                    "y1": {"type": "number", "description": "Y coordinate of arrow start (tail)"},
                    "x2": {"type": "number", "description": "X coordinate of arrow end (head)"},
                    "y2": {"type": "number", "description": "Y coordinate of arrow end (head)"},
                    "color": {"type": "string", "description": "Arrow color (default: 'black')"},
                    "line_width": {"type": "number", "description": "Line width (default: 2)"},
                    "head_width": {"type": "number", "description": "Arrow head width (default: 0.15)"},
                    "head_length": {"type": "number", "description": "Arrow head length (default: 0.1)"},
                    "label": {"type": "string", "description": "Label for the arrow"},
                    "label_position": {"type": "string", "description": "Where to place label: 'head', 'tail', 'middle' (default: 'head')"},
                    "label_color": {"type": "string", "description": "Label color (default: same as arrow)"}
                },
                "required": ["x1", "y1", "x2", "y2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draw_polygon",
            "description": "Draw a polygon. Use for: triangles, inclined planes, irregular shapes, wedges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "points": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "List of [x, y] coordinates for each vertex"
                    },
                    "fill_color": {"type": "string", "description": "Fill color"},
                    "edge_color": {"type": "string", "description": "Edge color (default: 'black')"},
                    "line_width": {"type": "number", "description": "Border width (default: 2)"},
                    "alpha": {"type": "number", "description": "Transparency 0-1 (default: 1)"},
                    "label": {"type": "string", "description": "Label at centroid"}
                },
                "required": ["points"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draw_arc",
            "description": "Draw an arc (part of a circle). Use for: angles, curved mirrors, lenses, partial circles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate of arc center"},
                    "y": {"type": "number", "description": "Y coordinate of arc center"},
                    "radius": {"type": "number", "description": "Arc radius"},
                    "start_angle": {"type": "number", "description": "Start angle in degrees"},
                    "end_angle": {"type": "number", "description": "End angle in degrees"},
                    "color": {"type": "string", "description": "Arc color (default: 'black')"},
                    "line_width": {"type": "number", "description": "Line width (default: 1.5)"},
                    "label": {"type": "string", "description": "Label for the arc (e.g., angle value)"},
                    "label_position": {"type": "string", "description": "'inside' or 'outside' the arc (default: 'inside')"}
                },
                "required": ["x", "y", "radius", "start_angle", "end_angle"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draw_text",
            "description": "Add a text label at any position. Use for: labels, values, descriptions, annotations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate"},
                    "y": {"type": "number", "description": "Y coordinate"},
                    "text": {"type": "string", "description": "Text content"},
                    "fontsize": {"type": "number", "description": "Font size (default: 14, minimum: 12 for readability)"},
                    "color": {"type": "string", "description": "Text color (default: 'black')"},
                    "fontweight": {"type": "string", "description": "'normal' or 'bold' (default: 'normal')"},
                    "ha": {"type": "string", "description": "Horizontal alignment: 'left', 'center', 'right' (default: 'center')"},
                    "va": {"type": "string", "description": "Vertical alignment: 'top', 'center', 'bottom' (default: 'center')"},
                    "rotation": {"type": "number", "description": "Rotation angle in degrees (default: 0)"},
                    "bbox": {"type": "boolean", "description": "Add background box around text (default: false)"}
                },
                "required": ["x", "y", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draw_angle_marker",
            "description": "Draw an angle marker (arc with label). Use for: showing angles in geometry/physics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vertex_x": {"type": "number", "description": "X coordinate of angle vertex"},
                    "vertex_y": {"type": "number", "description": "Y coordinate of angle vertex"},
                    "angle1": {"type": "number", "description": "First ray angle in degrees (from positive x-axis)"},
                    "angle2": {"type": "number", "description": "Second ray angle in degrees"},
                    "radius": {"type": "number", "description": "Arc radius (default: 0.5)"},
                    "label": {"type": "string", "description": "Angle label (e.g., '45°', 'θ')"},
                    "color": {"type": "string", "description": "Arc color (default: 'black')"}
                },
                "required": ["vertex_x", "vertex_y", "angle1", "angle2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_view",
            "description": "Set the visible area of the diagram. Call after all drawing to adjust view.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_min": {"type": "number", "description": "Minimum x value to show"},
                    "x_max": {"type": "number", "description": "Maximum x value to show"},
                    "y_min": {"type": "number", "description": "Minimum y value to show"},
                    "y_max": {"type": "number", "description": "Maximum y value to show"},
                    "equal_aspect": {"type": "boolean", "description": "Keep equal scaling for x and y (default: true)"}
                },
                "required": ["x_min", "x_max", "y_min", "y_max"]
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "finalize_diagram",
            "description": "Finalize and save the diagram. Call this LAST after all drawing is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "show_grid": {"type": "boolean", "description": "Show grid lines (default: false)"},
                    "show_axes": {"type": "boolean", "description": "Show x and y axes (default: false)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draw_circuit",
            "description": "Draw a proper circuit diagram with standard electrical symbols. Use this for all circuit diagrams instead of drawing with basic shapes. Supports series and parallel circuits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "circuit_type": {
                        "type": "string",
                        "enum": ["series", "parallel"],
                        "description": "Type of circuit: 'series' (all components in one loop) or 'parallel' (components in parallel branches)"
                    },
                    "components": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["resistor", "capacitor", "inductor", "lamp", "bulb", "switch", "ammeter", "voltmeter", "led", "diode", "fuse"],
                                    "description": "Component type"
                                },
                                "value": {"type": "string", "description": "Component value (e.g., '10Ω', '5μF', '100mH')"},
                                "label": {"type": "string", "description": "Component label (e.g., 'R1', 'C1', 'L1')"}
                            },
                            "required": ["type"]
                        },
                        "description": "List of circuit components (resistors, capacitors, etc.)"
                    },
                    "voltage": {"type": "number", "description": "Battery/source voltage in volts (default: 12)"},
                    "show_values": {"type": "boolean", "description": "Show component values on the diagram (default: true)"},
                    "show_current_direction": {"type": "boolean", "description": "Show current flow direction arrows (default: false)"}
                },
                "required": ["circuit_type", "components"]
            }
        }
    }
]


class DiagramBuilder:
    """
    A class that manages diagram construction through tool calls.
    
    The LLM calls tools to build up a diagram, and this class
    executes those calls using matplotlib.
    """
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.title = ""
        self.is_finalized = False
        self._elements = []  # Track what's been drawn for debugging
        self._circuit_image_data = None  # For circuit diagrams drawn with schemdraw
        self._view_set = False  # Track if view bounds were explicitly set  # For circuit diagrams drawn with schemdraw  # Track what's been drawn for debugging
        
    def reset(self):
        """Reset the builder for a new diagram."""
        if self.fig:
            plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.title = ""
        self.is_finalized = False
        self._elements = []
        self._circuit_image_data = None  # Clear any circuit diagram data
        self._view_set = False  # Reset view tracking  # Clear any circuit diagram data
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a drawing tool call.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Result dict with success status and any output
        """
        try:
            method = getattr(self, f"_tool_{tool_name}", None)
            if method is None:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
            
            result = method(**arguments)
            self._elements.append({"tool": tool_name, "args": arguments})
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Tool execution error: {tool_name} - {e}")
            return {"success": False, "error": str(e)}
    
    def _tool_create_canvas(self, title: str, width: float = 10, height: float = 8, 
                           background_color: str = "white") -> str:
        """Initialize the canvas."""
        self.reset()
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.fig.patch.set_facecolor(background_color)
        self.ax.set_facecolor(background_color)
        self.title = title
        
        # Set initial view with reasonable defaults (0-10 range)
        # This prevents empty diagrams when set_view is not called
        self.ax.set_xlim(-1, 11)
        self.ax.set_ylim(-1, 11)
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.axis('off')
        
        return f"Canvas created: {width}x{height} inches"
    
    def _tool_draw_circle(self, x: float, y: float, radius: float,
                         fill_color: str = "lightblue", edge_color: str = "black",
                         line_width: float = 2, label: str = None,
                         label_offset: List[float] = None) -> str:
        """Draw a circle with optional label."""
        if self.ax is None:
            raise ValueError("Canvas not created. Call create_canvas first.")
        
        circle = Circle((x, y), radius, facecolor=fill_color, edgecolor=edge_color,
                        linewidth=line_width)
        self.ax.add_patch(circle)
        
        if label:
            lx, ly = x, y
            if label_offset:
                lx += label_offset[0]
                ly += label_offset[1]
            # Larger font with white background for readability
            self.ax.text(lx, ly, label, ha='center', va='center', fontsize=13, 
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                 edgecolor='none', alpha=0.7))
        
        return f"Circle drawn at ({x}, {y}) with radius {radius}"
    
    def _tool_draw_rectangle(self, x: float, y: float, width: float, height: float,
                            angle: float = 0, centered: bool = False,
                            fill_color: str = "lightblue", edge_color: str = "black",
                            line_width: float = 2, label: str = None,
                            label_offset: list = None) -> str:
        """Draw a rectangle with optional label."""
        if self.ax is None:
            raise ValueError("Canvas not created. Call create_canvas first.")
        
        if centered:
            x = x - width / 2
            y = y - height / 2
        
        if angle != 0:
            # For rotated rectangles, use polygon
            cx, cy = x + width/2, y + height/2
            corners = np.array([
                [x, y], [x + width, y], [x + width, y + height], [x, y + height]
            ])
            # Rotate around center
            angle_rad = np.radians(angle)
            rot_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])
            corners_centered = corners - [cx, cy]
            corners_rotated = corners_centered @ rot_matrix.T + [cx, cy]
            rect = Polygon(corners_rotated, facecolor=fill_color, edgecolor=edge_color,
                          linewidth=line_width)
            center = (cx, cy)
        else:
            rect = Rectangle((x, y), width, height, facecolor=fill_color,
                            edgecolor=edge_color, linewidth=line_width)
            center = (x + width/2, y + height/2)
        
        self.ax.add_patch(rect)
        
        if label:
            # Calculate label position with optional offset
            label_x, label_y = center
            if label_offset and len(label_offset) >= 2:
                label_x += label_offset[0]
                label_y += label_offset[1]
            
            # Larger font with background for readability
            self.ax.text(label_x, label_y, label, ha='center', va='center',
                        fontsize=13, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                 edgecolor='none', alpha=0.7))
        
        return f"Rectangle drawn at ({x}, {y}) size {width}x{height}"
    
    def _tool_draw_line(self, x1: float, y1: float, x2: float, y2: float,
                       color: str = "black", line_width: float = 2,
                       line_style: str = "solid", label: str = None,
                       label_position: str = "middle") -> str:
        """Draw a line with optional label."""
        if self.ax is None:
            raise ValueError("Canvas not created. Call create_canvas first.")
        
        style_map = {"solid": "-", "dashed": "--", "dotted": ":"}
        ls = style_map.get(line_style, "-")
        
        self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=line_width, linestyle=ls)
        
        if label:
            if label_position == "start":
                lx, ly = x1, y1
            elif label_position == "end":
                lx, ly = x2, y2
            else:  # middle
                lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
            # Larger font with background
            self.ax.text(lx, ly, label, ha='center', va='bottom', fontsize=13,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                 edgecolor='none', alpha=0.7))
        
        return f"Line drawn from ({x1}, {y1}) to ({x2}, {y2})"
    
    def _tool_draw_arrow(self, x1: float, y1: float, x2: float, y2: float,
                        color: str = "black", line_width: float = 2,
                        head_width: float = 0.15, head_length: float = 0.1,
                        label: str = None, label_position: str = "head",
                        label_color: str = None) -> str:
        """Draw an arrow with optional label."""
        if self.ax is None:
            raise ValueError("Canvas not created. Call create_canvas first.")
        
        dx, dy = x2 - x1, y2 - y1
        self.ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color=color, lw=line_width,
                                       mutation_scale=15))
        
        if label:
            lcolor = label_color or color
            if label_position == "tail":
                lx, ly = x1 - 0.2, y1
            elif label_position == "middle":
                lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
                # Offset perpendicular to arrow for better visibility
                length = math.sqrt(dx**2 + dy**2)
                if length > 0:
                    # Perpendicular offset
                    perp_x = -dy / length * 0.3
                    perp_y = dx / length * 0.3
                    lx += perp_x
                    ly += perp_y
            else:  # head
                # Offset beyond arrow head
                length = math.sqrt(dx**2 + dy**2)
                if length > 0:
                    offset = 0.3
                    lx = x2 + offset * dx / length
                    ly = y2 + offset * dy / length
                else:
                    lx, ly = x2, y2
            
            # Use larger font and add white background for readability
            self.ax.text(lx, ly, label, ha='center', va='center', fontsize=13,
                        color=lcolor, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                 edgecolor='none', alpha=0.7))
        
        return f"Arrow drawn from ({x1}, {y1}) to ({x2}, {y2}) with label '{label}'"
    
    def _tool_draw_polygon(self, points: List[List[float]], fill_color: str = "tan",
                          edge_color: str = "black", line_width: float = 2,
                          alpha: float = 1.0, label: str = None) -> str:
        """Draw a polygon with optional label."""
        if self.ax is None:
            raise ValueError("Canvas not created. Call create_canvas first.")
        
        poly = Polygon(points, facecolor=fill_color, edgecolor=edge_color,
                      linewidth=line_width, alpha=alpha)
        self.ax.add_patch(poly)
        
        if label:
            # Place label at centroid
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            # Larger font with background
            self.ax.text(cx, cy, label, ha='center', va='center', fontsize=13,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                 edgecolor='none', alpha=0.7))
        
        return f"Polygon drawn with {len(points)} vertices"
    
    def _tool_draw_arc(self, x: float, y: float, radius: float,
                      start_angle: float, end_angle: float,
                      color: str = "black", line_width: float = 1.5,
                      label: str = None, label_position: str = "inside") -> str:
        """Draw an arc."""
        if self.ax is None:
            raise ValueError("Canvas not created. Call create_canvas first.")
        
        arc = Arc((x, y), 2*radius, 2*radius, angle=0,
                 theta1=start_angle, theta2=end_angle,
                 color=color, linewidth=line_width)
        self.ax.add_patch(arc)
        
        if label:
            # Place label at middle of arc
            mid_angle = math.radians((start_angle + end_angle) / 2)
            r = radius * (0.6 if label_position == "inside" else 1.3)
            lx = x + r * math.cos(mid_angle)
            ly = y + r * math.sin(mid_angle)
            self.ax.text(lx, ly, label, ha='center', va='center', fontsize=10)
        
        return f"Arc drawn at ({x}, {y})"
    
    def _tool_draw_text(self, x: float, y: float, text: str,
                       fontsize: float = 14, color: str = "black",
                       fontweight: str = "normal", ha: str = "center",
                       va: str = "center", rotation: float = 0,
                       bbox: bool = False) -> str:
        """Add text. Default fontsize=14 for readability."""
        if self.ax is None:
            raise ValueError("Canvas not created. Call create_canvas first.")
        
        # Ensure minimum readable font size
        fontsize = max(fontsize, 12)
        
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", 
                         edgecolor="gray", alpha=0.8) if bbox else None
        
        self.ax.text(x, y, text, fontsize=fontsize, color=color,
                    fontweight=fontweight, ha=ha, va=va, rotation=rotation,
                    bbox=bbox_props)
        
        return f"Text '{text}' added at ({x}, {y}) with fontsize {fontsize}"
    
    def _tool_draw_angle_marker(self, vertex_x: float, vertex_y: float,
                               angle1: float, angle2: float,
                               radius: float = 0.5, label: str = None,
                               color: str = "black") -> str:
        """Draw an angle marker arc with optional label."""
        if self.ax is None:
            raise ValueError("Canvas not created. Call create_canvas first.")
        
        # Ensure angles are in correct order
        if angle1 > angle2:
            angle1, angle2 = angle2, angle1
        
        arc = Arc((vertex_x, vertex_y), 2*radius, 2*radius, angle=0,
                 theta1=angle1, theta2=angle2, color=color, linewidth=1.5)
        self.ax.add_patch(arc)
        
        if label:
            mid_angle = math.radians((angle1 + angle2) / 2)
            lx = vertex_x + radius * 1.5 * math.cos(mid_angle)
            ly = vertex_y + radius * 1.5 * math.sin(mid_angle)
            # Larger font with background
            self.ax.text(lx, ly, label, ha='center', va='center', fontsize=13,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                 edgecolor='none', alpha=0.7))
        
        return f"Angle marker drawn at ({vertex_x}, {vertex_y}) from {angle1}° to {angle2}°"
    
    def _tool_set_view(self, x_min: float, x_max: float, y_min: float, y_max: float,
                      equal_aspect: bool = True) -> str:
        """Set the view bounds."""
        if self.ax is None:
            raise ValueError("Canvas not created. Call create_canvas first.")
        
        # Add small padding to ensure elements at edges are visible
        padding_x = (x_max - x_min) * 0.05
        padding_y = (y_max - y_min) * 0.05
        
        self.ax.set_xlim(x_min - padding_x, x_max + padding_x)
        self.ax.set_ylim(y_min - padding_y, y_max + padding_y)
        
        if equal_aspect:
            self.ax.set_aspect('equal', adjustable='datalim')
        
        # Mark that view was explicitly set
        self._view_set = True
        
        return f"View set to x:[{x_min}, {x_max}], y:[{y_min}, {y_max}]"
    
    def _tool_finalize_diagram(self, show_grid: bool = False, 
                               show_axes: bool = False) -> str:
        """Finalize the diagram."""
        if self.ax is None:
            raise ValueError("Canvas not created. Call create_canvas first.")
        
        if show_grid:
            self.ax.grid(True, alpha=0.3)
        
        if show_axes:
            self.ax.axhline(y=0, color='black', linewidth=0.5)
            self.ax.axvline(x=0, color='black', linewidth=0.5)
            self.ax.axis('on')
        else:
            self.ax.axis('off')
        
        if self.title:
            self.ax.set_title(self.title, fontsize=14, fontweight='bold')
        
        # Auto-fit view if no explicit limits were set
        # This ensures all drawn elements are visible
        x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()
        
        # Check if limits are default (not set by set_view)
        if not hasattr(self, '_view_set') or not self._view_set:
            # Let matplotlib autoscale, then add padding
            self.ax.autoscale_view()
            x_lim = self.ax.get_xlim()
            y_lim = self.ax.get_ylim()
            
            # Add 10% padding on all sides
            x_range = x_lim[1] - x_lim[0]
            y_range = y_lim[1] - y_lim[0]
            
            # Ensure minimum range to prevent tiny diagrams
            if x_range < 1:
                x_range = 10
            if y_range < 1:
                y_range = 10
            
            padding_x = x_range * 0.1
            padding_y = y_range * 0.1
            
            self.ax.set_xlim(x_lim[0] - padding_x, x_lim[1] + padding_x)
            self.ax.set_ylim(y_lim[0] - padding_y, y_lim[1] + padding_y)
        
        self.fig.tight_layout()
        self.is_finalized = True
        
        return "Diagram finalized"
    

    def _tool_draw_circuit(self, circuit_type: str, components: List[Dict[str, Any]],
                          voltage: float = 12, show_values: bool = True,
                          show_current_direction: bool = False) -> str:
        """Draw a circuit diagram using schemdraw for proper electrical symbols."""
        if not HAS_SCHEMDRAW:
            # Fallback to basic drawing if schemdraw not available
            return self._draw_circuit_fallback(circuit_type, components, voltage, show_values)
        
        # Close any existing matplotlib figure as we'll use schemdraw
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        
        # Create circuit with schemdraw
        with schemdraw.Drawing() as d:
            d.config(unit=3, fontsize=12)
            
            if circuit_type == 'series':
                self._draw_series_circuit(d, components, voltage, show_values, show_current_direction)
            elif circuit_type == 'parallel':
                self._draw_parallel_circuit(d, components, voltage, show_values, show_current_direction)
            else:
                raise ValueError(f"Unknown circuit type: {circuit_type}")
            
            # Convert schemdraw to matplotlib figure
            buf = io.BytesIO()
            d.save(buf, fmt='png', dpi=150)
            buf.seek(0)
            
            # Store the image data for later retrieval
            self._circuit_image_data = buf.getvalue()
            self.is_finalized = True
        
        return f"{circuit_type.title()} circuit drawn with {len(components)} component(s)"
    
    def _draw_series_circuit(self, d, components: List[Dict], voltage: float,
                            show_values: bool, show_current: bool):
        """Draw a series circuit with proper topology."""
        # Start with ground and battery on the left
        d += elm.Ground().at((0, 0))
        batt = d.add(elm.Battery().up().label(f'{voltage}V' if show_values else 'V').reverse())
        
        # Add current arrow if requested
        if show_current:
            d += elm.CurrentLabelInline(direction='in').at(batt.start).label('I')
        
        # Wire going right from battery
        d += elm.Line().right().length(d.unit * 0.5)
        
        # Add each component
        for comp in components:
            comp_type = comp.get('type', 'resistor')
            value = comp.get('value', '')
            label = comp.get('label', '')
            
            # Build display label
            if show_values and value and label:
                lbl = f'{label}\n{value}'
            elif show_values and value:
                lbl = str(value)
            elif label:
                lbl = label
            else:
                lbl = ''
            
            # Add the component
            self._add_component(d, comp_type, lbl, direction='right')
        
        # Complete the loop
        d += elm.Line().right().length(d.unit * 0.5)
        d += elm.Line().down().toy(0)
        d += elm.Line().left().tox(0)
    
    def _draw_parallel_circuit(self, d, components: List[Dict], voltage: float,
                              show_values: bool, show_current: bool):
        """Draw a parallel circuit with proper branch topology."""
        n = len(components) if components else 1
        spacing = d.unit * 1.2  # Space between parallel branches
        total_width = spacing * (n + 1)
        
        # Ground and battery on left
        d += elm.Ground().at((0, 0))
        batt = d.add(elm.Battery().up().length(d.unit * 1.5).label(f'{voltage}V' if show_values else 'V').reverse())
        
        if show_current:
            d += elm.CurrentLabelInline(direction='in').at(batt.start).label('I')
        
        battery_top = d.here
        
        # Top rail going right
        d += elm.Line().right().length(total_width)
        top_right = d.here
        
        # Draw parallel branches
        for i, comp in enumerate(components):
            comp_type = comp.get('type', 'resistor')
            value = comp.get('value', '')
            label = comp.get('label', '')
            
            # Build display label
            if show_values and value and label:
                lbl = f'{label}\n{value}'
            elif show_values and value:
                lbl = str(value)
            elif label:
                lbl = label
            else:
                lbl = ''
            
            # Position for this branch
            branch_x = battery_top[0] + (i + 1) * spacing
            
            # Draw branch: short line down, component, short line to bottom
            d.push()
            d += elm.Line().at((branch_x, battery_top[1])).down().length(d.unit * 0.3)
            self._add_component(d, comp_type, lbl, direction='down')
            d += elm.Line().down().toy(0)
            d.pop()
        
        # Bottom rail
        d += elm.Line().at((0, 0)).right().length(total_width)
        
        # Close on right side
        d += elm.Line().at(top_right).down().toy(0)
    
    def _add_component(self, d, comp_type: str, label: str, direction: str = 'right'):
        """Add a circuit component in the specified direction."""
        # Map direction to schemdraw method
        if direction == 'right':
            dir_method = lambda e: e.right()
        elif direction == 'down':
            dir_method = lambda e: e.down()
        elif direction == 'left':
            dir_method = lambda e: e.left()
        else:
            dir_method = lambda e: e.up()
        
        # Component mapping
        component_map = {
            'resistor': elm.Resistor,
            'capacitor': elm.Capacitor,
            'inductor': elm.Inductor,
            'lamp': elm.Lamp,
            'bulb': elm.Lamp,
            'switch': elm.Switch,
            'ammeter': elm.MeterA,
            'voltmeter': elm.MeterV,
            'led': elm.LED,
            'diode': elm.Diode,
            'fuse': elm.Fuse,
        }
        
        component_class = component_map.get(comp_type, elm.Resistor)
        element = component_class()
        element = dir_method(element)
        
        if label:
            element = element.label(label)
        
        d += element
    
    def _draw_circuit_fallback(self, circuit_type: str, components: List[Dict],
                              voltage: float, show_values: bool) -> str:
        """Fallback circuit drawing using matplotlib when schemdraw is not available."""
        if self.ax is None:
            self._tool_create_canvas(title=f"{circuit_type.title()} Circuit", width=10, height=6)
        
        # Simple representation using rectangles and lines
        if circuit_type == 'series':
            # Draw battery
            self._tool_draw_rectangle(0, 2, 0.5, 2, fill_color='white', edge_color='black')
            self._tool_draw_line(0.15, 2.2, 0.35, 2.2, line_width=3)  # Negative terminal
            self._tool_draw_line(0.1, 3.8, 0.4, 3.8, line_width=1)  # Positive terminal
            if show_values:
                self._tool_draw_text(0.7, 3, f'{voltage}V', fontsize=10)
            
            # Top wire
            self._tool_draw_line(0.25, 4, 0.25, 5, color='black')
            self._tool_draw_line(0.25, 5, 2, 5, color='black')
            
            # Components
            x_pos = 2
            for comp in components:
                # Draw resistor symbol (zigzag approximation with rectangle)
                self._tool_draw_rectangle(x_pos, 4.5, 1.5, 1, fill_color='#f0f0f0', edge_color='black')
                value = comp.get('value', '')
                label = comp.get('label', '')
                display = f"{label}={value}" if label and value else (value or label or '')
                if display and show_values:
                    self._tool_draw_text(x_pos + 0.75, 5.8, display, fontsize=9)
                x_pos += 2.5
            
            # Complete circuit
            self._tool_draw_line(x_pos - 1, 5, x_pos, 5, color='black')
            self._tool_draw_line(x_pos, 5, x_pos, 1, color='black')
            self._tool_draw_line(x_pos, 1, 0.25, 1, color='black')
            self._tool_draw_line(0.25, 1, 0.25, 2, color='black')
        
        self._tool_set_view(-1, 12, 0, 7)
        return f"Circuit drawn with {len(components)} components (fallback mode)"

    def get_image_bytes(self, format: str = "png", dpi: int = 150) -> bytes:
        """Get the diagram as image bytes."""
        # Check if we have circuit image data from schemdraw
        if hasattr(self, '_circuit_image_data') and self._circuit_image_data:
            return self._circuit_image_data
        
        if self.fig is None:
            raise ValueError("No diagram created")
        
        if not self.is_finalized:
            self._tool_finalize_diagram()
        
        buf = io.BytesIO()
        self.fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight',
                        facecolor=self.fig.get_facecolor())
        buf.seek(0)
        return buf.getvalue()
    
    def get_base64(self, format: str = "png", dpi: int = 150) -> str:
        """Get the diagram as base64 encoded string."""
        img_bytes = self.get_image_bytes(format=format, dpi=dpi)
        return base64.b64encode(img_bytes).decode('utf-8')


def get_diagram_tools() -> List[Dict]:
    """Get the tool definitions for the LLM."""
    return DIAGRAM_TOOLS


def get_diagram_system_prompt() -> str:
    """Get the system prompt explaining how to use diagram tools."""
    return """You are creating diagrams for JEE/NEET/CBSE exam papers.

⚠️ CRITICAL RULE: DO NOT SOLVE THE QUESTION
Your job is ONLY to show the SETUP - not the solution.

=== WHAT EXAM DIAGRAMS LOOK LIKE ===

Real exam diagrams are:
✓ MINIMAL - only essential elements
✓ CLEAN - no clutter, no decorations
✓ SIMPLE - clear lines, basic shapes
✓ INFORMATIVE - shows GIVEN information only
✓ CLEAR LABELS - readable, well-positioned, with units

=== ABSOLUTE RULES ===

1. NEVER SOLVE THE QUESTION
   ❌ No calculations
   ❌ No derived values (like F=ma results)
   ❌ No answers
   ✓ Only show what is EXPLICITLY GIVEN

2. MINIMAL ELEMENTS
   ❌ No extra shapes or decorations
   ❌ No formulas or equations in the diagram
   ❌ No explanatory text
   ✓ Only what's needed to understand the setup

3. LABEL ONLY GIVEN VALUES
   ✓ "m = 5 kg" (if mass is given as 5 kg)
   ✓ "θ = 30°" (if angle is given as 30°)
   ❌ "F = ma = 50 N" (calculated - DON'T show)
   ❌ "a = F/m = 10 m/s²" (calculated - DON'T show)

4. SHORT TITLE (2-3 words)
   ✓ "Force Diagram"
   ✓ "Circuit"
   ✓ "Number Line"
   ❌ "Diagram showing the forces acting on a body moving on an inclined plane"

=== VISUAL QUALITY RULES (CRITICAL!) ===

5. LABEL POSITIONING
   ✓ Labels MUST NOT overlap shapes or other labels
   ✓ Place labels OUTSIDE shapes, not on top
   ✓ Use label_offset to adjust position if needed
   ✓ Minimum fontsize: 14 (use 16 for important values)

6. SPACING AND LAYOUT
   ✓ Leave adequate space between elements (at least 1 unit gap)
   ✓ Don't crowd multiple elements together
   ✓ Use the FULL canvas area - spread things out
   ✓ Call set_view() at the end to ensure all elements are visible

7. CLARITY OVER COMPLETENESS
   ✓ If space is tight, use fewer labels but make them readable
   ✓ A clear diagram with 3 labels is better than a cluttered one with 10
   ✓ Arrows should be clearly distinguishable from each other

=== DIAGRAM GUIDELINES ===

PHYSICS - Forces:
- Object: simple box (2x2 units) or circle (radius 0.8)
- Arrows for forces (from object center, length proportional to force)
- Label: "F = 50 N" ONLY if 50 N is GIVEN
- Colors: Weight=red, Normal=blue, Friction=orange, Applied=green
- Label offset: place labels 0.3-0.5 units away from arrow heads

PHYSICS - Circuits:
- Use draw_circuit tool (produces best quality)
- Show components with GIVEN values only
- NO calculated equivalent resistance

PHYSICS - Inclined Planes:
- Draw triangle for incline (use draw_polygon)
- Object on the plane (box or circle depending on question)
- Show angle at the base
- Forces: Weight (down), Normal (perpendicular), Friction (along plane)

MATHS - Number Lines:
- Horizontal line with arrow at right
- Mark scale points clearly (fontsize 14)
- Show inequality region (shaded/bold)
- DO NOT solve for x

MATHS - Geometry:
- Draw shape with accurate proportions
- Label vertices (A, B, C) with fontsize 14
- Show GIVEN angles/sides only
- DO NOT calculate unknown values

=== TOOL USAGE ===
1. create_canvas(title='Short Title') - 2-3 words max
2. Draw shapes/objects first (rectangles, circles, polygons)
3. Add arrows for forces/vectors
4. Add labels with draw_text (fontsize >= 14)
5. Call set_view(x_min, x_max, y_min, y_max) to frame the diagram
6. finalize_diagram()

For circuits: draw_circuit directly (no canvas needed)

=== COLORS ===
- Lines: black (#000000)
- Objects: lightblue (#87CEEB), lightgray (#D3D3D3)
- Labels: black, fontsize 14-16
- Force arrows: red, blue, green, orange (one color per force type)
- Keep it simple - maximum 4-5 colors total

=== COORDINATE TIPS ===
- Standard canvas is 10x8 units
- Center your main object around (5, 4)
- Leave margins: don't draw below y=0.5 or above y=7.5
- Leave room for labels: if arrow goes to (8, 6), label at (8.5, 6.2)

Think: "Would a student looking at this for 5 seconds understand the setup?"
If labels are hard to read or overlapping, FIX THEM before finalizing."""
