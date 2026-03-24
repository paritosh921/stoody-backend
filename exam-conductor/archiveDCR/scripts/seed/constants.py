"""Shared constants for seed data generation."""

from __future__ import annotations

import struct

SUBJECTS = [
    "Mathematics", "Physics", "Chemistry", "Biology", "English",
    "Hindi", "Social Science", "Computer Science",
]

CLASSES = ["8A", "8B", "9A", "9B", "10A", "10B"]

FIRST_NAMES = [
    "Aarav", "Aditi", "Arjun", "Diya", "Ishaan", "Kavya", "Krishna",
    "Meera", "Neha", "Om", "Priya", "Rahul", "Riya", "Sai", "Tanvi",
    "Varun", "Aisha", "Dev", "Fatima", "Harsh", "Isha", "Jai", "Kiara",
    "Lakshmi", "Manav", "Nandini", "Pranav", "Rohan", "Sneha", "Vivek",
    "Ananya", "Aryan", "Bhavya", "Chirag", "Deepa", "Esha", "Gaurav",
    "Hina", "Ishan", "Jiya", "Kartik", "Lavanya", "Mohit", "Navya",
    "Ojas", "Pooja", "Rajat", "Sakshi", "Tushar", "Uma",
]

LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Reddy", "Gupta", "Verma",
    "Joshi", "Mishra", "Rao", "Nair", "Iyer", "Deshmukh", "Shah",
    "Mehta", "Pillai", "Banerjee", "Chatterjee", "Srinivasan", "Das",
]

VARIANTS = ["A", "B", "C", "D"]

SCORE_FSM_STATES = [
    "ai_draft", "teacher_reviewed", "finalized", "locked",
]

OBJECTION_STATES = [
    "filed", "assigned", "reviewing", "resolved",
]

# P05 pen coordinate frame: 14 bytes
# bookType(1) + bookSeq(1) + pageNo(2) + coordX(2) + coordY(2) +
# pressure(2) + penProp(1) + padding(1) + timestamp(2)
COORD_FRAME_FORMAT = "<BBHHHHBBH"
COORD_FRAME_SIZE = struct.calcsize(COORD_FRAME_FORMAT)  # 14 bytes

# Page dimensions in pen units (10 units/mm, A4 ~210x297mm)
PAGE_WIDTH_PU = 2100
PAGE_HEIGHT_PU = 2970
