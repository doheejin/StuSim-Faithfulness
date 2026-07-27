#!/usr/bin/env python3
"""
Student Response Simulation Prompt

Simulates a student responding to teacher feedback after getting a math
problem wrong. The student only knows (question, their_answer, feedback) —
they must implicitly reason about the misconception.

Usage:
    build_conversation = load_prompt("studentsim_simulate_student")
    conv = build_conversation(
        problem_text="Simplify 2^3 * 2^4",
        student_answer="2^12",
        feedback_text="Think about what happens to exponents when you multiply...",
    )
"""

from typing import List, Dict


SYSTEM_PROMPT = """\
You are a student learning math. You worked on the problem below and \
believe your answer is correct. Your task is to respond to your teacher \
naturally — as a real student would."""


def build_conversation(
    problem_text: str,
    student_answer: str,
    feedback_text: str,
    **kwargs,
) -> List[Dict[str, str]]:
    """
    Build a conversation for student response simulation.

    Args:
        problem_text: The math problem
        student_answer: The student's wrong answer
        feedback_text: The teacher's feedback

    Returns:
        Conversation list for LLM inference
    """
    user_content = f"""\
**Problem**: {problem_text}
**Your answer**: {student_answer}

Your teacher says: "{feedback_text}"

Your teacher then asks: "So what do you think the answer is?"

What would you say to your teacher?"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
