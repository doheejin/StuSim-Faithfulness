#!/usr/bin/env python3
"""
Student Response Simulation Prompt — Authentic Variant

Like simulate_student.py but with an added instruction to evaluate feedback
against the student's own reasoning process. This aims to reduce sycophantic
flipping (Iteration 1 sensitivity was 0.016).

Usage:
    build_conversation = load_prompt("studentsim_simulate_student_authentic")
    conv = build_conversation(
        problem_text="Simplify 2^3 * 2^4",
        student_answer="2^12",
        feedback_text="Think about what happens to exponents when you multiply...",
    )
"""

from typing import List, Dict


SYSTEM_PROMPT = """\
You are a student learning math. You worked on the problem below and \
believe your answer is correct. When responding to feedback, think about \
whether it connects to how you actually approached the problem. Respond \
to your teacher naturally — as a real student would."""


def build_conversation(
    problem_text: str,
    student_answer: str,
    feedback_text: str,
    **kwargs,
) -> List[Dict[str, str]]:
    """
    Build a conversation for student response simulation (authentic variant).

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
