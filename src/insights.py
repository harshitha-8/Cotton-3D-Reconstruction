from __future__ import annotations

import os


def generate_research_note(
    summary_text: str,
    cotton_metrics: str,
    user_prompt: str,
    model: str,
) -> str:
    prompt = "\n\n".join(
        [
            "You are helping write a concise academic technical note about cotton UAV 3D reconstruction.",
            "Use the provided reconstruction outputs to describe cotton depth structure, likely confidence, and next-step recommendations.",
            f"Reconstruction summary:\n{summary_text}",
            f"Cotton-focused metrics:\n{cotton_metrics}",
            f"User focus:\n{user_prompt or 'Summarize the result for a technical project note.'}",
        ]
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=model,
                input=prompt,
            )
            return response.output_text
        except Exception as exc:
            return fallback_research_note(summary_text, cotton_metrics, f"OpenAI call failed: {exc}")

    return fallback_research_note(summary_text, cotton_metrics, "OPENAI_API_KEY not configured")


def fallback_research_note(summary_text: str, cotton_metrics: str, status: str) -> str:
    return "\n".join(
        [
            "LLM research-note fallback",
            f"Status: {status}",
            "",
            "Interpretation:",
            "The current reconstruction is a monocular 3D approximation intended for exploratory cotton-structure analysis.",
            "The cotton-focused mask isolates bright elevated regions that may correspond to cotton bolls or high-reflectance canopy structures.",
            "",
            "Reconstruction summary:",
            summary_text,
            "",
            "Cotton-focused metrics:",
            cotton_metrics,
            "",
            "Recommended next step:",
            "Use overlapping UAV frames with a multi-view photogrammetry backend for stronger geometric validity, then compare this monocular estimate against the multi-view surface.",
        ]
    )
