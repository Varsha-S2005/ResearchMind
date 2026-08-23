import json

from backend.generation.gemini_llm import GeminiLLM


class GroundingCritic:
    """
    Verifies whether a generated answer is supported
    by the retrieved research-paper evidence.
    """

    def __init__(self):
        self.llm = GeminiLLM()

    def verify(
        self,
        question: str,
        answer: str,
        chunks: list[dict]
    ) -> dict:

        if not chunks:
            return {
                "verdict": "FAIL",
                "score": 0.0,
                "unsupported_claims": [
                    "No evidence was retrieved."
                ],
                "reason": (
                    "The answer cannot be verified because "
                    "no supporting research chunks were retrieved."
                )
            }

        evidence = self._build_evidence(chunks)

        prompt = f"""
You are a strict factual grounding critic for a
Retrieval-Augmented Generation (RAG) research assistant.

Your job is NOT to answer the question.

Your job is to determine whether the generated answer
is fully supported by the provided research-paper evidence.

QUESTION:
{question}

GENERATED ANSWER:
{answer}

RETRIEVED EVIDENCE:
{evidence}

Evaluation rules:

1. FIRST determine whether the retrieved evidence is relevant
   to the QUESTION.

2. The retrieved evidence must contain information that can
   reasonably answer the question.

3. If the retrieved evidence is irrelevant or does not contain
   information needed to answer the question, the verdict MUST
   be FAIL, even if the generated answer is factually supported
   by the retrieved evidence.

4. Check whether the generated answer actually answers the
   user's question.

5. Check every factual claim in the generated answer.

6. A claim is supported only if it can reasonably be inferred
   from the retrieved evidence.

7. Do not use outside knowledge.

8. Do not assume that a claim is correct simply because it
   sounds plausible.

9. Numerical values, names, dates, technical specifications,
   performance claims, and percentages require explicit
   supporting evidence.

10. If an important factual claim is unsupported, the answer
    should normally FAIL.

11. Minor wording differences are acceptable.

12. If the question is unrelated to the retrieved research
    evidence, the verdict MUST be FAIL.

13. If the evidence is relevant but the generated answer
    contains unsupported claims, the verdict MUST be FAIL.

14. Return ONLY valid JSON.

IMPORTANT:

A response must NOT receive PASS merely because its claims
appear in the retrieved evidence.

The question itself must be answerable from the retrieved
evidence.

For example, if the question asks about "quantum computing"
but the retrieved evidence is about "federated learning",
the verdict MUST be FAIL because the evidence is irrelevant
to the question.

When the evidence is irrelevant, explain the mismatch in
"reason" and identify it in "unsupported_claims".

Use exactly this JSON structure:

{{
    "verdict": "PASS" or "FAIL",
    "score": 0.0,
    "unsupported_claims": [],
    "reason": "short explanation"
}}
The score must be between 0 and 1.

PASS means:
- the retrieved evidence is relevant to the question, AND
- the answer addresses the question, AND
- the factual claims are sufficiently supported by the evidence.

FAIL means:
- the evidence is irrelevant to the question, OR
- the answer does not answer the question, OR
- one or more meaningful claims are not supported.
"""

        raw_response = self.llm.generate(prompt)

        return self._parse_response(raw_response)

    def _build_evidence(
        self,
        chunks: list[dict]
    ) -> str:

        evidence_parts = []

        for index, chunk in enumerate(chunks, start=1):

            evidence_parts.append(
                f"""
--- EVIDENCE {index} ---
Document: {chunk.get("document_id")}
Page: {chunk.get("page_number")}
Chunk: {chunk.get("chunk_id")}

{chunk.get("text", "")}
"""
            )

        return "\n".join(evidence_parts)

    def _parse_response(
        self,
        response: str
    ) -> dict:

        try:

            cleaned = response.strip()

            if cleaned.startswith("```"):
                cleaned = cleaned.replace(
                    "```json",
                    ""
                ).replace(
                    "```",
                    ""
                ).strip()

            result = json.loads(cleaned)

            verdict = result.get(
                "verdict",
                "FAIL"
            )

            score = result.get(
                "score",
                0.0
            )

            unsupported_claims = result.get(
                "unsupported_claims",
                []
            )

            reason = result.get(
                "reason",
                ""
            )

            if verdict not in ["PASS", "FAIL"]:
                verdict = "FAIL"

            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0

            score = max(
                0.0,
                min(1.0, score)
            )

            if not isinstance(
                unsupported_claims,
                list
            ):
                unsupported_claims = [
                    str(unsupported_claims)
                ]

            return {
                "verdict": verdict,
                "score": score,
                "unsupported_claims": unsupported_claims,
                "reason": reason
            }

        except Exception as exc:

            return {
                "verdict": "FAIL",
                "score": 0.0,
                "unsupported_claims": [
                    "Critic response could not be parsed."
                ],
                "reason": (
                    f"Grounding verification failed: {exc}"
                )
            }
