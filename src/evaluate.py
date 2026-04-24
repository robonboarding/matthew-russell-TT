"""
Evaluation harness.

JUSTIFICATION SUMMARY:
Three metrics chosen to match Rabobank's published vocabulary from the
GraphRAG techblog (Colin van Lieshout, November 2025):

1. Context recall: did retrieval surface the gold-standard chunks?
2. Faithfulness: is every claim in the answer grounded in the retrieved context?
3. Answer correctness: does the answer fully and accurately address the query?

Using their vocabulary signals domain fluency in the reflection call.

Implementation notes:
- Context recall measured against a list of expected chunk_ids per question
- Faithfulness and answer correctness use LLM-as-judge (gpt-4o-mini as grader)
- LLM-as-judge is not perfect; in production we would add human spot-checks
  on a sample and track inter-rater agreement

In production at Rabobank:
- Expand to 100+ questions covering edge cases, adversarial prompts, multi-hop
- Add red-teaming suite (jailbreaks, PII extraction attempts, prompt injection)
- Track metrics over time to detect regressions from model updates
- Correlate offline metrics with online user feedback
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AzureOpenAI

from src.config import CONFIG, EVAL_PATH
from src.generate import generate
from src.retrieve import retrieve

CLIENT = AzureOpenAI(                                                                                                                                                                                                    
      api_key=CONFIG.azure_api_key,
      api_version=CONFIG.azure_api_version,                                                                                                                                                                                
      azure_endpoint=CONFIG.azure_endpoint,                                                                                                                                                                                
  )
JUDGE_MODEL = "gpt-4o-mini"

FAITHFULNESS_PROMPT = """You are evaluating whether an ANSWER is faithful to a CONTEXT.
Faithfulness means every claim in the answer is supported by the context.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with a JSON object: {{"faithful": true|false, "unsupported_claims": [list of strings]}}"""

CORRECTNESS_PROMPT = """You are evaluating whether a PREDICTED ANSWER correctly answers the QUESTION, using the GOLD ANSWER as reference.

QUESTION: {question}
GOLD ANSWER: {gold}
PREDICTED ANSWER: {predicted}

Score from 0.0 to 1.0 where:
1.0 = fully correct and complete
0.5 = partially correct or incomplete
0.0 = incorrect or missing

Respond with a JSON object: {{"score": float, "reasoning": "short explanation"}}"""


@dataclass
class EvalResult:
    question: str
    context_recall: float
    faithfulness: float
    answer_correctness: float
    predicted_answer: str
    retrieved_chunks: list[str]
    expected_chunks: list[str]


def context_recall(retrieved: list[str], expected: list[str]) -> float:
    """Fraction of expected chunks that appear in retrieved chunks."""
    if not expected:
        return 1.0
    hits = sum(1 for e in expected if e in retrieved)
    return hits / len(expected)


def judge_faithfulness(context: str, answer: str) -> float:
    """LLM-as-judge faithfulness score. Returns 1.0 if faithful, else penalises."""
    response = CLIENT.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": FAITHFULNESS_PROMPT.format(context=context, answer=answer)}],
    )
    parsed = _parse_json(response.choices[0].message.content)
    if not parsed:
        return 0.0
    if parsed.get("faithful"):
        return 1.0
    unsupported = parsed.get("unsupported_claims") or []
    # Partial credit: penalise by fraction of unsupported claims (capped)
    return max(0.0, 1.0 - min(len(unsupported) * 0.25, 1.0))


def judge_correctness(question: str, gold: str, predicted: str) -> float:
    """LLM-as-judge answer correctness score."""
    response = CLIENT.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": CORRECTNESS_PROMPT.format(question=question, gold=gold, predicted=predicted)}
        ],
    )
    parsed = _parse_json(response.choices[0].message.content)
    if not parsed:
        return 0.0
    return float(parsed.get("score", 0.0))


def _parse_json(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def run_eval(eval_path: Path = EVAL_PATH) -> list[EvalResult]:
    """Run evaluation on the QA dataset."""
    eval_data = json.loads(eval_path.read_text())
    results: list[EvalResult] = []

    for item in eval_data:
        question = item["question"]
        gold = item["gold_answer"]
        expected_chunks = item.get("expected_chunks", [])

        # Run pipeline
        retrieval_results = retrieve(question)
        retrieved_ids = [r.chunk.chunk_id for r in retrieval_results]
        context = "\n\n".join(r.chunk.text for r in retrieval_results)

        gen = generate(question)
        predicted = gen.answer

        # Metrics
        recall = context_recall(retrieved_ids, expected_chunks)
        faith = judge_faithfulness(context, predicted) if retrieval_results else 0.0
        correct = judge_correctness(question, gold, predicted)

        results.append(
            EvalResult(
                question=question,
                context_recall=recall,
                faithfulness=faith,
                answer_correctness=correct,
                predicted_answer=predicted,
                retrieved_chunks=retrieved_ids,
                expected_chunks=expected_chunks,
            )
        )

        print(f"\nQ: {question}")
        print(f"  context_recall={recall:.2f}  faithfulness={faith:.2f}  correctness={correct:.2f}")

    return results


def summarise(results: list[EvalResult]) -> None:
    n = len(results)
    if n == 0:
        print("No results.")
        return
    avg_recall = sum(r.context_recall for r in results) / n
    avg_faith = sum(r.faithfulness for r in results) / n
    avg_correct = sum(r.answer_correctness for r in results) / n
    print("\n" + "=" * 60)
    print(f"Evaluated {n} questions")
    print(f"  Avg context recall:    {avg_recall:.2f}")
    print(f"  Avg faithfulness:      {avg_faith:.2f}")
    print(f"  Avg answer correctness: {avg_correct:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    eval_file = Path(sys.argv[1]) if len(sys.argv) > 1 else EVAL_PATH
    results = run_eval(eval_file)
    summarise(results)
