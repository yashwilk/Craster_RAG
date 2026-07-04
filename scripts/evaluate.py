"""
Runs the full multi-agent pipeline against a golden
Q&A dataset and measures answer quality using RAGAS metrics.

  faithfulness        does answer only use retrieved context?
                        (hallucination detection)

    answer_relevancy    does answer address the question?

    context_precision   are retrieved chunks relevant?
                        (retriever + reranker quality)

    context_recall      did we retrieve all relevant info?


"""


import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow
from rich.console import Console
from rich.table import Table


# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from craster_rag.agents.graph import run_pipeline


# logging
logging.basicConfig(
    level  = getattr(logging, settings.log_level),
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger  = logging.getLogger(__name__)
console = Console()


# path to golden Q&A dataset
GOLDEN_QA_PATH = Path("tests/eval/golden_qa_set.json")

def load_golden_qa(
    category : str = "",
    limit    : int = 0,
) -> list[dict]:

    if not GOLDEN_QA_PATH.exists():
        raise FileNotFoundError(
            f"Golden Q&A dataset not found: {GOLDEN_QA_PATH}\n"
            f"Expected at: tests/eval/golden_qa_set.json"
        )

    with open(GOLDEN_QA_PATH, encoding="utf-8") as f:
        qa_set = json.load(f)

    if category:
        qa_set = [q for q in qa_set if q["category"] == category]

    if limit:
        qa_set = qa_set[:limit]

    return qa_set



def run_evaluation(
    category : str = "",
    limit    : int = 0,
) -> dict:

    qa_set = load_golden_qa(category=category, limit=limit)

    console.print(
        f"\n[bold green]RAGAS Evaluation Pipeline[/bold green]\n"
        f"Questions: [cyan]{len(qa_set)}[/cyan] | "
        f"Category: [cyan]{category or 'all'}[/cyan]\n"
    )


    # ── Step 1: Run pipeline on all questions ───────────
    console.print("[bold]Step 1/3[/bold] Running pipeline on all questions...\n")

    results = []
    for qa in qa_set:
        try:
            result = run_pipeline(qa["question"])
            results.append({
                "id"              : qa["id"],
                "category"        : qa["category"],
                "question"        : qa["question"],
                "ground_truth"    : qa["ground_truth"],
                "answer"          : result.get("answer", ""),
                "contexts"        : [
                    chunk.content
                    for chunk in result.get("chunks", [])
                ],
                "can_answer"      : result.get("can_answer", False),
                "confidence_level": result.get("confidence_level", "none"),
                "sources"         : result.get("sources", []),
                "must_mention"    : qa.get("must_mention", []),
                "source_document" : qa.get("source_document", ""),
            })

        except Exception as e:
            logger.error(f"Pipeline failed for {qa['id']}: {e}")
            results.append({
                "id"              : qa["id"],
                "category"        : qa["category"],
                "question"        : qa["question"],
                "ground_truth"    : qa["ground_truth"],
                "answer"          : "",
                "contexts"        : [],
                "can_answer"      : False,
                "confidence_level": "error",
                "sources"         : [],
                "must_mention"    : qa.get("must_mention", []),
                "source_document" : qa.get("source_document", ""),
            })


    # ── Step 2: Compute simple metrics ──────────────────
    console.print(
        f"\n[bold]Step 2/3[/bold] Computing metrics...\n"
    )
    metrics = _compute_metrics(results)

    # ── Step 3: RAGAS scoring ───────────────────────────
    console.print("[bold]Step 3/3[/bold] RAGAS scoring...\n")

    ragas_scores = _run_ragas(results)
    metrics.update(ragas_scores)

    return {"metrics": metrics, "results": results}



def _compute_metrics(results: list[dict]) -> dict:
    """ Tracks:
        can_answer_rate   percentage of questions answered
        confidence_high   percentage with high confidence
        must_mention_rate percentage mentioning required terms
        category_breakdown per-category can_answer rate"""


    total = len(results)
    if total == 0:
        return {}

    answered        = sum(1 for r in results if r["can_answer"])
    high_confidence = sum(
        1 for r in results if r["confidence_level"] == "high"
    )


    # must_mention: does the answer contain the required keywords?
    must_mention_passes = 0
    for r in results:
        if not r["can_answer"]:
            continue
        answer   = r["answer"].lower()
        keywords = r["must_mention"]
        if all(kw.lower() in answer for kw in keywords):
            must_mention_passes += 1

    answered_count = max(answered, 1)

    # per-category can_answer rate
    categories: dict = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "answered": 0}
        categories[cat]["total"]    += 1
        categories[cat]["answered"] += 1 if r["can_answer"] else 0

    category_rates = {
        cat: round(v["answered"] / v["total"], 2)
        for cat, v in categories.items()
    }

    return {
        "total_questions"    : total,
        "can_answer_rate"    : round(answered / total, 3),
        "high_confidence_rate": round(high_confidence / answered_count, 3),
        "must_mention_rate"  : round(must_mention_passes / answered_count, 3),
        "category_rates"     : category_rates,
    }




def _run_ragas(results: list[dict]) -> dict:
    """
    Run RAGAS evaluation metrics.

    Only evaluates questions that were actually answered"""


    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

    except ImportError:
        logger.warning(
            "RAGAS not installed — skipping RAGAS scoring. "
            "pip install ragas datasets"
        )
        return {}


    # build RAGAS dataset — only answered questions with contexts
    ragas_data = {
        "question"    : [],
        "answer"      : [],
        "contexts"    : [],
        "ground_truth": [],
    }


    for r in results:
        if not r["can_answer"] or not r["contexts"] or not r["answer"]:
            continue

        ragas_data["question"].append(r["question"])
        ragas_data["answer"].append(r["answer"])
        ragas_data["contexts"].append(r["contexts"])
        ragas_data["ground_truth"].append(r["ground_truth"])

    if not ragas_data["question"]:
        logger.warning("No answered questions with contexts for RAGAS scoring")
        return {"ragas_sample_size": 0}

    try:
        dataset = Dataset.from_dict(ragas_data)

        score = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        score_dict = score.to_pandas().mean().to_dict()

        return {
            "ragas_sample_size"       : len(ragas_data["question"]),
            "ragas_faithfulness"      : round(score_dict.get("faithfulness", 0), 3),
            "ragas_answer_relevancy"  : round(score_dict.get("answer_relevancy", 0), 3),
            "ragas_context_precision" : round(score_dict.get("context_precision", 0), 3),
            "ragas_context_recall"    : round(score_dict.get("context_recall", 0), 3),
        }

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {"ragas_error": str(e)}



def _print_report(eval_output: dict) -> None:
    metrics = eval_output["metrics"]
    results = eval_output["results"]
    console.print("\n[bold green]Evaluation Complete![/bold green]\n")

    # ── Overall metrics table ───────────────────────────
    table = Table(
        show_header  = True,
        header_style = "bold cyan",
        title        = "Overall Metrics",
    )
    table.add_column("Metric",  style="dim")
    table.add_column("Score",   justify="right")

    table.add_row("Total questions",      str(metrics.get("total_questions", 0)))
    table.add_row("Can answer rate",      f"{metrics.get('can_answer_rate', 0):.1%}")
    table.add_row("High confidence rate", f"{metrics.get('high_confidence_rate', 0):.1%}")
    table.add_row("Must mention rate",    f"{metrics.get('must_mention_rate', 0):.1%}")

    if "ragas_faithfulness" in metrics:
        table.add_row("─" * 25,                 "─" * 10)
        table.add_row("[bold]RAGAS faithfulness[/bold]",
                      f"[bold]{metrics['ragas_faithfulness']:.3f}[/bold]")
        table.add_row("[bold]RAGAS answer relevancy[/bold]",
                      f"[bold]{metrics['ragas_answer_relevancy']:.3f}[/bold]")
        table.add_row("[bold]RAGAS context precision[/bold]",
                      f"[bold]{metrics['ragas_context_precision']:.3f}[/bold]")
        table.add_row("[bold]RAGAS context recall[/bold]",
                      f"[bold]{metrics['ragas_context_recall']:.3f}[/bold]")

    console.print(table)

    # ── Category breakdown ──────────────────────────────
    if metrics.get("category_rates"):
        cat_table = Table(
            show_header  = True,
            header_style = "bold cyan",
            title        = "Can Answer Rate by Category",
        )
        cat_table.add_column("Category", style="dim")
        cat_table.add_column("Rate",     justify="right")

        for cat, rate in metrics["category_rates"].items():
            colour = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"
            cat_table.add_row(cat, f"[{colour}]{rate:.1%}[/{colour}]")

        console.print("\n")
        console.print(cat_table)

    # ── Failed questions ────────────────────────────────
    failed = [r for r in results if not r["can_answer"]]
    if failed:
        console.print(
            f"\n[bold yellow]Questions not answered "
            f"({len(failed)}):[/bold yellow]"
        )
        for r in failed:
            console.print(
                f"  [dim]{r['id']}[/dim] {r['question'][:70]}..."
            )


def main():
    """
    Entry point. Parses args, runs evaluation, logs to MLflow.
    """
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation for Craster HR Policy RAG"
    )
    parser.add_argument(
        "--category",
        type    = str,
        default = "",
        help    = "Filter to specific category (e.g. leave_family)",
    )
    parser.add_argument(
        "--limit",
        type    = int,
        default = 0,
        help    = "Max questions to evaluate (0 = all)",
    )
    args = parser.parse_args()

    # MLflow tracking
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("craster-rag-eval")

    with mlflow.start_run(run_name="ragas_evaluation"):

        # log evaluation parameters
        mlflow.log_params({
            "category"        : args.category or "all",
            "limit"           : args.limit or "all",
            "chunk_size"      : settings.chunk_size,
            "chunk_overlap"   : settings.chunk_overlap,
            "embedding_model" : settings.embedding_model,
            "enable_reranking": settings.enable_reranking,
            "top_k_results"   : settings.top_k_results,
        })

        # run evaluation
        eval_output = run_evaluation(
            category = args.category,
            limit    = args.limit,
        )

        # print report
        _print_report(eval_output)

        # log metrics to MLflow (flat dict only, no nested)
        metrics = eval_output["metrics"]
        flat_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(flat_metrics)

        # log per-category rates as separate metrics
        for cat, rate in metrics.get("category_rates", {}).items():
            mlflow.log_metric(f"can_answer_{cat}", rate)

        console.print(
            f"\n[dim]MLflow run logged → "
            f"{settings.mlflow_tracking_uri}[/dim]"
        )
        console.print(
            "[dim]View results: make mlflow-ui[/dim]\n"
        )


if __name__ == "__main__":
    main()
