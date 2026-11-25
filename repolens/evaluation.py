"""Benchmark harness: does retrieval find the right code, and do answers cite it?

Data lives in ``eval/``:

* ``eval/repos.json`` - repositories pinned to exact commits, so results are
  reproducible,
* ``eval/questions.jsonl`` - hand-labelled questions with the file and line
  range that answers each one.

Retrieval metrics (per method: v1 baseline, BM25, vector, hybrid):

* **File Hit@k** - an expected file appears among the top-k chunks,
* **MRR@10** - mean reciprocal rank of the first chunk from an expected file,
* **Span Hit@5** - a top-5 chunk overlaps the expected line range (stricter:
  the right *code*, not just the right file).

Answer metrics (``--agent``; requires a model):

* **Correct citation** - the answer contains a *verified* citation into an
  expected file,
* **Evidence recall** - the investigation looked at the expected lines,
* **Grounding** - share of citations that are verified; plus invalid
  (hallucinated) citation count, steps and latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline_v1 import BaselineV1
from .config import Settings
from .ingest import download_github_repo
from .workspace import open_index

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"


@dataclass
class Expected:
    path: str
    start_line: int
    end_line: int


@dataclass
class Question:
    id: str
    repo: str
    question: str
    style: str
    expected: list[Expected]


def load_questions(path: Path) -> list[Question]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        raw = json.loads(line)
        questions.append(
            Question(
                raw["id"],
                raw["repo"],
                raw["question"],
                raw.get("style", "natural"),
                [Expected(**e) for e in raw["expected"]],
            )
        )
    return questions


def _overlaps(path: str, start: int, end: int, expected: list[Expected]) -> bool:
    return any(path == e.path and start <= e.end_line and e.start_line <= end for e in expected)


def score_ranking(results: list[tuple[str, int, int]], expected: list[Expected]) -> dict[str, float]:
    """``results`` are ``(path, start, end)`` in rank order."""
    paths = {e.path for e in expected}
    first_file = next((i for i, (p, _, _) in enumerate(results[:10], start=1) if p in paths), None)
    return {
        "file_hit@1": float(bool(results) and results[0][0] in paths),
        "file_hit@5": float(any(p in paths for p, _, _ in results[:5])),
        "mrr@10": 1.0 / first_file if first_file else 0.0,
        "span_hit@5": float(any(_overlaps(p, s, e, expected) for p, s, e in results[:5])),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: round(statistics.fmean(r[key] for r in rows), 3) for key in rows[0]}


def prepare_repo(spec: dict[str, str], settings: Settings):
    url = f"{spec['url']}/tree/{spec['ref']}"
    return download_github_repo(url, settings.cache_dir)


def run_retrieval(
    questions: list[Question], repos: dict[str, Any], methods: list[str], settings: Settings, semantic: bool
) -> dict[str, Any]:
    per_method: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
    by_style: dict[str, dict[str, list[dict[str, float]]]] = {}
    details: list[dict[str, Any]] = []
    timing: dict[str, list[float]] = {m: [] for m in methods}
    for repo_name, spec in repos.items():
        repo_questions = [q for q in questions if q.repo == repo_name]
        if not repo_questions:
            continue
        print(f"\n== {repo_name} ({len(repo_questions)} questions)", flush=True)
        source = prepare_repo(spec, settings)
        workspace = open_index(
            source,
            settings,
            semantic=semantic,
            progress=lambda m, f: (
                print(f"   [{f:4.0%}] {m}", flush=True) if ("Embedding" in m or f in (0.0, 1.0)) else None
            ),
        )
        index = workspace.index
        baseline = BaselineV1(source.root) if "v1" in methods else None
        for q in repo_questions:
            row = {"id": q.id, "style": q.style, "question": q.question}
            for method in methods:
                if method in {"vector", "hybrid"} and not index.has_embeddings:
                    continue
                started = time.perf_counter()
                if method == "v1":
                    ranked = [(r["path"], r["start_line"], r["end_line"]) for r in baseline.search(q.question, 10)]
                else:
                    hits = index.search(q.question, top_k=10, method=method, max_per_file=None)
                    ranked = [(h.chunk.path, h.chunk.start_line, h.chunk.end_line) for h in hits]
                timing[method].append(time.perf_counter() - started)
                scores = score_ranking(ranked, q.expected)
                per_method[method].append(scores)
                by_style.setdefault(q.style, {}).setdefault(method, []).append(scores)
                row[method] = {"top3": [f"{p}:{s}-{e}" for p, s, e in ranked[:3]], **scores}
            details.append(row)
    return {
        "summary": {m: mean_metrics(rows) for m, rows in per_method.items() if rows},
        "by_style": {
            style: {m: mean_metrics(r) for m, r in methods_rows.items()} for style, methods_rows in by_style.items()
        },
        "latency_ms": {m: round(1000 * statistics.fmean(t), 1) for m, t in timing.items() if t},
        "questions": len(details),
        "details": details,
    }


def run_answers(
    questions: list[Question],
    repos: dict[str, Any],
    settings: Settings,
    mode: str,
    limit: int | None,
    semantic: bool,
    use_cache: bool = True,
) -> dict[str, Any]:
    from .agent import Agent, AgentConfig
    from .citations import verify_citations
    from .llm import CachedClient, client_from_settings
    from .rag import answer_question

    # Re-running the benchmark replays identical requests from disk and costs no tokens.
    llm = CachedClient(client_from_settings(settings), settings.cache_dir / "llm-cache", enabled=use_cache)
    selected = questions
    rows = []
    for repo_name, spec in repos.items():
        repo_questions = [q for q in selected if q.repo == repo_name][: limit or None]
        if not repo_questions:
            continue
        source = prepare_repo(spec, settings)
        index = open_index(source, settings, semantic=semantic).index
        for q in repo_questions:
            started = time.perf_counter()
            tokens_before = llm.tokens_total
            if mode == "agent":
                result = Agent(
                    index,
                    llm,
                    AgentConfig.from_settings(settings),
                    label=f"{spec['url'].split('github.com/')[-1]}",
                ).ask(q.question)
                answer, evidence, grounding = result.answer, result.evidence, result.grounding
                steps, tool_calls = result.steps, result.tool_calls
            else:
                rag = answer_question(llm, index, q.question, max_context_tokens=settings.llm_max_context_tokens)
                answer, evidence = rag.answer, rag.evidence
                grounding = verify_citations(answer, index, evidence)
                steps, tool_calls = 1, 0
            elapsed = time.perf_counter() - started
            tokens = llm.tokens_total - tokens_before
            expected_paths = {e.path for e in q.expected}
            correct = any(c.status == "verified" and c.resolved_path in expected_paths for c in grounding.citations)
            recall = any(_overlaps(r.path, r.start_line, r.end_line, q.expected) for r in evidence.ranges)
            row = {
                "id": q.id,
                "correct_citation": float(correct),
                "evidence_recall": float(recall),
                "grounding": grounding.score if grounding.score is not None else 0.0,
                "cited": float(grounding.total > 0),
                "invalid_citations": grounding.count("invalid"),
                "steps": steps,
                "tool_calls": tool_calls,
                "seconds": round(elapsed, 1),
                "tokens": tokens,
                "answer": answer,
            }
            rows.append(row)
            print(
                f"   {q.id}: correct={correct} recall={recall} grounding={grounding.label} "
                f"{elapsed:.0f}s {tokens} tokens",
                flush=True,
            )
    numeric = [
        "correct_citation",
        "evidence_recall",
        "grounding",
        "cited",
        "invalid_citations",
        "steps",
        "tool_calls",
        "seconds",
        "tokens",
    ]
    summary = {k: round(statistics.fmean(r[k] for r in rows), 3) for k in numeric} if rows else {}
    return {
        "mode": mode,
        "model": settings.llm_model,
        "provider": settings.provider,
        "questions": len(rows),
        "summary": summary,
        "cache": {"hits": llm.hits, "misses": llm.misses, "tokens_spent": llm.tokens_spent},
        "details": rows,
    }


def retrieval_markdown(result: dict[str, Any], embedder: str | None) -> str:
    names = {
        "v1": "v1 baseline (line windows + BM25)",
        "bm25": "v2 BM25 (code-aware)",
        "vector": f"v2 vector ({embedder})",
        "hybrid": "v2 hybrid (BM25 + vector, RRF)",
    }
    lines = [
        f"Questions: {result['questions']}",
        "",
        "| Method | File Hit@1 | File Hit@5 | MRR@10 | Span Hit@5 | Latency |",
        "|---|---|---|---|---|---|",
    ]
    for method, m in result["summary"].items():
        lines.append(
            f"| {names.get(method, method)} | {m['file_hit@1']:.2f} | {m['file_hit@5']:.2f} | {m['mrr@10']:.2f} | {m['span_hit@5']:.2f} | {result['latency_ms'].get(method, 0):.0f} ms |"
        )
    lines += [
        "",
        "By question style (Span Hit@5):",
        "",
        "| Style | " + " | ".join(result["summary"]) + " |",
        "|---|" + "---|" * len(result["summary"]),
    ]
    for style, methods in sorted(result["by_style"].items()):
        lines.append(
            f"| {style} | "
            + " | ".join(f"{methods.get(m, {}).get('span_hit@5', 0):.2f}" for m in result["summary"])
            + " |"
        )
    return "\n".join(lines)


def answers_markdown(result: dict[str, Any]) -> str:
    s = result["summary"]
    return "\n".join(
        [
            f"Mode: {result['mode']} | provider: {result.get('provider', '?')} | model: {result['model']} | questions: {result['questions']}",
            "",
            "| Correct verified citation | Evidence recall | Grounding | Answers with citations | Invalid citations / answer | Avg steps | Avg latency | Avg tokens |",
            "|---|---|---|---|---|---|---|---|",
            f"| {s['correct_citation']:.2f} | {s['evidence_recall']:.2f} | {s['grounding']:.2f} | {s['cited']:.2f} | {s['invalid_citations']:.2f} | {s['steps']:.1f} | {s['seconds']:.0f} s | {s.get('tokens', 0):.0f} |",
        ]
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--repos", type=Path, default=EVAL_DIR / "repos.json")
    parser.add_argument("--methods", default="v1,bm25,vector,hybrid", help="Comma-separated retrieval methods")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip vector and hybrid methods")
    parser.add_argument("--agent", action="store_true", help="Also evaluate end-to-end answers (needs a model)")
    parser.add_argument("--mode", choices=["agent", "quick"], default="agent", help="Answer mode for --agent")
    parser.add_argument("--limit", type=int, help="Evaluate answers on the first N questions of each repository")
    parser.add_argument("--repo", help="Only evaluate one repository")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "results")
    parser.add_argument(
        "--provider", choices=["ollama", "groq", "openai", "anthropic"], help="Provider preset for --agent"
    )
    parser.add_argument("--model", help="Chat model for --agent")
    parser.add_argument("--no-cache", action="store_true", help="Do not reuse cached model responses")
    parser.add_argument("--skip-retrieval", action="store_true", help="Only run the answer evaluation")


def main(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.provider:
        settings.apply_provider(args.provider)
    if args.model:
        settings.llm_model = args.model
    repos = json.loads(args.repos.read_text(encoding="utf-8"))
    if args.repo:
        repos = {args.repo: repos[args.repo]}
    questions = [q for q in load_questions(args.questions) if q.repo in repos]
    args.output.mkdir(parents=True, exist_ok=True)
    semantic = not args.no_embeddings
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not semantic:
        methods = [m for m in methods if m not in {"vector", "hybrid"}]

    if not args.skip_retrieval:
        retrieval = run_retrieval(questions, repos, methods, settings, semantic)
        embedder = settings.embed_model if semantic else None
        table = retrieval_markdown(retrieval, embedder)
        (args.output / "retrieval.json").write_text(json.dumps(retrieval, indent=2), encoding="utf-8")
        (args.output / "retrieval.md").write_text(table + "\n", encoding="utf-8")
        print("\n" + table)

    if args.agent:
        print(f"\n== Answer evaluation ({args.mode}, {settings.llm_model})", flush=True)
        answers = run_answers(questions, repos, settings, args.mode, args.limit, semantic, not args.no_cache)
        table = answers_markdown(answers)
        model_slug = settings.llm_model.split("/")[-1].replace(":", "-")
        stem = f"answers-{settings.provider}-{model_slug}-{args.mode}"
        (args.output / f"{stem}.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")
        (args.output / f"{stem}.md").write_text(table + "\n", encoding="utf-8")
        print("\n" + table)
    return 0
