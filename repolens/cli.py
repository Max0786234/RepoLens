"""Command-line interface.

    repolens index  SOURCE                      build (or refresh) the index
    repolens search SOURCE "query"              retrieval only, no LLM needed
    repolens ask    SOURCE "question"           agent (default) or --mode quick
    repolens chat   SOURCE                      interactive conversation about the code
    repolens eval   [--agent]                   run the benchmark in eval/

SOURCE is a local folder, a .zip file or a GitHub URL.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time

from . import __version__
from .agent import Agent, AgentConfig
from .config import Settings
from .conversation import ConversationMemory, is_follow_up
from .ingest import IngestError
from .llm import LLMError, client_from_settings
from .rag import retrieve, stream_answer
from .workspace import open_workspace


def _print(text: str = "", **kwargs) -> None:
    print(text, flush=True, **kwargs)


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if getattr(args, "model", None):
        settings.llm_model = args.model
    if getattr(args, "base_url", None):
        settings.llm_base_url = args.base_url
    if getattr(args, "api_key", None):
        settings.llm_api_key = args.api_key
    if getattr(args, "max_steps", None):
        settings.max_agent_steps = args.max_steps
    return settings


def _progress(quiet: bool):
    last = {"message": ""}

    def report(message: str, fraction: float) -> None:
        if quiet or message == last["message"]:
            return
        last["message"] = message
        print(f"  [{fraction:>4.0%}] {message}", file=sys.stderr, flush=True)

    return report


def _open(args, settings):
    semantic = not args.no_embeddings
    workspace = open_workspace(
        args.source, settings, semantic, _progress(args.json), force=getattr(args, "rebuild", False)
    )
    if workspace.semantic_error and not args.json:
        print(f"  note: semantic search disabled ({workspace.semantic_error})", file=sys.stderr)
    return workspace


def cmd_index(args) -> int:
    settings = _settings(args)
    workspace = _open(args, settings)
    stats = workspace.index.stats()
    if args.json:
        _print(json.dumps({"source": workspace.source.label, **stats}, indent=2))
    else:
        langs = ", ".join(f"{k}: {v}" for k, v in stats["languages"].items())
        _print(
            f"Indexed {workspace.source.label}: {stats['files']} files, {stats['chunks']} chunks, {stats['symbols']} symbols"
        )
        _print(f"Languages: {langs}")
        _print(f"Semantic search: {'on (' + stats['embedder'] + ')' if stats['embeddings'] else 'off'}")
    return 0


def cmd_search(args) -> int:
    settings = _settings(args)
    workspace = _open(args, settings)
    hits = workspace.index.search(args.query, top_k=args.top_k, method=args.method)
    if args.json:
        _print(json.dumps([h.to_dict() for h in hits], indent=2))
        return 0
    for hit in hits:
        c = hit.chunk
        label = f"  [{c.kind} {c.symbol}]" if c.symbol else ""
        _print(
            f"{hit.rank:>2}. {c.path}:{c.start_line}-{c.end_line}{label}  score={hit.score:.3f} via {'+'.join(hit.sources)}"
        )
        if args.show_code:
            for n, line in enumerate(c.text.splitlines()[:12], start=c.start_line):
                _print(f"      {n:>5} | {line}")
    return 0


def _print_grounding(grounding) -> None:
    if not grounding.citations:
        _print("\nGrounding: the answer contains no line citations.")
        return
    _print(f"\nGrounding: {grounding.label}")
    marks = {"verified": "[ok]", "unverified": "[??]", "invalid": "[!!]"}
    for c in grounding.citations:
        reason = f" - {c.reason}" if c.reason else ""
        _print(f"  {marks[c.status]} {c.resolved_path or c.path}:{c.start_line}-{c.end_line}{reason}")


def cmd_ask(args) -> int:
    settings = _settings(args)
    workspace = _open(args, settings)
    llm = client_from_settings(settings)
    if not args.json:
        _print(f"Model: {settings.llm_model} @ {settings.llm_base_url}  |  mode: {args.mode}\n")

    if args.mode == "quick":
        rag = retrieve(workspace.index, args.question, top_k=args.top_k)
        if not args.json:
            for hit in rag.hits:
                _print(f"  -> retrieved {hit.chunk.path}:{hit.chunk.start_line}-{hit.chunk.end_line}")
            _print()
        started = time.perf_counter()
        try:
            for piece in stream_answer(llm, workspace.index, rag):
                if not args.json:
                    print(piece, end="", flush=True)
        except LLMError as exc:
            _print(f"\nError: {exc}")
            return 2
        if args.json:
            _print(
                json.dumps(
                    {
                        "answer": rag.answer,
                        "grounding": rag.grounding.to_dict(),
                        "hits": [h.to_dict() for h in rag.hits],
                    },
                    indent=2,
                )
            )
        else:
            _print()
            _print_grounding(rag.grounding)
            _print(f"\n({time.perf_counter() - started:.1f}s)")
        return 0

    agent = Agent(workspace.index, llm, AgentConfig(max_steps=settings.max_agent_steps), label=workspace.source.label)
    result = None
    for event in agent.run(args.question):
        if event.kind == "answer":
            result = event.data["result"]
        elif args.json:
            continue
        elif event.kind == "tool_call":
            prefix = "auto" if event.data.get("auto") else f"step {event.step}"
            _print(f"  -> [{prefix}] {event.message}")
        elif event.kind == "tool_result" and args.trace:
            for line in event.data.get("preview", "").splitlines()[:15]:
                _print(f"       {line}")
        elif event.kind in {"warning", "error"}:
            _print(f"  !! {event.message}")
    assert result is not None
    if args.json:
        _print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.stop_reason != "error" else 2
    _print("\n" + result.answer)
    _print_grounding(result.grounding)
    _print(
        f"\n({result.steps} steps, {result.tool_calls} tool calls, {result.elapsed:.1f}s, stop: {result.stop_reason})"
    )
    return 0 if result.stop_reason != "error" else 2


def cmd_chat(args) -> int:
    """Interactive conversation: follow-up questions keep the topic and earlier evidence."""
    settings = _settings(args)
    workspace = _open(args, settings)
    llm = client_from_settings(settings)
    agent = Agent(workspace.index, llm, AgentConfig(max_steps=settings.max_agent_steps), label=workspace.source.label)
    history: list[dict[str, str]] = []
    memory = ConversationMemory()
    stats = workspace.index.stats()
    _print(f"Chatting about {workspace.source.label} ({stats['files']} files) with {settings.llm_model} [{args.mode}].")
    _print("Ask anything about the code. Commands: /reset (new topic), /files (files examined), /quit\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            _print()
            return 0
        if not question:
            continue
        if question in {"/quit", "/exit", "/q"}:
            return 0
        if question == "/reset":
            history, memory = [], ConversationMemory()
            _print("  (conversation cleared)\n")
            continue
        if question == "/files":
            _print("  " + (", ".join(memory.files(30)) or "(none yet)") + "\n")
            continue
        if history and is_follow_up(question):
            _print("  (follow-up: searching with the previous question as context)")
        if args.mode == "quick":
            rag = retrieve(workspace.index, question, top_k=args.top_k, history=history)
            print("\nrepolens> ", end="", flush=True)
            try:
                for piece in stream_answer(llm, workspace.index, rag, history, memory):
                    print(piece, end="", flush=True)
            except LLMError as exc:
                _print(f"\nError: {exc}\n")
                continue
            answer, grounding, evidence = rag.answer, rag.grounding, rag.evidence
            _print()
        else:
            result = None
            for event in agent.run(question, history, memory):
                if event.kind == "tool_call":
                    prefix = "auto" if event.data.get("auto") else f"step {event.step}"
                    _print(f"  -> [{prefix}] {event.message}")
                elif event.kind in {"warning", "error"}:
                    _print(f"  !! {event.message}")
                elif event.kind == "answer":
                    result = event.data["result"]
            answer, grounding, evidence = result.answer, result.grounding, result.evidence
            _print(f"\nrepolens> {answer}")
        _print_grounding(grounding)
        _print()
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        memory.add_turn(evidence.ranges)


def cmd_eval(args) -> int:
    from .evaluation import main as eval_main

    return eval_main(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repolens", description="Evidence-first AI assistant for codebases.")
    parser.add_argument("--version", action="version", version=f"repolens {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("source", help="Local folder, .zip file or GitHub URL")
        p.add_argument("--no-embeddings", action="store_true", help="Keyword (BM25) search only")
        p.add_argument("--json", action="store_true", help="Machine-readable output")

    p = sub.add_parser("index", help="Build or refresh the index for a repository")
    common(p)
    p.add_argument("--rebuild", action="store_true", help="Ignore the cached index")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="Search code without an LLM")
    common(p)
    p.add_argument("query")
    p.add_argument("-k", "--top-k", type=int, default=8)
    p.add_argument("--method", choices=["auto", "bm25", "vector", "hybrid"], default="auto")
    p.add_argument("--show-code", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="Ask a question (agent or quick RAG)")
    common(p)
    p.add_argument("question")
    p.add_argument("--mode", choices=["agent", "quick"], default="agent")
    p.add_argument("--model", help="Chat model (default from REPOLENS_LLM_MODEL)")
    p.add_argument("--base-url", help="OpenAI-compatible base URL (default: local Ollama)")
    p.add_argument("--api-key", help="API key for hosted providers")
    p.add_argument("--max-steps", type=int, help="Agent tool-use budget")
    p.add_argument("-k", "--top-k", type=int, default=6, help="Chunks retrieved in quick mode")
    p.add_argument("--trace", action="store_true", help="Print tool results while investigating")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="Interactive conversation about a repository")
    common(p)
    p.add_argument("--mode", choices=["agent", "quick"], default="agent")
    p.add_argument("--model", help="Chat model (default from REPOLENS_LLM_MODEL)")
    p.add_argument("--base-url", help="OpenAI-compatible base URL (default: local Ollama)")
    p.add_argument("--api-key", help="API key for hosted providers")
    p.add_argument("--max-steps", type=int, help="Agent tool-use budget per question")
    p.add_argument("-k", "--top-k", type=int, default=6, help="Chunks retrieved in quick mode")
    p.set_defaults(func=cmd_chat, json=False)

    p = sub.add_parser("eval", help="Run the retrieval/agent benchmark")
    from .evaluation import add_arguments

    add_arguments(p)
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (IngestError, LLMError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
