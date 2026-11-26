"""RepoLens as a Model Context Protocol (MCP) server.

Lets any MCP client (desktop AI assistants, IDE agents, the MCP Inspector) use
RepoLens's code-search tools and its citation verifier:

    repolens mcp                      # open a repository later with the open_repository tool
    repolens mcp ./my-project         # or pre-load one (folder, .zip or GitHub URL)

Transport: JSON-RPC 2.0 messages, one per line, over stdin/stdout (the MCP stdio
transport). Implemented with the standard library only, like the rest of
RepoLens. Everything a client is shown is recorded in an evidence log, so
``verify_citations`` can check an answer written by the *client's* model the
same way RepoLens checks its own agent.

Logs go to stderr: stdout must carry protocol messages only.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, BinaryIO

from . import __version__
from .citations import verify_citations
from .config import Settings
from .ingest import IngestError
from .tools import EvidenceLog, ToolBox, overview_text
from .workspace import Workspace, open_workspace

SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST_VERSION = SUPPORTED_VERSIONS[0]

INSTRUCTIONS = (
    "RepoLens indexes a code repository and gives you read-only tools to investigate it. "
    "Start with open_repository (unless one is already open), then use search_code, read_file, "
    "find_symbol, find_callers/find_callees and grep to gather evidence. Cite code as path:start-end. "
    "Before finalising an answer, call verify_citations with your answer text: it reports which "
    "citations point at lines you were actually shown."
)

OPEN_REPOSITORY = {
    "name": "open_repository",
    "description": "Index a repository so the other tools can search it. Accepts a local folder, a .zip file "
    "or a public GitHub URL (branches/commits allowed). Replaces the currently open repository.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Folder path, .zip path or https://github.com/owner/repo"},
            "semantic": {
                "type": "boolean",
                "description": "Also build embeddings for semantic search (slower; needs an embedding server).",
            },
        },
        "required": ["source"],
    },
}
VERIFY_CITATIONS = {
    "name": "verify_citations",
    "description": "Check every path:line citation in a text against the open repository and against the code "
    "returned by RepoLens tools in this session. Reports Verified / Not inspected / Not found per citation.",
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The answer text containing citations."}},
        "required": ["text"],
    },
}


ANALYZE_CHANGE = {
    "name": "analyze_change",
    "description": "Analyse a unified diff (git diff output) against the open repository: changed files, the "
    "functions/classes the change touches, callers outside the change that may be affected, and tests that "
    "exercise the changed code. Use it to review or explain a change.",
    "inputSchema": {
        "type": "object",
        "properties": {"diff": {"type": "string", "description": "Unified diff text, e.g. the output of git diff."}},
        "required": ["diff"],
    },
}


class RepoLensMCPServer:
    def __init__(self, settings: Settings | None = None, semantic_default: bool = False) -> None:
        self.settings = settings or Settings.from_env()
        self.semantic_default = semantic_default
        self.workspace: Workspace | None = None
        self.evidence = EvidenceLog()
        self.toolbox: ToolBox | None = None
        self.protocol_version = LATEST_VERSION

    # ------------------------------------------------------------ repository

    def open(self, source: str, semantic: bool | None = None) -> str:
        use_semantic = self.semantic_default if semantic is None else bool(semantic)
        self.workspace = open_workspace(source, self.settings, semantic=use_semantic)
        self.evidence = EvidenceLog()
        self.toolbox = ToolBox(self.workspace.index, self.evidence)
        stats = self.workspace.index.stats()
        note = (
            f"\nNote: semantic search unavailable ({self.workspace.semantic_error})."
            if self.workspace.semantic_error
            else ""
        )
        return (
            f"Opened {self.workspace.source.label}: {stats['files']} files, {stats['chunks']} code passages, "
            f"{stats['symbols']} functions/classes. Search: {'hybrid' if stats['embeddings'] else 'keyword (BM25)'}.{note}\n\n"
            + overview_text(self.workspace.index, readme_lines=10)
        )

    # ------------------------------------------------------------ tools

    def tool_definitions(self) -> list[dict[str, Any]]:
        # The agent's tool schemas are reused so MCP clients get exactly the same tools.
        schemas = ToolBox(_EmptyIndex(), EvidenceLog()).schemas() if self.toolbox is None else self.toolbox.schemas()
        tools = [OPEN_REPOSITORY]
        for schema in schemas:
            function = schema["function"]
            tools.append(
                {
                    "name": function["name"],
                    "description": function["description"],
                    "inputSchema": function["parameters"],
                }
            )
        tools.append(ANALYZE_CHANGE)
        tools.append(VERIFY_CITATIONS)
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool; returns ``(text, is_error)``."""
        if name == "open_repository":
            source = str(arguments.get("source", "")).strip()
            if not source:
                return "open_repository requires 'source'.", True
            try:
                return self.open(source, arguments.get("semantic")), False
            except (IngestError, FileNotFoundError, ValueError) as exc:
                return f"Could not open {source!r}: {exc}", True
        if self.toolbox is None or self.workspace is None:
            return "No repository is open. Call open_repository first.", True
        if name == "analyze_change":
            from .change import ChangeError, analyze_change, parse_unified_diff

            try:
                changes = parse_unified_diff(str(arguments.get("diff", "")))
            except ChangeError as exc:
                return str(exc), True
            if not changes:
                return "The diff contains no file changes.", True
            report = analyze_change(self.workspace.index, changes)
            # Lines the client is shown count as evidence for verify_citations.
            self.evidence.ranges.extend(report.evidence.ranges)
            return report.summary_text(), False
        if name == "verify_citations":
            report = verify_citations(str(arguments.get("text", "")), self.workspace.index, self.evidence)
            if not report.citations:
                return "No path:line citations found in the text.", False
            labels = {"verified": "VERIFIED", "unverified": "NOT INSPECTED", "invalid": "NOT FOUND"}
            lines = [f"{report.label}."]
            for c in report.citations:
                reason = f" - {c.reason}" if c.reason else ""
                lines.append(f"{labels[c.status]}: {c.resolved_path or c.path}:{c.start_line}-{c.end_line}{reason}")
            return "\n".join(lines), False
        result = self.toolbox.call(name, arguments)
        return result.text, result.is_error

    # ------------------------------------------------------------ JSON-RPC

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC message. Returns the response, or ``None`` for notifications."""
        method = message.get("method")
        msg_id = message.get("id")
        is_notification = "id" not in message
        params = message.get("params") or {}
        try:
            if method == "initialize":
                requested = params.get("protocolVersion")
                self.protocol_version = requested if requested in SUPPORTED_VERSIONS else LATEST_VERSION
                result: Any = {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "repolens", "title": "RepoLens", "version": __version__},
                    "instructions": INSTRUCTIONS,
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.tool_definitions()}
            elif method == "tools/call":
                name = params.get("name")
                if not isinstance(name, str):
                    return _error(msg_id, -32602, "tools/call requires a tool name")
                known = {tool["name"] for tool in self.tool_definitions()}
                if name not in known:
                    return _error(msg_id, -32602, f"Unknown tool: {name}")
                text, is_error = self.call_tool(name, params.get("arguments") or {})
                result = {"content": [{"type": "text", "text": text}], "isError": is_error}
            elif isinstance(method, str) and method.startswith("notifications/"):
                return None
            else:
                return None if is_notification else _error(msg_id, -32601, f"Method not found: {method}")
        except Exception as exc:  # never let one bad request kill the server
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            return None if is_notification else _error(msg_id, -32603, f"Internal error: {exc}")
        return None if is_notification else {"jsonrpc": "2.0", "id": msg_id, "result": result}


class _EmptyIndex:
    """Lets ToolBox build its schemas before any repository is open."""

    files: list = []
    symbols: list = []

    def __getattr__(self, name):  # pragma: no cover - tools are never called on it
        raise RuntimeError("no repository open")


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def serve(server: RepoLensMCPServer, stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> None:
    """Read newline-delimited JSON-RPC messages until EOF."""
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer

    def send(payload: dict[str, Any]) -> None:
        stdout.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
        stdout.flush()

    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            send(_error(None, -32700, "Parse error"))
            continue
        batch = message if isinstance(message, list) else [message]
        for item in batch:
            if not isinstance(item, dict):
                send(_error(None, -32600, "Invalid request"))
                continue
            response = server.handle(item)
            if response is not None:
                send(response)


def main(source: str | None = None, semantic: bool = False) -> int:
    server = RepoLensMCPServer(semantic_default=semantic)
    if source:
        try:
            print(server.open(source).splitlines()[0], file=sys.stderr, flush=True)
        except (IngestError, FileNotFoundError, ValueError) as exc:
            print(f"repolens mcp: could not open {source!r}: {exc}", file=sys.stderr, flush=True)
    print(f"repolens {__version__} MCP server ready (stdio)", file=sys.stderr, flush=True)
    serve(server)
    return 0
