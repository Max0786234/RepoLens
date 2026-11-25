"""RepoLens web UI (Streamlit).

    streamlit run app.py

Everything substantive lives in the ``repolens`` package; this file is only
presentation and session state.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import streamlit as st

from repolens import __version__
from repolens.agent import Agent, AgentConfig
from repolens.citations import extract_citations
from repolens.config import PROVIDERS, Settings
from repolens.conversation import ConversationMemory, is_follow_up
from repolens.ingest import IngestError, RepoSource, download_github_repo, is_github_url, save_uploads
from repolens.llm import LLMError, make_chat_client
from repolens.rag import retrieve, stream_answer
from repolens.workspace import Workspace, open_index

APP_DIR = Path(__file__).resolve().parent
UPLOAD_TYPES = ["zip", "py", "js", "jsx", "ts", "tsx", "java", "kt", "go", "rs", "c", "h", "cpp", "hpp", "cs", "php",
                "rb", "swift", "scala", "sql", "sh", "md", "txt", "json", "yaml", "yml", "toml", "html", "css", "vue"]  # fmt: skip
EXAMPLE_QUESTIONS = [
    "Give me an overview of this repository's architecture.",
    "Where is the main entry point and what happens at startup?",
    "How is configuration loaded?",
    "How are errors handled?",
]
STATUS_ICON = {"verified": "✅", "unverified": "⚠️", "invalid": "❌"}
STATUS_HELP = {
    "verified": "the agent inspected these lines",
    "unverified": "exists, but the agent never looked at these lines",
    "invalid": "file or lines do not exist (hallucinated)",
}

st.set_page_config(page_title="RepoLens", page_icon="🔎", layout="wide")
st.markdown(
    """
    <style>
      .rl-badge {display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.8rem;font-weight:600;margin-right:6px}
      .rl-good {background:#dcfce7;color:#166534} .rl-mid {background:#fef9c3;color:#854d0e}
      .rl-bad {background:#fee2e2;color:#991b1b} .rl-none {background:#e5e7eb;color:#374151}
      .rl-step {font-family:ui-monospace,monospace;font-size:0.82rem}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers


def secret(name: str | None) -> str:
    if not name:
        return ""
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except Exception:  # no secrets.toml
        return ""


@st.cache_data(ttl=30, show_spinner=False)
def probe_models(base_url: str, api_key: str) -> tuple[list[str], str | None]:
    """Return ``(models, error)`` for the configured server."""
    try:
        client = make_chat_client(base_url, "probe", api_key or None, timeout=5)
        return client.list_models(), None
    except (LLMError, OSError, ValueError) as exc:
        return [], str(exc)


def init_state() -> None:
    defaults = {"workspace": None, "messages": [], "pending": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def grounding_badge(grounding: dict[str, Any] | None) -> str:
    if not grounding or not grounding.get("citations"):
        return '<span class="rl-badge rl-none">no line citations</span>'
    total = len(grounding["citations"])
    verified = grounding["verified"]
    css = "rl-good" if verified == total else "rl-mid" if verified else "rl-bad"
    text = f"{verified}/{total} citations verified"
    if grounding.get("invalid"):
        text += f" · {grounding['invalid']} invalid"
    return f'<span class="rl-badge {css}">{text}</span>'


def linkify(answer: str, source: RepoSource, index) -> str:
    """Turn ``path:12-30`` citations into GitHub links (outside code blocks)."""
    if not source.github:
        return answer
    parts = re.split(r"(```.*?```)", answer, flags=re.DOTALL)
    for i, part in enumerate(parts):
        if part.startswith("```"):
            continue
        for citation in extract_citations(part):
            resolved = index.resolve_path(citation.path)
            if not resolved:
                continue
            url = source.web_url(resolved, citation.start_line, citation.end_line)
            raw = citation.raw
            part = part.replace(f"`{raw}`", f"[`{raw}`]({url})").replace(f" {raw}", f" [`{raw}`]({url})")
        parts[i] = part
    return "".join(parts)


def render_citations(grounding: dict[str, Any], source: RepoSource) -> None:
    for c in grounding.get("citations", []):
        path = c.get("resolved_path") or c["path"]
        label = f"`{path}:{c['start_line']}-{c['end_line']}`"
        url = source.web_url(path, c["start_line"], c["end_line"]) if c["status"] != "invalid" else None
        if url:
            label = f"[{label}]({url})"
        st.markdown(f"{STATUS_ICON[c['status']]} {label} — {STATUS_HELP[c['status']]}")


def render_hits(hits: list[dict[str, Any]], source: RepoSource, limit: int = 8) -> None:
    for hit in hits[:limit]:
        title = f"**{hit['path']}** · lines {hit['start_line']}-{hit['end_line']}"
        if hit.get("symbol"):
            title += f" · `{hit['symbol']}`"
        if hit.get("sources"):
            title += f" · via {' + '.join(hit['sources'])}"
        url = source.web_url(hit["path"], hit["start_line"], hit["end_line"])
        st.markdown(title + (f" · [view on GitHub]({url})" if url else ""))
        st.code(
            hit["text"][:3000], language=hit.get("language") if hit.get("language") not in {"text", "rst"} else None
        )


def render_trace(trace: list[dict[str, Any]]) -> None:
    for event in trace:
        if event["kind"] == "tool_call":
            prefix = "auto" if event["data"].get("auto") else f"step {event['step']}"
            st.markdown(f'<div class="rl-step">→ [{prefix}] {event["message"]}</div>', unsafe_allow_html=True)
        elif event["kind"] == "tool_result":
            with st.container():
                st.code(event["data"].get("preview", "")[:1500], language=None)
        elif event["kind"] in {"warning", "error"}:
            st.warning(event["message"])


def render_message(message: dict[str, Any], workspace: Workspace) -> None:
    source = workspace.source
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
            return
        if message.get("mode") == "search":
            st.markdown(f"**Top results** ({message.get('elapsed', 0):.2f}s, no LLM used)")
            render_hits(message.get("hits", []), source)
            return
        st.markdown(linkify(message["content"], source, workspace.index))
        meta = message.get("meta", "")
        st.markdown(
            grounding_badge(message.get("grounding")) + f'<span class="rl-step">{meta}</span>', unsafe_allow_html=True
        )
        if message.get("grounding", {}).get("citations"):
            with st.expander("Citations"):
                render_citations(message["grounding"], source)
        if message.get("trace"):
            with st.expander("Investigation trace"):
                render_trace(message["trace"])
        if message.get("hits"):
            with st.expander("Retrieved evidence"):
                render_hits(message["hits"], source)


def conversation_markdown(workspace: Workspace) -> str:
    lines = [f"# RepoLens conversation: {workspace.source.label}", ""]
    for m in st.session_state.messages:
        if m["role"] == "user":
            lines += [f"## Q: {m['content']}", ""]
        elif m.get("mode") == "search":
            lines += [
                "Search results:",
                *[f"- {h['path']}:{h['start_line']}-{h['end_line']}" for h in m.get("hits", [])],
                "",
            ]
        else:
            g = m.get("grounding") or {}
            verified, total = g.get("verified", 0), len(g.get("citations", []))
            lines += [m["content"], "", f"_{verified}/{total} citations verified · {m.get('meta', '').strip()}_", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- sidebar

init_state()
settings = Settings.from_env()

with st.sidebar:
    st.markdown(f"### 🔎 RepoLens <small>v{__version__}</small>", unsafe_allow_html=True)
    st.caption("Evidence-first AI assistant for codebases.")

    provider_keys = list(PROVIDERS)
    default_provider = os.environ.get("REPOLENS_PROVIDER", "ollama")
    provider_key = st.selectbox(
        "Model provider",
        provider_keys,
        index=provider_keys.index(default_provider) if default_provider in provider_keys else 0,
        format_func=lambda k: PROVIDERS[k].name,
    )
    preset = PROVIDERS[provider_key]
    env_url = settings.llm_base_url if provider_key in {"ollama", "custom"} else preset.base_url
    base_url = st.text_input("Base URL", value=env_url, key=f"url-{provider_key}")
    api_key = ""
    if preset.needs_api_key or provider_key == "custom":
        api_key = st.text_input(
            "API key",
            value=secret(preset.api_key_env) or (settings.llm_api_key or ""),
            type="password",
            key=f"key-{provider_key}",
            help="Kept in this browser session only; never written to disk.",
        )
    models, probe_error = probe_models(base_url, api_key)
    chat_models = [m for m in models if "embed" not in m.lower()]
    default_model = settings.llm_model if provider_key == "ollama" else preset.default_model
    if chat_models:
        index_default = chat_models.index(default_model) if default_model in chat_models else 0
        model = st.selectbox("Model", chat_models, index=index_default, key=f"model-{provider_key}")
        st.success(f"Connected · {len(chat_models)} chat models", icon="🟢")
    else:
        model = st.text_input("Model", value=default_model, key=f"model-text-{provider_key}")
        if preset.needs_api_key and not api_key:
            st.info("Enter an API key, or use **Search only** mode (no model needed).", icon="🔑")
        elif probe_error:
            st.warning(f"Model server not reachable. {probe_error[:220]}", icon="🟠")
    if preset.notes:
        st.caption(preset.notes)

    st.divider()
    mode = st.radio(
        "Answer mode",
        ["agent", "quick", "search"],
        format_func={
            "agent": "🕵️ Agent - investigates with tools",
            "quick": "⚡ Quick - one search, one answer",
            "search": "🔍 Search only - no LLM",
        }.get,
        help="Agent is the most thorough. Quick is much faster on CPU-only machines. Search needs no model.",
    )
    embed_available = provider_key != "ollama" or any(settings.embed_model in m for m in models)
    semantic = st.toggle(
        "Semantic search (embeddings)",
        value=embed_available and settings.embed_backend != "none",
        help=f"Hybrid BM25 + vector search using `{settings.embed_model}`. Applied when a repository is indexed.",
    )
    if semantic and provider_key == "ollama" and models and not embed_available:
        st.caption(f"Run `ollama pull {settings.embed_model}` to enable embeddings.")
    with st.expander("Advanced"):
        max_steps = st.slider("Agent tool budget (steps)", 2, 10, settings.max_agent_steps)
        top_k = st.slider("Chunks retrieved (quick/search)", 3, 12, 6)

    settings.llm_base_url, settings.llm_model, settings.llm_api_key = base_url, model, api_key or None
    if provider_key != "ollama":
        # Hosted chat providers: use their embeddings endpoint only if explicitly configured.
        settings.embed_base_url = os.environ.get("REPOLENS_EMBED_BASE_URL", settings.embed_base_url)

    workspace: Workspace | None = st.session_state.workspace
    if workspace:
        st.divider()
        if st.session_state.messages:
            st.download_button(
                "⬇️ Export conversation (.md)",
                conversation_markdown(workspace),
                file_name="repolens-conversation.md",
                use_container_width=True,
            )
        if st.button("↩️ Load a different repository", use_container_width=True):
            st.session_state.workspace = None
            st.session_state.messages = []
            st.rerun()
    st.caption("[Source on GitHub](https://github.com/Max0786234/RepoLens)")


# ---------------------------------------------------------------- loading a repository


def build_workspace(make_source) -> None:
    with st.status("Preparing repository...", expanded=True) as status:
        try:
            status.update(label="Fetching source...")
            source: RepoSource = make_source()
            bar = st.progress(0.0)
            note = st.empty()

            def progress(message: str, fraction: float) -> None:
                bar.progress(min(1.0, max(0.0, fraction)))
                note.caption(message)

            workspace = open_index(source, settings, semantic=semantic, progress=progress)
        except (IngestError, FileNotFoundError, ValueError) as exc:
            status.update(label="Could not load repository", state="error")
            st.error(str(exc))
            return
        stats = workspace.index.stats()
        status.update(
            label=f"Indexed {source.label}: {stats['files']} files, {stats['chunks']} chunks", state="complete"
        )
        if workspace.semantic_error:
            st.warning(f"Semantic search unavailable, using keyword search only. ({workspace.semantic_error})")
    st.session_state.workspace = workspace
    st.session_state.messages = []
    time.sleep(0.3)
    st.rerun()


workspace = st.session_state.workspace

if workspace is None:
    st.title("🔎 RepoLens")
    st.markdown("#### Ask questions about any codebase and get answers backed by **verified line citations**.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**🧠 Agentic investigation**  \nThe agent searches, reads files and follows symbols before answering.")
    c2.markdown("**🎯 Hybrid code search**  \nStructure-aware chunks, BM25 + embeddings fused with RRF.")
    c3.markdown(
        "**✅ Verified grounding**  \nEvery `file:line` citation is checked against what the agent actually read."
    )
    st.write("")
    tab_github, tab_upload, tab_sample = st.tabs(["🐙 GitHub repository", "📦 Upload ZIP / files", "✨ Try a sample"])
    with tab_github:
        url = st.text_input("Public GitHub URL", placeholder="https://github.com/psf/requests")
        st.caption("Branches and commits work too: `https://github.com/owner/repo/tree/<ref>`")
        if st.button("Index repository", type="primary", disabled=not url):
            if not is_github_url(url):
                st.error("That does not look like a GitHub repository URL.")
            else:
                build_workspace(
                    lambda: download_github_repo(url, settings.cache_dir, token=settings.github_token or None)
                )
    with tab_upload:
        uploads = st.file_uploader("Project ZIP or source files", type=UPLOAD_TYPES, accept_multiple_files=True)
        st.caption("Archives are extracted safely: path traversal, symlinks and oversized files are rejected.")
        if st.button("Index upload", type="primary", disabled=not uploads):
            build_workspace(lambda: save_uploads([(f.name, f.getvalue()) for f in uploads], settings.cache_dir))
    with tab_sample:
        st.markdown("Index **RepoLens itself** - a quick way to see the agent at work.")
        if st.button("Index RepoLens", type="primary"):
            build_workspace(lambda: RepoSource(APP_DIR, "RepoLens (this project)", "local"))
    st.stop()


# ---------------------------------------------------------------- chat

stats = workspace.index.stats()
header, metrics = st.columns([3, 2])
with header:
    link = (
        f"https://github.com/{workspace.source.github['owner']}/{workspace.source.github['repo']}"
        if workspace.source.github
        else None
    )
    st.markdown(f"## {'[' + workspace.source.label + '](' + link + ')' if link else workspace.source.label}")
    langs = " · ".join(f"{k} {v}" for k, v in list(stats["languages"].items())[:6])
    st.caption(
        f"{langs}  |  search: {'hybrid (BM25 + ' + settings.embed_model + ')' if stats['embeddings'] else 'BM25 keyword'}"
    )
with metrics:
    m1, m2, m3 = st.columns(3)
    m1.metric("Files", f"{stats['files']:,}")
    m2.metric("Chunks", f"{stats['chunks']:,}")
    m3.metric("Symbols", f"{stats['symbols']:,}")

for message in st.session_state.messages:
    render_message(message, workspace)

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(question, use_container_width=True):
            st.session_state.pending = question
            st.rerun()

prompt = st.chat_input("Ask about this codebase, e.g. 'How does authentication work?'")
if st.session_state.pending:
    prompt, st.session_state.pending = st.session_state.pending, None

if prompt:
    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages if m.get("mode") != "search"
    ]
    # Evidence from earlier answers in this conversation (files the AI already looked at).
    memory = ConversationMemory()
    for m in st.session_state.messages:
        memory.ranges.extend(ConversationMemory.from_list(m.get("evidence")).ranges)
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message(st.session_state.messages[-1], workspace)
    index = workspace.index
    started = time.perf_counter()

    if mode == "search":
        hits = index.search(prompt, top_k=top_k)
        message = {"role": "assistant", "mode": "search", "content": "", "hits": [h.to_dict() for h in hits],
                   "elapsed": time.perf_counter() - started}  # fmt: skip
        st.session_state.messages.append(message)
        render_message(message, workspace)
        st.stop()

    llm = make_chat_client(
        settings.llm_base_url, settings.llm_model, settings.llm_api_key, settings.llm_timeout, settings.ollama_num_ctx
    )
    with st.chat_message("assistant"):
        if history and is_follow_up(prompt):
            st.caption("↪ Follow-up detected: searching with the previous question as context.")
        if mode == "quick":
            rag = retrieve(index, prompt, top_k=top_k, history=history)
            with st.expander(f"Retrieved {len(rag.hits)} chunks", expanded=False):
                render_hits([h.to_dict() for h in rag.hits], workspace.source)
            try:
                st.write_stream(stream_answer(llm, index, rag, history, memory))
                error = None
            except LLMError as exc:
                error = str(exc)
                st.error(error)
            message = {
                "role": "assistant",
                "mode": "quick",
                "content": rag.answer or f"**The model could not be reached:** {error}",
                "grounding": rag.grounding.to_dict(),
                "hits": [h.to_dict() for h in rag.hits],
                "evidence": [r.__dict__ for r in rag.evidence.ranges][:300],
                "meta": f"quick · {settings.llm_model} · {time.perf_counter() - started:.1f}s",
            }
        else:
            agent = Agent(index, llm, AgentConfig(max_steps=max_steps), label=workspace.source.label)
            result = None
            with st.status("Investigating the codebase...", expanded=True) as status:
                for event in agent.run(prompt, history, memory):
                    if event.kind == "tool_call":
                        prefix = "auto" if event.data.get("auto") else f"step {event.step}"
                        st.markdown(f'<div class="rl-step">→ [{prefix}] {event.message}</div>', unsafe_allow_html=True)
                        status.update(label=f"{prefix}: {event.data.get('tool')}")
                    elif event.kind == "status":
                        status.update(label=f"Step {event.step}: {event.message}")
                    elif event.kind in {"warning", "error"}:
                        st.warning(event.message)
                    elif event.kind == "answer":
                        result = event.data["result"]
                status.update(
                    label=f"Investigated in {result.steps} steps · {result.tool_calls} tool calls · {result.elapsed:.1f}s",
                    state="error" if result.stop_reason == "error" else "complete",
                    expanded=False,
                )
            hits = next(
                (e.data.get("hits", []) for e in result.events if e.kind == "tool_result" and e.data.get("hits")), []
            )
            message = {
                "role": "assistant",
                "mode": "agent",
                "content": result.answer,
                "grounding": result.grounding.to_dict(),
                "trace": [
                    e.to_dict() for e in result.events if e.kind in {"tool_call", "tool_result", "warning", "error"}
                ],
                "hits": hits,
                "evidence": [r.__dict__ for r in result.evidence.ranges][:300],
                "meta": f" agent · {settings.llm_model} · {result.steps} steps · {result.tool_calls} tools · {result.elapsed:.1f}s",
            }
    st.session_state.messages.append(message)
    st.rerun()
