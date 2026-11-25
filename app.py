"""RepoLens web UI (Streamlit).

    streamlit run app.py

Everything substantive lives in the ``repolens`` package; this file is only
presentation and session state.
"""

from __future__ import annotations

import html
import os
import time
from pathlib import Path
from typing import Any

import streamlit as st

from repolens import __version__
from repolens.agent import Agent, AgentConfig
from repolens.citations import linkify_citations
from repolens.config import PROVIDERS, Settings
from repolens.conversation import ConversationMemory, is_follow_up
from repolens.ingest import IngestError, Limits, RepoSource, download_github_repo, is_github_url, save_uploads
from repolens.llm import LLMError, make_chat_client
from repolens.rag import retrieve, stream_answer
from repolens.workspace import Workspace, open_index

APP_DIR = Path(__file__).resolve().parent
REPO_URL = "https://github.com/Max0786234/RepoLens"
UPLOAD_TYPES = ["zip", "py", "js", "jsx", "ts", "tsx", "java", "kt", "go", "rs", "c", "h", "cpp", "hpp", "cs", "php",
                "rb", "swift", "scala", "sql", "sh", "md", "txt", "json", "yaml", "yml", "toml", "html", "css", "vue"]  # fmt: skip
EXAMPLE_QUESTIONS = [
    "Overview of the architecture",
    "Where is the main entry point?",
    "How is configuration loaded?",
    "How are errors handled?",
]
MODE_LABELS = {"agent": "Agent", "quick": "Quick answer", "search": "Search only"}
MODE_CAPTIONS = {
    "agent": "Investigates step by step with tools. Most thorough.",
    "quick": "One search, one answer. Fastest AI mode.",
    "search": "Shows the most relevant code. No AI model used.",
}
STATUS_LABEL = {"verified": "Verified", "unverified": "Not inspected", "invalid": "Not found"}
STATUS_HELP = {
    "verified": "The agent read these lines before answering.",
    "unverified": "The lines exist, but the agent never looked at them.",
    "invalid": "This file or line range does not exist in the repository.",
}
USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/manage_search:"

LOGO_SVG = """<svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="1" y="1" width="22" height="22" rx="6" fill="#1d4ed8"/>
<circle cx="10.5" cy="10.5" r="4.6" stroke="white" stroke-width="2"/>
<path d="M14 14l4.2 4.2" stroke="white" stroke-width="2" stroke-linecap="round"/>
</svg>"""

CSS = """
<style>
/* Theme-neutral styling: works in Streamlit's light AND dark themes (and with browser dark modes).
   Text inherits the theme colour; borders/backgrounds are translucent; accents are mid-tone. */
:root {
  --rl-border: rgba(127, 127, 127, 0.28); --rl-soft: rgba(127, 127, 127, 0.07);
  --rl-accent: #3b6ff5;
}
code, pre, .rl-mono { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important; }
/* Keep the header/toolbar container (it holds the sidebar open button); hide only decoration. */
[data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
.block-container { padding-top: 3.6rem; max-width: 1180px; }
h1, h2, h3, h4 { letter-spacing: -0.015em; }

.rl-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
.rl-brand .name { font-weight: 700; font-size: 1.15rem; }
.rl-brand .ver { font-size: .72rem; opacity: .7; border: 1px solid var(--rl-border); border-radius: 6px; padding: 1px 6px; }
.rl-section { text-transform: uppercase; letter-spacing: .08em; font-size: .7rem; font-weight: 600; opacity: .65; margin: 18px 0 6px; }
.rl-eyebrow { text-transform: uppercase; letter-spacing: .09em; font-size: .75rem; font-weight: 600; color: var(--rl-accent); margin-bottom: 8px; }
.rl-hero h1 { font-size: 2.35rem; line-height: 1.18; font-weight: 700; margin: 0 0 12px; padding: 0; }
.rl-lead { font-size: 1.06rem; line-height: 1.6; opacity: .8; max-width: 760px; margin-bottom: 26px; }

.rl-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 30px; }
.rl-card { border: 1px solid var(--rl-border); border-radius: 12px; padding: 16px 18px; background: var(--rl-soft); }
.rl-card .step { font-size: .72rem; font-weight: 600; color: var(--rl-accent); letter-spacing: .06em; text-transform: uppercase; }
.rl-card h4 { margin: 6px 0 6px; font-size: 1rem; font-weight: 600; padding: 0; }
.rl-card p { margin: 0; opacity: .8; font-size: .9rem; line-height: 1.55; }

.rl-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.rl-stat { border: 1px solid var(--rl-border); border-radius: 10px; padding: 10px 14px; background: var(--rl-soft); }
.rl-stat .v { font-size: 1.35rem; font-weight: 600; }
.rl-stat .k { font-size: .75rem; opacity: .7; }
.rl-tags { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 4px; }
.rl-tag { font-size: .75rem; background: var(--rl-soft); border: 1px solid var(--rl-border); border-radius: 6px; padding: 2px 8px; }

.rl-chip { display: inline-flex; align-items: center; gap: 6px; font-size: .78rem; font-weight: 600; border-radius: 999px; padding: 2px 10px; margin-right: 8px; }
.rl-chip::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.rl-ok { background: rgba(22, 163, 74, 0.14); color: #22a35a; }
.rl-warn { background: rgba(217, 119, 6, 0.16); color: #d98a1a; }
.rl-bad { background: rgba(220, 38, 38, 0.14); color: #e04848; }
.rl-none { background: rgba(127, 127, 127, 0.14); color: inherit; opacity: .8; }
.rl-meta { font-size: .78rem; opacity: .7; }
.rl-cite { display: flex; align-items: baseline; gap: 10px; padding: 6px 0; border-bottom: 1px solid var(--rl-border); }
.rl-cite:last-child { border-bottom: none; }
.rl-cite .why { font-size: .82rem; opacity: .7; }
.rl-cite a, .rl-cite .rl-mono { font-size: .84rem; }
.rl-step { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .8rem; opacity: .9; padding: 2px 0; }
.rl-step b { color: var(--rl-accent); font-weight: 500; }
.rl-note { font-size: .82rem; opacity: .7; }

@media (max-width: 760px) {
  .rl-grid, .rl-stats { grid-template-columns: 1fr; }
  .rl-hero h1 { font-size: 1.8rem; }
}
</style>
"""

GUIDE = [
    (
        "Answer modes",
        "**Agent** investigates like an engineer: it searches the code, opens files, finds where functions are "
        "defined and used, and only then answers. It is the most thorough mode and takes a few seconds to a minute.  \n"
        "**Quick answer** runs one search and writes an answer from the best matches. Fastest, good for simple "
        "lookups.  \n**Search only** shows the most relevant pieces of code without using an AI model at all.",
    ),
    (
        "Citations and verification",
        "Answers reference code as `file:line-range`. RepoLens keeps a log of every line the agent was shown and "
        "checks each reference against it:  \n"
        "- **Verified** – the agent read these lines before answering.  \n"
        "- **Not inspected** – the lines exist, but the agent never looked at them.  \n"
        "- **Not found** – the file or line range does not exist (a hallucinated reference).  \n"
        "Verified means the evidence was seen; it does not prove every sentence about it is right, so open the "
        "linked lines when it matters.",
    ),
    (
        "Follow-up questions",
        "This is a conversation. Short follow-ups such as *“Are there tests for it?”* are searched together with "
        "your previous question, and the agent remembers which files it already examined.",
    ),
    (
        "Investigation steps and retrieved code",
        "Under each answer, **Investigation steps** lists every tool call the agent made (searches, file reads, "
        "definition lookups, who-calls-what queries) and what it found. **Retrieved code** shows the code passages the answer was built "
        "from, with links to GitHub.",
    ),
    (
        "How search works",
        "Code is split along its structure (functions, classes, methods, documentation sections) rather than "
        "fixed-size blocks. Keyword search understands identifiers such as `parse_header` or `getUserName`; when "
        "an embedding model is available it is combined with semantic search for paraphrased questions.",
    ),
    (
        "Supported sources, languages and limits",
        "Public GitHub repositories (any branch or commit), ZIP archives and individual files. Structure-aware "
        "parsing for Python, JavaScript/TypeScript, Go, Rust, Java, Kotlin, C#, Swift, C/C++, Ruby and PHP; other "
        "text files are indexed too. Dependencies, build output, binaries and secret files (such as `.env`) are "
        "skipped. The public demo limits repository size and the number of AI answers per session.",
    ),
    (
        "Privacy",
        "The agent can only read files from the repository you load; it cannot run code or access anything else. "
        "Code passages are sent to the configured AI model to produce answers. Do not upload code you are not "
        "allowed to share.",
    ),
]

st.set_page_config(
    page_title="RepoLens",
    page_icon=":material/manage_search:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


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


def section(label: str) -> None:
    st.markdown(f'<div class="rl-section">{html.escape(label)}</div>', unsafe_allow_html=True)


def grounding_chip(grounding: dict[str, Any] | None) -> str:
    if not grounding or not grounding.get("citations"):
        return '<span class="rl-chip rl-none">No line citations</span>'
    total = len(grounding["citations"])
    verified = grounding["verified"]
    css = "rl-ok" if verified == total else "rl-warn" if verified else "rl-bad"
    text = f"{verified} of {total} citations verified"
    if grounding.get("invalid"):
        text += f" · {grounding['invalid']} not found"
    return f'<span class="rl-chip {css}">{text}</span>'


def linkify(answer: str, source: RepoSource, index) -> str:
    """Turn ``path:12-30`` citations into GitHub links (outside code blocks)."""
    if not source.github:
        return answer

    def url_for(path: str, start: int, end: int) -> str | None:
        resolved = index.resolve_path(path)
        return source.web_url(resolved, start, end) if resolved else None

    return linkify_citations(answer, url_for)


def render_citations(grounding: dict[str, Any], source: RepoSource) -> None:
    rows = []
    css = {"verified": "rl-ok", "unverified": "rl-warn", "invalid": "rl-bad"}
    for c in grounding.get("citations", []):
        path = c.get("resolved_path") or c["path"]
        ref = html.escape(f"{path}:{c['start_line']}-{c['end_line']}")
        url = source.web_url(path, c["start_line"], c["end_line"]) if c["status"] != "invalid" else None
        ref_html = (
            f'<a class="rl-mono" href="{url}" target="_blank">{ref}</a>'
            if url
            else f'<span class="rl-mono">{ref}</span>'
        )
        rows.append(
            f'<div class="rl-cite"><span class="rl-chip {css[c["status"]]}">{STATUS_LABEL[c["status"]]}</span>'
            f'{ref_html}<span class="why">{STATUS_HELP[c["status"]]}</span></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_hits(hits: list[dict[str, Any]], source: RepoSource, limit: int = 8) -> None:
    for hit in hits[:limit]:
        title = f"**{hit['path']}** · lines {hit['start_line']}–{hit['end_line']}"
        if hit.get("symbol"):
            title += f" · `{hit['symbol']}`"
        url = source.web_url(hit["path"], hit["start_line"], hit["end_line"])
        st.markdown(title + (f" · [Open on GitHub]({url})" if url else ""))
        language = hit.get("language") if hit.get("language") not in {"text", "rst"} else None
        st.code(hit["text"][:3000], language=language)


def step_label(event: dict[str, Any] | Any) -> str:
    data = event["data"] if isinstance(event, dict) else event.data
    step = event["step"] if isinstance(event, dict) else event.step
    message = event["message"] if isinstance(event, dict) else event.message
    prefix = "Initial search" if data.get("auto") else f"Step {step}"
    return f'<div class="rl-step"><b>{prefix}</b> · {html.escape(message)}</div>'


def render_trace(trace: list[dict[str, Any]]) -> None:
    for event in trace:
        if event["kind"] == "tool_call":
            st.markdown(step_label(event), unsafe_allow_html=True)
        elif event["kind"] == "tool_result":
            st.code(event["data"].get("preview", "")[:1500], language=None)
        elif event["kind"] in {"warning", "error"}:
            st.warning(event["message"])


def render_message(message: dict[str, Any], workspace: Workspace) -> None:
    source = workspace.source
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "user":
            st.markdown(message["content"])
            return
        if message.get("mode") == "search":
            st.markdown(
                f'<span class="rl-meta">Search results · {message.get("elapsed", 0):.2f}s · no AI model used</span>',
                unsafe_allow_html=True,
            )
            render_hits(message.get("hits", []), source)
            return
        st.markdown(linkify(message["content"], source, workspace.index))
        meta = html.escape(message.get("meta", "").strip())
        st.markdown(
            grounding_chip(message.get("grounding")) + f'<span class="rl-meta">{meta}</span>', unsafe_allow_html=True
        )
        citations = message.get("grounding", {}).get("citations", [])
        if citations:
            with st.expander(f"Sources ({len(citations)})"):
                render_citations(message["grounding"], source)
        calls = [e for e in message.get("trace", []) if e["kind"] == "tool_call"]
        if calls:
            with st.expander(f"Investigation steps ({len(calls)})"):
                render_trace(message["trace"])
        if message.get("hits"):
            with st.expander(f"Retrieved code ({len(message['hits'])})"):
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


def load_secrets_into_env() -> None:
    """Streamlit Cloud keeps configuration in st.secrets: expose top-level values as env vars."""
    try:
        items = dict(st.secrets)
    except Exception:  # no secrets.toml locally
        return
    for key, value in items.items():
        if isinstance(value, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(value)


# ---------------------------------------------------------------- configuration

load_secrets_into_env()
init_state()
settings = Settings.from_env()
# Public demo mode (e.g. Streamlit Community Cloud): one shared free-tier key, so cap usage.
PUBLIC_DEMO = os.environ.get("REPOLENS_PUBLIC_DEMO", "").lower() in {"1", "true", "yes"}
MAX_QUESTIONS = int(os.environ.get("REPOLENS_MAX_QUESTIONS", "8" if PUBLIC_DEMO else "0"))
LIMITS = Limits(max_files=2_500, max_total_bytes=25_000_000, max_download_bytes=60_000_000) if PUBLIC_DEMO else Limits()
SAMPLE_REPOS = {
    "psf/requests": ("https://github.com/psf/requests", "Python HTTP client"),
    "pallets/flask": ("https://github.com/pallets/flask", "Python web framework"),
    "expressjs/express": ("https://github.com/expressjs/express", "Node.js web framework"),
}


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown(
        f'<div class="rl-brand">{LOGO_SVG}<span class="name">RepoLens</span><span class="ver">v{__version__}</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("Codebase Q&A with verified citations")

    section("Model")
    provider_keys = list(PROVIDERS)
    default_provider = os.environ.get("REPOLENS_PROVIDER", "ollama")
    if PUBLIC_DEMO:
        # Public demo: the provider is fixed by the server and no connection settings are shown.
        provider_key = default_provider if default_provider in provider_keys else "groq"
    else:
        provider_key = st.selectbox(
            "Provider",
            provider_keys,
            index=provider_keys.index(default_provider) if default_provider in provider_keys else 0,
            format_func=lambda k: PROVIDERS[k].name,
        )
    preset = PROVIDERS[provider_key]
    env_url = settings.llm_base_url if provider_key in {"ollama", "custom"} else preset.base_url
    # The server's key is used server-side only. It is NEVER put into a widget: widget values
    # are sent to the browser, so pre-filling a password field would leak the key to visitors.
    server_key = secret(preset.api_key_env) or (settings.llm_api_key or "")
    if PUBLIC_DEMO:
        base_url, api_key = env_url, server_key
    else:
        base_url = st.text_input("Base URL", value=env_url, key=f"url-{provider_key}")
        api_key = server_key
        if preset.needs_api_key or provider_key == "custom":
            typed = st.text_input(
                "API key",
                value="",
                type="password",
                key=f"key-{provider_key}",
                placeholder="Using the key configured on the server" if server_key else "Paste your API key",
                help="Optional if the server has a key. Kept in this session's memory only; never written to disk.",
            )
            api_key = typed or server_key
    models, probe_error = probe_models(base_url, api_key)
    non_chat = ("embed", "whisper", "guard", "orpheus", "tts", "allam")
    chat_models = [m for m in models if not any(word in m.lower() for word in non_chat)]
    default_model = settings.llm_model if provider_key == "ollama" else preset.default_model
    if chat_models:
        index_default = chat_models.index(default_model) if default_model in chat_models else 0
        model = st.selectbox(
            "Model" if not PUBLIC_DEMO else f"Model ({preset.name})",
            chat_models,
            index=index_default,
            key=f"model-{provider_key}",
        )
        st.markdown('<span class="rl-chip rl-ok">Connected</span>', unsafe_allow_html=True)
    else:
        model = st.text_input("Model", value=default_model, key=f"model-text-{provider_key}")
        if preset.needs_api_key and not api_key:
            st.info("Add an API key, or use Search only mode (no model needed).")
        elif probe_error:
            st.warning(f"Model server not reachable. {probe_error[:220]}")

    section("Answer mode")
    mode = st.radio(
        "Answer mode",
        ["agent", "quick", "search"],
        format_func=MODE_LABELS.get,
        captions=[MODE_CAPTIONS[m] for m in ("agent", "quick", "search")],
        label_visibility="collapsed",
    )
    embed_available = provider_key != "ollama" or any(settings.embed_model in m for m in models)
    if PUBLIC_DEMO:
        semantic = settings.embed_backend != "none"
    else:
        section("Search")
        semantic = st.toggle(
            "Semantic search",
            value=embed_available and settings.embed_backend != "none",
            help=f"Combine keyword search with embeddings (`{settings.embed_model}`) to match paraphrased "
            "questions. Applied when a repository is loaded.",
        )
        if semantic and provider_key == "ollama" and models and not embed_available:
            st.caption(f"Run `ollama pull {settings.embed_model}` to enable it.")
    with st.expander("Advanced"):
        step_cap = 4 if PUBLIC_DEMO else 10
        max_steps = st.slider(
            "Agent step budget",
            2,
            step_cap,
            min(settings.max_agent_steps, step_cap),
            help="Maximum number of investigation rounds before the agent must answer.",
        )
        top_k = st.slider(
            "Code passages per search", 3, 12, 6, help="How many passages Quick answer and Search only retrieve."
        )

    if provider_key not in {"ollama", "custom"}:
        settings.apply_provider(provider_key)  # token budgets for rate-limited tiers
    settings.llm_base_url, settings.llm_model, settings.llm_api_key = base_url, model, api_key or None
    if provider_key != "ollama":
        # Hosted chat providers: use their embeddings endpoint only if explicitly configured.
        settings.embed_base_url = os.environ.get("REPOLENS_EMBED_BASE_URL", settings.embed_base_url)

    workspace: Workspace | None = st.session_state.workspace
    if workspace:
        section("Session")
        if MAX_QUESTIONS:
            used = sum(1 for m in st.session_state.messages if m["role"] == "assistant" and m.get("mode") != "search")
            st.caption(f"AI answers used: {used} of {MAX_QUESTIONS}")
        if st.session_state.messages:
            st.download_button(
                "Export conversation",
                conversation_markdown(workspace),
                file_name="repolens-conversation.md",
                use_container_width=True,
                help="Download the questions, answers and citation results as Markdown.",
            )
        if st.button("Load a different repository", use_container_width=True):
            st.session_state.workspace = None
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        f'<div class="rl-note" style="margin-top:22px"><a href="{REPO_URL}" target="_blank">Source code</a> · '
        "MIT licensed</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- loading a repository


def build_workspace(make_source) -> None:
    with st.status("Preparing repository", expanded=True) as status:
        try:
            status.update(label="Downloading source")
            source: RepoSource = make_source()
            bar = st.progress(0.0)
            note = st.empty()

            def progress(message: str, fraction: float) -> None:
                bar.progress(min(1.0, max(0.0, fraction)))
                note.caption(message)

            workspace = open_index(source, settings, semantic=semantic, progress=progress)
        except (IngestError, FileNotFoundError, ValueError) as exc:
            status.update(label="Could not load the repository", state="error")
            st.error(str(exc))
            return
        stats = workspace.index.stats()
        status.update(label=f"Indexed {source.label} · {stats['files']} files", state="complete")
        if workspace.semantic_error:
            st.warning(f"Semantic search unavailable, using keyword search only. ({workspace.semantic_error})")
    st.session_state.workspace = workspace
    st.session_state.messages = []
    time.sleep(0.3)
    st.rerun()


def render_landing() -> None:
    st.markdown(
        """
        <div class="rl-hero">
          <div class="rl-eyebrow">Codebase Q&amp;A</div>
          <h1>Understand any codebase,<br>with sources you can check.</h1>
          <div class="rl-lead">RepoLens reads a repository and investigates your question the way an engineer
          would &mdash; searching the code, opening files, following definitions and usages &mdash; then answers
          with references to the exact lines. Every reference is checked against the code it actually read.</div>
        </div>
        <div class="rl-grid">
          <div class="rl-card"><div class="step">Step 1</div><h4>Load a repository</h4>
            <p>Paste a public GitHub link, upload a ZIP, or pick an example. The code is split into functions,
            classes and sections and indexed for search.</p></div>
          <div class="rl-card"><div class="step">Step 2</div><h4>Ask in plain English</h4>
            <p>The agent searches, reads files, finds definitions and usages, and collects evidence before it
            writes an answer. Ask follow-up questions like in a chat.</p></div>
          <div class="rl-card"><div class="step">Step 3</div><h4>Check the sources</h4>
            <p>Answers cite file and line ranges. Each citation is marked Verified, Not inspected or Not found,
            and links straight to the code on GitHub.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Load a repository")
    tab_github, tab_upload, tab_examples = st.tabs(["GitHub URL", "Upload", "Examples"])
    with tab_github:
        url = st.text_input(
            "Public GitHub repository",
            placeholder="https://github.com/psf/requests",
            help="Any public repository. Branches and commits work too: https://github.com/owner/repo/tree/<ref>",
        )
        if st.button("Load repository", type="primary", disabled=not url):
            if not is_github_url(url):
                st.error("That does not look like a GitHub repository URL.")
            else:
                build_workspace(
                    lambda: download_github_repo(url, settings.cache_dir, LIMITS, token=settings.github_token or None)
                )
    with tab_upload:
        uploads = st.file_uploader(
            "Project ZIP or source files",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help="Archives are extracted safely: path traversal, symlinks and oversized files are rejected.",
        )
        if st.button("Load upload", type="primary", disabled=not uploads):
            build_workspace(lambda: save_uploads([(f.name, f.getvalue()) for f in uploads], settings.cache_dir, LIMITS))
    with tab_examples:
        st.caption("Well-known open-source projects, ready to explore.")
        cols = st.columns(len(SAMPLE_REPOS) + 1)
        for col, (name, (sample_url, blurb)) in zip(cols, SAMPLE_REPOS.items()):
            with col:
                if st.button(name, use_container_width=True, help=blurb):
                    build_workspace(
                        lambda u=sample_url: download_github_repo(
                            u, settings.cache_dir, LIMITS, token=settings.github_token or None
                        )
                    )
        with cols[-1]:
            if st.button("RepoLens (this app)", use_container_width=True, help="The source code of this application"):
                build_workspace(lambda: RepoSource(APP_DIR, "RepoLens", "local"))

    st.markdown("#### Guide")
    left, right = st.columns(2)
    for i, (title, body) in enumerate(GUIDE):
        with left if i % 2 == 0 else right, st.expander(title):
            st.markdown(body)


workspace = st.session_state.workspace

if workspace is None:
    render_landing()
    st.stop()


# ---------------------------------------------------------------- chat

stats = workspace.index.stats()
link = (
    f"https://github.com/{workspace.source.github['owner']}/{workspace.source.github['repo']}"
    if workspace.source.github
    else None
)
title = html.escape(workspace.source.label)
title_html = (
    f'<a href="{link}" target="_blank" style="color:inherit;text-decoration:none">{title}</a>' if link else title
)
langs = "".join(f'<span class="rl-tag">{html.escape(k)} · {v}</span>' for k, v in list(stats["languages"].items())[:6])
search_kind = "Hybrid search (keyword + semantic)" if stats["embeddings"] else "Keyword search"
header, metrics = st.columns([3, 2], vertical_alignment="bottom")
with header:
    st.markdown(
        f'<div class="rl-eyebrow">Repository</div><h2 style="margin:0">{title_html}</h2>'
        f'<div class="rl-tags">{langs}</div><div class="rl-meta">{search_kind} · '
        f"{MODE_LABELS[mode]} mode · {html.escape(settings.llm_model if mode != 'search' else 'no AI model')}</div>",
        unsafe_allow_html=True,
    )
with metrics:
    st.markdown(
        f'<div class="rl-stats"><div class="rl-stat"><div class="v">{stats["files"]:,}</div><div class="k">Files</div></div>'
        f'<div class="rl-stat"><div class="v">{stats["chunks"]:,}</div><div class="k">Code passages</div></div>'
        f'<div class="rl-stat"><div class="v">{stats["symbols"]:,}</div><div class="k">Functions &amp; classes</div></div></div>',
        unsafe_allow_html=True,
    )
st.write("")

for message in st.session_state.messages:
    render_message(message, workspace)

if not st.session_state.messages:
    st.markdown('<div class="rl-section">Suggested questions</div>', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(question, use_container_width=True):
            st.session_state.pending = question
            st.rerun()
    with st.expander("How to read the answers"):
        st.markdown(GUIDE[1][1])
        st.markdown(GUIDE[3][1])

prompt = st.chat_input("Ask about this codebase, e.g. How does authentication work?")
if st.session_state.pending:
    prompt, st.session_state.pending = st.session_state.pending, None

asked = sum(1 for m in st.session_state.messages if m["role"] == "assistant" and m.get("mode") != "search")
if prompt and MAX_QUESTIONS and mode != "search" and asked >= MAX_QUESTIONS:
    st.warning(
        f"This public demo allows {MAX_QUESTIONS} AI answers per session to share a free API quota fairly. "
        "Search only mode still works, or run RepoLens locally (see the GitHub README) for unlimited use."
    )
    prompt = None

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
        settings.llm_base_url,
        settings.llm_model,
        settings.llm_api_key,
        settings.llm_timeout,
        settings.ollama_num_ctx,
        settings.llm_max_output_tokens,
        settings.reasoning_effort,
    )
    if hasattr(llm, "limiter"):
        llm.limiter.on_wait = lambda seconds, reason: st.toast(f"{reason} Waiting {seconds:.0f}s.")
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        if history and is_follow_up(prompt):
            st.caption("Follow-up detected: searching together with your previous question.")
        if mode == "quick":
            rag = retrieve(index, prompt, top_k=top_k, history=history)
            with st.expander(f"Retrieved code ({len(rag.hits)})", expanded=False):
                render_hits([h.to_dict() for h in rag.hits], workspace.source)
            try:
                st.write_stream(stream_answer(llm, index, rag, history, memory, settings.llm_max_context_tokens))
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
                "meta": f"Quick answer · {settings.llm_model} · {time.perf_counter() - started:.1f}s",
            }
        else:
            agent = Agent(
                index, llm, AgentConfig.from_settings(settings, max_steps=max_steps), label=workspace.source.label
            )
            result = None
            with st.status("Investigating", expanded=True) as status:
                for event in agent.run(prompt, history, memory):
                    if event.kind == "tool_call":
                        st.markdown(step_label(event), unsafe_allow_html=True)
                        stage = "Initial search" if event.data.get("auto") else f"Step {event.step}"
                        status.update(label=f"Investigating · {stage}: {event.data.get('tool')}")
                    elif event.kind == "status":
                        status.update(label=f"Investigating · step {event.step}: {event.message.rstrip('.')}")
                    elif event.kind in {"warning", "error"}:
                        st.warning(event.message)
                    elif event.kind == "answer":
                        result = event.data["result"]
                status.update(
                    label=f"Investigation complete · {result.steps} steps · {result.tool_calls} tool calls · "
                    f"{result.elapsed:.1f}s",
                    state="error" if result.stop_reason == "error" else "complete",
                    expanded=False,
                )
            hits = next(
                (e.data.get("hits", []) for e in result.events if e.kind == "tool_result" and e.data.get("hits")), []
            )
            tokens = result.usage.get("prompt_tokens", 0) + result.usage.get("completion_tokens", 0)
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
                "meta": f"Agent · {settings.llm_model} · {result.steps} steps · {tokens:,} tokens · "
                f"{result.elapsed:.1f}s",
            }
    st.session_state.messages.append(message)
    st.rerun()
