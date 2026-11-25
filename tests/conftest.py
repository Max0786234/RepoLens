from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from repolens.index import CodeIndex

SAMPLE_FILES = {
    "README.md": """
        # Sample Shop

        A tiny shop used by the RepoLens tests.

        ## Installation

        Run `pip install sample-shop` to install it.
    """,
    "shop/__init__.py": "",
    "shop/auth.py": '''
        """Authentication helpers."""
        import hashlib

        SECRET_SALT = "pepper"


        def hash_password(password: str) -> str:
            """Hash a password with a salt."""
            return hashlib.sha256((SECRET_SALT + password).encode()).hexdigest()


        class LoginManager:
            """Keeps track of logged-in users."""

            def __init__(self):
                self.sessions = {}

            def login(self, user, password):
                if hash_password(password) != user.password_hash:
                    raise PermissionError("bad credentials")
                self.sessions[user.name] = True
                return True

            def logout(self, user):
                self.sessions.pop(user.name, None)
    ''',
    "shop/cart.py": '''
        from shop.auth import hash_password


        def compute_total(items, discount=0.0):
            """Sum item prices and apply a percentage discount."""
            subtotal = sum(item["price"] * item["qty"] for item in items)
            return round(subtotal * (1 - discount), 2)


        def apply_coupon(total, code):
            if code == "SAVE10":
                return total * 0.9
            return total
    ''',
    "web/server.js": """
        const express = require('express');

        /**
         * Create the HTTP server and register routes.
         */

        function createServer(port) {
          const app = express();
          app.get('/health', (req, res) => {
            res.send('ok');
          });
          return app.listen(port);
        }

        class RateLimiter {
          constructor(limit) {
            this.limit = limit;
          }

          allow(clientId) {
            return true;
          }
        }

        module.exports = { createServer, RateLimiter };
    """,
    "tests/test_cart.py": """
        from shop.cart import compute_total


        def test_compute_total():
            assert compute_total([{"price": 2, "qty": 3}]) == 6
    """,
    ".env": "API_KEY=super-secret",
    "node_modules/lib/index.js": "module.exports = 1;",
    "image.png": "not really a png",
}


def write_repo(root: Path, files: dict[str, str] = SAMPLE_FILES) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return root


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    return write_repo(tmp_path / "repo")


@pytest.fixture
def sample_index(sample_repo: Path) -> CodeIndex:
    return CodeIndex.build(sample_repo)


class FakeEmbedder:
    """Deterministic bag-of-words embedder: semantic search without a model server."""

    name = "fake-embedder"
    vocabulary = ["password", "login", "hash", "total", "price", "discount", "server", "http", "route", "install", "limit", "coupon"]  # fmt: skip

    def __init__(self) -> None:
        self.document_calls = 0

    def _vector(self, text: str, other: float) -> np.ndarray:
        lower = text.lower()
        counts = [lower.count(word) for word in self.vocabulary]
        # An extra "other" dimension keeps vocabulary-free documents from looking similar to everything.
        vec = np.array([*counts, other], dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def embed_documents(self, texts, progress=None):
        self.document_calls += len(texts)
        return np.stack([self._vector(t, other=1.0) for t in texts])

    def embed_query(self, text):
        return self._vector(text, other=0.01)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
