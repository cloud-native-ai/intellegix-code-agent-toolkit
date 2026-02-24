"""Shared test helpers for the automated OpenCode loop test suite.

Fixtures are in conftest.py. This module contains non-fixture helpers
(mock builders, output stream builders) used across multiple test files.
"""

import io
import json
from unittest.mock import MagicMock


# --- OpenCode output builders ---

def build_opencode_output(
    result_text: str,
    is_error: bool = False,
) -> str:
    """Build a realistic output string matching OpenCode -f json format.

    ``opencode -p <prompt> -f json`` outputs a single JSON object::

        {"response": "<assistant text>"}

    ``is_error`` is accepted for API compatibility but has no effect on the
    output format, since OpenCode does not distinguish error responses in its
    JSON output.
    """
    return json.dumps({"response": result_text})


# Keep a backward-compatible alias so that existing test call-sites can be
# migrated incrementally.  The *session_id*, *cost*, and *duration_ms*
# arguments are silently ignored because OpenCode does not include them in its
# output.
def build_ndjson_stream(
    session_id: str,
    cost: float,
    turns: int,
    result_text: str,
    is_error: bool = False,
    duration_ms: int = 10000,
) -> str:
    """Backward-compatible alias for :func:`build_opencode_output`.

    .. deprecated::
        Use :func:`build_opencode_output` directly.  The *session_id*, *cost*,
        *turns*, and *duration_ms* parameters are no longer reflected in the
        output because OpenCode uses a simpler ``{"response": "..."}`` format.
    """
    return build_opencode_output(result_text, is_error=is_error)


# --- Subprocess mock helpers ---

def mock_playwright_result(synthesis: str = "Keep going") -> MagicMock:
    """Build a mock subprocess result for Playwright research."""
    return MagicMock(
        returncode=0,
        stdout=json.dumps({
            "synthesis": synthesis,
            "models": ["perplexity-research"],
            "citations": [],
            "execution_time_ms": 30000,
        }),
        stderr="",
    )


def mock_playwright_error(error: str = "Browser timeout") -> MagicMock:
    """Build a mock subprocess result for Playwright error."""
    return MagicMock(
        returncode=0,
        stdout=json.dumps({
            "error": error,
            "synthesis": "",
            "execution_time_ms": 0,
        }),
        stderr="",
    )


def mock_git_log_result() -> MagicMock:
    """Mock result for git log (not a git repo)."""
    return MagicMock(returncode=128, stdout="", stderr="not a git repo")


def make_subprocess_dispatcher(
    opencode_result=None,
    opencode_side_effect=None,
    research_result=None,
    research_side_effect=None,
    # Backward-compatible aliases
    claude_result=None,
    claude_side_effect=None,
):
    """Create a subprocess.run mock that dispatches based on command.

    - git commands -> mock_git_log_result()
    - opencode commands -> opencode_result or raise opencode_side_effect
    - council_browser commands -> research_result or raise research_side_effect

    The *claude_result* / *claude_side_effect* kwargs are accepted as
    backward-compatible aliases for *opencode_result* / *opencode_side_effect*.
    """
    # Allow old call-sites that still pass claude_result / claude_side_effect
    if opencode_result is None and claude_result is not None:
        opencode_result = claude_result
    if opencode_side_effect is None and claude_side_effect is not None:
        opencode_side_effect = claude_side_effect

    def side_effect(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if isinstance(cmd, list) and cmd:
            if cmd[0] == "git":
                return mock_git_log_result()
            if cmd[0] == "opencode":
                # Handle preflight check (opencode --version)
                if len(cmd) >= 2 and cmd[1] == "--version":
                    return MagicMock(returncode=0, stdout="opencode 1.0.0-test\n", stderr="")
                if opencode_side_effect is not None:
                    raise opencode_side_effect
                return opencode_result
            if "council_browser" in str(cmd):
                if research_side_effect is not None:
                    raise research_side_effect
                return research_result
        # Default fallback
        if opencode_result is not None:
            return opencode_result
        return MagicMock(returncode=0, stdout="", stderr="")

    return side_effect


def make_research_dispatcher(playwright_result=None, playwright_side_effect=None):
    """Create a subprocess.run mock for research bridge tests.

    Git log calls get a no-op result. Council browser calls get playwright_result
    or raise playwright_side_effect.
    """
    def side_effect(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if isinstance(cmd, list) and cmd and cmd[0] == "git":
            return mock_git_log_result()
        if playwright_side_effect is not None:
            raise playwright_side_effect
        return playwright_result

    return side_effect


# --- Popen mock for capturing OpenCode output ---

class MockPopen:
    """Mock subprocess.Popen that yields output lines from stdout.

    Used for testing _invoke_opencode which reads stdout until EOF.
    """

    def __init__(self, output_stream: str, returncode: int = 0) -> None:
        self.stdout = io.StringIO(output_stream)
        self.stderr = io.StringIO("")
        self.returncode = returncode
        self.pid = 99999

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        pass


def make_popen_factory(
    output_stream: str, returncode: int = 0
):
    """Create a factory for subprocess.Popen mock (returns MockPopen)."""
    def factory(*args, **kwargs):
        return MockPopen(output_stream, returncode)
    return factory


def make_popen_dispatcher(
    opencode_output: str | None = None,
    opencode_returncode: int = 0,
    opencode_side_effect: Exception | None = None,
    # Backward-compatible aliases
    claude_ndjson: str | None = None,
    claude_returncode: int | None = None,
    claude_side_effect: Exception | None = None,
):
    """Create a side_effect for subprocess.Popen mock.

    ``opencode`` commands return MockPopen with the given output stream.
    Non-opencode Popen calls (e.g. taskkill) return a no-op MockPopen.

    The *claude_ndjson* / *claude_returncode* / *claude_side_effect* kwargs are
    accepted as backward-compatible aliases.
    """
    # Allow old call-sites that still use claude_* kwargs
    if opencode_output is None and claude_ndjson is not None:
        opencode_output = claude_ndjson
    if claude_returncode is not None:
        opencode_returncode = claude_returncode
    if opencode_side_effect is None and claude_side_effect is not None:
        opencode_side_effect = claude_side_effect

    def factory(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if isinstance(cmd, list) and cmd and cmd[0] == "opencode":
            if opencode_side_effect is not None:
                raise opencode_side_effect
            return MockPopen(opencode_output or "", opencode_returncode)
        # taskkill or other subprocess.Popen calls
        return MockPopen("", 0)
    return factory
