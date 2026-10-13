"""Unit tests for tools.py — no API keys required.

Run with: python -m pytest tests/ -v
"""

import os
import sys
import json
import tempfile

# Add project root to path so we can import tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools


# ═══════════════════════════════════════════════════════════
#  URL CLEANING
# ═══════════════════════════════════════════════════════════

def test_clean_url_strips_whitespace():
    assert tools._clean_url("  https://example.com  ") == "https://example.com"


def test_clean_url_extracts_from_markdown():
    assert tools._clean_url("[link](https://example.com)") == "https://example.com"


def test_clean_url_strips_trailing_punctuation():
    assert tools._clean_url("https://example.com.") == "https://example.com"
    assert tools._clean_url("https://example.com,") == "https://example.com"


def test_clean_url_extracts_from_text():
    assert tools._clean_url("check this https://example.com out") == "https://example.com"


def test_clean_url_handles_empty():
    assert tools._clean_url("") == ""
    assert tools._clean_url(None) is None


# ═══════════════════════════════════════════════════════════
#  WORKSPACE
# ═══════════════════════════════════════════════════════════

def test_workspace_roundtrip(tmp_path, monkeypatch):
    """Set workspace, write a file, read it back, list it."""
    monkeypatch.setattr(tools, "WORKSPACE_FILE", str(tmp_path / "ws.json"))

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    result = tools.set_workspace(str(project_dir))
    assert "✅" in result

    assert tools.get_workspace() == str(project_dir)

    result = tools.write_project_file("hello.txt", "world")
    assert "✅" in result
    assert (project_dir / "hello.txt").read_text() == "world"

    assert tools.read_project_file("hello.txt") == "world"

    files = tools.list_project_files()
    assert "hello.txt" in files


def test_workspace_blocks_escape(tmp_path, monkeypatch):
    """Cannot read files outside the workspace."""
    monkeypatch.setattr(tools, "WORKSPACE_FILE", str(tmp_path / "ws.json"))

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    tools.set_workspace(str(project_dir))

    result = tools.read_project_file("../../../etc/passwd")
    assert "❌" in result


def test_workspace_no_workspace_set(tmp_path, monkeypatch):
    """Operations fail cleanly when no workspace is set."""
    monkeypatch.setattr(tools, "WORKSPACE_FILE", str(tmp_path / "missing.json"))

    result = tools.read_project_file("anything.txt")
    assert "❌" in result
    assert "No workspace" in result


# ═══════════════════════════════════════════════════════════
#  CODE EXECUTION
# ═══════════════════════════════════════════════════════════

def test_run_python_code_success():
    result = tools.run_python_code("print('hello')")
    assert "hello" in result
    assert "Exit: 0" in result


def test_run_python_code_error():
    result = tools.run_python_code("raise ValueError('boom')")
    assert "ValueError" in result


# ═══════════════════════════════════════════════════════════
#  PROMPTS
# ═══════════════════════════════════════════════════════════

def test_build_system_prompt():
    from prompts import build_system_prompt
    prompt = build_system_prompt("/tmp/workspace", "test memory")
    assert "Super Agent" in prompt
    assert "/tmp/workspace" in prompt
    assert "test memory" in prompt
    assert "Never lie about tool results" in prompt


# ═══════════════════════════════════════════════════════════
#  SKILLS
# ═══════════════════════════════════════════════════════════

def test_skills_save_and_get(tmp_path, monkeypatch):
    """Save a skill, retrieve it by name."""
    import skills
    monkeypatch.setattr(skills, "SKILLS_JSON", str(tmp_path / "skills.json"))
    monkeypatch.setattr(skills, "SKILLS_DIR", str(tmp_path / "chroma"))

    result = skills.save_skill(
        name="test_skill_xyz",
        description="a test",
        when_to_use="testing",
        steps="1. do thing",
        code_template="",
        tags=["test"],
    )
    assert "✅" in result

    retrieved = skills.get_skill("test_skill_xyz")
    assert "test_skill_xyz" in retrieved
    assert "a test" in retrieved

    listing = skills.list_skills()
    assert "test_skill_xyz" in listing

    result = skills.delete_skill("test_skill_xyz")
    assert "Deleted" in result or "🗑" in result

    assert "No skill named" in skills.get_skill("test_skill_xyz")