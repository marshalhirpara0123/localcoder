"""Core logic: model backends, sandboxed tools, JSON parsing. No UI code here."""
import difflib
import json
import subprocess
from pathlib import Path

import requests

SYSTEM = """You are a coding assistant working inside a project folder.
Respond ONLY with one JSON object per turn, in one of these forms:

{"tool": "list_files", "args": {}}
{"tool": "read_file", "args": {"path": "..."}}
{"tool": "write_file", "args": {"path": "...", "content": "FULL new file content"}}
{"tool": "run_command", "args": {"cmd": "..."}}
{"done": "final message to the user"}

Rules:
- Read a file before editing it.
- After editing, run the tests or the program to verify.
- Do one action per turn.
- When finished, reply with the done form."""

SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build"}
BLOCKED = ["rm -rf", "sudo", "mkfs", "shutdown", "reboot", "format ", ":(){"]


# ---------------------------------------------------------------- backends
class OllamaBackend:
    def __init__(self, model, temperature=0.2, base_url=""):
        self.model, self.temperature = model, temperature
        self.url = (base_url or "http://localhost:11434").rstrip("/") + "/api/chat"

    def chat(self, messages):
        r = requests.post(self.url, timeout=600, json={
            "model": self.model, "messages": messages, "stream": False,
            "format": "json", "options": {"temperature": self.temperature}})
        r.raise_for_status()
        return r.json()["message"]["content"]


class OpenAICompatBackend:
    """LM Studio (port 1234), llama.cpp server, etc."""

    def __init__(self, model, temperature=0.2, base_url=""):
        self.model, self.temperature = model, temperature
        self.base_url = (base_url or "http://localhost:1234/v1").rstrip("/")

    def chat(self, messages):
        r = requests.post(f"{self.base_url}/chat/completions", timeout=600, json={
            "model": self.model, "messages": messages,
            "temperature": self.temperature})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def make_backend(kind, model, temperature, base_url=""):
    cls = OpenAICompatBackend if kind == "openai" else OllamaBackend
    return cls(model, temperature, base_url)


def list_ollama_models(base_url=""):
    """Returns installed Ollama model names, or [] if Ollama is not reachable."""
    try:
        url = (base_url or "http://localhost:11434").rstrip("/") + "/api/tags"
        r = requests.get(url, timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def extract_json(text):
    """Models sometimes wrap JSON in ```fences``` or add text around it."""
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s >= 0 and e > s:
            return json.loads(cleaned[s:e + 1])
        raise


# ------------------------------------------------------------------- tools
class Tools:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def safe(self, path):
        p = (self.root / path).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("Path is outside the project folder")
        return p

    def list_files(self):
        files = [str(p.relative_to(self.root)) for p in self.root.rglob("*")
                 if p.is_file()
                 and not SKIP_DIRS & set(p.relative_to(self.root).parts)]
        return "\n".join(sorted(files))[:4000] or "(empty project)"

    def read_file(self, path, start=1, end=None):
        lines = self.safe(path).read_text(errors="replace").splitlines()
        return "\n".join(lines[max(1, int(start)) - 1:end])[:8000]

    def preview_write(self, path, content):
        """Returns (is_new, unified_diff). Does NOT write anything."""
        p = self.safe(path)
        old = p.read_text(errors="replace") if p.exists() else ""
        diff = "".join(difflib.unified_diff(
            old.splitlines(True), content.splitlines(True),
            f"a/{path}", f"b/{path}"))
        return (not p.exists()), diff

    def write_file(self, path, content):
        p = self.safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return "File written."

    @staticmethod
    def is_blocked(cmd):
        return any(b in cmd for b in BLOCKED)

    def run_command(self, cmd):
        if self.is_blocked(cmd):
            return "Blocked: command is not allowed."
        try:
            r = subprocess.run(cmd, shell=True, cwd=self.root,
                               capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return "Command timed out after 60s."
        return ((r.stdout + r.stderr) or "(no output)")[-4000:]
