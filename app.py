"""Local AI Coding Agent - Streamlit web UI.   Run:  streamlit run app.py"""
import subprocess
import sys
from pathlib import Path

import streamlit as st

import core

st.set_page_config(page_title="Local Coder", page_icon="💻", layout="wide")
ss = st.session_state

# ------------------------------------------------------------ session state
defaults = {
    "history": [],      # messages sent to the model
    "log": [],          # what is shown on screen: dicts {kind, text, detail}
    "pending": None,    # an action waiting for Approve / Reject
    "running": False,   # is the agent loop active
    "steps": 0,
    "bad_json": 0,
    "folder": str(Path.cwd() / "test_project"),
}
for k, v in defaults.items():
    ss.setdefault(k, v)


def add_log(kind, text, detail=None):
    ss.log.append({"kind": kind, "text": text, "detail": detail})


def reset_chat():
    ss.history, ss.log, ss.pending = [], [], None
    ss.running, ss.steps, ss.bad_json = False, 0, 0


def browse_folder():
    """Opens a native folder dialog (works because Streamlit runs on your PC)."""
    code = ("import tkinter as tk\nfrom tkinter import filedialog\n"
            "r=tk.Tk();r.withdraw();r.attributes('-topmost',True)\n"
            "print(filedialog.askdirectory() or '')")
    try:
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=120).stdout.strip()
        if out:
            ss.folder = out
            reset_chat()
    except Exception:
        pass


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    backend_kind = st.selectbox(
        "Backend", ["ollama", "openai"],
        format_func=lambda x: "Ollama" if x == "ollama" else "LM Studio / OpenAI-compatible")
    base_url = st.text_input("Server URL (optional)", "",
                             placeholder="default for the chosen backend")

    if backend_kind == "ollama":
        installed = core.list_ollama_models(base_url)
        if installed:
            model = st.selectbox("Model", installed)
        else:
            st.warning("Can't reach Ollama. Is it running?")
            model = st.text_input("Model", "qwen2.5-coder:7b")
    else:
        model = st.text_input("Model", "local-model")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_steps = st.slider("Max steps per request", 3, 30, 15)
    max_history = st.slider("Messages kept in context", 4, 40, 14)
    confirm_cmds = st.checkbox("Ask before running commands", True,
                               help="Strongly recommended.")

    st.divider()
    st.header("📁 Project")
    st.text_input("Project folder", key="folder")
    st.button("Browse...", on_click=browse_folder)

    # --- Drag & Drop File Upload Feature ---
    st.divider()
    st.header("📤 Upload Files")
    uploaded_files = st.file_uploader(
        "Drag and drop files to add to project",
        accept_multiple_files=True,
        key="project_file_uploader"
    )
    if uploaded_files:
        if st.button("Save Uploaded Files to Project"):
            target_folder = Path(ss.folder).expanduser()
            if target_folder.is_dir():
                saved_count = 0
                for file in uploaded_files:
                    save_path = target_folder / file.name
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())
                    saved_count += 1
                st.success(f"Successfully saved {saved_count} file(s) to `{target_folder.name}`!")
                add_log("tool", f"📤 Uploaded {saved_count} file(s) to project folder.")
            else:
                st.error("Invalid project folder.")

    st.button("🗑️ Clear conversation", on_click=reset_chat)

# ------------------------------------------------------------------- header
st.title("💻 Local AI Coder")
st.caption("Runs 100% on your machine - no cloud, no token limits.")

folder = Path(ss.folder).expanduser()
if not folder.is_dir():
    st.warning("Choose a valid project folder in the sidebar to begin.")
    st.stop()
tools = core.Tools(folder)
st.info(f"📂 Working in: `{tools.root}`")


# ---------------------------------------------------------- approve / reject
def resolve(approved):
    p = ss.pending
    if not p:
        return
    if not approved:
        result = "User rejected the action."
        add_log("tool", f"❌ Rejected {p['tool']}")
    else:
        try:
            if p["tool"] == "write_file":
                result = tools.write_file(**p["args"])
            else:
                result = tools.run_command(p["args"]["cmd"])
        except Exception as e:
            result = f"Error: {e}"
        add_log("tool", f"✅ Approved {p['tool']}", result)
    ss.history.append({"role": "user", "content": f"Tool result:\n{result}"})
    ss.pending = None
    ss.running = True  # resume the agent loop


# --------------------------------------------------------------- agent loop
def advance(backend, status):
    """Runs agent steps until it finishes, needs approval, or hits the limit."""
    while ss.running and ss.pending is None:
        if ss.steps >= max_steps:
            add_log("error", "Stopped: step limit reached.")
            ss.running = False
            return
        ss.steps += 1

        status.update(label=f"Thinking... (step {ss.steps}/{max_steps})")
        context = [{"role": "system", "content": core.SYSTEM}] + ss.history[-max_history:]
        try:
            reply = backend.chat(context)
        except Exception as e:
            add_log("error", f"Model error: {e}  (Is the model server running?)")
            ss.running = False
            return
        ss.history.append({"role": "assistant", "content": reply})

        try:
            action = core.extract_json(reply)
            ss.bad_json = 0
        except Exception:
            ss.bad_json += 1
            if ss.bad_json >= 3:
                add_log("error", "Model keeps returning invalid JSON. Try a stronger model.")
                ss.running = False
                return
            ss.history.append({"role": "user",
                               "content": "Invalid JSON. Reply with one valid JSON object."})
            continue

        if isinstance(action.get("done"), str):
            add_log("ai", action["done"])
            ss.running = False
            return

        name, args = action.get("tool"), action.get("args") or {}
        try:
            if name == "write_file":
                is_new, diff = tools.preview_write(args["path"], args["content"])
                if not is_new and not diff:
                    result = "No changes (file already identical)."
                else:
                    ss.pending = {"tool": name, "args": args, "diff": diff, "is_new": is_new}
                    return
            elif name == "run_command":
                if tools.is_blocked(args["cmd"]):
                    result = "Blocked: command is not allowed."
                elif confirm_cmds:
                    ss.pending = {"tool": name, "args": args}
                    return
                else:
                    result = tools.run_command(args["cmd"])
            elif name == "list_files":
                result = tools.list_files()
            elif name == "read_file":
                result = tools.read_file(**args)
            else:
                result = "Unknown tool. Use: list_files, read_file, write_file, run_command."
        except Exception as e:
            result = f"Error: {e}"

        label = f"🔧 {name} {args.get('path') or args.get('cmd') or ''}".strip()
        add_log("tool", label, result)
        status.write(label)
        ss.history.append({"role": "user", "content": f"Tool result:\n{result}"})


# ------------------------------------------------------------ chat rendering
log_box = st.container()

if prompt := st.chat_input("Ask the local AI to fix a bug or write code...",
                           disabled=ss.pending is not None):
    add_log("user", prompt)
    ss.history.append({"role": "user", "content": prompt})
    ss.running, ss.steps, ss.bad_json = True, 0, 0

if ss.running and ss.pending is None:
    backend = core.make_backend(backend_kind, model, temperature, base_url)
    with st.status("Working...", expanded=True) as status:
        advance(backend, status)
        status.update(label="Waiting for your approval" if ss.pending else "Done",
                      state="running" if ss.pending else "complete", expanded=False)


def render_log():
    for e in ss.log:
        if e["kind"] == "user":
            with st.chat_message("user"):
                st.write(e["text"])
        elif e["kind"] == "ai":
            with st.chat_message("assistant"):
                st.write(e["text"])
        elif e["kind"] == "error":
            st.error(e["text"])
        else:  # tool
            if e["detail"]:
                with st.expander(e["text"]):
                    st.code(e["detail"], language="text")
            else:
                st.caption(e["text"])


with log_box:
    if not ss.log:
        st.write("👋 Try: *The test is failing, please fix it*")
    render_log()

# -------------------------------------------------------- approval request UI
if ss.pending:
    p = ss.pending
    with st.container(border=True):
        if p["tool"] == "write_file":
            st.subheader(f"{'🆕 Create' if p['is_new'] else '✏️ Edit'} `{p['args']['path']}`")
            if p["diff"]:
                st.code(p["diff"], language="diff")
            if p["is_new"]:
                st.code(p["args"]["content"])
        else:
            st.subheader("⚠️ Run this command?")
            st.code(p["args"]["cmd"], language="bash")
        c1, c2, _ = st.columns([1, 1, 4])
        c1.button("✅ Approve", key="approve", type="primary",
                  on_click=resolve, args=(True,))
        c2.button("❌ Reject", key="reject", on_click=resolve, args=(False,))
