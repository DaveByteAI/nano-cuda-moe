#!/usr/bin/env python3
"""A coding agent in the terminal, thinking with this machine's model (a running bl-server).

    python3 agent/cli.py [--url http://127.0.0.1:8080] [--root DIR] [--think low] [--yes] [--prompt "..."]

Type a request; the model reads, searches, edits and runs commands in DIR (default: the current directory) until it
answers. Files outside DIR are out of reach; every shell command asks first (y / n / a = allow for the session)
unless --yes; edits print their diff. Commands: /reset, /think low|medium|xhigh|off, /exit; Ctrl+C stops an answer.
--prompt runs one request and exits. bl-server needs function calling (this repository's) and enough context: the
web page's 32K fills after a few files; start it with --ctx 131072 for this. Python standard library only.
"""
import argparse
import difflib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    import readline   # noqa: F401  (line editing and history for input())
except ImportError:
    pass

OUT_MAX = 12000
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
DIM, BOLD, CYAN, RED, END = ("\033[2m", "\033[1m", "\033[36m", "\033[31m", "\033[0m") if sys.stdout.isatty() else ("",) * 5

TOOLS = [
    {"type": "function", "function": {"name": "list_dir", "description": "List a directory.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "relative to the working directory"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read lines of a text file, numbered.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer", "description": "first line (1-based), default 1"},
                                                     "count": {"type": "integer", "description": "lines, default 200, at most 400"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "search", "description": "grep -rnE a regular expression (at most 100 matching lines).",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "description": "default: the working directory"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace one exact, unique occurrence of old_text with new_text.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a whole file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command in the working directory (the user may refuse it); returns its output.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer", "description": "default 120"}}, "required": ["command"]}}},
]

SYSTEM = """You are a coding agent working in a terminal, in the directory {root} ({os}; today is {date}).
Use the tools to look at the files and to act; do not guess what a file says. Keep each step small: read what you
need, change it, check it (build, run, test) when that makes sense. `edit_file` needs old_text to appear exactly once:
copy it from read_file output without the line numbers. Shell commands are shown to the user, who may refuse one.
When the task is done, or you need the user, answer in plain text without calling a tool: say briefly what you did."""


def clip(s, n=OUT_MAX):
    return s if len(s) <= n else s[:n // 2] + f"\n... [{len(s) - n} characters cut] ...\n" + s[-n // 2:]


class Session:
    def __init__(self, a):
        self.a, self.root = a, pathlib.Path(a.root).resolve()
        self.think = a.think if a.think != "off" else ""
        self.allow_all = a.yes
        self.seen = []
        self.reset()

    def reset(self):
        self.msgs = [{"role": "system", "content": SYSTEM.format(root=self.root, os=platform.platform(), date=time.strftime("%Y-%m-%d"))}]

    # ---- tools
    def path(self, rel):
        p = (self.root / rel).resolve()
        if p != self.root and not str(p).startswith(str(self.root) + os.sep): raise ValueError(f"{rel}: outside {self.root}")
        return p

    def show_diff(self, p, old, new):
        d = list(difflib.unified_diff(old.splitlines(), new.splitlines(), str(p.relative_to(self.root)), str(p.relative_to(self.root)), lineterm="", n=1))
        for l in d[:60]:
            print((CYAN if l.startswith("+") else RED if l.startswith("-") else DIM) + "    " + l + END)
        if len(d) > 60: print(DIM + f"    ... {len(d) - 60} more diff lines" + END)

    def tool(self, name, args):
        try:
            if name == "list_dir":
                p = self.path(args.get("path", "."))
                return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir() if x.name != ".git")) or "(empty)"
            if name == "read_file":
                lines = self.path(args["path"]).read_text(errors="replace").splitlines()
                start, count = max(1, int(args.get("start", 1))), min(400, int(args.get("count", 200)))
                body = "\n".join(f"{i}\t{l}" for i, l in enumerate(lines[start - 1:start - 1 + count], start))
                return clip(body) + f"\n[lines {start}-{min(len(lines), start + count - 1)} of {len(lines)}]"
            if name == "search":
                p = subprocess.run(["grep", "-rnE", "--exclude-dir=.git", "--exclude-dir=build", "--exclude-dir=node_modules",
                                    args["pattern"], str(self.path(args.get("path", ".")))], capture_output=True, text=True, timeout=60)
                lines = p.stdout.replace(str(self.root) + "/", "").splitlines()
                return clip("\n".join(lines[:100]) + (f"\n[{len(lines)} matches, first 100]" if len(lines) > 100 else "")) or "no match"
            if name == "edit_file":
                p = self.path(args["path"])
                s = p.read_text()
                n = s.count(args["old_text"])
                if n != 1: return f"error: old_text found {n} times; it must be exactly once (include more context)"
                new = s.replace(args["old_text"], args["new_text"])
                self.show_diff(p, s, new)
                p.write_text(new)
                return "edited"
            if name == "write_file":
                p = self.path(args["path"])
                old = p.read_text(errors="replace") if p.exists() else ""
                p.parent.mkdir(parents=True, exist_ok=True)
                self.show_diff(p, old, args["content"])
                p.write_text(args["content"])
                return f"written ({len(args['content'])} characters)"
            if name == "bash":
                cmd = args["command"]
                if not self.allow_all:
                    ans = input(f"{BOLD}  run? [y/N/a]{END} ").strip().lower()
                    if ans == "a": self.allow_all = True
                    elif ans != "y": return "the user refused this command"
                p = subprocess.run(["bash", "-c", cmd], cwd=self.root, capture_output=True, text=True,
                                   timeout=int(args.get("timeout_s", 120)))
                out = (p.stdout + p.stderr).strip()
                print(DIM + "    " + "\n    ".join(clip(out, 1200).splitlines()[:20]) + END)
                return clip(f"exit {p.returncode}\n{out}")
            return f"error: no tool {name}"
        except subprocess.TimeoutExpired:
            return "error: timed out"
        except Exception as e:
            return f"error: {e}"

    # ---- the model
    def request(self):
        """one streamed reply: reasoning and text printed as they come; the message and the context used"""
        body = {"messages": self.msgs, "tools": TOOLS, "stream": True, "max_tokens": self.a.max_tokens,
                "temperature": 0.6, "top_p": 0.95, "top_k": 20}
        if self.think: body["reasoning_effort"] = self.think
        req = urllib.request.Request(self.a.url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        msg, calls, used, where = {"role": "assistant", "content": "", "reasoning_content": ""}, [], 0, None
        with OPENER.open(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]": continue
                d = json.loads(line[6:])
                if "error" in d: raise RuntimeError(d["error"].get("message", d))
                if d.get("usage"): used = d["usage"]["prompt_tokens"] + d["usage"]["completion_tokens"]
                delta = d["choices"][0]["delta"]
                for key, style in (("reasoning_content", DIM), ("content", "")):
                    piece = delta.get(key)
                    if piece:
                        if where != key: print("\n" if where else "", end=""); where = key
                        print(style + piece + (END if style else ""), end="", flush=True)
                        msg[key] += piece
                calls = delta.get("tool_calls") or calls
        if where: print()
        if calls: msg["tool_calls"] = [{k: c[k] for k in ("id", "type", "function")} for c in calls]
        if not msg["reasoning_content"]: del msg["reasoning_content"]
        return msg, used

    def trim(self):
        """past the budget: the system prompt, a note of what was looked at, the last messages without reasoning"""
        tail = [{k: v for k, v in x.items() if k != "reasoning_content"} for x in self.msgs[-10:]]
        while tail and tail[0]["role"] == "tool": tail.pop(0)
        note = {"role": "user", "content": "(The earlier part of this conversation was cut to fit the context. Tool calls made "
                                           "there:\n" + "\n".join(self.seen[-60:]) + ")"}
        self.msgs = self.msgs[:1] + [note] + tail
        self.seen = []
        print(DIM + "  (history trimmed to fit the context)" + END)

    def turn(self, text):
        self.msgs.append({"role": "user", "content": text})
        for _ in range(self.a.max_steps):
            try:
                msg, used = self.request()
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code == 400 and "longer than the context" in detail and len(self.msgs) > 4:
                    self.trim()
                    continue
                print(RED + f"  server: {e.code} {detail}" + END)
                return
            self.msgs.append(msg)
            if used > 0.6 * self.a.ctx: self.trim()
            if not msg.get("tool_calls"): return
            for c in msg["tool_calls"]:
                name = c["function"]["name"]
                try: args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError: args = {}
                shown = args.get("command") or args.get("path") or args.get("pattern") or ""
                print(f"{CYAN}● {name}{END} {shown}")
                out = self.tool(name, args)
                self.seen.append(f"- {name} {json.dumps(args, ensure_ascii=False)[:160]}")
                if out.startswith("error"): print(RED + "  " + out[:300] + END)
                self.msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        print(DIM + f"  (stopped after {self.a.max_steps} steps)" + END)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--root", default=".", help="the working directory; files outside it are out of reach")
    ap.add_argument("--think", default="low", help="reasoning effort: low, medium, xhigh, off")
    ap.add_argument("--yes", action="store_true", help="run shell commands without asking")
    ap.add_argument("--ctx", type=int, default=32768, help="the server's --ctx (the history is trimmed at 60%%)")
    ap.add_argument("--max-tokens", type=int, default=8192, help="per reply, thinking included")
    ap.add_argument("--max-steps", type=int, default=40, help="tool rounds per request")
    ap.add_argument("--prompt", help="run this one request and exit")
    a = ap.parse_args()
    s = Session(a)
    if a.prompt:
        s.turn(a.prompt)
        return
    print(f"{BOLD}bl agent{END} - {s.root} - thinking {a.think}; /reset, /think MODE, /exit; Ctrl+C stops an answer")
    while True:
        try:
            text = input(f"\n{BOLD}> {END}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text: continue
        if text in ("/exit", "/quit"): return
        if text == "/reset": s.reset(); print(DIM + "  new conversation" + END); continue
        if text.startswith("/think"):
            mode = text.split()[1] if len(text.split()) > 1 else "low"
            s.think = "" if mode == "off" else mode
            print(DIM + f"  thinking: {mode}" + END)
            continue
        try:
            s.turn(text)
        except KeyboardInterrupt:
            print(RED + "\n  (stopped)" + END)
            while s.msgs[-1]["role"] != "user" and s.msgs[-1].get("tool_calls"):   # an unanswered call: drop it
                s.msgs.pop()


if __name__ == "__main__":
    main()
