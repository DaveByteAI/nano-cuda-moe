#!/usr/bin/env python3
"""The engine optimizing itself: this machine's Qwen3.8 (a frozen bl-server, "the brain") drives a coding loop on a
copy of the repository, judged by bench/eval.py.

    python3 agent/run.py [--hours 8] [--max-evals 30] [--think medium] [--task "..."]

- the work happens in a git worktree (../boundless-agent-<time>, branch agent/<time>); the agent may change src/ and
  include/ only; it has no shell: list, read, search, edit, write, build, diff, evaluate, finish
- the brain is a copy of build/bl-server taken at start, so a broken change cannot break the agent itself
- one GPU: evaluate stops the brain, runs the examiner on the worktree, starts the brain again; the brain's next
  request re-reads the conversation
- a change is kept (committed on the branch) when the examiner passes it and it is >= 3% faster on the corpus decode or
  the 32K prompt without the other getting worse beyond the noise; otherwise the worktree goes back to the last kept
  commit. The journal of every evaluation stays in the system prompt
- the chat server on port 8080 is stopped at start and started again at the end
Logs: logs/agent/<time>/ (transcript.jsonl, journal.md, eval-N.json). Python standard library only.
"""
import argparse
import json
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import eval as examiner   # noqa: E402

MODEL = ROOT / "models" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
EDITABLE = ("src/", "include/")
# A change is measured against the best build so far in turns (eval.py --base): 3 window runs of each (the decode
# kernels on fixed text) and 2 runs of a 32K prompt. Kept when the windows gain >= KEEP (median of the pairs) with no
# pair slower, the prompt not losing more than LOSS; or the prompt gains >= KEEP with every pair >= 1%, the windows not
# losing more than LOSS. (One process against another an hour apart differs by up to ~5%: hence the pairs.)
KEEP, LOSS = 0.03, 0.01
LOOK_NUDGE, LOOK_LIMIT = 15, 25   # read/search/list calls since the last change: a reminder, then no more
TOOL_OUT_MAX = 12000
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

TOOLS = [
    {"type": "function", "function": {"name": "list_dir", "description": "List a directory of the repository.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "relative to the repository root, e.g. src/cuda"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read lines of a file, numbered.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer", "description": "first line (1-based), default 1"},
                                                     "count": {"type": "integer", "description": "lines, default 200, at most 400"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "search", "description": "grep -rnE a regular expression in the repository (or a sub-path); at most 100 matching lines.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "description": "default: the whole repository"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace one exact, unique occurrence of old_text with new_text in a file under src/ or include/.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a whole file under src/ or include/.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "build", "description": "Compile the repository; returns the errors, if any. Cheap: use it before evaluate.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "git_diff", "description": "Your changes since the last kept version.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "evaluate", "description": "Judge the current changes with the examiner (2-4 minutes): correctness gate, then speed. Kept if faster, otherwise reverted.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string", "description": "one line: what you changed and why it should be faster"}}, "required": ["summary"]}}},
    {"type": "function", "function": {"name": "finish", "description": "Stop working.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


def sh(cmd, cwd=None, timeout=600):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def clip(s, n=TOOL_OUT_MAX):
    return s if len(s) <= n else s[:n // 2] + f"\n... [{len(s) - n} characters cut] ...\n" + s[-n // 2:]


class Brain:
    """the frozen bl-server the agent thinks with"""
    def __init__(self, home, port, ctx, log):
        self.home, self.port, self.ctx, self.log, self.proc = home, port, ctx, log, None

    def start(self):
        out = open(self.log, "a")
        self.proc = subprocess.Popen([str(self.home / "build" / "bl-server"), "--model", str(MODEL), "--host", "127.0.0.1",
                                      "--port", str(self.port), "--ctx", str(self.ctx), "--web", "none"],
                                     stdout=out, stderr=out, start_new_session=True)
        for _ in range(300):
            if self.proc.poll() is not None: raise RuntimeError(f"the brain exited (see {self.log})")
            try:
                OPENER.open(f"http://127.0.0.1:{self.port}/health", timeout=2).read()
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("the brain did not come up in 300 s")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try: self.proc.wait(60)
            except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait()
        self.proc = None

    def chat(self, messages, think, max_tokens):
        body = {"messages": messages, "tools": TOOLS, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
        if think: body["reasoning_effort"] = think
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with OPENER.open(req, timeout=3600) as r:
            return json.load(r)


class Agent:
    def __init__(self, a):
        self.a = a
        self.stamp = time.strftime("%Y%m%d-%H%M%S")
        self.logs = ROOT / "logs" / "agent" / self.stamp
        self.logs.mkdir(parents=True)
        self.work = ROOT.parent / f"boundless-agent-{self.stamp}"
        self.branch = f"agent/{self.stamp}"
        self.brain = Brain(ROOT / "agent" / "brain", a.port, a.ctx, self.logs / "brain.log")
        self.journal, self.n_eval, self.done = [], 0, False
        self.gain_w = self.gain_p = 1.0   # the kept changes' product, against the start
        self.best_dir = ROOT / "agent" / "best"   # the best build so far: build/bl-run, data/ (the base of every A/B)
        self.seen = []   # the tool calls since the last trim, one line each (kept when the history is trimmed)
        self.looks = 0   # read/search/list calls since the last edit or evaluation
        self.t_end = time.time() + a.hours * 3600

    # ---- logging
    def note(self, kind, **kw):
        with open(self.logs / "transcript.jsonl", "a") as f:
            f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "kind": kind, **kw}, ensure_ascii=False) + "\n")

    def say(self, s):
        print(time.strftime("%H:%M:%S"), s, flush=True)

    # ---- setup and teardown
    def setup(self):
        sh(["pkill", "-x", "bl-server"])   # the chat server (port 8080): the GPU is ours until the end
        while subprocess.run(["pgrep", "-x", "bl-server"], capture_output=True).returncode == 0: time.sleep(1)
        home = ROOT / "agent" / "brain"   # the brain: today's build, frozen
        shutil.rmtree(home, ignore_errors=True)
        (home / "build").mkdir(parents=True)
        shutil.copy2(ROOT / "build" / "bl-server", home / "build" / "bl-server")
        (home / "data").symlink_to(ROOT / "data")
        rc, out = sh(["git", "-C", str(ROOT), "worktree", "add", "-b", self.branch, str(self.work), "HEAD"])
        if rc: raise RuntimeError(out)
        self.say(f"worktree {self.work} (branch {self.branch}); baseline build and evaluation ...")
        report = self.examine("baseline", extra=["--no-speed"])
        if report["verdict"] != "pass": raise RuntimeError("the baseline does not pass the examiner: " + json.dumps(report)[:2000])
        shutil.rmtree(self.best_dir, ignore_errors=True)
        (self.best_dir / "build").mkdir(parents=True)
        shutil.copy2(self.work / "build" / "bl-run", self.best_dir / "build" / "bl-run")
        (self.best_dir / "data").symlink_to(ROOT / "data")
        self.say("baseline: builds and passes the correctness gate")
        self.brain.start()

    def teardown(self):
        self.brain.stop()
        subprocess.Popen([str(ROOT / "build" / "bl-server"), "--model", str(MODEL), "--host", "0.0.0.0", "--port", "8080",
                          "--ctx", "32768"], stdout=open(ROOT / "logs" / "server.log", "w"), stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=ROOT)
        rc, kept = sh(["git", "-C", str(self.work), "log", "--oneline", "main..HEAD"])
        summary = f"# Agent run {self.stamp}\n\nbranch {self.branch}, worktree {self.work}\n\nkept commits:\n{kept or '(none)'}\n\n" + \
                  "\n".join(self.journal)
        (self.logs / "journal.md").write_text(summary + "\n")
        self.say(f"done: {self.n_eval} evaluations besides the baseline; journal {self.logs / 'journal.md'}; the chat server is starting on 8080")

    # ---- the examiner
    def examine(self, name=None, extra=()):
        if name is None:
            self.n_eval += 1
            name = str(self.n_eval)
        out = self.logs / f"eval-{name}.json"
        p = subprocess.run([sys.executable, str(ROOT / "bench" / "eval.py"), "--repo", str(self.work), "--out", str(out)] + list(extra),
                           capture_output=True, text=True, timeout=7200)
        if not out.exists(): return {"verdict": "fail", "error": clip(p.stdout + p.stderr, 4000)}
        return json.loads(out.read_text())

    def judge(self, summary):
        rc, diff = sh(["git", "-C", str(self.work), "diff"])
        (self.logs / f"eval-{self.n_eval + 1}.patch").write_text(diff)   # kept or not, for a human to look at later
        self.brain.stop()
        try:
            r = self.examine(extra=["--base", str(self.best_dir)])
        finally:
            self.brain.start()
        keep, why, ab = False, "", r.get("ab", {})
        if r.get("verdict") != "pass":
            why = "failed: " + ("build" if not r.get("build", {}).get("ok") else
                                "correctness gate" if not r.get("correct", {}).get("ok") else "measurement run")
        else:
            w, p = ab["windows_gain"], ab["prompt_gain"]
            keep = (w["median"] >= KEEP and w["min"] >= 0 and p["median"] > -LOSS) or \
                   (p["median"] >= KEEP and p["min"] >= 0.01 and w["median"] > -LOSS)
            why = f"windows {w['median']:+.1%} (pairs {w['min']:+.1%}..{w['max']:+.1%}), 32K prompt {p['median']:+.1%} " \
                  f"(pairs {p['min']:+.1%}..{p['max']:+.1%})"
            if keep:
                self.gain_w *= 1 + w["median"]
                self.gain_p *= 1 + p["median"]
        if keep:
            sh(["git", "-C", str(self.work), "add", "-A", "src", "include"])
            sh(["git", "-C", str(self.work), "-c", "user.name=agent", "-c", "user.email=agent@localhost", "commit", "-q",
                "-m", f"agent: {summary}\n\n{why}"])
            shutil.copy2(self.work / "build" / "bl-run", self.best_dir / "build" / "bl-run")   # the new base
        else:
            sh(["git", "-C", str(self.work), "reset", "-q", "--hard"])
            sh(["git", "-C", str(self.work), "clean", "-q", "-fd", "src", "include"])
        line = f"- eval {self.n_eval}: {'KEPT' if keep else 'reverted'} - {summary} - {why}"
        self.journal.append(line)
        self.say(line)
        brief = {"verdict": r.get("verdict"), "kept": keep, "why": why,
                 "rule": f"kept when the windows gain >= {KEEP:.0%} with no pair slower (prompt > -{LOSS:.0%}), or the 32K "
                         f"prompt gains >= {KEEP:.0%} with every pair >= 1% (windows > -{LOSS:.0%})"}
        if r.get("verdict") == "pass":
            brief["measured"] = {"windows_tok_s": ab["windows"], "prompt_32k_s": ab["prompt_32k_s"]}
        else:
            brief["details"] = {k: r[k] for k in ("build", "correct", "ab", "error") if k in r}
        if not keep: brief["note"] = "the worktree is back at the last kept version"
        return clip(json.dumps(brief, ensure_ascii=False, indent=1))

    # ---- the tools
    def path(self, rel, write=False):
        p = (self.work / rel).resolve()
        if not str(p).startswith(str(self.work)): raise ValueError("outside the repository")
        if write and not any(str(p).startswith(str(self.work / e)) for e in EDITABLE): raise ValueError("only src/ and include/ may be changed")
        return p

    def tool(self, name, args):
        if name in ("read_file", "search", "list_dir"):
            self.looks += 1
            if self.looks > LOOK_LIMIT:
                return (f"error: {self.looks - 1} look-around calls since your last change. Make a change now (edit_file / "
                        "write_file), build and evaluate; reading resumes after that.")
        elif name in ("edit_file", "write_file", "evaluate"):
            self.looks = 0
        out = self.tool_(name, args)
        if self.looks == LOOK_NUDGE:
            out += f"\n[{LOOK_NUDGE} look-around calls since your last change: decide on one change and make it]"
        return out

    def tool_(self, name, args):
        try:
            if name == "list_dir":
                p = self.path(args.get("path", "."))
                return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir() if x.name not in (".git", "build")))
            if name == "read_file":
                lines = self.path(args["path"]).read_text(errors="replace").splitlines()
                start, count = max(1, int(args.get("start", 1))), min(400, int(args.get("count", 200)))
                body = "\n".join(f"{i}\t{l}" for i, l in enumerate(lines[start - 1:start - 1 + count], start))
                return clip(body) + (f"\n[lines {start}-{min(len(lines), start + count - 1)} of {len(lines)}]")
            if name == "search":
                rc, out = sh(["grep", "-rnE", "--exclude-dir=build", "--exclude-dir=.git", args["pattern"], str(self.path(args.get("path", ".")))])
                out = out.replace(str(self.work) + "/", "")
                lines = out.splitlines()
                return clip("\n".join(lines[:100]) + (f"\n[{len(lines)} matches, first 100]" if len(lines) > 100 else "")) or "no match"
            if name == "edit_file":
                p = self.path(args["path"], write=True)
                s = p.read_text()
                n = s.count(args["old_text"])
                if n != 1: return f"error: old_text found {n} times; it must be exactly once (include more context)"
                p.write_text(s.replace(args["old_text"], args["new_text"]))
                return "edited"
            if name == "write_file":
                p = self.path(args["path"], write=True)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(args["content"])
                return f"written ({len(args['content'])} characters)"
            if name == "build":
                r = examiner.build(self.work, jobs=4)   # (the brain holds 43 GB of RAM meanwhile)
                return "build ok" if r["ok"] else "build FAILED:\n" + r["log"]
            if name == "git_diff":
                rc, out = sh(["git", "-C", str(self.work), "diff"])
                return clip(out) or "no changes"
            if name == "evaluate":
                if self.n_eval >= self.a.max_evals: return "error: no evaluations left; call finish"
                return self.judge(args.get("summary", "(no summary)"))
            if name == "finish":
                self.done = True
                self.journal.append(f"- finished: {args.get('summary', '')}")
                return "ok"
            return f"error: no tool {name}"
        except Exception as e:
            return f"error: {e}"

    # ---- the loop
    def system(self):
        prompt = (ROOT / "agent" / "system_prompt.md").read_text()
        j = "\n".join(self.journal) or "(no evaluation yet)"
        return prompt + f"\n\n# Journal (every evaluation so far)\n\nthe kept changes so far: windows {self.gain_w - 1:+.1%}, 32K prompt " \
                        f"{self.gain_p - 1:+.1%} against the start\n{j}\n\nEvaluations left: {self.a.max_evals - self.n_eval}; " \
                        f"time left: {max(0, (self.t_end - time.time()) / 3600):.1f} h."

    def trim(self, msgs):
        """the history past its budget: the system prompt (with the journal), the task, what was looked at, the last
        messages without their reasoning (the next request re-reads it all once)"""
        tail = [{k: v for k, v in x.items() if k != "reasoning_content"} for x in msgs[-8:]]
        while tail and tail[0]["role"] == "tool": tail.pop(0)
        looked = "\n".join(self.seen[-80:])
        self.seen = []
        self.say(f"history trimmed ({len(msgs)} messages -> {len(tail) + 3})")
        msgs[0]["content"] = self.system()
        return msgs[:2] + [{"role": "user", "content": "(The earlier part of this session was cut to fit the context. The tool calls "
                                                       "you made there, for reference - do not repeat them without a reason:\n" + looked + ")"}] + tail

    def run(self):
        task = self.a.task or "Make the engine faster without changing its results. Start by reading the code; measure your ideas with evaluate."
        msgs = [{"role": "system", "content": self.system()}, {"role": "user", "content": task}]
        idle, failures = 0, 0
        while not self.done and time.time() < self.t_end and self.n_eval < self.a.max_evals:
            try:
                r = self.brain.chat(msgs, self.a.think, self.a.max_tokens)
            except urllib.error.HTTPError as e:   # 400: most likely the context is full
                failures += 1
                self.say(f"brain refused the request ({e}); trimming the history ({failures}/3)")
                if failures > 3: break
                msgs = self.trim(msgs)
                continue
            except Exception as e:
                failures += 1
                self.say(f"brain request failed ({e}); restarting it ({failures}/3)")
                if failures > 3: break
                self.brain.stop(); self.brain.start()
                continue
            failures = 0
            m = r["choices"][0]["message"]
            used = r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]
            self.say(f"  ({used} tokens in context, {r['usage']['completion_tokens']} generated)")
            self.note("assistant", content=m.get("content"), reasoning=m.get("reasoning_content"), tool_calls=m.get("tool_calls"), tokens=used)
            msgs.append({k: m[k] for k in ("role", "content", "reasoning_content", "tool_calls") if m.get(k) is not None})
            calls = m.get("tool_calls") or []
            if not calls:
                idle += 1
                if idle > 3: break
                msgs.append({"role": "user", "content": "Use the tools to continue (call finish when you are done)."})
                continue
            idle = 0
            for c in calls:
                name = c["function"]["name"]
                try: args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError: args = {}
                self.say(f"{name} {json.dumps(args, ensure_ascii=False)[:160]}")
                out = self.tool(name, args)
                if name in ("read_file", "search", "list_dir"): self.seen.append(f"- {name} {json.dumps(args, ensure_ascii=False)[:200]}")
                self.note("tool", name=name, args=args, result=out[:4000])
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
                if name == "evaluate":   # the brain restarted anyway: refresh the journal (the whole history is re-read)
                    msgs[0]["content"] = self.system()
            if used > 0.55 * self.a.ctx:
                msgs = self.trim(msgs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=8)
    ap.add_argument("--max-evals", type=int, default=30)
    ap.add_argument("--think", default="low", help="the brain's reasoning effort: low, medium, xhigh, or '' (off)")
    ap.add_argument("--max-tokens", type=int, default=16384, help="per brain reply (thinking included)")
    ap.add_argument("--ctx", type=int, default=131072, help="the brain's context")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--task", default="", help="the first instruction (default: make the engine faster)")
    a = ap.parse_args()
    ag = Agent(a)
    try:
        ag.setup()
        ag.run()
    finally:
        ag.teardown()


if __name__ == "__main__":
    main()
