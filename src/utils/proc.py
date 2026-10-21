import json, os, sys, subprocess, signal
from datetime import datetime


try:
    import psutil  # optional, but best for process trees and stats
except Exception:
    psutil = None


def describe_exit(rc: int) -> str:
    if rc is None:
        return "still running"
    if rc == 0:
        return "exited normally (0)"
    if rc > 0:
        return f"exited with status {rc}"
    sig = -rc
    try:
        name = signal.Signals(sig).name
    except ValueError:
        name = f"SIG({sig})"
    return f"terminated by signal {sig} ({name})"


def snapshot_process(proc):
    """
    Build a best-effort diagnostic snapshot for a subprocess.Popen-like object.
    Includes returncode (after proc.poll()), cmdline, children, open files,
    cpu/mem info, and a tail of kernel messages that commonly include OOM/segfaults.
    """
    # Ensure returncode is current
    try:
        rc = proc.poll()
    except Exception:
        rc = None

    # obtain a human-readable returncode (e.g. "SIGTERM" for -15)
    if rc is not None:
        state = "exited"
        if rc >= 0:
            rc_str = str(rc)
        else:
            # Negative return codes are usually signals; convert to human-readable form
            try:
                rc_human = f"{signal.Signals(-rc).name}"
            except ValueError:
                # If the signal is not recognized, fallback to a generic message
                rc_human = "unknown signal"
            rc_str = f"{rc} ({rc_human})"
    else:
        state = "running"
        rc_str = str(getattr(proc, 'returncode', None))

    snap = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "pid": getattr(proc, "pid", None),
        "args": getattr(proc, "args", None),
        "state": state,
        "returncode": rc_str,
        "children": [],
        "cmdline": None,
        "open_files": [],
        "cpu_times": None,
        "memory_info": None,
        "limits": None,
        "proc_status": None,
    }

    pid = snap["pid"]
    if pid is None:
        return snap

    if psutil:
        try:
            p = psutil.Process(pid)
            # cmdline / cpu / mem
            try:
                snap["cmdline"] = p.cmdline()
            except Exception:
                pass
            try:
                snap["cpu_times"] = p.cpu_times()._asdict()
            except Exception:
                pass
            try:
                mi = p.memory_info()
                snap["memory_info"] = {
                    k: getattr(mi, k)
                    for k in dir(mi)
                    if not k.startswith("_") and isinstance(getattr(mi, k), (int, float))
                }
            except Exception:
                pass
            # open files
            try:
                snap["open_files"] = [f.path for f in p.open_files()]
            except Exception:
                pass
            # children (recursive)
            try:
                snap["children"] = [{"pid": c.pid, "cmd": c.cmdline()} for c in p.children(recursive=True)]
            except Exception:
                pass
        except Exception as e:
            snap["psutil_error"] = str(e)
    else:
        # /proc fallbacks (Linux)
        base = f"/proc/{pid}"
        try:
            with open(os.path.join(base, "cmdline"), "rb") as f:
                snap["cmdline"] = (
                    f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
                )
        except Exception:
            pass
        for name, key in (("status", "proc_status"), ("limits", "limits")):
            try:
                with open(os.path.join(base, name), "r") as f:
                    snap[key] = f.read()
            except Exception:
                pass
        # children via pgrep
        try:
            out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
            if out.returncode == 0:
                snap["children"] = [
                    {"pid": int(x)} for x in out.stdout.strip().splitlines() if x.strip().isdigit()
                ]
        except Exception:
            pass

    # Kernel tail (best-effort; may require privileges)
    try:
        d = subprocess.run(
            ["dmesg", "--ctime", "--kernel", "--color=never"],
            capture_output=True, text=True, timeout=2
        )
        if d.returncode == 0:
            lines = d.stdout.splitlines()[-200:]
            tail = "\n".join(
                ln for ln in lines
                if any(k in ln.lower() for k in ("segfault", "oom", "killed process"))
            )
            if tail:
                snap["kernel_tail"] = tail
    except Exception:
        pass

    return snap


def print_snapshot(snap, *, stream=sys.stderr, prefix="[timeout] "):
    """Pretty-print the snapshot JSON to the chosen stream."""
    try:
        print(f"{prefix}Process snapshot:", file=stream)
        print(json.dumps(snap, indent=2), file=stream)
    except Exception:
        pass


def _kill_children_psutil(p):
    for c in p.children(recursive=True):
        try:
            c.terminate()
        except Exception:
            pass
    try:
        psutil.wait_procs(p.children(recursive=True), timeout=2)
    except Exception:
        pass
    for c in p.children(recursive=True):
        try:
            c.kill()
        except Exception:
            pass


def _kill_children_fallback(pid):
    # best-effort using pgrep + signals
    try:
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, timeout=1)
        if out.returncode == 0:
            for x in out.stdout.strip().splitlines():
                if not x.strip().isdigit():
                    continue
                cpid = int(x)
                try:
                    os.kill(cpid, signal.SIGTERM)
                except Exception:
                    pass
            # small grace
            try:
                subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, timeout=1)
            except Exception:
                pass
            # escalate
            out2 = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, timeout=1)
            if out2.returncode == 0:
                for x in out2.stdout.strip().splitlines():
                    if not x.strip().isdigit():
                        continue
                    cpid = int(x)
                    try:
                        os.kill(cpid, signal.SIGKILL)
                    except Exception:
                        pass
    except Exception:
        pass


def kill_process_tree(proc):
    """
    Always try to terminate the process AND its children.
    Graceful first, then force. Uses psutil if available; otherwise falls back
    to pgrep + signals.
    """
    pid = getattr(proc, "pid", None)
    if pid is None:
        return

    if psutil:
        try:
            p = psutil.Process(pid)
        except Exception:
            p = None
        if p is not None:
            try:
                _kill_children_psutil(p)
            except Exception:
                pass
            # parent graceful -> force
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            return

    # Fallback path (no psutil or failed)
    try:
        _kill_children_fallback(pid)
    except Exception:
        pass
    # Parent graceful -> force
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def snapshot_process_and_kill(proc, *, stream=sys.stderr):
    """
    Convenience: take a snapshot (printed to stream) and then kill the process tree.
    """
    snap = snapshot_process(proc)
    print_snapshot(snap, stream=stream)
    kill_process_tree(proc)
