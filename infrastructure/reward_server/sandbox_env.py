"""Build a minimal, explicit environment for the subprocesses that execute
untrusted, LLM-generated kernel/reference source (compile.py, verify.py,
profile.py, baseline.py).

Security background: these subprocesses run arbitrary code
supplied by an LLM backend, which this design treats as untrusted.
Previously each handler called `os.environ.copy()`
and handed the *entire* parent environment to that untrusted code — meaning
any secret or credential-bearing variable present in the reward server's own
process (now or in the future) would be readable by the generated kernel
simply via `os.environ`. This module replaces that with an explicit
allowlist: only variables the child actually needs to run (locate the Python
interpreter and its native/shared libraries, resolve tempdirs, get sane
locale defaults) are copied from the parent; nothing else crosses the
boundary. Callers still add their own `NKI_*` payload variables on top.
"""

from __future__ import annotations

import os

# Variables the child interpreter and the Neuron toolchain (numpy / torch /
# neuronxcc, invoked via `sys.executable`) need to start up and locate their
# native libraries. Deliberately does NOT include the full parent
# environment — see module docstring.
_ALLOWED_EXACT = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
    }
)

# Prefixes for toolchain-specific variables that may be set at the OS/AMI
# level (e.g. by the Neuron DLAMI) and that the compiler/runtime needs, but
# whose exact names aren't enumerable in advance. None of these are secrets
# — they configure the Neuron compiler/runtime, not authentication.
_ALLOWED_PREFIXES = ("NEURON_", "AWS_NEURON_")


def build_subprocess_env(extra: dict[str, str]) -> dict[str, str]:
    """Build the environment for a child process running untrusted source.

    Starts from an explicit allowlist of the parent's environment (never a
    full copy) and overlays the caller-supplied `extra` variables, which are
    always included regardless of name (they're the handler's own
    `NKI_*` payload-passing variables, not parent secrets).

    Args:
        extra (dict[str, str]): Variables the caller needs the child to see
            in addition to the allowlisted parent variables, e.g. the
            `NKI_KERNEL_FILE`/`NKI_INPUT_SPECS` variables each handler uses
            to hand the child its inputs.

    Returns:
        dict[str, str]: The environment to pass as `subprocess.run(..., env=...)`.
    """
    env: dict[str, str] = {
        key: value
        for key, value in os.environ.items()
        if key in _ALLOWED_EXACT or key.startswith(_ALLOWED_PREFIXES)
    }
    env.update(extra)
    return env
