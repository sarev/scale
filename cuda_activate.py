#!/usr/bin/env python3
"""
Session-local CUDA/cuDNN/TensorRT environment initialiser for Windows and Linux.

- Discovers CUDA, cuDNN, TensorRT in common install paths.
- On Windows, finds the correct MSVC bin (cl.exe) folder by scanning VS trees.
- On Linux, checks for gcc/clang (host compiler) and common cuDNN/TensorRT locations.
- Scrubs stale CUDA/cuDNN/TensorRT fragments from PATH unless --no-clean is set.
- Emits shell-specific exports so you can apply to the current shell.

If run without --shell, prints a short "How to use" guide with examples.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# ---------------- generic helpers ----------------


def _existing(paths: Iterable[Path]) -> List[Path]:
    return [p for p in paths if p and p.exists()]


def _strip_path_entries(path_value: str, needles: Tuple[str, ...], sep: str | None = None) -> str:
    sep = sep or os.pathsep
    parts = path_value.split(sep) if path_value else []
    keep = []
    for d in parts:
        norm = d.lower()
        if any(n.lower() in norm for n in needles):
            continue
        keep.append(d)
    return sep.join(keep)


def _split_mixed_path(raw: str) -> List[str]:
    """
    Split PATH safely when it may contain both ';' and ':' while preserving Windows
    drive prefixes like 'D:\\...'. Returns a list of raw segments (no trimming/normalisation).
    """
    if not raw:
        return []
    parts: List[str] = []
    buf: List[str] = []
    n = len(raw)
    for i, ch in enumerate(raw):
        if ch in (';', ':'):
            if ch == ':' and i >= 1:
                prev = raw[i - 1]
                nxt = raw[i + 1] if i + 1 < n else ''
                # Detect drive letter pattern "C:\" or "C:/"
                if prev.isalpha() and nxt in ('\\', '/'):
                    buf.append(ch)
                    continue
            # split here
            seg = ''.join(buf).strip()
            if seg:
                parts.append(seg)
            buf = []
        else:
            buf.append(ch)
    last = ''.join(buf).strip()
    if last:
        parts.append(last)
    return parts


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _emit(shell: str, name: str, value: str) -> str:
    if shell == "bash":
        safe = value.replace('"', r'\"')
        return f'export {name}="{safe}"'
    if shell == "powershell":
        safe = value.replace('`', '``').replace('"', '`"')
        return f'$env:{name} = "{safe}"'
    if shell == "cmd":
        return f'set {name}={value}'
    raise ValueError(f"unsupported shell: {shell}")


def _print_howto() -> None:
    me = Path(sys.argv[0]).name or "cuda_activate.py"
    print(
        f"""How to use
==========
Git Bash / Linux bash:
  eval "$(python {me} --shell bash)"

PowerShell:
  python {me} --shell powershell | Invoke-Expression

Windows cmd.exe:
  for /f "usebackq delims=" %i in (`python {me} --shell cmd`) do %i

Optional flags:
  --cuda     "<path-to-CUDA-home>"       # e.g. Windows: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4
                                         #      Linux:  /usr/local/cuda-12.4  (or let /usr/local/cuda symlink decide)
  --cudnn    "<path-to-cuDNN-home>"      # e.g. Windows: C:\\Program Files\\NVIDIA\\CUDNN\\v9.1
  --tensorrt "<path-to-TensorRT-home>"   # e.g. Windows: C:\\Program Files\\NVIDIA\\TensorRT\\v10.13.3.9
  --vsbin    "<MSVC bin folder>"         # Windows only; otherwise auto-discovered via cl.exe
  --torch-arch 8.6                       # sets TORCH_CUDA_ARCH_LIST
  --no-clean                             # do not scrub CUDA/cuDNN/TensorRT from PATH
""".strip()
    )


def _note(msg: str) -> None:
    print(msg, file=sys.stderr)

# ---------------- platform detection ----------------


def _is_windows() -> bool:
    return os.name == "nt"


def _is_linux() -> bool:
    return sys.platform.startswith("linux")

# ---------------- Windows-specific helpers ----------------


def _windows_program_files() -> Tuple[Path, Path]:
    pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    pfx86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    return pf, pfx86


def _find_latest(dir_: Path, prefix: str = "v") -> Optional[Path]:
    if not dir_.exists():
        return None
    candidates: List[Tuple[Tuple[int, ...], Path]] = []
    for child in dir_.iterdir():
        if child.is_dir():
            name = child.name
            if name.startswith(prefix):
                parts = tuple(int(x) for x in re.findall(r"\d+", name))
                candidates.append((parts, child))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _is_64bit_os() -> bool:
    arch = (os.environ.get("PROCESSOR_ARCHITECTURE", "") + " " + os.environ.get("PROCESSOR_ARCHITEW6432", "")).lower()
    return "64" in arch


def _find_msvc_cl_bin(pfx86: Path) -> Optional[Path]:
    """
    Search 'Program Files (x86)/Microsoft Visual Studio/**/VC/Tools/MSVC/*/bin/*/x64/cl.exe'
    Prefer Hostx64\\x64 on 64-bit Windows, else Hostx86\\x64 (32-bit host).
    """
    root = pfx86 / "Microsoft Visual Studio"
    if not root.exists():
        return None

    preferred_host = "Hostx64" if _is_64bit_os() else "Hostx86"
    candidates: List[Tuple[Tuple[int, ...], Path, bool]] = []

    for cl in root.rglob("VC/Tools/MSVC/*/bin/*/x64/cl.exe"):
        bin_dir = cl.parent
        ver_folder = bin_dir.parents[4].name  # .../MSVC/<ver>/bin/Host*/x64
        ver_parts = tuple(int(x) for x in re.findall(r"\d+", ver_folder)) or (0,)
        is_pref = (bin_dir.parts[-2] == preferred_host)
        candidates.append((ver_parts, bin_dir, is_pref))

    if not candidates:
        return None

    # sort: preferred host first, then highest version
    candidates.sort(key=lambda t: (not t[2], t[0]), reverse=True)
    return candidates[0][1]


def _which_cygpath() -> Optional[str]:
    for exe in ("cygpath", "cygpath.exe"):
        p = shutil.which(exe)
        if p:
            return p
    return None


def _to_bash_path_windows(p: Path) -> str:
    cp = _which_cygpath()
    if cp:
        try:
            import subprocess
            txt = subprocess.check_output([cp, "-u", str(p)], text=True).strip()
            if txt:
                return txt
        except Exception:
            pass
    # fallback: /c/...
    s = str(p).replace("\\", "/")
    if re.match(r"^[A-Za-z]:", s):
        drive = s[0].lower()
        s = f"/{drive}{s[2:]}"
    return s


def _normalise_existing_path_for_shell(shell: str, raw_path: str) -> str:
    """
    Windows + bash: convert any 'C:\\...' to '/c/...', unify with ':'.
    Other shells: leave separator as-is (we already use os.pathsep).
    Always dedupe while preserving order and drop empties.
    """
    parts = _split_mixed_path(raw_path)
    if not parts:
        return ""
    if _is_windows() and shell == "bash":
        norm: List[str] = []
        for seg in parts:
            s = seg.strip().strip('"')
            # Windows absolute path? Convert to bash form.
            if re.match(r"^[A-Za-z]:[\\/]", s):
                norm.append(_to_bash_path_windows(Path(s)))
            else:
                # Already /c/... or a plain unix-style path
                norm.append(s.replace("\\", "/"))
        norm = _dedupe_preserve_order([s for s in norm if s])
        return ":".join(norm)
    # Non-bash shells: keep original order; just dedupe exact strings
    parts = _dedupe_preserve_order([p for p in parts if p])
    # Rejoin with native separator (on Windows this is ';', on Linux ':')
    return os.pathsep.join(parts)


# ---------------- Linux-specific helpers ----------------


def _find_cuda_linux(explicit: Optional[Path]) -> Optional[Path]:
    if explicit:
        return explicit if explicit.exists() else None
    # Prefer /usr/local/cuda symlink if present
    cuda = Path("/usr/local/cuda")
    if cuda.exists():
        return cuda.resolve()
    # Else pick highest /usr/local/cuda-*
    base = Path("/usr/local")
    candidates: List[Tuple[Tuple[int, ...], Path]] = []
    for child in base.glob("cuda-*"):
        if child.is_dir():
            parts = tuple(int(x) for x in re.findall(r"\d+", child.name))
            candidates.append((parts, child))
    if candidates:
        candidates.sort()
        return candidates[-1][1]
    # Some distros use /opt/cuda-*
    base = Path("/opt")
    candidates = []
    for child in base.glob("cuda-*"):
        if child.is_dir():
            parts = tuple(int(x) for x in re.findall(r"\d+", child.name))
            candidates.append((parts, child))
    if candidates:
        candidates.sort()
        return candidates[-1][1]
    return None


def _find_cudnn_linux() -> Tuple[Optional[Path], List[Path]]:
    """
    Returns (cudnn_home, cudnn_lib_dirs). We consider home as the include root where cudnn*.h lives.
    """
    include_candidates = [Path("/usr/include"), Path("/usr/local/cuda/include")]
    for inc in include_candidates:
        if any(inc.glob("cudnn*.h")):
            # lib dirs commonly in:
            libs = _existing([
                Path("/usr/lib/x86_64-linux-gnu"),
                Path("/usr/local/cuda/lib64"),
                Path("/usr/local/lib"),
                Path("/usr/lib"),
            ])
            # Filter those that actually have libcudnn*
            libs = [d for d in libs if list(d.glob("libcudnn*.so*"))]
            return inc, libs
    # try pure library-first discovery
    libs = [d for d in [
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/lib"),
        Path("/usr/lib"),
    ] if list(d.glob("libcudnn*.so*"))]
    return None, libs


def _find_tensorrt_linux() -> Tuple[Optional[Path], List[Path]]:
    """
    Returns (trt_home, trt_lib_dirs). home is where NvInfer headers might be; lib dirs contain libnvinfer.
    """
    libs = [d for d in [
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/lib"),
        Path("/usr/lib"),
        Path("/usr/local/tensorrt/lib"),
        Path("/usr/local/TensorRT/lib"),
    ] if list(d.glob("libnvinfer*.so*"))]
    home_candidates = [Path("/usr/include"), Path("/usr/include/x86_64-linux-gnu"),
                       Path("/usr/local/include"), Path("/usr/local/TensorRT/include")]
    home = next((d for d in home_candidates if list(d.glob("NvInfer*.h"))), None)
    return home, libs


def _which_host_compiler_linux() -> Optional[str]:
    # Prefer gcc, fall back to clang
    for c in ("gcc", "cc", "clang"):
        if shutil.which(c):
            return c
    return None

# ---------------- data model ----------------


@dataclass
class EnvPlan:
    shell: str
    cuda_home: Optional[Path]
    cudnn_home: Optional[Path]
    trt_home: Optional[Path]
    vs_bin: Optional[Path]         # Windows only
    host_compiler: Optional[str]   # Linux only (name)
    torch_arch: Optional[str]
    bins: List[Path]
    libs: List[Path]

# ---------------- planners ----------------


def plan_windows(shell: str,
                 cuda: Optional[Path], cudnn: Optional[Path],
                 trt: Optional[Path], vsbin: Optional[Path],
                 torch_arch: Optional[str], no_clean: bool) -> EnvPlan:

    pf, pfx86 = _windows_program_files()

    # CUDA
    if not cuda:
        cuda_root = pf / "NVIDIA GPU Computing Toolkit" / "CUDA"
        cuda = _find_latest(cuda_root) or cuda_root
    # cuDNN
    if not cudnn:
        cudnn_root = pf / "NVIDIA" / "CUDNN"
        cudnn = _find_latest(cudnn_root)
    # TensorRT
    if not trt:
        trt_root = pf / "NVIDIA" / "TensorRT"
        trt = _find_latest(trt_root)
    # MSVC bin (required)
    if not vsbin:
        vsbin = _find_msvc_cl_bin(pfx86)

    # Build PATH segments
    bins: List[Path] = []
    libs: List[Path] = []

    if cuda and cuda.exists():
        bins += _existing([cuda / "bin"])
        libs += _existing([cuda / "lib", cuda / "lib" / "x64", cuda / "libnvvp", cuda / "extras" / "CUPTI" / "lib64"])

    if cudnn and cudnn.exists():
        bins += _existing([cudnn / "bin", cudnn / "bin" / (cuda.name.lstrip("v") if cuda else "")])
        libs += _existing([cudnn / "lib", cudnn / "lib" / (cuda.name.lstrip("v") if cuda else "") / "x64"])

    if trt and trt.exists():
        bins += _existing([trt / "bin"])
        libs += _existing([trt / "lib"])

    if vsbin and vsbin.exists():
        bins.append(vsbin)

    return EnvPlan(shell, cuda, cudnn, trt, vsbin, None, torch_arch, bins, libs)


def plan_linux(shell: str,
               cuda: Optional[Path], cudnn: Optional[Path],
               trt: Optional[Path], torch_arch: Optional[str], no_clean: bool) -> EnvPlan:

    cuda_home = _find_cuda_linux(cuda)
    cudnn_home, cudnn_libs = _find_cudnn_linux()
    trt_home, trt_libs = _find_tensorrt_linux()

    # honour CLI overrides
    if cudnn and cudnn.exists():
        cudnn_home = cudnn
    if trt and trt.exists():
        trt_home = trt

    bins: List[Path] = []
    libs: List[Path] = []

    if cuda_home and cuda_home.exists():
        bins += _existing([cuda_home / "bin"])
        libs += _existing([cuda_home / "lib64", cuda_home / "extras" / "CUPTI" / "lib64"])
    libs += cudnn_libs
    libs += trt_libs

    host = _which_host_compiler_linux()

    return EnvPlan(shell, cuda_home, cudnn_home, trt_home, None, host, torch_arch, bins, libs)

# ---------------- emission ----------------


def _to_shell_path(shell: str, p: Path) -> str:
    if _is_windows() and shell == "bash":
        return _to_bash_path_windows(p)
    return str(p)


def emit_env(plan: EnvPlan, no_clean: bool) -> int:
    # Validate and log to stderr; emit exports to stdout
    if _is_windows():
        if not plan.cuda_home or not plan.cuda_home.exists():
            _note(f"WARNING: CUDA not found at: {plan.cuda_home}")
        if not plan.vs_bin:
            _note("ERROR: Could not locate MSVC 'cl.exe'. Install Visual Studio 2022 Build Tools (C++ workload).")
            return 2
        if not plan.vs_bin.exists():
            _note(f"ERROR: MSVC bin not found at: {plan.vs_bin}")
            return 2
    else:
        if not plan.cuda_home or not plan.cuda_home.exists():
            _note(f"WARNING: CUDA not found at: {plan.cuda_home or '<auto>'}")
        if not plan.host_compiler:
            _note("ERROR: No host compiler found (gcc/clang). Install build-essential or clang.")
            return 2

    # Build exports
    shell = plan.shell
    # IMPORTANT: In Git Bash (MSYS) we must emit ':'-separated PATHs, even on Windows.
    path_sep = ":" if shell == "bash" else os.pathsep

    # Scrub PATH if required
    PATH = _normalise_existing_path_for_shell(shell, os.environ.get("PATH", ""))
    if not no_clean:
        # Scrub against the normalised representation
        PATH = _strip_path_entries(PATH, ("CUDA", "CUDNN", "TensorRT"), sep=path_sep)

    out: List[str] = []

    # CUDA vars
    if plan.cuda_home and plan.cuda_home.exists():
        ch = _to_shell_path(shell, plan.cuda_home)
        out.append(_emit(shell, "CUDA_HOME", ch))
        out.append(_emit(shell, "CUDA_PATH", ch))
        out.append(_emit(shell, "CUDA_TOOLKIT_ROOT", ch))
        out.append(_emit(shell, "CUDA_TOOLKIT_ROOT_DIR", ch))
        if _is_windows():
            out.append(_emit(shell, "CUDA_BIN_PATH", path_sep.join([ch, _to_shell_path(shell, plan.cuda_home / "bin")])))
            out.append(_emit(shell, "CUDA_LIB_PATH", path_sep.join(filter(None, [
                ch,
                _to_shell_path(shell, plan.cuda_home / "lib"),
                _to_shell_path(shell, plan.cuda_home / "lib" / "x64"),
                _to_shell_path(shell, plan.cuda_home / "libnvvp"),
                _to_shell_path(shell, plan.cuda_home / "extras" / "CUPTI" / "lib64"),
            ]))))

    # cuDNN/TRT homes for convenience (may be None on Linux package installs)
    if plan.cudnn_home and plan.cudnn_home.exists():
        out.append(_emit(shell, "CUDNN_HOME", _to_shell_path(shell, plan.cudnn_home)))
    if plan.trt_home and plan.trt_home.exists():
        out.append(_emit(shell, "TRT_HOME", _to_shell_path(shell, plan.trt_home)))

    if plan.torch_arch:
        out.append(_emit(shell, "TORCH_CUDA_ARCH_LIST", plan.torch_arch))

    # LD_LIBRARY_PATH (useful for Linux and MSYS bash)
    lib_paths = [p for p in plan.libs if p.exists()]
    if lib_paths:
        libs = path_sep.join(_to_shell_path(shell, p) for p in lib_paths)
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        # Always treat LD_LIBRARY_PATH as ':'-separated in bash; on Windows PowerShell/cmd it’s harmless anyway.
        if shell == "bash":
            ld = libs if not existing_ld else ":".join([libs, existing_ld])
        else:
            ld = libs if not existing_ld else path_sep.join([libs, existing_ld])

        out.append(_emit(shell, "LD_LIBRARY_PATH", ld))

    # PATH last: include bins + libs to make DLL/.so discovery easy in dev shells
    bin_paths = [p for p in plan.bins if p.exists()]
    all_bins = path_sep.join(_to_shell_path(shell, p) for p in (bin_paths + lib_paths))
    PATH_new = all_bins if not PATH else path_sep.join([PATH, all_bins])
    out.append(_emit(shell, "PATH", PATH_new))

    # Summary
    _note("")
    _note(f"Platform   = {'Windows' if _is_windows() else 'Linux'}")
    _note(f"CUDA_HOME  = {plan.cuda_home}")
    _note(f"CUDNN_HOME = {plan.cudnn_home}")
    _note(f"TRT_HOME   = {plan.trt_home}")
    if _is_windows():
        _note(f"VS bin     = {plan.vs_bin}")
    else:
        _note(f"Host CC    = {plan.host_compiler}")
    if plan.torch_arch:
        _note(f"TORCH_CUDA_ARCH_LIST = {plan.torch_arch}")
    _note("")

    print("\n".join(out))
    return 0

# ---------------- main ----------------


def main() -> int:
    p = argparse.ArgumentParser(description="Emit shell commands to activate CUDA/cuDNN/TensorRT paths for this session.")
    p.add_argument("--shell", choices=["bash", "powershell", "cmd"], required=False, help="Target shell for output.")
    p.add_argument("--cuda", type=Path, help="Explicit CUDA home.")
    p.add_argument("--cudnn", type=Path, help="Explicit cuDNN home.")
    p.add_argument("--tensorrt", type=Path, help="Explicit TensorRT home.")
    p.add_argument("--vsbin", type=Path, help="(Windows) MSVC bin path (Host*/x64).")
    p.add_argument("--torch-arch", dest="torch_arch", help="Set TORCH_CUDA_ARCH_LIST (e.g. 8.6).")
    p.add_argument("--no-clean", action="store_true", help="Do not scrub existing CUDA/cuDNN/TensorRT fragments from PATH.")
    args = p.parse_args()

    if not args.shell:
        _print_howto()
        return 0

    if _is_windows():
        plan = plan_windows(args.shell, args.cuda, args.cudnn, args.tensorrt, args.vsbin, args.torch_arch, args.no_clean)
    elif _is_linux():
        plan = plan_linux(args.shell, args.cuda, args.cudnn, args.tensorrt, args.torch_arch, args.no_clean)
    else:
        _note("ERROR: Unsupported platform. Only Windows and Linux are supported.")
        return 2

    return emit_env(plan, args.no_clean)


if __name__ == "__main__":
    raise SystemExit(main())
