#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

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
    """
    Filter a list of paths to only include existing ones.

    Parameters:
    - `paths`: An iterable of `Path` objects to filter.

    Returns:
    - A list of `Path` objects that exist on the file system.
    """

    return [p for p in paths if p and p.exists()]


def _strip_path_entries(path_value: str, needles: Tuple[str, ...], sep: str | None = None) -> str:
    """
    Strip unwanted path entries from a given PATH value.

    This function removes any directory paths containing specified needles from the input PATH string.
    The needle strings are matched case-insensitively.

    Parameters:
    - `path_value`: The original PATH value as a string.
    - `needles`: A tuple of directory names to be removed from the PATH.
    - `sep`: The path separator character (default: platform-specific).

    Returns:
    - The modified PATH value with unwanted entries stripped.
    """

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
    Split a mixed PATH string into raw segments.

    This function safely splits a PATH string that may contain both ';' and ':' while preserving Windows drive prefixes like 'D:\\...'.
    It returns a list of raw segments without trimming or normalisation.

    Parameters:
    - `raw`: The PATH string to split, possibly containing both ';' and ':'.

    Returns:
    - A list of raw PATH segments.
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
    """
    Remove duplicate items from an iterable while preserving the original order.

    This function takes an iterable of strings, removes any duplicates, and returns a new list with the unique
    items in their original order.

    Parameters:
    - `items`: The input iterable of strings to deduplicate.

    Returns:
    - A new list containing the unique items from the input iterable, in their original order.
    """

    seen: set[str] = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _emit(shell: str, name: str, value: str) -> str:
    """
    Emit a shell-specific export statement to set an environment variable.

    This function generates a string that sets the value of an environment variable in the specified shell.
    It takes into account the quoting requirements for each shell type.

    Parameters:
    - `shell`: The type of shell to generate the export statement for (bash, powershell, or cmd).
    - `name`: The name of the environment variable to set.
    - `value`: The value to assign to the environment variable.

    Returns:
    - A string containing the shell-specific export statement to set the environment variable.

    Raises:
    - `ValueError`: If the specified shell is not supported.
    """

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
    """
    Print a how-to guide for using the script.

    This function prints a formatted guide on how to use the script, including examples for Git Bash, PowerShell,
    and Windows cmd.exe. It also documents optional flags that can be used to customize the script's behavior.

    Optional Flags:
    - `--cuda`: Specify the path to the CUDA home directory (e.g. Windows: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4, Linux: /usr/local/cuda-12.4).
    - `--cudnn`: Specify the path to the cuDNN home directory (e.g. Windows: C:\\Program Files\\NVIDIA\\CUDNN\\v9.1).
    - `--tensorrt`: Specify the path to the TensorRT home directory (e.g. Windows: C:\\Program Files\\NVIDIA\\TensorRT\\v10.13.3.9).
    - `--vsbin`: Specify the MSVC bin folder (Windows only; otherwise auto-discovered via cl.exe).
    - `--torch-arch`: Set the TORCH_CUDA_ARCH_LIST to a specific value (e.g. 8.6).
    - `--no-clean`: Prevent the script from scrubbing CUDA/cuDNN/TensorRT from the PATH environment variable.
    """

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
    """
    Print a message to the standard error stream.

    Parameters:
    - `msg`: The message string to be printed.
    """

    print(msg, file=sys.stderr)

# ---------------- platform detection ----------------


def _is_windows() -> bool:
    """
    Determine whether the system is running on Windows.

    Returns:
        `True` if the system is Windows, `False` otherwise.
    """

    return os.name == "nt"


def _is_linux() -> bool:
    """
    Determine if the current platform is Linux.

    Returns:
        `True` if the platform is Linux, `False` otherwise.
    """

    return sys.platform.startswith("linux")

# ---------------- Windows-specific helpers ----------------


def _windows_program_files() -> Tuple[Path, Path]:
    """
    Return the paths to the Windows Program Files and x86 Program Files directories.

    These directories are used to search for CUDA, cuDNN, and TensorRT installations.

    Returns:
        - `pf`: The path to the 64-bit Program Files directory.
        - `pfx86`: The path to the 32-bit Program Files (x86) directory.
    """

    pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    pfx86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    return pf, pfx86


def _find_latest(dir_: Path, prefix: str = "v") -> Optional[Path]:
    """
    Find the latest directory within a given directory tree.

    This function searches for directories whose names start with the specified prefix (default: "v"),
    followed by one or more numeric version parts. It returns the path to the directory with the highest
    version number, or `None` if no matching directories are found.

    Parameters:
    - `dir_`: The directory to search within.
    - `prefix`: The prefix for versioned directory names (default: "v").

    Returns:
    - The path to the latest matching directory, or `None` if none are found.
    """

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
    """
    Determine if the current operating system is 64-bit.

    This function checks the `PROCESSOR_ARCHITECTURE` and `PROCESSOR_ARCHITEW6432` environment variables to
    determine the OS architecture.

    Parameters:
    - None

    Returns:
    - `True` if the OS is 64-bit, `False` otherwise
    """

    arch = (os.environ.get("PROCESSOR_ARCHITECTURE", "") + " " + os.environ.get("PROCESSOR_ARCHITEW6432", "")).lower()
    return "64" in arch


def _find_msvc_cl_bin(pfx86: Path) -> Optional[Path]:
    """
    Search for the MSVC cl.exe compiler binary in the specified Visual Studio installation.

    This function searches for `cl.exe` in the 'Program Files (x86)/Microsoft Visual Studio/**/VC/Tools/MSVC/*/bin/*/x64/cl.exe'
    directory, preferring the 'Hostx64' directory on 64-bit Windows and 'Hostx86' on 32-bit hosts.

    Parameters:
    - `pfx86`: The path to the 'Program Files (x86)' directory.

    Returns:
    - The path to the MSVC cl.exe compiler binary if found, or `None` otherwise.
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
    """
    Find the path to the Cygwin `cygpath` executable.

    This function searches for the `cygpath` or `cygpath.exe` executable in the system's PATH and returns its full
    path if found, otherwise returns `None`.

    Returns:
        - The full path to the `cygpath` executable if found, or `None` if not.
    """

    for exe in ("cygpath", "cygpath.exe"):
        p = shutil.which(exe)
        if p:
            return p
    return None


def _to_bash_path_windows(p: Path) -> str:
    """
    Convert a Windows path to a bash-compatible path.

    This function attempts to use the `cygpath` utility to convert the given Windows path to a Unix-style path.
    If `cygpath` is not available, it falls back to manually replacing backslashes with forward slashes and
    adjusting the drive letter if necessary.

    Parameters:
    - `p`: The Windows path to be converted.

    Returns:
    - The bash-compatible path as a string, or an empty string if conversion failed.
    """

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
    Normalise an existing path for a specific shell.

    This function converts Windows-style paths to Unix-style paths for bash, and leaves other shells' separators as-is.
    It also deduplicates the path segments while preserving order and drops any empty strings.

    Parameters:
    - `shell`: The name of the shell (e.g. "bash", "cmd").
    - `raw_path`: The original path string.

    Returns:
    - A normalised path string suitable for the specified shell.
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
    """
    Find the CUDA installation location on Linux.

    This function searches for the CUDA installation directory in common locations, preferring explicit paths,
    then `/usr/local/cuda`, and finally `/usr/local` or `/opt` directories with names matching the pattern
    `cuda-<version>`, where `<version>` is a sequence of digits. The most recent version found is returned.

    Parameters:
    - `explicit`: An optional explicit path to the CUDA installation directory.

    Returns:
    - The path to the CUDA installation directory, or `None` if not found.
    """

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
    Find the cuDNN installation on a Linux system.

    Returns:
        - A tuple containing the cuDNN home directory (`cudnn_home`) and a list of cuDNN library
          directories (`cudnn_lib_dirs`).

    The cuDNN home directory is determined by finding the include root where the `cudnn*.h` header files
    reside. The cuDNN library directories are identified by searching for the presence of `libcudnn*.so*`
    files in common locations.

    Parameters:
        - None

    Returns:
        - A tuple of two values:
            - `cudnn_home`: The path to the cuDNN home directory, or `None` if not found.
            - `cudnn_lib_dirs`: A list of paths to cuDNN library directories, or an empty list if not found.
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
    Returns the TensorRT home directory and a list of directories containing the libnvinfer library.

    The TensorRT home directory is where NvInfer headers might be located. The returned list of directories
    contains the libnvinfer library.

    Parameters:
    - None

    Returns:
    - A tuple containing two values:
      - `trt_home`: The path to the TensorRT home directory, or `None` if not found.
      - `trt_lib_dirs`: A list of paths to directories containing the libnvinfer library.
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
    """
    Determine the host compiler on Linux.

    Prefer `gcc` and fall back to `clang`.

    Returns:
        - The name of the host compiler (either "gcc" or "clang") if found, otherwise `None`.
    """

    for c in ("gcc", "cc", "clang"):
        if shutil.which(c):
            return c
    return None

# ---------------- data model ----------------


@dataclass
class EnvPlan:
    """
    Represents a plan for configuring the environment.

    This class encapsulates the necessary information to configure the CUDA, cuDNN, and TensorRT environments
    for Windows and Linux systems. It provides a structured way to represent the environment configuration,
    including shell, CUDA home, cuDNN home, TensorRT home, MSVC bin path (Windows only), host compiler (Linux only),
    torch architecture, and paths to binaries and libraries.

    Parameters:
    - `shell`: The shell type (e.g., 'bash', 'zsh').
    - `cuda_home`: The path to the CUDA installation directory (optional).
    - `cudnn_home`: The path to the cuDNN installation directory (optional).
    - `trt_home`: The path to the TensorRT installation directory (optional).
    - `vs_bin`: The path to the MSVC bin directory (Windows only, optional).
    - `host_compiler`: The name of the host compiler (Linux only, optional).
    - `torch_arch`: The torch architecture (optional).
    - `bins`: A list of paths to binaries.
    - `libs`: A list of paths to libraries.

    Notes:
    This class is abstract and should not be instantiated directly. Instead, it should be used as a data model
    to represent the environment configuration.
    """

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

    """
    Construct an environment plan for Windows.

    This function populates the `EnvPlan` data model with settings for the specified shell and CUDA, cuDNN,
    and TensorRT installations.

    Parameters:
    - `shell`: The target shell (e.g., "cmd" or "bash").
    - `cuda`: The path to the CUDA installation directory (or `None` to discover it).
    - `cudnn`: The path to the cuDNN installation directory (or `None` to discover it).
    - `trt`: The path to the TensorRT installation directory (or `None` to discover it).
    - `vsbin`: The path to the MSVC bin directory (or `None` to discover it).
    - `torch_arch`: The Torch architecture (or `None` for default).
    - `no_clean`: Whether to preserve stale CUDA/cuDNN/TensorRT fragments in the PATH environment variable.

    Returns:
    - An `EnvPlan` instance representing the constructed environment configuration.
    """

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

    """
    Create a Linux environment plan.

    This function configures the environment for a Linux system by discovering CUDA, cuDNN, and TensorRT libraries,
    and determining the host compiler. The resulting `EnvPlan` object can be used to emit shell-specific exports.

    Parameters:
    - `shell`: The target shell (e.g., "bash", "zsh").
    - `cuda`: The path to a custom CUDA installation (optional).
    - `cudnn`: The path to a custom cuDNN installation (optional).
    - `trt`: The path to a custom TensorRT installation (optional).
    - `torch_arch`: The target architecture for PyTorch (optional).
    - `no_clean`: Whether to preserve stale CUDA/cuDNN/TensorRT fragments in the PATH environment variable.

    Returns:
    - An `EnvPlan` object representing the configured environment.
    """

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
    """
    Convert a Path object to a shell-specific path string.

    This function takes a `Path` object and a shell name, and returns the path as a string suitable for use
    in the specified shell.

    Parameters:
    - `shell`: The name of the shell (e.g. "bash", "cmd").
    - `p`: The `Path` object to convert.

    Returns:
    - A string representing the path in the specified shell.
    """

    if _is_windows() and shell == "bash":
        return _to_bash_path_windows(p)
    return str(p)


def emit_env(plan: EnvPlan, no_clean: bool) -> int:
    # Validate and log to stderr; emit exports to stdout
    """
    Emit shell-specific exports to activate the desired environment.

    This function validates and logs to stderr, then emits exports to stdout. It checks for the presence of CUDA,
    cuDNN, and TensorRT libraries in common install paths, and scrubs stale fragments from the PATH environment
    variable unless `no_clean` is set.

    Parameters:
    - `plan`: The `EnvPlan` object representing the environment configuration.
    - `no_clean`: A boolean indicating whether to clean up stale fragments from the PATH environment variable.

    Returns:
    - `0` if the operation was successful, or an error number.

    Notes:
    - This function raises an error if CUDA is not found on Windows or no host compiler is found on Linux.
    - It emits shell-specific exports to stdout, including CUDA_HOME, CUDNN_HOME, TRT_HOME, LD_LIBRARY_PATH, and PATH.
    """

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
    """
    Emit shell commands to activate CUDA/cuDNN/TensorRT paths for this session.

    This is the main entry point of the program. It parses command-line arguments and determines the target shell,
    CUDA, cuDNN, and TensorRT environments, as well as the MSVC bin path (on Windows) or host compiler (on Linux).
    The program then generates a plan to configure the environment and emits shell-specific commands to activate it.

    Parameters:
    - `args`: The parsed command-line arguments.

    Returns:
    - An exit code indicating success (0) or failure (2).

    Notes:
    - If no target shell is specified, a "How to use" guide is printed and the program exits with success.
    - If an unsupported platform is detected, an error message is printed and the program exits with failure.
    """

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
