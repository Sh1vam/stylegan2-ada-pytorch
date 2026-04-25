# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import glob
import shutil
import platform
import hashlib
import importlib
from pathlib import Path

import torch
import torch.utils.cpp_extension

from torch.utils.file_baton import FileBaton

#----------------------------------------------------------------------------
# Global options.

verbosity = 'brief'  # 'none' | 'brief' | 'full'

#----------------------------------------------------------------------------
# Cross-platform compiler detection
# ---------------------------------------------------------------------------
#
# Original code only handled Windows (os.name == 'nt') — on Linux and macOS
# the compiler block was skipped entirely and missing tools produced cryptic
# errors deep inside torch.utils.cpp_extension.load().
#
# This rewrite handles all three platforms explicitly:
#
#   Windows  – look for cl.exe (MSVC); same as before
#   Linux    – look for gcc/g++ and nvcc; patch PATH if nvcc is in a
#              non-standard CUDA toolkit location (/usr/local/cuda/bin)
#   macOS    – look for clang; warn clearly that CUDA/nvcc is unavailable
#              (Apple dropped CUDA in 2019), so CUDA plugins will never build
#              and the callers must use the pure-Python reference fallbacks
#
# ---------------------------------------------------------------------------

# Common locations where nvcc lives when it is NOT already on $PATH
_NVCC_SEARCH_PATHS = [
    '/usr/local/cuda/bin',          # standard CUDA toolkit install
    '/usr/local/cuda-*/bin',        # versioned: cuda-11.x, cuda-12.x …
    '/usr/cuda/bin',                # some Debian/Ubuntu packages
    '/opt/cuda/bin',                # Arch Linux
]


def _find_compiler_bindir():
    """Return the directory that contains the C/C++ compiler, or None.

    Called only when the compiler is NOT already on PATH.  The return value
    is appended to PATH so subsequent subprocess calls can find the compiler.
    """
    system = platform.system()

    # ── Windows ──────────────────────────────────────────────────────────────
    if system == 'Windows':
        patterns = [
            'C:/Program Files (x86)/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
            'C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
            'C:/Program Files (x86)/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
            'C:/Program Files (x86)/Microsoft Visual Studio */vc/bin',
        ]
        for pattern in patterns:
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[-1]
        return None  # MSVC truly not found

    # ── Linux ─────────────────────────────────────────────────────────────────
    if system == 'Linux':
        # gcc/g++ — these must be on PATH; we cannot helpfully "find" them
        # in a non-standard location, so just return None and let the caller
        # raise the right error message.
        for cc in ('gcc', 'g++', 'cc'):
            if shutil.which(cc):
                return None   # already on PATH, nothing to add

        # If we reach here neither gcc nor g++ are on PATH.
        return None  # caller will raise a clear error

    # ── macOS ─────────────────────────────────────────────────────────────────
    if system == 'Darwin':
        # Apple ships clang as the system C++ compiler.
        # It is normally at /usr/bin/clang after `xcode-select --install`.
        for cc in ('clang++', 'clang', 'g++', 'gcc'):
            if shutil.which(cc):
                return None   # already on PATH

        # Xcode command-line tools missing
        return None  # caller will raise a clear error

    return None  # unknown platform


def _check_compilers():
    """Inspect the current environment and either fix it or raise a clear error.

    Returns a warning string (non-fatal) or None (all good).
    """
    system = platform.system()
    sep = ';' if system == 'Windows' else ':'

    # ── Windows ──────────────────────────────────────────────────────────────
    if system == 'Windows':
        import subprocess
        if subprocess.call('where cl.exe >nul 2>nul', shell=True) != 0:
            bindir = _find_compiler_bindir()
            if bindir is None:
                raise RuntimeError(
                    'Could not find MSVC (cl.exe) on this Windows machine.\n'
                    'Install "Desktop development with C++" via Visual Studio Installer,\n'
                    'or the standalone "Build Tools for Visual Studio".\n'
                    f'See _find_compiler_bindir() in "{__file__}".'
                )
            os.environ['PATH'] += sep + bindir
        return None  # nvcc is handled by CUDA toolkit installer on Windows

    # ── Linux ─────────────────────────────────────────────────────────────────
    if system == 'Linux':
        # Check for C++ compiler
        if not shutil.which('gcc') and not shutil.which('g++') and not shutil.which('c++'):
            raise RuntimeError(
                'No C++ compiler found on this Linux system.\n'
                'Install one with:  sudo apt-get install build-essential\n'
                'or:               sudo yum groupinstall "Development Tools"'
            )

        # Check for nvcc — may be in a well-known non-PATH location
        if not shutil.which('nvcc'):
            found_nvcc_dir = None
            for pattern in _NVCC_SEARCH_PATHS:
                for nvcc_dir in sorted(glob.glob(pattern)):
                    if os.path.isfile(os.path.join(nvcc_dir, 'nvcc')):
                        found_nvcc_dir = nvcc_dir
                        break
                if found_nvcc_dir:
                    break

            if found_nvcc_dir:
                # nvcc exists but is not on PATH — add it
                os.environ['PATH'] = found_nvcc_dir + sep + os.environ.get('PATH', '')
                return None  # fixed silently

            # nvcc truly missing: CUDA plugin will fall back to Python reference
            return (
                'nvcc not found — the CUDA toolkit is not installed or not on PATH.\n'
                'The custom CUDA ops (bias_act, upfirdn2d) will fall back to their\n'
                'slower pure-Python reference implementations automatically.'
            )

        return None  # both gcc and nvcc are on PATH

    # ── macOS ─────────────────────────────────────────────────────────────────
    if system == 'Darwin':
        # Check for Clang (or GCC via Homebrew)
        if not shutil.which('clang++') and not shutil.which('clang') \
                and not shutil.which('g++') and not shutil.which('gcc'):
            raise RuntimeError(
                'No C++ compiler found on this macOS system.\n'
                'Run:  xcode-select --install\n'
                'or install GCC via Homebrew:  brew install gcc'
            )

        # macOS has no CUDA/nvcc — Apple dropped support in 2019.
        # This is expected; the CUDA ops will use Python fallbacks.
        return (
            'macOS detected — CUDA (nvcc) is not available on Apple hardware.\n'
            'The custom CUDA ops (bias_act, upfirdn2d) will automatically fall\n'
            'back to their pure-Python reference implementations.  Training will\n'
            'work but will be significantly slower than on a Linux+CUDA machine.'
        )

    # ── Unknown platform ──────────────────────────────────────────────────────
    return f'Unknown platform "{system}" — compiler checks skipped.'


#----------------------------------------------------------------------------
# PyTorch 2.x-compatible build-directory resolver.
#
# torch.utils.cpp_extension._get_build_directory() was a private, undocumented
# helper that was REMOVED in PyTorch 2.0.  The public replacement is
# get_default_build_root() (available since PyTorch 1.12).
#
# Resolution order:
#   1. $TORCH_EXTENSIONS_DIR  (explicit env override — highest priority)
#   2. Old private API         (PyTorch < 2.0 backward compat)
#   3. New public API          (PyTorch >= 1.12)
#   4. /tmp fallback           (always writable — last resort)
#----------------------------------------------------------------------------

def _get_build_dir(module_name, verbose):
    """Return a per-module build directory compatible with PyTorch 1.x and 2.x."""

    # 1. Explicit env override
    ext_dir = os.environ.get('TORCH_EXTENSIONS_DIR', '')
    if ext_dir:
        d = os.path.join(ext_dir, module_name)
        os.makedirs(d, exist_ok=True)
        return d

    # 2. Old private API (PyTorch < 2.0)
    try:
        d = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose)  # pylint: disable=protected-access
        os.makedirs(d, exist_ok=True)
        return d
    except AttributeError:
        pass  # removed in PyTorch 2.0 → fall through

    # 3. Public API (PyTorch >= 1.12)
    try:
        root = torch.utils.cpp_extension.get_default_build_root()
        d = os.path.join(root, module_name)
        os.makedirs(d, exist_ok=True)
        return d
    except AttributeError:
        pass  # very old PyTorch → last resort

    # 4. /tmp fallback (always writable on Linux/macOS/Windows WSL)
    d = os.path.join('/tmp', 'torch_ext_fallback', module_name)
    os.makedirs(d, exist_ok=True)
    return d


#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA plugins.
#----------------------------------------------------------------------------

_cached_plugins = dict()


def get_plugin(module_name, sources, **build_kwargs):
    assert verbosity in ['none', 'brief', 'full']

    # Already cached?
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    # Print status.
    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)

    try:
        # ── Compiler / environment check (cross-platform) ──────────────────
        warning = _check_compilers()
        if warning and verbosity != 'none':
            print(f'\n  ⚠️  {warning}')

        verbose_build = (verbosity == 'full')

        # ── Incremental build using an MD5 digest of the source files ──────
        # Only active when TORCH_EXTENSIONS_DIR is set (signals that the user
        # wants the cached-build optimisation) AND all sources live in one dir.
        source_dirs_set = set(os.path.dirname(s) for s in sources)
        if len(source_dirs_set) == 1 and ('TORCH_EXTENSIONS_DIR' in os.environ):
            all_source_files = sorted(
                x for x in Path(list(source_dirs_set)[0]).iterdir() if x.is_file()
            )

            hash_md5 = hashlib.md5()
            for src in all_source_files:
                with open(src, 'rb') as f:
                    hash_md5.update(f.read())

            # FIX: use our compat helper instead of the removed private API
            build_dir = _get_build_dir(module_name, verbose_build)
            digest_build_dir = os.path.join(build_dir, hash_md5.hexdigest())

            if not os.path.isdir(digest_build_dir):
                os.makedirs(digest_build_dir, exist_ok=True)
                baton = FileBaton(os.path.join(digest_build_dir, 'lock'))
                if baton.try_acquire():
                    try:
                        for src in all_source_files:
                            shutil.copyfile(src, os.path.join(digest_build_dir, os.path.basename(src)))
                    finally:
                        baton.release()
                else:
                    baton.wait()

            digest_sources = [
                os.path.join(digest_build_dir, os.path.basename(x)) for x in sources
            ]
            torch.utils.cpp_extension.load(
                name=module_name,
                build_directory=build_dir,
                verbose=verbose_build,
                sources=digest_sources,
                **build_kwargs,
            )
        else:
            torch.utils.cpp_extension.load(
                name=module_name,
                verbose=verbose_build,
                sources=sources,
                **build_kwargs,
            )

        module = importlib.import_module(module_name)

    except:
        if verbosity == 'brief':
            print('Failed!')
        raise

    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')

    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------
