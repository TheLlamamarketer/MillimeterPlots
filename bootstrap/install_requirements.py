"""
Cross-platform installer script that creates a virtual environment in `bootstrap/.venv`
and installs packages from `bootstrap/requirements.txt`.

Run:
  python bootstrap/install_requirements.py

It will use the current Python executable to create the venv.
"""
import os
import sys
import subprocess


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(root, ".venv")

    python_exe = sys.executable

    if not os.path.isdir(venv_dir):
        print(f"Creating venv at {venv_dir}")
        run([python_exe, "-m", "venv", venv_dir])

    # Detect venv layout rather than relying solely on os.name. Some Python
    # distributions (msys/mingw) create POSIX-style venvs (bin/) even on
    # Windows hosts, which caused FileNotFoundError for Scripts\python.exe.
    scripts_python = os.path.join(venv_dir, "Scripts", "python.exe")
    scripts_pip = os.path.join(venv_dir, "Scripts", "pip.exe")
    bin_python = os.path.join(venv_dir, "bin", "python")
    bin_pip = os.path.join(venv_dir, "bin", "pip")

    if os.path.isfile(scripts_python):
        venv_python = scripts_python
        venv_pip = scripts_pip
    elif os.path.isfile(bin_python):
        venv_python = bin_python
        venv_pip = bin_pip
    else:
        # Fall back to assuming Windows layout if nothing is found; we'll
        # emit a clear error if the expected executables are missing later.
        venv_python = scripts_python
        venv_pip = scripts_pip

    print("Upgrading pip in venv...")
    if not os.path.isfile(venv_python):
        raise SystemExit(f"Virtualenv python not found at {venv_python!r}. The venv may have a different layout; check {venv_dir!r}.")
    run([venv_python, "-m", "pip", "install", "--upgrade", "pip"])

    req = os.path.join(root, "requirements.txt")
    print(f"Installing from {req}...")
    run([venv_pip, "install", "-r", req])

    print("\nDone. Activate the venv:\n")
    # Print activation instructions matching the actual venv layout we detected
    if os.path.isfile(scripts_python):
        print(r"PowerShell: .\bootstrap\.venv\Scripts\Activate.ps1")
        print(r"cmd: .\bootstrap\.venv\Scripts\activate.bat")
    elif os.path.isfile(bin_python):
        print("source bootstrap/.venv/bin/activate")
    else:
        # Unknown layout — show both suggestions
        print(r"PowerShell: .\bootstrap\.venv\Scripts\Activate.ps1  (if present)")
        print(r"bash: source bootstrap/.venv/bin/activate  (if present)")


if __name__ == "__main__":
    main()
