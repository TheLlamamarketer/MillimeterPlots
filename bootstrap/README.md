# Bootstrap helper

## bootstrap — creating an environment and installing dependencies

This directory contains helpers to create a Python virtual environment in `bootstrap/.venv` and install the third-party packages used by the `Functions/` code in this repository.

What is here

- `requirements.txt` — list of packages used by the project.
- `install_requirements.ps1` — PowerShell helper (Windows-focused) to create `.venv` and install packages.
- `install_requirements.py` — cross-platform Python script that does the same using the `venv` module.

Why we use a venv

- Isolation: avoids polluting or depending on the system Python and prevents package-version conflicts.
- Reproducibility: a venv plus `requirements.txt` lets others recreate the environment on their machine.

Important portability note

- A virtual environment is not a portable, moveable bundle. It contains absolute paths and binaries built for the system/ABI where it was created. You should not copy `.venv` between machines/OSes expecting it to work. Instead, use the `requirements.txt` (or an offline wheelhouse — see below) to recreate the environment on the target machine.

Quick start — PowerShell (recommended on Windows)

1. Allow the local script to run for this PowerShell session (once):

    ```powershell
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    ```

2. Run the bootstrap script (creates `bootstrap/.venv` and installs packages):

    ```powershell
    .\bootstrap\install_requirements.ps1
    ```

If you prefer the Python script (cross-platform):

```powershell
python bootstrap/install_requirements.py
```

Activation instructions (use the one that matches your venv layout)

- Windows PowerShell (Windows CPython / typical Windows venv):

```powershell
. .\bootstrap\.venv\Scripts\Activate.ps1
# then you should see the venv in your prompt and can run python/pip
```

- Command Prompt (cmd.exe):

```cmd
.\bootstrap\.venv\Scripts\activate.bat
```

- POSIX-style shells (Git Bash, msys, WSL, macOS, Linux):

```bash
source bootstrap/.venv/bin/activate
```

Detecting MSYS / POSIX-style venv on Windows

- Some Python installations (for example MSYS2/MINGW Python) create venvs with a `bin/` layout even on Windows. That creates files like `bootstrap/.venv/bin/python` and `bootstrap/.venv/bin/Activate.ps1` instead of `Scripts\`.
- If `install_requirements.ps1` or `install_requirements.py` tries to run `...\Scripts\python.exe` and fails, inspect `bootstrap/.venv` and use the `bin/` activation command shown above.

Useful diagnostic commands

```powershell
# which python is on PATH and what executable will be used
python --version
python -c "import sys; print(sys.executable)"

# list the venv layout
Get-ChildItem -Path .\bootstrap\.venv -Force
Get-ChildItem -Path .\bootstrap\.venv\Scripts -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path .\bootstrap\.venv\bin -Force -ErrorAction SilentlyContinue
```

Troubleshooting pip missing

- If `python -m pip` fails inside the venv with "No module named pip", try bootstrapping pip:

```powershell
# use ensurepip (preferred if present)
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

- If `ensurepip` is not available in your Python distribution, you can use the `get-pip.py` script from the official source (requires network):

```powershell
# run in PowerShell from repo root
curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py
python get-pip.py
```

Offline / portable installs — wheelhouse

If you need to install the same packages on machines without Internet access or you want to prepare packages for a specific OS/ABI, create a wheelhouse (downloaded wheels) and ship that folder.

Create a wheelhouse (on a machine with internet and the desired Python/OS):

```powershell
# download wheels/sources into bootstrap/packages
python -m pip download -r .\bootstrap\requirements.txt -d .\bootstrap\packages
```

Install from the wheelhouse on a target machine (offline):

```powershell
python -m venv .venv
# PowerShell activate (or use the appropriate activate command for your shell/layout)
. .\.venv\Scripts\Activate.ps1
# install from the local folder without contacting PyPI
python -m pip install --no-index --find-links .\bootstrap\packages -r .\bootstrap\requirements.txt
```

Notes about wheelhouses

- Wheels are platform-specific. If you need to support multiple platforms (Windows/Linux/macOS) you must build/download wheels for each platform and keep separate wheelhouses (for example `packages-win`, `packages-linux`).
- Pure-Python packages work across platforms; compiled extensions do not.

Recreating a clean venv (safe workflow)

```powershell
# remove old venv (if present)
Remove-Item -Recurse -Force .\bootstrap\.venv
# create a fresh one and install
python -m venv .\bootstrap\.venv
. .\bootstrap\.venv\Scripts\Activate.ps1   # or source bootstrap/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r .\bootstrap\requirements.txt
```

Docker (alternative for maximum reproducibility)

- If you want absolute reproducibility across machines, create a Docker image that contains the Python runtime and all dependencies. This requires Docker but avoids differences between host machines.

Example minimal Dockerfile (optional)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY bootstrap/requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt
COPY . .
CMD ["python", "-c", "print('container ready')"]
```

FAQ

- Q: Can I copy `.venv` to another machine?

  - A: Generally no. venvs contain interpreter paths and platform-specific binaries. Use `requirements.txt` or a wheelhouse instead.

- Q: Why did `install_requirements.ps1` fail to dot-source `Activate.ps1`?

  - A: Because your venv uses a `bin/` layout (MSYS/mingw/WSL style). Use `source bootstrap/.venv/bin/activate` or recreate the venv with a Windows CPython to get `Scripts\`.
  