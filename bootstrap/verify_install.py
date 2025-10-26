"""
Simple verification script: import each package listed in requirements.txt and report success/failure.

Usage:
  # after creating and activating venv (or system Python)
  python bootstrap/verify_install.py

The script maps some pip package names to their import names (e.g. opencv-python -> cv2).
"""
import importlib
import sys
import os


ROOT = os.path.dirname(os.path.abspath(__file__))
REQ = os.path.join(ROOT, "requirements.txt")

# common mappings from pip package name -> import name
COMMON_IMPORT_MAP = {
    "opencv-python": "cv2",
    "opencv_python": "cv2",
}


def parse_requirements(path):
    pkgs = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            # remove extras and version specifiers (keeps package base)
            for sep in ("==", ">=", "<=", ">", "<", "~="):
                if sep in ln:
                    ln = ln.split(sep, 1)[0]
                    break
            ln = ln.split("[", 1)[0]
            pkgs.append(ln.strip())
    return pkgs


def import_name_for(pkg_name: str) -> str:
    # explicit mapping
    if pkg_name in COMMON_IMPORT_MAP:
        return COMMON_IMPORT_MAP[pkg_name]
    # many packages import with same name
    return pkg_name.replace("-", "_")


def main():
    if not os.path.exists(REQ):
        print("requirements.txt not found at:", REQ)
        sys.exit(2)

    pkgs = parse_requirements(REQ)
    failures = []
    print("Verifying imports for packages listed in:", REQ)
    for pkg in pkgs:
        imp_name = import_name_for(pkg)
        try:
            importlib.import_module(imp_name)
            print(f"OK   : {pkg} (import as '{imp_name}')")
        except Exception as e:
            print(f"FAIL : {pkg} (import as '{imp_name}') -> {e.__class__.__name__}: {e}")
            failures.append((pkg, imp_name, str(e)))

    if failures:
        print("\nSome imports failed. Install the requirements into your venv and re-run this script.")
        print("You can create/activate the venv using the provided install scripts in this folder.")
        sys.exit(1)
    else:
        print("\nAll imports succeeded.")
        sys.exit(0)


if __name__ == "__main__":
    main()
