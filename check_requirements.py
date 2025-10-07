#!/usr/bin/env python3
"""
Utility script to check if requirements.txt matches the current virtual environment
Usage: python check_requirements.py
"""

import subprocess
import sys
from pathlib import Path

def get_installed_packages():
    """Get currently installed packages and versions"""
    result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=freeze'],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error getting installed packages: {result.stderr}")
        return {}

    packages = {}
    for line in result.stdout.strip().split('\n'):
        if '==' in line:
            name, version = line.split('==', 1)
            packages[name.lower()] = version
    return packages

def get_requirements_packages():
    """Get packages from requirements.txt"""
    requirements_file = Path(__file__).parent / 'requirements.txt'
    if not requirements_file.exists():
        print("requirements.txt not found!")
        return {}

    packages = {}
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '>=' in line:
                    name = line.split('>=')[0]
                    version = ">=requirement"
                elif '==' in line:
                    name, version = line.split('==', 1)
                else:
                    continue
                packages[name.lower()] = version
    return packages

def main():
    print("Checking requirements.txt against installed packages...")
    print("=" * 60)

    installed = get_installed_packages()
    required = get_requirements_packages()

    # Check for missing packages
    missing = []
    version_mismatches = []

    for req_pkg, req_version in required.items():
        if req_pkg not in installed:
            missing.append(f"{req_pkg}=={req_version}")
        elif req_version.startswith('==') and req_version != f"=={installed[req_pkg]}":
            version_mismatches.append(f"{req_pkg}: required={req_version}, installed=={installed[req_pkg]}")

    # Check for extra packages (core AI/ML packages only)
    important_packages = ['torch', 'torchvision', 'torchreid', 'ultralytics', 'opencv-python',
                         'numpy', 'flask', 'sqlalchemy', 'pillow', 'gdown', 'tensorboard']

    extra = []
    for pkg in important_packages:
        if pkg in installed and pkg not in required:
            extra.append(f"{pkg}=={installed[pkg]}")

    # Report results
    if not missing and not version_mismatches and not extra:
        print("[OK] Requirements.txt is up to date!")
    else:
        if missing:
            print("[ERROR] Missing packages in environment:")
            for pkg in missing:
                print(f"   {pkg}")
            print()

        if version_mismatches:
            print("[WARNING] Version mismatches:")
            for mismatch in version_mismatches:
                print(f"   {mismatch}")
            print()

        if extra:
            print("[INFO] Important packages not in requirements.txt:")
            for pkg in extra:
                print(f"   {pkg}")
            print()

    print("=" * 60)
    print("Core functionality test...")

    try:
        import flask, cv2, numpy, torch, torchvision, ultralytics, sqlalchemy, torchreid
        print("[OK] All core dependencies working!")
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")

if __name__ == "__main__":
    main()