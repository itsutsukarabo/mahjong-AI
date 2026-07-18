"""
新PC展開後の個別ファイルSHA256照合スクリプト。
リポジトリルートで実行:
    python phase2/migration/verify_checksums.py
"""
import hashlib
import sys
from pathlib import Path

CHECKSUM_FILE = Path(__file__).parent / "checksums_20260718.txt"
REPO_ROOT = Path(__file__).parent.parent.parent


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not CHECKSUM_FILE.exists():
        print(f"ERROR: {CHECKSUM_FILE} not found")
        sys.exit(1)

    lines = CHECKSUM_FILE.read_text(encoding="utf-8").splitlines()

    in_file_section = False
    entries = []
    for line in lines:
        if line.strip() == "[FILE_SHA256]":
            in_file_section = True
            continue
        if line.startswith("[") and in_file_section:
            break
        if in_file_section and line.strip() and not line.startswith("#"):
            parts = line.split(None, 1)
            if len(parts) == 2:
                entries.append((parts[0], parts[1].strip()))

    ok = 0
    ng = 0
    missing = 0
    for expected_hash, rel_path in entries:
        full_path = REPO_ROOT / rel_path
        if not full_path.exists():
            print(f"MISSING  {rel_path}")
            missing += 1
            continue
        actual = sha256_file(full_path)
        if actual.lower() == expected_hash.lower():
            print(f"OK       {rel_path}")
            ok += 1
        else:
            print(f"MISMATCH {rel_path}")
            print(f"  expected: {expected_hash}")
            print(f"  actual:   {actual}")
            ng += 1

    print(f"\nResult: {ok} OK / {ng} MISMATCH / {missing} MISSING (total {len(entries)})")
    if ng > 0 or missing > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
