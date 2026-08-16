#!/usr/bin/env python3
"""
Generic gated/normal HuggingFace dataset downloader via hf-mirror.

Why not snapshot_download: hf-mirror rejects HEAD requests, so we list files
with huggingface_hub (GET works) and GET each file with requests + Bearer token.

Usage:
  python download_hf_dataset.py --repo hunglc007/ThyroidXL \
      --dest data/thyroid/thyroidxl --subdirs train/images train/labels test/images test/labels

Token is read from ~/.huggingface/token (created by the user, never echoed).
"""

import argparse
import concurrent.futures
import pathlib
import sys

import requests

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def load_token(token_file):
    p = pathlib.Path(token_file)
    if not p.exists():
        sys.exit(f"token file not found: {token_file}")
    return p.read_text(encoding="utf-8-sig").strip()


def list_repo_files(repo, token):
    from huggingface_hub import list_repo_files
    return list_repo_files(repo_id=repo, repo_type="dataset", token=token)


def download_one(item, base, dest_root, headers, retries):
    rel, = (item,)
    dest = dest_root / rel
    if dest.exists() and dest.stat().st_size > 0:
        return ("skip", rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = base + rel
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=180, headers=headers)
            if r.status_code in (401, 403):
                return ("auth", f"{rel}: HTTP {r.status_code} - {r.text[:80]}")
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            return ("ok", rel)
        except Exception as e:  # noqa: BLE001 - 下载重试兜底
            last_err = e
    return ("fail", f"{rel}: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--subdirs", nargs="+", default=None,
                    help="only download files under these prefixes")
    ap.add_argument("--token-file", default=str(pathlib.Path.home() / ".huggingface" / "token"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--endpoint", default="official",
                    choices=["official", "mirror"],
                    help="official=huggingface.co (works for gated datasets); mirror=hf-mirror.com")
    args = ap.parse_args()

    token = load_token(args.token_file)
    headers = {"Authorization": f"Bearer {token}", "User-Agent": UA}
    host = "huggingface.co" if args.endpoint == "official" else "hf-mirror.com"
    base = f"https://{host}/datasets/{args.repo}/resolve/main/"
    dest_root = pathlib.Path(args.dest).resolve()

    print(f"listing {args.repo} ...")
    files = list_repo_files(repo=args.repo, token=token)
    if args.subdirs:
        files = [f for f in files
                 if any(f == s or f.startswith(s + "/") for s in args.subdirs)]
    print(f"files to download: {len(files)}")
    if not files:
        sys.exit("empty file list - check subdirs or access approval")

    tasks = [(f,) for f in files]
    ok = skip = 0
    auth_fail = []
    fails = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, *t, base, dest_root, headers, args.retries): t[0]
                for t in tasks}
        for done, fut in enumerate(concurrent.futures.as_completed(futs), start=1):
            status, rel = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            elif status == "auth":
                auth_fail.append(rel)
            else:
                fails.append(rel)
            if done % 500 == 0 or done == len(tasks):
                print(f"  progress {done}/{len(tasks)} (ok={ok}, skip={skip}, "
                      f"auth={len(auth_fail)}, fail={len(fails)})")

    total = sum(p.stat().st_size for p in dest_root.rglob("*") if p.is_file())
    print(f"\nDONE: ok={ok}, skipped={skip}, failed={len(fails)}")
    print(f"total size: {total / 1e6:.1f} MB -> {dest_root}")
    if auth_fail:
        print(f"AUTH/BLOCKED ({len(auth_fail)}): dataset access still pending approval?")
        for r in auth_fail[:3]:
            print(f"  {r}")
    for f in fails[:5]:
        print(f"  FAIL: {f}")


if __name__ == "__main__":
    main()
