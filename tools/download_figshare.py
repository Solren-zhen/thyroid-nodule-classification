"""figshare 文件下载（支持断点续传 + 进度日志）。

用法：
  python tools/download_figshare.py --url https://ndownloader.figshare.com/files/36506100 \
      --dest data/thyroid/thywise/thywise_us_images.zip --log logs/thywise_dl.log
"""
import argparse
import sys
import time
import urllib.request
from pathlib import Path

CHUNK = 256 * 1024


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else None

    # 已存在部分大小（断点续传）
    got = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": "python", "Range": f"bytes={got}-"}
    total = 0
    t0 = time.time()
    last = 0
    mode = "ab" if got else "wb"

    def log(msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    log(f"开始下载（已存在 {human(got)}，断点续传）→ {dest}")
    try:
        req = urllib.request.Request(args.url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, mode) as out:
            length = r.headers.get("Content-Length")
            if length:
                total = got + int(length)
            while True:
                b = r.read(CHUNK)
                if not b:
                    break
                out.write(b)
                got += len(b)
                now = time.time()
                if now - last >= 15:
                    speed = (got) / max(now - t0, 1e-6)
                    eta = (total - got) / speed / 60 if total > got else 0
                    log(f"{human(got)} / {human(total)}  {human(speed)}/s  ETA {eta:.0f}min")
                    last = now
        log(f"完成：{dest} ({human(got)})")
    except Exception as e:  # noqa: BLE001 - 下载中断时保存进度并退出
        log(f"中断/出错：{type(e).__name__} {e}（已保存 {human(got)}，可重跑续传）")
        sys.exit(1)


if __name__ == "__main__":
    main()
