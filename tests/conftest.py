"""pytest 共享配置：把仓库根目录加入 sys.path。

脚本通过 `sys.path.insert(0, 仓库根)` 导入 `config` / `models` / `data`，
本 conftest 让 pytest 与脚本行为一致（且不依赖仓库文件夹名）。
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
