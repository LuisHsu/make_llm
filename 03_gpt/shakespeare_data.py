"""階段 3 的資料：data/tinyshakespeare.txt（Karpathy char-rnn repo 的 tiny shakespeare）。

下載：uv run python 03_gpt/shakespeare_data.py
"""

import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
TEXT_PATH = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"


def download(path: Path = TEXT_PATH) -> Path:
    """下載 tiny shakespeare 並核對 sha256；檔案已存在且內容正確時不重新下載。"""
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == SHA256:
        return path
    with urllib.request.urlopen(URL) as resp:
        raw = resp.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"tinyshakespeare.txt 的 sha256 不符：{digest}（預期 {SHA256}）")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def load_text(path: Path = TEXT_PATH) -> str:
    """整份文字（1115394 個字元，65 種字元）。"""
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，先執行：uv run python 03_gpt/shakespeare_data.py")
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    p = download()
    text = load_text(p)
    print(f"{p}：{len(text)} 個字元，{len(set(text))} 種字元")
