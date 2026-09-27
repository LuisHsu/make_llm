import sys
from pathlib import Path

# 資料夾名稱以數字開頭，不能當 package import；讓測試能直接 `import engine` / `import nn`
sys.path.insert(0, str(Path(__file__).parent))
