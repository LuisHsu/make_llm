"""階段 0 環境檢查：PyTorch 與可用的加速器（CUDA／MPS），以及矩陣乘法速度比較。

執行：uv run python 00_env/check_env.py
任何一項檢查失敗時以非零狀態碼結束。沒有加速器不算失敗（只用 CPU 也能完成前幾個階段）。
"""

import platform
import sys
import time

import numpy as np

N = 4096  # 矩陣邊長
REPEATS = 5
SEED = 0

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def bench(fn, sync) -> float:
    """回傳 REPEATS 次中最快的一次（秒）；前面先跑一次 warmup。"""
    fn()
    sync()
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        sync()
        best = min(best, time.perf_counter() - t0)
    return best


print(f"Python   {sys.version.split()[0]} ({sys.executable})")
print(f"Platform {platform.platform()}")
check("Python 3.12", sys.version_info[:2] == (3, 12), sys.version.split()[0])
check("在專案 .venv 內執行", ".venv" in sys.executable, sys.executable)

# 同一組輸入給每個後端，最後比對結果，確認每個後端真的有算出東西
rng = np.random.default_rng(SEED)
a_np = rng.standard_normal((N, N), dtype=np.float32)
b_np = rng.standard_normal((N, N), dtype=np.float32)

results: dict[str, float] = {}
outputs: dict[str, np.ndarray] = {}

import torch  # noqa: E402

print(f"\ntorch    {torch.__version__}")

a_cpu, b_cpu = torch.from_numpy(a_np), torch.from_numpy(b_np)
results["torch cpu"] = bench(lambda: a_cpu @ b_cpu, lambda: None)
outputs["torch cpu"] = (a_cpu @ b_cpu).numpy()

# 依 CLAUDE.md §3 的順序找加速器：cuda → mps
accelerators = []
if torch.cuda.is_available():
    accelerators.append(("cuda", torch.cuda.synchronize))
    print(f"CUDA     {torch.version.cuda}，{torch.cuda.get_device_name(0)}")
if torch.backends.mps.is_available():
    accelerators.append(("mps", torch.mps.synchronize))
    print("MPS      可用")
if not accelerators:
    print("加速器   沒有找到 CUDA 或 MPS，之後的階段會在 CPU 上執行")

for name, sync in accelerators:
    device = torch.device(name)
    a_dev, b_dev = a_cpu.to(device), b_cpu.to(device)
    check(f"tensor 位於 {name}", a_dev.device.type == name, str(a_dev.device))
    results[f"torch {name}"] = bench(lambda: a_dev @ b_dev, sync)
    outputs[f"torch {name}"] = (a_dev @ b_dev).cpu().numpy()

    # autograd 在加速器上能跑，且梯度正確：d/dx sum(x*x) = 2x
    x = torch.randn(8, device=device, requires_grad=True)
    (x * x).sum().backward()
    check(f"{name} autograd", torch.allclose(x.grad, 2 * x.detach()))

    # bf16 可用
    y = torch.ones(4, 4, device=device, dtype=torch.bfloat16)
    check(f"{name} bfloat16", (y @ y).dtype == torch.bfloat16 and (y @ y)[0, 0].item() == 4.0)

# ---- 結果一致性 ----
print()
ref = outputs["torch cpu"]
for name, out in outputs.items():
    if name == "torch cpu":
        continue
    rel_err = float(np.abs(out - ref).max() / np.abs(ref).max())
    # GPU 的 float32 matmul 可能用 TF32 等較低精度，容許誤差放寬到 1e-2
    check(f"{name} 結果與 torch cpu 一致", rel_err < 1e-2, f"max rel err {rel_err:.2e}")

# ---- 速度 ----
print(f"\n{N}x{N} float32 矩陣乘法（{REPEATS} 次取最快）")
base = results["torch cpu"]
for name, sec in results.items():
    print(f"  {name:<10} {sec * 1000:8.1f} ms   ({base / sec:5.1f}x vs torch cpu)")

print()
if failures:
    print(f"失敗 {len(failures)} 項：{failures}")
    sys.exit(1)
print("全部通過")
