import sys

import matplotlib.pyplot as plt
import pandas as pd


path = sys.argv[1] if len(sys.argv) > 1 else "parameter_sweep.csv"
results = pd.read_csv(path)
if results.empty:
    raise RuntimeError("The sweep CSV contains no measurements.")

results["Config"] = (
    "BM=" + results.BM.astype(str)
    + ", BN=" + results.BN.astype(str)
    + ", BK=" + results.BK.astype(str)
    + ", TM=" + results.TM.astype(str)
    + ", TN=" + results.TN.astype(str)
)
results = results.sort_values("GFLOPS")

plt.figure(figsize=(12, 7))
plt.barh(results["Config"], results["GFLOPS"], color="darkorange")
plt.xlabel("GFLOP/s")
plt.ylabel("Tile configuration")
plt.title(
    f"Parameter sensitivity for {results.M.iloc[0]} x {results.N.iloc[0]} x {results.K.iloc[0]} SGEMM"
)
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("parameter_sweep_gflops.png", dpi=160)
plt.show()

print(results[["Config", "Threads", "ElapsedMs", "GFLOPS"]].to_string(index=False))