"""Time draw() + PDF save against structure size.

    python benchmarks/bench.py
"""
import io
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.build import mx2

import inkatoms as ia


def run(n_side):
    atoms = mx2("MoS2", size=(n_side, n_side, 1), vacuum=3.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    t0 = time.perf_counter()
    art = ia.draw(ax, atoms, view="z", rotate="-40x")
    t1 = time.perf_counter()
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    t2 = time.perf_counter()
    plt.close(fig)
    pieces = len(art.collection.get_paths()) if art.collection is not None else len(art.patches)
    return len(atoms), pieces, t1 - t0, t2 - t1, buf.tell() / 1e6


if __name__ == "__main__":
    print(f"{'atoms':>7} {'pieces':>8} {'draw s':>7} {'save s':>7} {'PDF MB':>7}")
    for n_side in (6, 18, 40, 58, 82):
        n, p, td, ts, mb = run(n_side)
        print(f"{n:7d} {p:8d} {td:7.2f} {ts:7.2f} {mb:7.2f}")
