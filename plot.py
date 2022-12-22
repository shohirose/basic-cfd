import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot(dir: Path):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.linspace(-1, 1, 21)
    for file in dir.iterdir():
        if not file.is_file():
            continue
        q = np.loadtxt(file)
        _, n = file.stem.split("_")
        ax.plot(x, q, "o-", label=f"n={n}")

    ax.set_xlabel("x")
    ax.set_ylabel("q")
    ax.set_title(dir.stem)
    ax.legend()
    fig.savefig(Path(f"{dir.stem}.png"))


if __name__ == "__main__":
    dirs = ["FTCS", "Lax", "Lax-Wendroff", "Upwind"]
    for dir in dirs:
        plot(Path(f"build/{dir}"))
