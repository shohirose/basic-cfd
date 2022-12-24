import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


def get_timestep(filename: str) -> int:
    p = re.compile("[0-9]+$")
    result = p.search(filename)
    return int(result.group())


def plot(dir: Path):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.linspace(0, 2, 21)
    for file in dir.iterdir():
        if not file.is_file():
            continue
        q = np.loadtxt(file)
        n = get_timestep(file.stem)
        ax.plot(x, q, "o-", label=f"n={n}")

    ax.set_xlabel("x")
    ax.set_ylabel("q")
    ax.set_title(dir.stem)
    ax.legend()
    fig.savefig(Path(f"{dir.stem}.png"))


if __name__ == "__main__":
    dirs = ["FTCS", "Lax", "Lax-Wendroff", "Upwind1", "Upwind2"]
    for dir in dirs:
        plot(Path(f"build/{dir}"))
