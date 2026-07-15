import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

os.makedirs("results/plots", exist_ok=True)

for file in glob.glob("results/csv/*.csv"):

    df = pd.read_csv(file)

    plt.figure(figsize=(8,5))

    plt.plot(
        df["step"],
        df["value"],
        linewidth=2
    )

    plt.title(os.path.basename(file))
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid()

    output = (
        "results/plots/"
        + os.path.basename(file).replace(".csv",".png")
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output)
