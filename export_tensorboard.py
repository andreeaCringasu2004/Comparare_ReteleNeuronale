from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os
import glob


log_files = glob.glob(
    "lightning_logs/**/events.out.tfevents.*",
    recursive=True
)

os.makedirs("results/csv", exist_ok=True)

for log_file in log_files:

    print("Processing:", log_file)

    ea = EventAccumulator(log_file)
    ea.Reload()

    tags = ea.Tags()["scalars"]

    for tag in tags:
        events = ea.Scalars(tag)

        df = pd.DataFrame([
            {
                "step": e.step,
                "wall_time": e.wall_time,
                "value": e.value
            }
            for e in events
        ])

        name = tag.replace("/", "_")

        output = f"results/csv/{name}.csv"

        df.to_csv(output, index=False)

        print("Saved:", output)
