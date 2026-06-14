import pandas as pd

df = pd.read_json(LOG_FILE)

baseline = df.iloc[0]["accuracy"]

df["improvement"] = (

    df["accuracy"]

    - baseline
)

df = df.sort_values(

    by="accuracy",

    ascending=False
)
df.to_csv(

    "experiment_report.csv",

    index=False
)

print(df)