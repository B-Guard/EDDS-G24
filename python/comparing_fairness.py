import os

import pandas as pd
import sqlite3
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns


def read_sqlite_subset(
    db_name: str,
    table_name: str,
    alpha: float = 0.1,
) -> pd.DataFrame:
    conn = sqlite3.connect(db_name)
    query = """
        SELECT 
            m.metric as metric, 
            m.opt as opt,  
            m.data - f.data AS data_difference
        FROM "{table}" AS m
        JOIN "{table}" AS f
        ON m.metric = f.metric 
        AND m.metric != "hr" 
        AND m.metric != "recall" 
        AND m.metric != "precision"
        AND m.opt = f.opt 
        AND m.alpha = f.alpha
        WHERE m.alpha = ?
        AND m."group" = 'Male' 
        AND f."group" = 'Female'
    """.format(table=table_name)
    df = pd.read_sql_query(query, conn, params=(alpha,))
    conn.close()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, required=True)
    parser.add_argument("--table_name", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    df = read_sqlite_subset(args.db_name, args.table_name, args.alpha)

    df = df.pivot(index="opt", columns="metric", values="data_difference")

    # Map the opt names to more readable names
    row_names_map = {
        "SEPopt": "P-PGPR",
        "LIRopt": "R-PGPR",
        "ETDopt": "D-PGPR",
        "baseline": "PGPR",
        "ETD_LIR_opt": "DR-PGPR",
        "ETD_SEP_opt": "DP-PGPR",
        "SEP_LIR_opt": "PR-PGPR",
        "ETD_SEP_LIR_opt": "DPR-PGPR",
    }
    df = df.rename(index=row_names_map)

    # Force order
    df = df.reindex(
        [
            "PGPR",
            "R-PGPR",
            "P-PGPR",
            "D-PGPR",
            "DR-PGPR",
            "PR-PGPR",
            "DP-PGPR",
            "DPR-PGPR",
        ]
    )

    # Rename columns
    df = df.rename(
        columns={
            "ndcg": "ΔNDCG",
            "LIR": "ΔLIR",
            "SEP": "ΔSEP",
            "ETD": "ΔETD",
        }
    )

    # Force order
    df = df[["ΔNDCG", "ΔLIR", "ΔSEP", "ΔETD"]]
    print(df)

    # Plot as simple table
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.round(3).values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.2)
    plt.title(f"{args.table_name}, α={args.alpha}")
    plt.tight_layout()
    plt.show()
