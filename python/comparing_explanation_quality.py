import os
import pandas as pd
import sqlite3
import argparse
import matplotlib.pyplot as plt


def read_sqlite_subset(
    db_name: str,
    table_name: str,
    alpha: float = 0.1,
) -> pd.DataFrame:
    conn = sqlite3.connect(db_name)
    query = """
        SELECT 
            t.metric as metric,
            t.opt as opt,
            t.data as data
        FROM "{table}" AS t
        WHERE t.alpha = ?
        AND t."group" = 'Overall'
        AND t.metric != "hr"
        AND t.metric != "recall"
        AND t.metric != "precision"
        AND t.metric != "ndcg"
    """.format(table=table_name)
    df = pd.read_sql_query(query, conn, params=(alpha,))
    conn.close()
    return df


def plot_table(db_name: str, table_name: str, alpha: float):
    df = read_sqlite_subset(db_name, table_name, alpha)

    df = df.pivot(index="opt", columns="metric", values="data")

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
            "DP-PGPR",
            "PR-PGPR",
            "DPR-PGPR",
        ]
    )

    # Add a new column EQ as sum of the other columns
    df["EQ"] = df.sum(axis=1)

    # Force order of columns
    df = df[["EQ", "LIR", "SEP", "ETD"]]
    print(df)

    # Plot as table
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
    plt.title(f"Explanation quality {table_name}, α={alpha}")
    plt.tight_layout()
    output_dir = os.path.join("data", "img", "EQ")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"{table_name}_a{str(alpha)[-1]}.png"))
    # plt.show()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, required=True)
    parser.add_argument("--table_name", type=str)
    parser.add_argument("--alpha", type=float)
    args = parser.parse_args()

    if args.table_name is not None:
        if args.alpha is not None:
            plot_table(args.db_name, args.table_name, args.alpha)
        else:
            for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
                plot_table(args.db_name, args.table_name, alpha)
    else:
        for tab_name in [
            "lastfm_10-12-1",
            "lastfm_25-50-1",
            "ml1m_10-12-1",
            "ml1m_25-50-1",
        ]:
            for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
                plot_table(args.db_name, tab_name, alpha)
