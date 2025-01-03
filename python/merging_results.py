#!/usr/bin/env python3

import os
import pandas as pd
import sqlite3
import argparse
import matplotlib.pyplot as plt

RESULTS_DIR = "data"


def get_all_results(subdir: str) -> list:
    directory = os.path.join(RESULTS_DIR, subdir)
    results = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            results.append(pd.read_csv(os.path.join(directory, file)))
    return results


def merge_results(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dataframes, axis=0, ignore_index=True)


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    # Drop all rows where "alpha" is not in [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    df = df[df["alpha"].isin([0, 0.1, 0.2, 0.3, 0.4, 0.5])]

    # Drop all rows where "group" is not "Overall", "Male" or "Female"
    df = df[df["group"].isin(["Overall", "Male", "Female"])]

    return df


def save_to_sqlite(df: pd.DataFrame, db_name: str, table_name: str):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()


def percentual_change(df: pd.DataFrame, opt: str):
    # Find all rows where pd[opt] is opt
    opt_rows = df[(df["opt"] == opt) & (df["group"] == "Overall")]
    # Find all rows where pd[opt] is baseline
    baseline_rows = df[(df["opt"] == "baseline") & (df["group"] == "Overall")]

    # Create a new dataframe, computing the percentual change of the "data"
    # column for each "apha" and "metric" between the rows where df[opt] is opt
    # and the rows where df[opt] is baseline
    percentual_change = pd.DataFrame()
    for alpha in opt_rows["alpha"].unique():
        for metric in ["ndcg", "LIR", "SEP", "ETD"]:
            opt_data = opt_rows[
                (opt_rows["alpha"] == alpha) & (opt_rows["metric"] == metric)
            ]["data"].values[0]
            baseline_data = baseline_rows[
                (baseline_rows["alpha"] == alpha) & (baseline_rows["metric"] == metric)
            ]["data"].values[0]
            pct_change = 100 * (opt_data - baseline_data) / baseline_data
            percentual_change = percentual_change.append(
                {
                    "alpha": alpha,
                    "metric": metric,
                    "percentage_increase": pct_change,
                },
                ignore_index=True,
            )
            # Also add pct_change to the original dataframe
            df.loc[
                (df["opt"] == opt)
                & (df["alpha"] == alpha)
                & (df["metric"] == metric)
                & (df["group"] == "Overall"),
                "percentage_increase",
            ] = pct_change
    return percentual_change, df


def plot_heatmap(df: pd.DataFrame):
    # Transform the dataframe into a table, where the columns are given by
    # alpha, the rows are given by metric.
    df = df.pivot(index="metric", columns="alpha", values="data")

    # Plot heatmap
    plt.imshow(df, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Data")
    plt.xticks(range(len(df.columns)), df.columns, rotation=45)
    plt.yticks(range(len(df.index)), df.index)
    plt.xlabel("Alpha")
    plt.ylabel("Metric")
    plt.title("Heatmap of Metrics by Alpha")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge CSV results and save to SQLite database."
    )
    parser.add_argument(
        "-subdir",
        type=str,
        help="Subdirectory within the results directory containing the CSV files",
    )
    args = parser.parse_args()

    dataframes = get_all_results(args.subdir)
    df = clean_results(merge_results(dataframes))

    # Save merged results to a CSV file
    df.to_csv(f"{args.subdir.replace('/', '_')}.csv", index=False)

    # Print unique opt
    print(df["opt"].unique())
    
    pc_lir, df = percentual_change(df, "LIRopt")
    pc_sep, df = percentual_change(df, "SEPopt")
    pc_etd, df = percentual_change(df, "ETDopt")
    pc_esl, df = percentual_change(df, "ETD_SEP_LIR_opt")
    pc_el, df = percentual_change(df, "ETD_LIR_opt")
    pc_es, df = percentual_change(df, "ETD_SEP_opt")
    pc_sl, df = percentual_change(df, "SEP_LIR_opt")

    # plot_heatmap(df)
    print(df.head(20))

    save_to_sqlite(df, "results.db", args.subdir.replace("/", "_"))
