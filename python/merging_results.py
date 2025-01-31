#!/usr/bin/env python3

import os
import pandas as pd
import sqlite3
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from PIL import Image

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


def plot_heatmap(
    df: pd.DataFrame,
    lower: float,
    upper: float,
    title: str = "",
    show_cbar: bool = True,
    show: bool = False,
):
    filename = "_".join(reversed(title.split(" "))).lower()
    # Transform the dataframe into a table, where the columns are given by
    # alpha, the rows are given by opt.
    df = df.pivot(index="opt", columns="alpha", values="percentage_increase")

    # Create a custom colormap
    cmap = plt.get_cmap("PiYG")
    norm = TwoSlopeNorm(
        vmin=-max(upper, abs(lower)),
        vcenter=0,
        vmax=max(upper, abs(lower)),
    )

    row_names_map = {"SEPopt": "P-PGPR", "LIRopt": "R-PGPR", "ETDopt": "D-PGPR"}

    # Rename row labels according to the mapping
    df.rename(index=row_names_map, inplace=True)

    # Reorder rows to be R-P-D
    df = df.reindex(["R-PGPR", "P-PGPR", "D-PGPR"])

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df,
        cmap=cmap,
        norm=norm,
        annot=True,
        fmt=".1f",
        cbar=show_cbar,
        cbar_kws={
            "label": "Percentage Increase",
            "location": "bottom",
            "shrink": 0.95,
        }
        if show_cbar
        else None,
        square=True,
        annot_kws={"size": 24},
    )
    plt.xlabel("Alpha", fontsize=16) if show_cbar else plt.xlabel("")
    plt.ylabel("")  # Remove Y axis label
    plt.title(title, fontsize=24)
    plt.xticks(fontsize=16) if show_cbar else plt.xticks(ticks=[])
    plt.yticks(
        rotation=0, fontsize=16
    )  # Make row labels horizontal and increase font size
    plt.tight_layout()
    os.makedirs(os.path.join('data', 'img'), exist_ok=True)
    plt.savefig(
        f"{os.path.join('data', 'img', filename)}.png"
    ) if not show else plt.show()


def concatenate_images(image_files, output_path):
    """
    Stacks images vertically and saves the result
    """
    images: list[Image.Image] = [Image.open(img) for img in image_files]
    widths, heights = zip(*(img.size for img in images))

    total_height = sum(heights)
    max_width = max(widths)

    concatenated_image = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for img in images:
        concatenated_image.paste(img, (0, y_offset))
        y_offset += img.height

    concatenated_image.save(output_path)

def filter4plot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # Translate the SQL query to pandas
    filtered_df = df[
        (df["group"] == "Overall")
        & (df["opt"].isin(["LIRopt", "SEPopt", "ETDopt"]))
        & (df["metric"] == metric)
    ]
    return filtered_df[["alpha", "percentage_increase", "opt"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-subdir",
        type=str,
        help="Subdirectory within the results directory containing the CSV files",
    )
    args = parser.parse_args()

    dataframes = get_all_results(args.subdir)
    df = clean_results(merge_results(dataframes))

    # Save merged results to a CSV file

    # Print unique opt
    print(df["opt"].unique())

    pc_lir, df = percentual_change(df, "LIRopt")
    pc_sep, df = percentual_change(df, "SEPopt")
    pc_etd, df = percentual_change(df, "ETDopt")
    pc_esl, df = percentual_change(df, "ETD_SEP_LIR_opt")
    pc_el, df = percentual_change(df, "ETD_LIR_opt")
    pc_es, df = percentual_change(df, "ETD_SEP_opt")
    pc_sl, df = percentual_change(df, "SEP_LIR_opt")

    df.to_csv(f"{args.subdir.replace('/', '_')}.csv", index=False)

    print(df.head(20))
    print()
    print(filter4plot(df, "ndcg"))

    # save_to_sqlite(df, "results.db", args.subdir.replace("/", "_"))

    image_files = []
    metrics = ["ndcg", "LIR", "SEP", "ETD"]

    for metric in metrics:
        title = f"{metric.upper()} {args.subdir.split('/')[0].upper()}"
        plot_heatmap(
            filter4plot(df, metric),
            -21,  #
            224,  #
            title=title,
            show_cbar=True if metric == "ETD" else False,
            # show=True,
        )
        image_files.append(
            os.path.join("data", "img", "_".join(reversed(title.split(" "))).lower())
            + ".png"
        )

    concatenate_images(
        image_files,
        f"{os.path.join('data', 'img', args.subdir.replace('/', '_').lower())}.png",
    )
