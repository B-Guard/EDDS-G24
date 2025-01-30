import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, norm
from matplotlib.colors import TwoSlopeNorm



def load_and_label_csv_avg(file_path: str, run_id: str, dataset_name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df['run_id'] = run_id
        df['dataset_name'] = dataset_name
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def process_dataset_avg(base_dir: str, dataset: str) -> list:
    dataset_dir = os.path.join(base_dir, dataset)
    all_dfs = []

    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} does not exist.")
        return []

    for run_id in os.listdir(dataset_dir):
        run_dir = os.path.join(dataset_dir, run_id)
        if os.path.isdir(run_dir):
            for file_name in os.listdir(run_dir):
                if file_name.endswith('avg.csv'):
                    file_path = os.path.join(run_dir, file_name)
                    df = load_and_label_csv_avg(file_path, run_id, dataset)
                    if df is not None:
                        all_dfs.append(df)

    return all_dfs

def load_all_avg_datasets(base_dir: str, datasets: list) -> pd.DataFrame:
    all_dfs = []

    for dataset in datasets:
        dataset_dfs = process_dataset_avg(base_dir, dataset)
        all_dfs.extend(dataset_dfs)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        print("No data was loaded.")
        return pd.DataFrame()

def clean_results_avg(df: pd.DataFrame) -> pd.DataFrame:
   
    
    valid_alpha_values = {0, 0.1, 0.2, 0.3, 0.4, 0.5}
    valid_groups = {"Overall", "Male", "Female"}
    valid_metrics = {'ndcg', 'LIR', 'SEP', 'ETD'}
    
   
    df = df[df["alpha"].isin(valid_alpha_values)]
    df = df[df["metric"].isin(valid_metrics)]

    
    df = df[~df["run_id"].eq("ootb")]  
    df["run_id"] = df["run_id"].str.extract(r"(Run \d+)")[0]
    
   
    df = df[df["group"].isin(valid_groups)]

    return df

def calculate_percentual_change(df: pd.DataFrame, optimizations: list) -> pd.DataFrame:
    results = []

    for dataset in df['dataset_name'].unique():
        for run_id in df['run_id'].unique():
            for alpha in df['alpha'].unique():
                for metric in df['metric'].unique():
                    baseline_value = df[(df['dataset_name'] == dataset) &
                                        (df['run_id'] == run_id) &
                                        (df['alpha'] == alpha) &
                                        (df['metric'] == metric) &
                                        (df['opt'] == 'baseline') &
                                        (df['group'] == 'Overall')]['data'].values

                    baseline_value = baseline_value[0]

                    for opt in optimizations:
                        opt_value = df[(df['dataset_name'] == dataset) &
                                       (df['run_id'] == run_id) &
                                       (df['alpha'] == alpha) &
                                       (df['metric'] == metric) &
                                       (df['opt'] == opt) &
                                       (df['group'] == 'Overall')]['data'].values

                        opt_value = opt_value[0]
                        pct_change = 100 * (opt_value - baseline_value) / baseline_value

                        results.append({
                            'dataset_name': dataset,
                            'run_id': run_id,
                            'alpha': alpha,
                            'metric': metric,
                            'opt': opt,
                            'percentage_increase': pct_change})

    return pd.DataFrame(results)

def plot_heatmap(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    lower: float,
    upper: float,
    title: str = "",
    run_id: str = None,
    agg_method: str = None,
    show_cbar: bool = True,
    ax=None):

    if run_id:
        plot_df = df[(df['dataset_name'] == dataset) & (df['run_id'] == run_id) & (df['metric'] == metric)]
    elif agg_method:
        plot_df = df[(df['dataset_name'] == dataset) & (df['metric'] == metric)]
        if agg_method == 'max':
            plot_df = plot_df.groupby(['alpha', 'opt'])['percentage_increase'].max().reset_index()
        elif agg_method == 'min':
            plot_df = plot_df.groupby(['alpha', 'opt'])['percentage_increase'].min().reset_index()
        elif agg_method == 'avg':
            plot_df = plot_df.groupby(['alpha', 'opt'])['percentage_increase'].mean().reset_index()
        elif agg_method == 'median':
            plot_df = plot_df.groupby(['alpha', 'opt'])['percentage_increase'].median().reset_index()
        else:
            raise ValueError("Invalid aggregation method. Choose from 'max', 'min', 'avg', 'median'.")
    else:
        raise ValueError("Either run_id or agg_method must be specified.")

    plot_df = plot_df[plot_df['alpha'] != 0.0]
    
    cmap = sns.color_palette("PiYG", as_cmap=True)
    norm = TwoSlopeNorm(vmin=lower, vcenter=0, vmax=upper)

    row_names_map = {"SEPopt": "P-PGPR", "LIRopt": "R-PGPR", "ETDopt": "D-PGPR"}
    plot_df['opt'] = plot_df['opt'].map(row_names_map)
    heatmap_data = plot_df.pivot_table(index='opt', columns='alpha', values='percentage_increase')
    heatmap_data = heatmap_data.reindex(["R-PGPR", "P-PGPR", "D-PGPR"])

    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        norm=norm,
        annot=True,
        fmt=".1f",
        cbar=show_cbar,
        cbar_kws={
            "label": "Percentage Increase",
            "location": "bottom",
            "shrink": 0.85,
            "orientation": "horizontal",
            "pad": 0.2,
        } if show_cbar else None,
        square=True,
        annot_kws={"size": 12},
        ax=ax)
    ax.set_xlabel("Alpha", fontsize=14) if show_cbar else ax.set_xlabel("")
    ax.set_ylabel("")  
    ax.set_title(title, fontsize=16, pad=10)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12, rotation=0)

def plot_all_heatmaps(df: pd.DataFrame, run_id: str = None, agg_method: str = None, show=True):

    datasets = df['dataset_name'].unique()
    metrics = df['metric'].unique()

    lower, upper = -50, 300  
    tick_interval = 50  

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), constrained_layout=True)
    
    cmap = sns.color_palette("PiYG", as_cmap=True)
    norm = plt.Normalize(vmin=lower, vmax=upper)  

    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            title = f"{metric.upper()} {dataset.upper()}"

            plot_heatmap(
                df,
                dataset=dataset,
                metric=metric,
                lower=lower,
                upper=upper,
                title=title,
                run_id=run_id,
                agg_method=agg_method,
                show_cbar=False,
                ax=ax)

            if i == 3:  
                alpha_values = sorted(val for val in df[df['dataset_name'] == dataset]['alpha'].unique() if val != 0.0)
                xtick_positions = [pos + 0.5 for pos in range(len(alpha_values))]
                ax.set_xticks(xtick_positions)
                ax.set_xticklabels(
                    [f"{alpha:.1f}" for alpha in alpha_values],
                    fontsize=12,
                    rotation=0,
                    ha="center",
                )
            else:
                ax.set_xticks([])

            ax.set_ylabel("")

            ax.set_title(title, fontsize=16, pad=15)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes,
        location="bottom",
        shrink=0.9,
        aspect=30,
        pad=0.1)
    
    cbar.set_label("Percentage Increase (%)", fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    ticks = np.arange(lower, upper + 1, tick_interval)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(tick)}" for tick in ticks])

    if show:
        plt.show()

def find_best_alpha(data, ndcg_threshold=-10, min_explanation_increase=5, return_most_used_alpha=True):
    
    results = []

    for dataset, dataset_group in data.groupby('dataset_name'):
        dataset_results = []

        ndcg_group = dataset_group.query("metric == 'ndcg'")
        
        for (opt, run_id), group in ndcg_group.groupby(['opt', 'run_id']):
            if opt == 'baseline':  
                continue

            group = group[group['alpha'] != 0.0]
            
            explanation_metrics = [metric for metric in ['LIR', 'SEP', 'ETD'] if metric in opt]

            explanation_data = dataset_group[
                (dataset_group['metric'].isin(explanation_metrics)) &
                (dataset_group['opt'] == opt) &
                (dataset_group['run_id'] == run_id) &
                (dataset_group['alpha'] != 0.0)]

            explanation_pivot = explanation_data.pivot_table(
                index=['alpha'],
                columns=['metric'],
                values='percentage_increase',
                aggfunc='mean').reset_index()

            filtered = group[group['percentage_increase'] >= ndcg_threshold]

            if not filtered.empty:
                merged = pd.merge(
                    filtered[['alpha', 'percentage_increase']],
                    explanation_pivot,
                    on='alpha',
                    how='left')

                merged = merged.sort_values(by='alpha').reset_index(drop=True)

                for metric in explanation_metrics:
                    merged[f'{metric}_increment'] = merged[metric].diff()

                merged['valid_alpha'] = merged[[f'{metric}_increment' for metric in explanation_metrics]].max(axis=1) >= min_explanation_increase

                merged = merged[merged['valid_alpha']]

                if not merged.empty:
                    best_row = merged.loc[merged['alpha'].idxmax()]

                    dataset_results.append({
                        'opt': opt,
                        'run_id': run_id,
                        'dataset_name': dataset,
                        'alpha': best_row['alpha'],
                        'ndcg_change': best_row['percentage_increase'],
                        'total_explanation_quality': best_row[explanation_metrics].sum()})
                else:
                    default_row = group[group['alpha'] == 0.1].iloc[0]
                    dataset_results.append({
                        'opt': opt,
                        'run_id': run_id,
                        'dataset_name': dataset,
                        'alpha': 0.1,
                        'ndcg_change': default_row['percentage_increase'],
                        'total_explanation_quality': explanation_pivot.loc[
                            explanation_pivot['alpha'] == 0.1, explanation_metrics
                        ].sum(axis=1).values[0]})
            else:
                default_row = group[group['alpha'] == 0.1].iloc[0]
                dataset_results.append({
                    'opt': opt,
                    'run_id': run_id,
                    'dataset_name': dataset,
                    'alpha': 0.1,
                    'ndcg_change': default_row['percentage_increase'],
                    'total_explanation_quality': explanation_pivot.loc[
                        explanation_pivot['alpha'] == 0.1, explanation_metrics
                    ].sum(axis=1).values[0]})

        dataset_results_df = pd.DataFrame(dataset_results)

        if return_most_used_alpha:
            dataset_results_df = dataset_results_df[dataset_results_df['alpha'] >= 0.1]

            most_used_alpha = (
                dataset_results_df.groupby(['opt'])['alpha']
                .agg(lambda x: x.value_counts().idxmax())
                .reset_index())
            most_used_alpha['dataset_name'] = dataset

            results.append(most_used_alpha)
        else:
            results.append(dataset_results_df)

    final_results = pd.concat(results, ignore_index=True)
    return final_results

def extract_data_with_most_used_alpha_avg(data, best_alpha_values, only_overall = True):
  
    data = data.loc[data["group"] == "Overall"] if only_overall else data

    extracted_data = pd.merge(
        data,
        best_alpha_values,
        on=['opt', 'dataset_name', 'alpha'],
        how='right')
    
    extracted_data = pd.concat([data.loc[(data["alpha"] == 0) & (data["opt"] == "baseline")],extracted_data])

    return extracted_data

def create_ndcg_table(ndcg_data):
    row_names_map = {
    "SEPopt": "P-PGPR",
    "LIRopt": "R-PGPR",
    "ETDopt": "D-PGPR",
    "ETD_LIR_opt": "DR-PGPR",
    "ETD_SEP_opt": "DP-PGPR",
    "SEP_LIR_opt": "PR-PGPR",
    "ETD_SEP_LIR_opt": "DPR-PGPR",
    "baseline": "PGPR"}

    
    ndcg_data = ndcg_data[ndcg_data["metric"] == "ndcg"]
    ndcg_means = ndcg_data.groupby(['opt', 'dataset_name'])['data'].mean().reset_index()

    ndcg_means['Model'] = ndcg_means['opt'].map(row_names_map)
   
    pivoted_ndcg = ndcg_means.pivot(index="Model", columns="dataset_name", values="data").reset_index()

    pivoted_ndcg["ml1m"] = pivoted_ndcg["ml1m"].apply(lambda x: f"{x:.2f}")
    pivoted_ndcg["lastfm"] = pivoted_ndcg["lastfm"].apply(lambda x: f"{x:.2f}")
    baseline = pivoted_ndcg[pivoted_ndcg["Model"] == "PGPR"]

    ordered_ours_models = ["R-PGPR", "P-PGPR", "D-PGPR", "DP-PGPR", "PR-PGPR", "DR-PGPR", "DPR-PGPR"]
    pivoted_ndcg = pivoted_ndcg.iloc[:,[0,2,1]]
    pivoted_ndcg = pivoted_ndcg.set_index("Model").reindex(ordered_ours_models).reset_index()

    baseline_part = pd.DataFrame({
        "Baseline Model": ["FM", "BPR", "NMF", "CKE", "CFKG", "KGAT", "PGPR"],
        "Baseline ML1M": [0.34, "-", 0.34, 0.34, 0.27, 0.34, baseline["ml1m"].values[0]],
        "Baseline LASTFM": [0.14, "-", 0.13, 0.11, 0.03, 0.12, baseline["lastfm"].values[0]]})
   
    combined_table = pd.concat([
        baseline_part.reset_index(drop=True),
        pivoted_ndcg.rename(columns={"ml1m": "Ours ML1M", "lastfm": "Ours LASTFM"}).reset_index(drop=True)], axis=1)

    return combined_table

def plot_ndcg_table(df):
    fig, ax = plt.subplots(figsize=(6, 2)) 
    ax.axis('off') 

    # Add the table
    table = plt.table(cellText=df.values,  
                    colLabels=df.columns,  
                    loc='center', 
                    cellLoc='center') 

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))  
    for key, cell in table.get_celld().items():
        cell.set_height(0.15)
    
    plt.show()

def create_explanation_quality_table(data: pd.DataFrame) -> pd.DataFrame:

    row_names_map = {
        "SEPopt": "P-PGPR",
        "LIRopt": "R-PGPR",
        "ETDopt": "D-PGPR",
        "baseline": "PGPR",
        "ETD_LIR_opt": "DR-PGPR",
        "ETD_SEP_opt": "DP-PGPR",
        "SEP_LIR_opt": "PR-PGPR",
        "ETD_SEP_LIR_opt": "DPR-PGPR"}

    rows = data[data["metric"].isin(["LIR", "SEP", "ETD"])].copy()

    rows["opt"] = rows["opt"].map(row_names_map)

    avg = rows.groupby(["opt", "dataset_name", "metric"])["data"].mean().unstack(level=2)

    avg["EQ"] = avg[["LIR", "SEP", "ETD"]].sum(axis=1)

    row_order = [
        "PGPR", "R-PGPR", "P-PGPR", "D-PGPR",
        "DR-PGPR", "DP-PGPR", "PR-PGPR", "DPR-PGPR"]
    dataset_order = ["ml1m", "lastfm"]

    avg = avg.reindex(
        pd.MultiIndex.from_product([row_order, dataset_order], names=["opt", "dataset_name"]))

    df_final = avg.unstack(level=1)

    df_final = df_final.loc[row_order]

    metric_order = ["EQ", "LIR", "SEP", "ETD"]
    column_order = [f"{dataset} {metric}" for dataset in dataset_order for metric in metric_order]

    df_final.columns = [f"{col[1]} {col[0]}" for col in df_final.columns] 
    df_final = df_final[column_order]  

    return df_final

def plot_explanation_quality_table(df: pd.DataFrame) -> None:
 
    df_rounded = df.copy().map(lambda x: f"{x:.2f}")  

    fig, ax = plt.subplots(figsize=(len(df.columns) * 1.2, len(df) * 0.6))  
    ax.axis("off")  

    table = plt.table(
        cellText=df_rounded.values,      
        colLabels=df_rounded.columns,   
        rowLabels=df_rounded.index,    
        cellLoc="center",             
        loc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df_rounded.columns))))  
    for key, cell in table.get_celld().items():
        cell.set_height(0.07)
   
    plt.show()

def clean_table_5_data(df):
   
    df = df[df["group"] != "Overall"]

    df = df[df["metric"].isin(["ndcg", "LIR", "SEP", "ETD"])]

    pivot_df = df.pivot_table(
        index=["alpha", "metric", "opt", "run_id", "dataset_name"],
        columns="group",
        values="data").reset_index()

    pivot_df["Diff"] = pivot_df["Male"] - pivot_df["Female"]

    diff_df = pivot_df.drop(columns=["Male", "Female"]).melt(
        id_vars=["alpha", "metric", "opt", "run_id", "dataset_name"],
        value_name="data",
        var_name="group")
    
    diff_df.sort_values(by=["dataset_name", "opt", "metric", "run_id"], inplace=True)

    return diff_df

def create_delta_explanation_table(data: pd.DataFrame) -> pd.DataFrame:

    row_names_map = {
        "SEPopt": "P-PGPR",
        "LIRopt": "R-PGPR",
        "ETDopt": "D-PGPR",
        "baseline": "PGPR",
        "ETD_LIR_opt": "DR-PGPR",
        "ETD_SEP_opt": "DP-PGPR",
        "SEP_LIR_opt": "PR-PGPR",
        "ETD_SEP_LIR_opt": "DPR-PGPR"}
    
    df = data.copy()

    df["opt"] = df["opt"].map(row_names_map)

    avg_data = (df.groupby(["opt", "dataset_name", "metric"])["data"].mean().reset_index())

    pivot_df = avg_data.pivot(index="opt", columns=["dataset_name", "metric"], values="data")

    pivot_df.columns = [f"{dataset} Δ{metric}" for dataset, metric in pivot_df.columns]

    row_order = ["PGPR", "R-PGPR", "P-PGPR", "D-PGPR","DR-PGPR", "DP-PGPR", "PR-PGPR", "DPR-PGPR"]
    dataset_order = ["ml1m", "lastfm"]
    metric_order = ["ndcg", "LIR", "SEP", "ETD"]
    column_order = [f"{dataset} Δ{metric}" for dataset in dataset_order for metric in metric_order]

    pivot_df = pivot_df.reindex(row_order)
    column_order = [col for col in column_order if col in pivot_df.columns]
    pivot_df = pivot_df[column_order]

    return pivot_df

def plot_fairness_table(df: pd.DataFrame,round = False) -> None:

    if round:
      df_rounded = df.copy().map(lambda x: f"{x:.3f}")  
    else:
      df_rounded = df.copy()

    fig, ax = plt.subplots(figsize=(len(df.columns) * 1.2, len(df) * 0.6))  
    ax.axis("off")  

    table = plt.table(
        cellText=df_rounded.values,      
        colLabels=df_rounded.columns,   
        rowLabels=df_rounded.index,     
        cellLoc="center",              
        loc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df_rounded.columns))))
    for key, cell in table.get_celld().items():
        cell.set_height(0.07)
  
    plt.show()

def load_and_label_csv_with_opt(file_path, run_id, dataset_name, chunk_size=100000):
   
    file_name = os.path.basename(file_path)
    opt_match = re.match(r'^([^a-z]*)', file_name)
    opt = opt_match.group(1).strip() if opt_match else "baseline"

    for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype={'alpha': 'float32', 'data': 'float32'}):
        chunk['run_id'] = run_id
        chunk['dataset_name'] = dataset_name
        chunk['opt'] = opt

        if "alpha" in chunk.columns:
            chunk["alpha"] = pd.to_numeric(chunk["alpha"], errors="coerce").fillna(0.0).astype("float32")
        else:
            chunk["alpha"] = 0.0  

        yield chunk  

def clean_results_dist(df):

    df = df.copy()

    if "alpha" in df.columns:
        df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce").fillna(0.0).astype("float32")

        valid_alphas = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        df = df[df["alpha"].isin(valid_alphas)]

    else:
        df["alpha"] = 0.0 

    df.loc[:, "opt"] = df["opt"].astype(str).str.rstrip("_")
    mappings = {
        'ETD': 'ETDopt',
        'SEP': 'SEPopt',
        'LIR': 'LIRopt',
        'ETD_SEP': 'ETD_SEP_opt',
        'ETD_LIR': 'ETD_LIR_opt',
        'SEP_LIR': 'SEP_LIR_opt',
        'ETD_SEP_LIR': 'ETD_SEP_LIR_opt',
        '': 'baseline'}
    df.loc[:, "opt"] = df["opt"].replace(mappings)

    df = df[(df["group"].isin(["Male", "Female"])) &
        (df["metric"].isin(["LIR", "SEP", "ETD", "ndcg"])) &
        (df["run_id"] != "ootb")]

    df.loc[:, "run_id"] = df["run_id"].str.extract(r'(Run \d+)')[0]
    
    return df

def process_all_files(base_dir, datasets):
    for dataset in datasets:
        dataset_dir = os.path.join(base_dir, dataset)
        for run_id in os.listdir(dataset_dir):
            run_dir = os.path.join(dataset_dir, run_id)
            if os.path.isdir(run_dir):
                for file_name in os.listdir(run_dir):
                    if file_name.endswith("distribution.csv"):
                        file_path = os.path.join(run_dir, file_name)
                        for chunk in load_and_label_csv_with_opt(file_path, run_id, dataset):
                            yield clean_results_dist(chunk)

def extract_data_with_most_used_alpha_dist(data, best_alpha_values):
  
    data["alpha"] = pd.to_numeric(data["alpha"], errors="coerce").fillna(0.0).astype(float).round(1)
    best_alpha_values["alpha"] = best_alpha_values["alpha"].astype(float).round(1)

    extracted_data = pd.merge(
        data,
        best_alpha_values,
        on=['opt', 'dataset_name', 'alpha'],
        how='right')

    baseline_rows = data.loc[data["opt"] == "baseline"]
    extracted_data = pd.concat([baseline_rows, extracted_data], ignore_index=True)

    return extracted_data

def perform_kruskal_test(data: pd.DataFrame) -> pd.DataFrame:
   
    results = []

    grouped = data.groupby(["opt", "run_id", "dataset_name", "metric"])
 
    data["opt"] = data["opt"].str.strip().str.lower()
    data["group"] = data["group"].str.strip().str.title()
    data["metric"] = data["metric"].str.strip().str.lower()

    for (opt, run_id, dataset_name, metric), group in grouped:
        male_data = group[group["group"] == "Male"]["data"]
        female_data = group[group["group"] == "Female"]["data"]

        if len(male_data) > 1 and len(female_data) > 1:
            stat, p_value = kruskal(male_data, female_data)
            significance = "Significant" if p_value < 0.05 else "Not Significant"

            results.append({
                "opt": opt,
                "run_id": run_id,
                "dataset_name": dataset_name,
                "metric": metric,
                "statistic": stat,
                "p_value": p_value,
                "significance": significance})
        else:
            print(f"Skipping opt={opt}, run_id={run_id}, dataset={dataset_name}, metric={metric} (not enough data)")

    return pd.DataFrame(results)

def perform_stouffer_test(kruskal_results: pd.DataFrame) -> pd.DataFrame:
   
    results = []

    kruskal_results["significance"] = kruskal_results["significance"].str.strip().str.lower()
    kruskal_results["opt"] = kruskal_results["opt"].str.strip().str.lower()
    kruskal_results["dataset_name"] = kruskal_results["dataset_name"].str.strip().str.lower()
    kruskal_results["metric"] = kruskal_results["metric"].str.strip().str.lower()

    grouped = kruskal_results.groupby(["opt", "dataset_name", "metric"])

    for (opt, dataset_name, metric), group in grouped:

        p_values = group["p_value"].tolist()

        z_scores = [norm.ppf(1 - p) for p in p_values if p > 0]

        sig_count = (group["significance"] == "significant").sum()
        non_sig_count = (group["significance"] == "not significant").sum()

        if z_scores:
            combined_z = sum(z_scores) / len(z_scores)
            combined_p = 1 - norm.cdf(combined_z)

            overall_significance = "Significant" if combined_p < 0.05 else "Not Significant"

            results.append({
                "opt": opt,
                "dataset_name": dataset_name,
                "metric": metric,
                "significant_runs": sig_count,
                "not_significant_runs": non_sig_count,
                "combined_p_value": combined_p,
                "overall_significance": overall_significance})

    return pd.DataFrame(results)

def mark_significant_values(data: pd.DataFrame, stouffer_results: pd.DataFrame) -> pd.DataFrame:

    row_names_map = {
    "sepopt": "P-PGPR",
    "liropt": "R-PGPR",
    "etdopt": "D-PGPR",
    "baseline": "PGPR",
    "etd_lir_opt": "DR-PGPR",
    "etd_sep_opt": "DP-PGPR",
    "sep_lir_opt": "PR-PGPR",
    "etd_sep_lir_opt": "DPR-PGPR"}

    reverse_row_names_map = {v: k for k, v in row_names_map.items()}

    significant_dict = {
        (row['opt'], row['dataset_name'], row['metric']): row['overall_significance'] == "Significant"
        for _, row in stouffer_results.iterrows()}

    updated_data = data.copy()

    updated_data = updated_data.astype(str)

    for opt_display_name, row in updated_data.iterrows():
        opt = reverse_row_names_map.get(opt_display_name, None)
        if opt is None:
            continue

        for col in updated_data.columns:
            try:
                dataset, metric = col.split(' Δ') 
                metric = metric.lower()
            except ValueError:
                continue

            if significant_dict.get((opt, dataset, metric), False):
                updated_data.at[opt_display_name, col] = f"{float(row[col]):.3f}*" 
            else:
                updated_data.at[opt_display_name, col] = f"{float(row[col]):.3f}"
    return updated_data