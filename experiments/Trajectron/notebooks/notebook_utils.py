"""
Modules to transform results.json to pd.DataFrame for jupyter notebook
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def metric2float(df):
    return df.applymap(lambda x: float(x.split()[0]) if not pd.isna(x) else 0)


def fill_zero_with_mean(col):
    zero_idx = np.where(col.values == 0)[0]
    col[zero_idx] = col.mean()
    return col


def load_results_dict(path):
    with open(os.path.join(path, "results.json"), "r") as f:
        results_dict = json.load(f)
        model_name = results_dict["model"]
        results = results_dict["results"]
    return results


def append_mean_and_std(df, axis=1):
    df = df.applymap(lambda x: x if not isinstance(x, str) else float(x.split(" ")[0]))
    df_fillna = df.apply(lambda x: x.fillna(x.mean()), axis=axis)
    mean = df_fillna.mean(axis=axis).round(2)
    std = df_fillna.std(axis=axis).round(2)
    if axis == 0:
        df.loc["mean"] = mean
        df.loc["std"] = std
    elif axis == 1:
        df.loc[:, "mean"] = mean
        df.loc[:, "std"] = std
    return df


def country_or_road_class_dict2df(results, k_show="5", ph_show="12", metric_show="ADE"):
    models = [os.path.basename(model) for model in results.keys()]
    df = pd.DataFrame(columns=range(5), index=models)

    for model, eval_dict in results.items():
        for i, (eval_dir, metric_dict) in enumerate(eval_dict.items()):
            for conventional_metric, ph_dict in metric_dict.items():
                for ph, k_dict in ph_dict.items():
                    for k, GE_dict in k_dict.items():
                        for metric, value in GE_dict.items():
                            if (
                                k == k_show
                                and ph == ph_show
                                and conventional_metric == metric_show
                            ):
                                if metric == "AB":
                                    df.loc[os.path.basename(model), i] = value

    info = f'min{metric_show}_{k_show}_[{"%" if metric_show == "MR" else "m"}]'
    GE = "SDRGE [%]"

    cols = pd.MultiIndex.from_tuples(
        [
            (info, "keine", "in distribution"),
            (info, "Land", "USA <-> SGP"),
            (info, "Straßenart", "innerhalb Stadtverkehr"),
            (info, "Straßenart", "zu Kreisverkehr"),
            (info, "Straßenart", "zu Schnellstraße"),
        ],
        names=["Metrik", "Änderung", "Details"],
    )

    cols_GE = pd.MultiIndex.from_tuples(
        [
            (GE, "Land", "USA <-> SGP"),
            (GE, "Straßenart", "innerhalb Stadtverkehr"),
            (GE, "Straßenart", "zu Kreisverkehr"),
            (GE, "Straßenart", "zu Schnellstraße"),
        ],
        names=["Metrik", "Änderung", "Details"],
    )

    df = metric2float(df)
    df.columns = cols
    df_GE = pd.DataFrame(index=models, columns=cols_GE)
    df.index.name = "Training"
    df_GE.index.name = df.index.name

    for _, Aenderung, Detail in cols:
        if Aenderung == "keine":
            continue
        df_GE[(GE, Aenderung, Detail)] = (
            (df[(info, Aenderung, Detail)] - df[(info, "keine", "in distribution")])
            / df[(info, Aenderung, Detail)]
            * 100
        ).round(2)

    return pd.concat([df, df_GE], axis=1)


def driving_characteristic_dict2df(
    results, k_show="5", ph_show="12", metric_show="ADE"
):
    models = [os.path.basename(model) for model in results.keys()]
    characteristics = [
        os.path.basename(os.path.dirname(os.path.dirname(key)))
        for key in results[list(results.keys())[0]].keys()
    ]
    categories = [
        os.path.basename(os.path.dirname(key))
        for key in results[list(results.keys())[0]].keys()
    ]

    cols = pd.MultiIndex.from_tuples(
        list(zip(characteristics, categories)),
        names=["Fahrcharakteristik", "Kategorie"],
    )
    idx = models
    df = pd.DataFrame(index=idx, columns=cols)

    for model, eval_dict in results.items():
        for eval_dir, metric_dict in eval_dict.items():
            for conventional_metric, ph_dict in metric_dict.items():
                for ph, k_dict in ph_dict.items():
                    for k, GE_dict in k_dict.items():
                        for metric, value in GE_dict.items():
                            if (
                                k == k_show
                                and ph == ph_show
                                and conventional_metric == metric_show
                            ):
                                if metric == "AB":
                                    df.loc[
                                        os.path.basename(model),
                                        (
                                            os.path.basename(
                                                os.path.dirname(
                                                    os.path.dirname(eval_dir)
                                                )
                                            ),
                                            os.path.basename(os.path.dirname(eval_dir)),
                                        ),
                                    ] = value

    return df.T


def ood_driving_characteristic_dict2df(
    results, k_show="5", ph_show="12", metric_show="ADE"
):

    domains = [os.path.basename(model) for model in results.keys()]
    cols = pd.MultiIndex.from_product(
        [domains, domains], names=["Training", "Evaluation"]
    )

    characteristics = [
        os.path.basename(os.path.dirname(os.path.dirname(key)))
        for key in results[list(results.keys())[0]].keys()
    ]
    categories = [
        os.path.basename(os.path.dirname(key))
        for key in results[list(results.keys())[0]].keys()
    ]

    characteristics = characteristics[: int(len(characteristics) / 2)]
    categories = categories[: int(len(categories) / 2)]

    idx = pd.MultiIndex.from_tuples(
        list(zip(characteristics, categories)),
        names=["Fahrcharakteristik", "Kategorie"],
    )

    df = pd.DataFrame(index=idx, columns=cols)

    for model, eval_dict in results.items():
        for eval_dir, metric_dict in eval_dict.items():
            for conventional_metric, ph_dict in metric_dict.items():
                for ph, k_dict in ph_dict.items():
                    for k, GE_dict in k_dict.items():
                        for metric, value in GE_dict.items():
                            if (
                                k == k_show
                                and ph == ph_show
                                and conventional_metric == metric_show
                            ):
                                if metric == "AB":
                                    eval_dir_subfolders = eval_dir.split("/")

                                    train_domain = os.path.basename(model)
                                    eval_domain = eval_dir_subfolders[-1].split(
                                        "_middle"
                                    )[0]
                                    category = eval_dir_subfolders[-2]
                                    characteristic = eval_dir_subfolders[-3]

                                    df.loc[
                                        (characteristic, category),
                                        (train_domain, eval_domain),
                                    ] = value

    return df


def speed_zone_dict2df(results, k_show="5", ph_show="12", metric_show="ADE"):

    eval_sets = [key.split("_")[-1] for key in results[list(results.keys())[0]].keys()]
    models = [os.path.basename(key) for key in results.keys()]
    df = pd.DataFrame(columns=eval_sets, index=models)

    for model, eval_dict in results.items():
        for eval_dir, metric_dict in eval_dict.items():
            for conventional_metric, ph_dict in metric_dict.items():
                for ph, k_dict in ph_dict.items():
                    for k, GE_dict in k_dict.items():
                        for metric, value in GE_dict.items():
                            if (
                                k == k_show
                                and ph == ph_show
                                and conventional_metric == metric_show
                            ):
                                if metric == "AB":
                                    df.loc[
                                        os.path.basename(model), eval_dir.split("_")[-1]
                                    ] = value
    df = metric2float(df)
    SDRGE_slow = ((df["slow"] - df["middle"]) / df["middle"] * 100).round(2)
    SDRGE_fast = ((df["fast"] - df["middle"]) / df["middle"] * 100).round(2)
    df["SDRGE: middle -> slow"] = SDRGE_slow
    df["SDRGE: middle ->fast"] = SDRGE_fast

    df.columns = pd.MultiIndex.from_tuples(
        [
            ("minADE_5 [m]", "slow"),
            ("minADE_5 [m]", "middle"),
            ("minADE_5 [m]", "fast"),
            ("SDRGE [%]", "middle ->slow"),
            ("SDRGE [%]", "middle ->fast"),
        ],
        names=["Metrik", "Geschwindigkeitsbereich"],
    )

    return df


def ablation_dict2df(results, k_show="5", ph_show="12", metric_show="ADE"):
    models = [model.split("/")[-2:] for model in results.keys()]
    idx = pd.MultiIndex.from_tuples(models, names=["Model", "Training"])
    eval_sets = [
        os.path.basename(key) for key in results[list(results.keys())[0]].keys()
    ]

    df = pd.DataFrame(columns=eval_sets, index=idx)
    df.columns.name = "Evaluation"

    for model, eval_dict in results.items():
        for eval_dir, metric_dict in eval_dict.items():
            for conventional_metric, ph_dict in metric_dict.items():
                for ph, k_dict in ph_dict.items():
                    for k, GE_dict in k_dict.items():
                        for metric, value in GE_dict.items():
                            if (
                                k == k_show
                                and ph == ph_show
                                and conventional_metric == metric_show
                            ):
                                if metric == "AB":
                                    df.loc[
                                        tuple(model.split("/")[-2:]),
                                        os.path.basename(eval_dir),
                                    ] = value

    return df


def Verbesserung_dict2df(results, k_show="5", ph_show="12", metric_show="ADE"):
    models = [
        (model.split("/")[2] + "_" + model.split("/")[1], model.split("/")[-1])
        if "Verbesserung" in model
        else tuple(model.split("/"))
        for model in results.keys()
    ]
    idx = pd.MultiIndex.from_tuples(models, names=["Model", "Training"])
    eval_sets = [
        os.path.basename(key) for key in results[list(results.keys())[0]].keys()
    ]

    df = pd.DataFrame(columns=eval_sets, index=idx)
    df.columns.name = "Evaluation"

    for model, eval_dict in results.items():
        for eval_dir, metric_dict in eval_dict.items():
            for conventional_metric, ph_dict in metric_dict.items():
                for ph, k_dict in ph_dict.items():
                    for k, GE_dict in k_dict.items():
                        for metric, value in GE_dict.items():
                            if (
                                k == k_show
                                and ph == ph_show
                                and conventional_metric == metric_show
                            ):
                                if metric == "AB":
                                    model_tuple = (
                                        (
                                            model.split("/")[2]
                                            + "_"
                                            + model.split("/")[1],
                                            model.split("/")[-1],
                                        )
                                        if "Verbesserung" in model
                                        else tuple(model.split("/"))
                                    )
                                    df.loc[
                                        model_tuple, os.path.basename(eval_dir)
                                    ] = value

    return df


def visualize_ablation(
    df,
    normal_version,
    ablation_version,
    vorzeichen="Zuwachs",
    figsize=(10, 5),
    fontsize=15,
):

    df_ablation = df.loc[ablation_version]
    df_normal = df.loc[normal_version]

    if len(df_ablation) < len(df_normal):
        df_ablation = np.tile(df_ablation.values, (len(df_normal), 1))

    if len(df_ablation) > len(df_normal):
        df_normal = np.tile(df_normal.values, (len(df_ablation), 1))

    df_advantage = ((df_ablation - df_normal) / df_normal * 100).round(2)

    fig, ax = plt.subplots(figsize=figsize)
    df_advantage.T.plot.barh(ax=ax, fontsize=fontsize)
    ax.legend(loc="upper center", bbox_to_anchor=(1.2, 0.5), fontsize=fontsize)
    ax.set_ylabel("Evaluations-Domäne", fontsize=fontsize)
    ax.set_xlabel(f"minADE_5 {vorzeichen} [%]", fontsize=fontsize)
    return fig, ax, df_advantage


def get_out_of_distribution_df(path_Trajectron, path_MANTRA):
    results_Trajectron = load_results_dict(path_Trajectron)
    results_MANTRA = load_results_dict(path_MANTRA)

    df_Trajectron = ood_driving_characteristic_dict2df(results_Trajectron)
    df_MANTRA = ood_driving_characteristic_dict2df(results_MANTRA)

    train_col = df_Trajectron.columns.get_level_values("Training")
    eval_col = df_Trajectron.columns.get_level_values("Evaluation")

    df_Trajectron.columns = pd.MultiIndex.from_tuples(
        list(zip(["Trajectron++" for _ in range(len(train_col))], train_col, eval_col)),
        names=["Model", "Training", "Evaluation"],
    )
    df_MANTRA.columns = pd.MultiIndex.from_tuples(
        list(zip(["MANTRA" for _ in range(len(train_col))], train_col, eval_col)),
        names=["Model", "Training", "Evaluation"],
    )

    df = pd.concat([df_Trajectron, df_MANTRA], axis=1)
    df = metric2float(df)
    df = df.fillna(0)
    return df


def sort_by(df, dataset="Evaluation"):
    if dataset == "Training":
        df = df.reorder_levels(["Model", "Training", "Evaluation"], axis=1)
        resort_cols = [
            (model, train, eval)
            for model in df.columns.get_level_values("Model").unique()
            for train in df.columns.get_level_values("Training").unique()
            for eval in df.columns.get_level_values("Evaluation").unique()
        ]
    elif dataset == "Evaluation":
        df = df.reorder_levels(["Model", "Evaluation", "Training"], axis=1)
        resort_cols = [
            (model, eval, train)
            for model in df.columns.get_level_values("Model").unique()
            for eval in df.columns.get_level_values("Evaluation").unique()
            for train in df.columns.get_level_values("Training").unique()
        ]
    return df[resort_cols]


def sort_ablation(df, by="Training"):
    if by == "Training":
        df = df.reorder_levels(["Training", "Model"], axis=0)
        resort_cols = [
            (train, model)
            for train in df.index.get_level_values("Training").unique()
            for model in df.index.get_level_values("Model").unique()
        ]
    elif by == "Model":
        df = df.reorder_levels(["Model", "Training"], axis=0)
        resort_cols = [
            (model, train)
            for model in df.index.get_level_values("Model").unique()
            for train in df.index.get_level_values("Training").unique()
        ]
    return df.loc[resort_cols]


def TDRGE_version(df, relative=True):
    df = sort_by(df, dataset="Evaluation")
    eval_domains = df.columns.get_level_values("Evaluation").unique()
    models = df.columns.get_level_values("Model").unique()

    df_TDRGE_abs = pd.DataFrame(index=df.index, columns=df.columns)
    df_TDRGE_rel = pd.DataFrame(index=df.index, columns=df.columns)

    for model in models:
        for eval_domain in eval_domains:
            in_dist = eval_domain
            out_of_dist = [domain for domain in eval_domains if domain != eval_domain][
                0
            ]

            TDRGE_abs = (
                df[(model, eval_domain, out_of_dist)]
                - df[(model, eval_domain, in_dist)]
            )
            TDRGE_rel = TDRGE_abs / df[(model, eval_domain, in_dist)] * 100
            df_TDRGE_abs[(model, eval_domain, out_of_dist)] = TDRGE_abs
            df_TDRGE_rel[(model, eval_domain, out_of_dist)] = TDRGE_rel

    df_TDRGE_abs = df_TDRGE_abs.dropna(axis=1, how="all").round(2)
    df_TDRGE_rel = df_TDRGE_rel.dropna(axis=1, how="all").round(2)
    df_TDRGE_rel = df_TDRGE_rel.fillna(0).applymap(
        lambda x: f"({'+' if x > 0 or int(x) == 0 else ''}{str(int(x))} %)"
    )
    if relative:
        return df_TDRGE_rel
    else:
        return df_TDRGE_abs


def visualize_out_of_distribution(
    df, figsize=(15, 50), fontsize=15, hspace=0.5, wspace=0.5
):
    sort_by(df, dataset="Training")
    models = df.columns.get_level_values("Model").unique()
    characteristics = df.index.get_level_values("Fahrcharakteristik").unique()

    fig, ax = plt.subplots(len(characteristics), 2, figsize=figsize)
    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    offset = 6
    for i, characteristic in enumerate(characteristics):
        for j, model in enumerate(models):
            for k, (train, eval) in enumerate(df[model].columns):
                entry = df.loc[characteristic, (model, train, eval)]
                yticks_str = entry.index
                yticks_num_spaced = np.arange(0, len(yticks_str) * offset, offset)
                yticks_num = np.arange(0, len(yticks_str) * offset)

                ax[i, j].barh(
                    y=yticks_num_spaced + k,
                    width=entry.values,
                    label=f"train: {train}, eval: {eval}",
                )

                yticks_str_spaced = []
                categories = df.loc[characteristic].index

                for c in range(len(yticks_num) - len(yticks_num_spaced)):
                    yticks_str_spaced.append("")
                    if c < len(categories):
                        yticks_str_spaced.append("")
                        yticks_str_spaced.append(categories[c])
                ax[i, j].set_yticklabels(yticks_str_spaced, fontsize=fontsize)

                if i == 0 and j == 0:
                    ax[i, j].legend(
                        loc="upper center", bbox_to_anchor=(1, 1.5), fontsize=fontsize
                    )

                ax[i, j].set_title(model, fontsize=fontsize)
                ax[i, j].set_ylabel(characteristic, fontsize=fontsize)
                ax[i, j].set_xlabel("minADE_5 [m]", fontsize=fontsize)
    return fig, ax


def rename_df(df):
    df = df.rename(index={"nuScenes_Boston": "nuScenes-Boston"})
    df = df.rename(index={"nuScenes_Onenorth": "nuScenes-Onenorth"})
    df = df.rename(index={"nuScenes_Queenstown": "nuScenes-Queenstown"})
    df = df.rename(index={"lyft_level_5": "lyft level 5"})

    df = df.rename(columns={"nuScenes_Boston": "nuScenes-Boston"})
    df = df.rename(columns={"nuScenes_Onenorth": "nuScenes-Onenorth"})
    df = df.rename(columns={"nuScenes_Queenstown": "nuScenes-Queenstown"})
    df = df.rename(columns={"lyft_level_5": "lyft level 5"})

    df = df.rename(index={"nuScenes_Boston_middle": "nuScenes-Boston"})
    df = df.rename(index={"nuScenes_Onenorth_middle": "nuScenes-Onenorth"})
    df = df.rename(index={"nuScenes_Queenstown_middle": "nuScenes-Queenstown"})
    df = df.rename(index={"lyft_level_5_middle": "lyft level 5"})
    df = df.rename(index={"lyft_level_5_middle_train": "lyft level 5"})

    df = df.rename(columns={"nuScenes_Boston_middle": "nuScenes-Boston"})
    df = df.rename(columns={"nuScenes_Onenorth_middle": "nuScenes-Onenorth"})
    df = df.rename(columns={"nuScenes_Queenstown_middle": "nuScenes-Queenstown"})
    df = df.rename(columns={"lyft_level_5_middle": "lyft level 5"})
    df = df.rename(columns={"lyft_level_5_middle_train": "lyft level 5"})

    df = df.rename(index={"KITTI_fast": "KITTI_Highway"})
    df = df.rename(columns={"KITTI_fast": "KITTI_Highway"})

    df = df.rename(index={"robot_no_map": "Trajectron++, keine Karte"})
    df = df.rename(index={"robot_no_edge": "Trajectron++, keine Interaktion"})
    df = df.rename(index={"int_ee_me": "Trajectron++, kein Robot"})
    df = df.rename(index={"MANTRA_no_map": "MANTRA, keine Karte"})
    df = df.rename(index={"MANTRA_online_writing": "MANTRA, online writing"})

    df = df.rename(index={"00_15": "0°-15°"})
    df = df.rename(index={"15_45": "15°-45°"})
    df = df.rename(index={"45_75": "45°-75°"})
    df = df.rename(index={"75_360": "75°+  °"})
    df = df.rename(columns={"00_15": "0°-15°"})
    df = df.rename(columns={"15_45": "15°-45°"})
    df = df.rename(columns={"45_75": "45°-75°"})
    df = df.rename(columns={"75_360": "75°+  °"})

    return df


def rename_df_for_word(df):

    df = df.rename(columns={"nuScenes-Boston": "nS-Bo"})
    df = df.rename(columns={"nuScenes-Queenstown": "nS-Qu"})
    df = df.rename(columns={"nuScenes-Onenorth": "nS-On"})
    df = df.rename(columns={"lyft level 5": "l5"})
    return df
