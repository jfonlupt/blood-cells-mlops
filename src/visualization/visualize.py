import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.decomposition import IncrementalPCA

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data"
BARCELONA_DATA_PATH = "PBC_dataset_normal_DIB"
MUNICH_DATA_PATH = "PKG-AML-Cytomorphology_LMU"
ABBREVIATIONS = DATA_PATH / MUNICH_DATA_PATH / "abbreviations.txt"
FIGURES_PATH = BASE_DIR / "reports" / "figures"
TYPE_DICT = {
    # BARCELONA dataset abbreviations
    "BA": "BAS",
    "EO": "EOS",
    "ERB": "EBO",
    "MMY": "MMZ",
    "MY": "MYB",
    "PMY": "PMO",
    "LY": "LY",
    "MO": "MON",
    "BNE": "NGB",
    "SNE": "NGS",
    "PLATELET": "PLT",
    # MUNICH dataset abbreviations
    "BAS": "BAS",
    "EBO": "EBO",
    "EOS": "EOS",
    "LYA": "LY",
    "LYT": "LY",
    "MMZ": "MMZ",
    "MON": "MON",
    "MYB": "MYB",
    "MYO": "MYO",
    "NGB": "NGB",
    "NGS": "NGS",
    "PMB": "PMO",
    "PMO": "PMO",
}
R_COLS = [f"R{i}" for i in range(256)]
G_COLS = [f"G{i}" for i in range(256)]
B_COLS = [f"B{i}" for i in range(256)]
GRAY_COLS = [f"Gray{i}" for i in range(256)]


def compute_stats_image(img: np.array):
    data = {}
    data["size_x"] = img.shape[1]
    data["size_y"] = img.shape[0]
    data["brightness"] = np.mean(img)
    data["contrast"] = np.std(img)
    data["sharpness"] = cv2.Laplacian(cv2.resize(img, (100, 100)), cv2.CV_64F).var()

    hist = cv2.calcHist([img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    for j in range(256):
        data[f"R{j}"] = hist[0, j]
        data[f"G{j}"] = hist[1, j]
        data[f"B{j}"] = hist[2, j]
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    for j in range(256):
        data[f"Gray{j}"] = hist_gray[j][0]
    return data


def load_data():
    data = {
        "dataset": [],
        "path": [],
        "original_type": [],
        "type": [],
        "format": [],
        "size_x": [],
        "size_y": [],
        "brightness": [],
        "contrast": [],
        "sharpness": [],
    }
    for i in range(256):
        data[f"R{i}"] = []
        data[f"G{i}"] = []
        data[f"B{i}"] = []
        data[f"Gray{i}"] = []

    for root, _, files in os.walk(DATA_PATH):
        for file in [f for f in files if (f.endswith((".tiff", ".jpg"))) and not f.startswith(".")]:
            cell_type = file.split("_")[0].upper()
            if cell_type not in TYPE_DICT:
                continue

            file_path = str(Path(root) / file)
            data["path"].append(file_path)
            data["dataset"].append("BARCELONA" if BARCELONA_DATA_PATH in file_path else "MUNICH")
            data["original_type"].append(file.split("_")[0].upper())
            data["type"].append(TYPE_DICT[cell_type])
            data["format"].append("tiff" if file.endswith(".tiff") else "jpg")

            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Unable to read image {file_path}. Skipping.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            computed_stats = compute_stats_image(img)
            data["size_x"].append(computed_stats["size_x"])
            data["size_y"].append(computed_stats["size_y"])
            data["brightness"].append(computed_stats["brightness"])
            data["contrast"].append(computed_stats["contrast"])
            data["sharpness"].append(computed_stats["sharpness"])
            for i in range(256):
                data[f"R{i}"].append(computed_stats[f"R{i}"])
                data[f"G{i}"].append(computed_stats[f"G{i}"])
                data[f"B{i}"].append(computed_stats[f"B{i}"])
                data[f"Gray{i}"].append(computed_stats[f"Gray{i}"])

    return pd.DataFrame(data)


def normalize_images(df: pd.DataFrame):
    return df


def create_histograms(
    df: pd.DataFrame, normalize: bool = False, normalize_stats: dict | None = None
):
    colors = np.zeros((len(df), 768))
    gray = np.zeros((len(df), 256))

    for i, row in enumerate(df.itertuples()):
        img = Image.open(row.path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if normalize and normalize_stats is not None:
            img = normalize_image(img, normalize_stats["source"], normalize_stats["target"])
        colors[i] = np.array(img.histogram())
        gray[i] = np.array(img.convert("L").histogram())

    colors_df = pd.DataFrame(
        colors,
        columns=[f"R{i}" for i in range(256)]
        + [f"G{i}" for i in range(256)]
        + [f"B{i}" for i in range(256)],
    )
    gray_df = pd.DataFrame(gray, columns=[f"Gray{i}" for i in range(256)])
    full_df = pd.concat([df, colors_df, gray_df], axis=1)
    hists = (
        full_df.drop(["path", "original_type", "format", "mode", "size_x", "size_y"], axis=1)
        .groupby(["dataset", "type"])
        .mean()
    )
    return hists, full_df


def load_histograms():
    if not (
        (DATA_PATH / "blood_cells_histograms.csv").exists()
        and (DATA_PATH / "blood_cells_data_with_histograms.csv").exists()
    ):
        return None
    hists = pd.read_csv(DATA_PATH / "blood_cells_histograms.csv", index_col=[0, 1])
    full_df = pd.read_csv(DATA_PATH / "blood_cells_data_with_histograms.csv")
    return hists, full_df


def load_histograms_normalized():
    if (
        not (DATA_PATH / "blood_cells_histograms_norm.csv").exists()
        or not (DATA_PATH / "blood_cells_data_with_histograms_norm.csv").exists()
    ):
        return None
    hists = pd.read_csv(DATA_PATH / "blood_cells_histograms_norm.csv", index_col=[0, 1])
    full_df = pd.read_csv(DATA_PATH / "blood_cells_data_with_histograms_norm.csv")
    return hists, full_df


def load_abbreviations():
    with Path(ABBREVIATIONS).open() as f:
        lines = f.readlines()
    abbr_df = pd.DataFrame(
        [line.strip().split(" ") for line in lines[2:-3]],
        columns=["Abbreviation", "Name", "Details"],
    ).set_index("Abbreviation")
    abbr_df.fillna("", inplace=True)
    return abbr_df


def map_type(original_type: str, abbr_df: pd.DataFrame):
    if original_type in TYPE_DICT:
        return TYPE_DICT[original_type]
    if original_type in abbr_df.index:
        return abbr_df.loc[original_type].Name.lower()
    return original_type


def distribution(df: pd.DataFrame, show=False):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="type", hue="dataset", order=df["type"].value_counts().index)
    plt.title("Distribution of Blood Cell Types")
    plt.xlabel("Blood Cell Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(FIGURES_PATH / "cell_type_distribution.png", dpi=300, bbox_inches="tight")


def weighted_mean_std_agg(hist_cols, df):
    vals = np.arange(256)
    counts = df[hist_cols].values.sum(axis=0)
    total = counts.sum()
    mean = (counts * vals).sum() / total
    var = (counts * (vals - mean) ** 2).sum() / total
    return mean, np.sqrt(var)


def normalize_channel(channel_array, src_mean, src_std, tgt_mean, tgt_std):
    out = (channel_array.astype(np.float32) - src_mean) / (src_std + 1e-6)
    out = out * tgt_std + tgt_mean
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_image(img: Image, stats: dict, target_stats: dict):
    img_shape = img.size[::-1] + (3,)
    img_data = np.array(img.convert("RGB").get_flattened_data())

    map_channel = {"R": 0, "G": 1, "B": 2}
    for ch_name, ch_idx in map_channel.items():
        src_mean, src_std = stats[ch_name]
        tgt_mean, tgt_std = target_stats[ch_name]
        channel_array = np.array([pixel[ch_idx] for pixel in img_data])
        img_data[:, ch_idx] = normalize_channel(channel_array, src_mean, src_std, tgt_mean, tgt_std)
    return Image.fromarray(img_data.reshape(img_shape).astype(np.uint8))


def visualize_histograms(hists: pd.DataFrame, name: str = "blood_cells_histograms", show=False):
    r_cols = [f"R{i}" for i in range(256)]
    g_cols = [f"G{i}" for i in range(256)]
    b_cols = [f"B{i}" for i in range(256)]
    gray_cols = [f"Gray{i}" for i in range(256)]

    datasets = hists.index.get_level_values("dataset").unique()
    types = hists.index.get_level_values("type").unique()

    fig, axes = plt.subplots(
        len(types),
        len(datasets),
        figsize=(4 * len(datasets), 3 * len(types)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, typ in enumerate(types):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            if i == 0:
                ax.set_title(dataset)
            if j == 0:
                ax.set_ylabel(typ)
            if (dataset, typ) not in hists.index:
                plt.setp(ax.spines.values(), visible=False)
                continue
            row = hists.loc[(dataset, typ)]
            ax.plot(row[r_cols].values, color="red", label="R")
            ax.plot(row[g_cols].values, color="green", label="G")
            ax.plot(row[b_cols].values, color="blue", label="B")
            ax.plot(row[gray_cols].values, color="black", label="Gray", linestyle="--")

    axes[0, 0].legend()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(FIGURES_PATH / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def visualize_histograms_by_dataset(hists: pd.DataFrame, hists_norm: pd.DataFrame, show=False):
    r_cols = [f"R{i}" for i in range(256)]
    g_cols = [f"G{i}" for i in range(256)]
    b_cols = [f"B{i}" for i in range(256)]
    gray_cols = [f"Gray{i}" for i in range(256)]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True, squeeze=False)
    ax = axes[0, 0]
    ax.set_title("BARCELONA")
    ax.set_ylabel("original")

    row = hists.loc[("BARCELONA", "basophil")]
    ax.plot(row[r_cols].values, color="red", label="R")
    ax.plot(row[g_cols].values, color="green", label="G")
    ax.plot(row[b_cols].values, color="blue", label="B")
    ax.plot(row[gray_cols].values, color="black", label="Gray", linestyle="--")

    ax = axes[0, 1]
    ax.set_title("MUNICH")

    row = hists.loc[("MUNICH", "basophil")]
    ax.plot(row[r_cols].values, color="red", label="R")
    ax.plot(row[g_cols].values, color="green", label="G")
    ax.plot(row[b_cols].values, color="blue", label="B")
    ax.plot(row[gray_cols].values, color="black", label="Gray", linestyle="--")

    ax = axes[1, 0]
    ax.set_ylabel("normalized")

    row = hists_norm.loc[("BARCELONA", "basophil")]
    ax.plot(row[r_cols].values, color="red", label="R")
    ax.plot(row[g_cols].values, color="green", label="G")
    ax.plot(row[b_cols].values, color="blue", label="B")
    ax.plot(row[gray_cols].values, color="black", label="Gray", linestyle="--")

    ax = axes[1, 1]
    row = hists_norm.loc[("MUNICH", "basophil")]
    ax.plot(row[r_cols].values, color="red", label="R")
    ax.plot(row[g_cols].values, color="green", label="G")
    ax.plot(row[b_cols].values, color="blue", label="B")
    ax.plot(row[gray_cols].values, color="black", label="Gray", linestyle="--")

    if show:
        plt.show()
    else:
        plt.savefig(
            FIGURES_PATH / "blood_cells_histograms_by_dataset.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)


def visualize_samples(
    df: pd.DataFrame,
    types: list,
    datasets: list,
    normalize: bool = False,
    normalize_stats: dict | None = None,
    show=False,
):
    fig, axes = plt.subplots(
        len(types),
        len(datasets),
        figsize=(4 * len(datasets), 3 * len(types)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, typ in enumerate(types):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            ax.set(xticks=[], yticks=[])
            if i == 0:
                ax.set_title(dataset)
            if j == 0:
                ax.set_ylabel(typ)

            imgs = df[(df.dataset == dataset) & (df.type == typ)]
            if len(imgs) == 0:
                plt.setp(ax.spines.values(), visible=False)
                continue
            img = Image.open(imgs.sample(n=1, random_state=42).iloc[0].path)
            if normalize:
                img = normalize_image(
                    img, normalize_stats["source"][dataset], normalize_stats["target"]
                )
            ax.imshow(img.resize((200, 200)))
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(
            FIGURES_PATH / f"blood_cells_samples{'_norm' if normalize else ''}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def compute_brightness_contrast(df: pd.DataFrame):
    gray_cols = [f"Gray{i}" for i in range(256)]
    vals = np.arange(256)

    def brightness_contrast(row):
        counts = row[gray_cols].values.astype(float)
        total = counts.sum()
        mean = (counts * vals).sum() / total
        std = np.sqrt((counts * (vals - mean) ** 2).sum() / total)
        return pd.Series({"brightness": mean, "contrast": std})

    df[["brightness", "contrast"]] = df.apply(brightness_contrast, axis=1)
    return df


def visualize_brightness_contrast(df: pd.DataFrame, df_norm: pd.DataFrame, show=False):
    fig, axes = plt.subplots(2, 2, figsize=(10, 4))
    sns.boxplot(data=df, x="dataset", y="brightness", ax=axes[0, 0])
    axes[0, 0].set_title("Brightness by Dataset (Original)")
    sns.boxplot(data=df, x="dataset", y="contrast", ax=axes[0, 1])
    axes[0, 1].set_title("Contrast by Dataset (Original)")
    sns.boxplot(data=df_norm, x="dataset", y="brightness", ax=axes[1, 0])
    axes[1, 0].set_title("Brightness by Dataset (Normalized)")
    sns.boxplot(data=df_norm, x="dataset", y="contrast", ax=axes[1, 1])
    axes[1, 1].set_title("Contrast by Dataset (Normalized)")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(
            FIGURES_PATH / "blood_cells_brightness_contrast.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)


def visualize_brightness_by_type(
    df: pd.DataFrame,
    title="Blood Cells Brightness by Type",
    name="blood_cells_brightness_by_type",
    show=False,
):
    plt.figure()
    sns.boxplot(data=df, hue="dataset", x="type", y="brightness")
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    if show:
        plt.show()
    else:
        plt.savefig(FIGURES_PATH / f"{name}.png", dpi=300, bbox_inches="tight")


def visualize_contrast_by_type(
    df: pd.DataFrame,
    title="Blood Cells Contrast by Type",
    name="blood_cells_contrast_by_type",
    show=False,
):
    plt.figure()
    sns.boxplot(data=df, hue="dataset", x="type", y="contrast")
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    if show:
        plt.show()
    else:
        plt.savefig(FIGURES_PATH / f"{name}.png", dpi=300, bbox_inches="tight")


def load_images_data(
    df: pd.DataFrame,
    size: int = 100,
    sample_size: int = 1000,
    normalize: bool = False,
    normalize_stats: dict | None = None,
):
    def _load_one(item, size=size):
        i, path = item
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if normalize and normalize_stats is not None:
            img = normalize_image(
                Image.fromarray(img),
                normalize_stats["source"][df.iloc[i]["dataset"]],
                normalize_stats["target"],
            )
            img = np.array(img)
        return i, img

    df_sample = df.sample(sample_size, random_state=42)
    images = np.empty((len(df_sample), size, size, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, img in ex.map(_load_one, enumerate(df_sample["path"])):
            images[i] = img

    return images, df_sample


def pca(images: np.ndarray, n_components: int = 2):
    images = images.reshape(len(images), -1).astype(np.float32) / 255.0
    pca = IncrementalPCA(n_components=n_components)
    return pca.fit_transform(images)


def visualize_pca(
    data_pca: np.ndarray,
    df_sample: pd.DataFrame,
    n_components: int = 2,
    by: str = "dataset",
    title="PCA Visualization",
    name="blood_cells_pca",
    show=False,
):
    if n_components == 2:
        fig = plt.figure()
        sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=df_sample[by])
        plt.title(title)
        if show:
            plt.show()
        else:
            plt.savefig(FIGURES_PATH / f"{name}.png", dpi=300, bbox_inches="tight")
    elif n_components == 3:
        categories = df_sample[by]
        uniques = categories.unique()
        palette = dict(zip(uniques, sns.color_palette(n_colors=len(uniques))))

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for cat in uniques:
            mask = (categories == cat).values
            ax.scatter(
                data_pca[mask, 0],
                data_pca[mask, 1],
                data_pca[mask, 2],
                color=palette[cat],
                label=cat,
                s=20,
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        if show:
            plt.show()
        else:
            plt.savefig(FIGURES_PATH / f"{name}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
    else:
        raise ValueError("n_components must be 2 or 3")


def main(show=False):
    df = load_data()
    abbr_df = load_abbreviations()
    df["type"] = df.original_type.map(lambda x: map_type(x, abbr_df))
    hists, full_df = load_histograms() or create_histograms(df)
    print(df.head())
    distribution(df, show=show)
    visualize_samples(
        df, types=df["type"].unique(), datasets=df["dataset"].unique(), normalize=False, show=show
    )
    visualize_histograms(hists, show=show)

    stats = {}
    for dataset, sub in full_df.groupby("dataset"):
        stats[dataset] = {
            ch: weighted_mean_std_agg([f"{ch}{i}" for i in range(256)], sub)
            for ch in ["R", "G", "B"]
        }

    target_stats = {
        ch: weighted_mean_std_agg([f"{ch}{i}" for i in range(256)], full_df)
        for ch in ["R", "G", "B"]
    }

    visualize_samples(
        full_df,
        types=full_df["type"].unique(),
        datasets=full_df["dataset"].unique(),
        normalize=True,
        normalize_stats={"source": stats, "target": target_stats},
        show=show,
    )
    hists_norm, full_df_norm = load_histograms_normalized() or create_histograms(
        df, normalize=True, normalize_stats={"source": stats, "target": target_stats}
    )
    visualize_histograms(hists_norm, name="blood_cells_histograms_norm", show=show)

    visualize_histograms_by_dataset(hists, hists_norm, show=show)

    full_df = compute_brightness_contrast(full_df)
    full_df_norm = compute_brightness_contrast(full_df_norm)

    visualize_brightness_contrast(full_df, full_df_norm, show=show)
    visualize_brightness_by_type(
        full_df,
        title="Blood Cells Brightness by Type (Original)",
        name="blood_cells_brightness_by_type",
        show=show,
    )
    visualize_brightness_by_type(
        full_df_norm,
        title="Blood Cells Brightness by Type (Normalized)",
        name="blood_cells_brightness_by_type_norm",
        show=show,
    )
    visualize_contrast_by_type(
        full_df,
        title="Blood Cells Contrast by Type (Original)",
        name="blood_cells_contrast_by_type",
        show=show,
    )
    visualize_contrast_by_type(
        full_df_norm,
        title="Blood Cells Contrast by Type (Normalized)",
        name="blood_cells_contrast_by_type_norm",
        show=show,
    )

    images, df_sample = load_images_data(full_df, size=100, sample_size=1000, normalize=False)
    data_pca_2d = pca(images, n_components=2)
    visualize_pca(
        data_pca_2d,
        df_sample,
        n_components=2,
        by="dataset",
        title="PCA Visualization by Dataset (Original)",
        name="blood_cells_pca_by_dataset",
        show=show,
    )
    visualize_pca(
        data_pca_2d,
        df_sample,
        n_components=2,
        by="type",
        title="PCA Visualization by Type (Original)",
        name="blood_cells_pca_by_type",
        show=show,
    )
    data_pca_3d = pca(images, n_components=3)
    visualize_pca(
        data_pca_3d,
        df_sample,
        n_components=3,
        by="dataset",
        title="PCA Visualization by Dataset (Original)",
        name="blood_cells_pca_by_dataset_3d",
        show=show,
    )
    visualize_pca(
        data_pca_3d,
        df_sample,
        n_components=3,
        by="type",
        title="PCA Visualization by Type (Original)",
        name="blood_cells_pca_by_type_3d",
        show=show,
    )

    images_norm, df_sample_norm = load_images_data(
        full_df_norm,
        size=100,
        sample_size=1000,
        normalize=True,
        normalize_stats={"source": stats, "target": target_stats},
    )
    data_pca_norm_2d = pca(images_norm, n_components=2)
    visualize_pca(
        data_pca_norm_2d,
        df_sample_norm,
        n_components=2,
        by="dataset",
        title="PCA Visualization by Dataset (Normalized)",
        name="blood_cells_pca_by_dataset_norm",
        show=show,
    )
    visualize_pca(
        data_pca_norm_2d,
        df_sample_norm,
        n_components=2,
        by="type",
        title="PCA Visualization by Type (Normalized)",
        name="blood_cells_pca_by_type_norm",
        show=show,
    )
    data_pca_norm_3d = pca(images_norm, n_components=3)
    visualize_pca(
        data_pca_norm_3d,
        df_sample_norm,
        n_components=3,
        by="dataset",
        title="PCA Visualization by Dataset (Normalized)",
        name="blood_cells_pca_by_dataset_norm_3d",
        show=show,
    )
    visualize_pca(
        data_pca_norm_3d,
        df_sample_norm,
        n_components=3,
        by="type",
        title="PCA Visualization by Type (Normalized)",
        name="blood_cells_pca_by_type_norm_3d",
        show=show,
    )


if __name__ == "__main__":
    show = False
    main(show=show)
