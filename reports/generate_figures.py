from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("darkgrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfbeb",
            "axes.facecolor": "#fcfbeb",
            "savefig.facecolor": "#fcfbeb",
            "axes.edgecolor": "#fcfbeb",
            "grid.color": "#ffffff",
            "font.size": 10,
        }
    )

    df_train = pd.read_csv(DATA_DIR / "train_final.csv")

    df_train["reservation_status_date"] = pd.to_datetime(
        df_train["reservation_status_date"]
    )

    df_train = df_train.drop(
        columns=["country", "reservation_status_date", "assigned_room_type"]
    )

    X = df_train.drop(columns="is_canceled")
    y = df_train["is_canceled"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    transformer = joblib.load(MODELS_DIR / "preprocessing_transformer.pkl")
    model = joblib.load(MODELS_DIR / "random_forest_model.pkl")

    X_test_transformed = transformer.transform(X_test)
    feature_names = transformer.get_feature_names_out()
    feature_names = [name.split("__", 1)[-1] for name in feature_names]
    X_test_transformed = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index,
    )

    y_pred = model.predict(X_test_transformed)
    y_score = model.predict_proba(X_test_transformed)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Not Canceled", "Canceled"]))
    print("AUC:", roc_auc_score(y_test, y_score))

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(16, 5),
        gridspec_kw={"width_ratios": [1, 1.15]},
    )

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not_Canceled", "Canceled"],
        yticklabels=["Not_Canceled", "Canceled"],
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted Cancellations")
    axes[0].set_ylabel("True Cancellations")

    fpr, tpr, _ = roc_curve(y_test, y_score, drop_intermediate=False)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#4f79c7", label="random model")
    axes[1].plot([0, 0], [1, 0], linestyle="--", color="#b5b5b5", label="ideal model")
    axes[1].plot([1, 1], linestyle="--", color="#b5b5b5")
    axes[1].plot(fpr, tpr, color="red", linewidth=2, label="model_Grid_refined")
    axes[1].set_title("Receiver Operating Characteristic (ROC) Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("Recall")
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model-evaluation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    best_estimator = getattr(model, "best_estimator_", model)
    importances = pd.Series(
        best_estimator.feature_importances_,
        index=X_test_transformed.columns,
    ).sort_values(ascending=False)

    top_importances = importances.head(10)

    fig, ax = plt.subplots(figsize=(16, 2.2))
    sns.barplot(
        x=top_importances.values,
        y=top_importances.index,
        orient="h",
        color="#8fc7bd",
        ax=ax,
    )
    ax.set_title("Features Importance - Random Forest Classifier")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelbottom=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature-importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
