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


def format_feature_name(feature_name):
    readable_names = {
        "deposit_type_Non Refund": "Deposit type: Non Refund",
        "total_of_special_requests": "Total special requests",
        "lead_time": "Lead time",
        "adr": "Average daily rate (ADR)",
        "arrival_date_day_of_month": "Arrival day of month",
        "arrival_date_week_number": "Arrival week number",
        "stays_in_week_nights": "Week nights",
        "arrival_date_month": "Arrival month",
        "previous_cancellations": "Previous cancellations",
        "stays_in_weekend_nights": "Weekend nights",
        "required_car_parking_spaces": "Required parking spaces",
        "market_segment_Online TA": "Market segment: Online TA",
        "booking_changes": "Booking changes",
        "customer_type_Transient": "Customer type: Transient",
        "market_segment_Groups": "Market segment: Groups",
        "distribution_channel_TA/TO": "Distribution channel: TA/TO",
    }
    return readable_names.get(feature_name, feature_name.replace("_", " ").title())


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
        stratify=y,
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

    roc_auc = roc_auc_score(y_test, y_score)

    print(classification_report(y_test, y_pred, target_names=["Not Canceled", "Canceled"]))
    print("AUC:", roc_auc)

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1, 1.08], "wspace": 0.25},
    )
    fig.suptitle(
        "Random Forest Model Evaluation",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )

    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm / cm.sum(axis=1, keepdims=True)
    cm_labels = [
        [f"{count:,}\n{percent:.1%}" for count, percent in zip(row, percent_row)]
        for row, percent_row in zip(cm, cm_percent)
    ]
    sns.heatmap(
        cm,
        annot=cm_labels,
        fmt="",
        cmap="Blues",
        linewidths=1.2,
        linecolor="#fcfbeb",
        cbar_kws={"label": "Number of bookings"},
        xticklabels=["Not canceled", "Canceled"],
        yticklabels=["Not canceled", "Canceled"],
        ax=axes[0],
        annot_kws={"fontsize": 12},
    )
    axes[0].set_title("Confusion Matrix", fontsize=14, pad=14)
    axes[0].set_xlabel("Predicted booking status")
    axes[0].set_ylabel("Actual booking status")
    axes[0].tick_params(axis="both", labelsize=10)

    fpr, tpr, _ = roc_curve(y_test, y_score, drop_intermediate=False)
    axes[1].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#6f88c9",
        linewidth=1.8,
        label="Random classifier",
    )
    axes[1].plot(
        [0, 0, 1],
        [0, 1, 1],
        linestyle="--",
        color="#b5b5b5",
        linewidth=1.8,
        label="Ideal classifier",
    )
    axes[1].plot(
        fpr,
        tpr,
        color="#c0392b",
        linewidth=2.6,
        label=f"Random Forest (AUC = {roc_auc:.2f})",
    )
    axes[1].set_title("ROC Curve", fontsize=14, pad=14)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate / Recall")
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.04)
    axes[1].grid(color="#ffffff", linewidth=1.2)
    axes[1].legend(loc="lower right", frameon=True, fontsize=10)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(top=0.82, wspace=0.28)
    fig.savefig(FIGURES_DIR / "model-evaluation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    best_estimator = getattr(model, "best_estimator_", model)
    if hasattr(best_estimator, "named_steps"):
        best_estimator = best_estimator.named_steps["rf"]

    importances = pd.Series(
        best_estimator.feature_importances_,
        index=X_test_transformed.columns,
    ).sort_values(ascending=False)

    top_importances = importances.head(10).rename(index=format_feature_name)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x=top_importances.values,
        y=top_importances.index,
        orient="h",
        color="#77b8ad",
        ax=ax,
    )
    max_importance = top_importances.max()
    ax.set_title(
        "Top 10 Feature Importances - Random Forest Classifier",
        fontsize=15,
        pad=16,
    )
    ax.set_xlabel("Relative importance")
    ax.set_ylabel("")
    ax.set_xlim(0, max_importance * 1.16)
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", color="#ffffff", linewidth=1.2)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)

    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{value:.3f}" for value in top_importances.values],
            padding=5,
            fontsize=10,
            color="#333333",
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature-importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
