# Hotel Booking Cancellation Prediction

Machine learning classification project for predicting whether a hotel booking is likely to be canceled.

This project is part of my Data Analytics and Data Science portfolio. It focuses on framing a practical business problem, cleaning and preparing hotel booking data, evaluating classification models, and interpreting the drivers of cancellation risk.

## Project Overview

Hotel cancellations can affect revenue planning, staffing, inventory management, and overbooking strategy. The goal of this project is to predict cancellation risk using booking-related information available at booking time or shortly after booking.

The final portfolio version avoids using variables that may reveal information from after the booking outcome, such as reservation status dates or later operational assignment fields.

## Business Question

Can we predict whether a hotel booking will be canceled using booking characteristics such as lead time, deposit type, market segment, customer type, previous cancellations, and special requests?

## Dataset

The dataset contains hotel booking records with information about:

- booking dates and stay duration,
- guest composition,
- booking channel and market segment,
- room type,
- deposit type,
- prior customer behavior,
- special requests,
- cancellation status.

The data dictionary is available in [reports/dataset_dictionary.md](reports/dataset_dictionary.md).

## Modeling Approach

The project includes:

- exploratory data analysis,
- target distribution analysis,
- feature availability and leakage review,
- preprocessing with scaling and categorical encoding,
- SMOTE resampling for class imbalance,
- baseline model comparison,
- Random Forest model tuning with SMOTE inside the cross-validation pipeline,
- classification report and ROC-AUC evaluation,
- feature importance interpretation.

## Final Model Performance

Final tuned Random Forest pipeline test performance:

| Metric | Value |
|---|---:|
| Accuracy | 0.85 |
| Weighted F1-score | 0.85 |
| ROC-AUC score | 0.90 |
| Canceled precision | 0.78 |
| Canceled recall | 0.71 |
| Canceled F1-score | 0.74 |

The model provides a useful balance between precision and recall for the cancellation class. When it predicts a cancellation, it is correct about 78% of the time, and it identifies about 71% of actual cancellations. This trade-off matters in a hotel business context because false positives and false negatives can have different operational and revenue-management costs.

## Model Evaluation Visuals

### Confusion Matrix and ROC Curve

![Random Forest model evaluation](reports/figures/model-evaluation.png)

### Feature Importance

![Top 10 feature importances](reports/figures/feature-importance.png)

## Key Predictors

The feature-importance results are consistent with the exploratory analysis: cancellation behavior is shaped by booking timing, deposit policy, price level, customer engagement signals, calendar effects, and past behavior.

| Top driver | Business interpretation |
|---|---|
| Lead time | Bookings made far in advance tend to carry different cancellation risk because plans can change before the stay date. |
| Deposit type: Non Refund | Deposit policy is strongly linked to cancellation behavior in this dataset. |
| Average daily rate (ADR) | Price level may contain cancellation-risk signal and can also reflect seasonality, hotel type, room type, demand level, or customer segment. |
| Total special requests | Guests with special requests may show stronger booking intent because they are already planning specific details of the stay. |
| Arrival day of month | Calendar timing may capture demand, seasonality, or operational patterns related to cancellation behavior. |
| Arrival week number | Week-of-year effects may reflect seasonality and travel-demand patterns. |
| Week nights | Stay duration during the week can reflect trip purpose and booking context. |
| Arrival month | Month-level seasonality can influence booking and cancellation behavior. |
| Previous cancellations | Past customer behavior is predictive because guests with prior cancellations may be more likely to cancel again. |
| Booking changes | Modified bookings may indicate uncertainty, but they can also show active planning, making this a useful behavioral signal. |

## Repository Structure

```text
hotel-booking-cancellation-prediction/
  data/
    train_final.csv
    test_final.csv
  models/
    preprocessing_transformer.pkl
    random_forest_model.pkl
  notebooks/
    01_hotel_booking_cancellation_analysis.ipynb
  reports/
    dataset_dictionary.md
    generate_figures.py
    figures/
      model-evaluation.png
      feature-importance.png
  README.md
  requirements.txt
```

## How To Run

1. Clone the repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Open and run the notebook:

```text
notebooks/01_hotel_booking_cancellation_analysis.ipynb
```

## Current Status

Portfolio-ready version. Possible next improvements:

- create a short portfolio website page,
- optionally package the preprocessing and model inference workflow as a Python script.

## Author

Created by **Lissette Valdes**.

- GitHub: [github.com/luthien4](https://github.com/luthien4)
- Portfolio: [luthien4.github.io/LissetteDoesWebPortfolio.github.io](https://luthien4.github.io/LissetteDoesWebPortfolio.github.io/)
- LinkedIn: [lissette-valdes-valdes-b987651](https://www.linkedin.com/in/lissette-valdes-valdes-b987651/)
