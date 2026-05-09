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
- Random Forest model tuning,
- classification report and ROC-AUC evaluation,
- feature importance interpretation.

## Final Model Performance

Final Random Forest test performance:

| Metric | Value |
|---|---:|
| Accuracy | 0.84 |
| Weighted F1-score | 0.83 |
| ROC-AUC score | 0.87 |
| Canceled precision | 0.81 |
| Canceled recall | 0.61 |
| Canceled F1-score | 0.70 |

The model is relatively precise when it flags a booking as likely to be canceled, but it still misses some actual cancellations. This trade-off matters in a hotel business context because false positives and false negatives can have different operational and revenue-management costs.

## Model Evaluation Visuals

### Confusion Matrix and ROC Curve

![Random Forest model evaluation](reports/figures/model-evaluation.png)

### Feature Importance

![Top 10 feature importances](reports/figures/feature-importance.png)

## Key Predictors

The strongest cancellation-risk drivers included:

- deposit type,
- total special requests,
- lead time,
- previous cancellations,
- required parking spaces,
- market segment,
- booking changes,
- customer type,
- distribution channel.

These features suggest that cancellation behavior is shaped by customer commitment signals, booking timing, past behavior, and sales channel patterns.

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

Portfolio-ready draft. Next improvements:

- simplify the notebook for smoother reading,
- create a short portfolio website page,
- optionally package the preprocessing and model inference workflow as a Python script.

## Author

Created by **Lissette Valdes**.

- GitHub: [github.com/luthien4](https://github.com/luthien4)
- Portfolio: [luthien4.github.io/LissetteDoesWebPortfolio.github.io](https://luthien4.github.io/LissetteDoesWebPortfolio.github.io/)
- LinkedIn: [lissette-valdes-valdes-b987651](https://www.linkedin.com/in/lissette-valdes-valdes-b987651/)
