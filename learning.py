
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# STEP 4: MACHINE LEARNING
# Predict book prices using Linear Regression
# ==========================================

INPUT_FILE = Path("books_cleaned.csv")
OUTPUT_PREDICTIONS = Path("test_predictions.csv")
OUTPUT_RESULTS = Path("model_results.txt")
OUTPUT_CHART = Path("charts/model_predictions.png")

OUTPUT_CHART.parent.mkdir(
    parents=True,
    exist_ok=True
)


def train_price_model():

    # 1. Load the cleaned dataset
    if not INPUT_FILE.exists():
        print("ERROR: books_cleaned.csv was not found.")
        print("Run Steps 1 and 2 first.")
        return

    df = pd.read_csv(INPUT_FILE)

    print("=" * 55)
    print("STEP 4: MACHINE LEARNING")
    print("=" * 55)

    print(f"\nBooks loaded: {len(df)}")

    # 2. Check required columns
    required_columns = [
        "price_gbp",
        "rating",
        "category",
        "stock_quantity"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    # The scraper in Step 1 does not collect stock_quantity.
    # If absent, create it as missing; the model pipeline
    # will handle the missing values.
    if "stock_quantity" in missing_columns:
        df["stock_quantity"] = np.nan
        missing_columns.remove("stock_quantity")

    if missing_columns:
        print(
            "ERROR: Missing required columns:",
            missing_columns
        )
        return

    # 3. Convert numerical fields
    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    df["stock_quantity"] = pd.to_numeric(
        df["stock_quantity"],
        errors="coerce"
    )

    # 4. Remove rows without a valid target price
    df = df.dropna(subset=["price_gbp"])
    df = df[df["price_gbp"] >= 0]

    # Keep only plausible ratings; missing ratings are allowed.
    df.loc[
        ~df["rating"].between(1, 5),
        "rating"
    ] = np.nan

    df["category"] = (
        df["category"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    if len(df) < 10:
        print(
            "ERROR: At least 10 books with valid prices "
            "are required for this exercise."
        )
        return

    # 5. Select features (X) and target (y)
    feature_columns = [
        "rating",
        "category",
        "stock_quantity"
    ]

    X = df[feature_columns]
    y = df["price_gbp"]

    print(f"Usable records: {len(df)}")
    print("\nInput features:")
    print(feature_columns)
    print("Target variable: price_gbp")

    # 6. Build preprocessing pipelines
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                ["rating", "stock_quantity"]
            ),
            (
                "categorical",
                categorical_pipeline,
                ["category"]
            )
        ]
    )

    # 7. Create the Linear Regression model
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("regression", LinearRegression())
        ]
    )

    # 8. Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # 9. Train the model
    model.fit(X_train, y_train)

    print("\nModel training completed.")

    # 10. Predict prices for the test set
    predictions = model.predict(X_test)

    # 11. Evaluate the model
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nMODEL EVALUATION")
    print("-" * 40)
    print(f"Mean Absolute Error (MAE): £{mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): £{rmse:.2f}")
    print(f"R-squared (R²): {r2:.4f}")

    # 12. Save predictions
    results_df = pd.DataFrame({
        "actual_price_gbp": y_test.to_numpy(),
        "predicted_price_gbp": predictions,
        "absolute_error_gbp": np.abs(
            y_test.to_numpy() - predictions
        )
    })

    results_df.to_csv(
        OUTPUT_PREDICTIONS,
        index=False
    )

    # 13. Save evaluation metrics
    with open(
        OUTPUT_RESULTS,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("BOOK MARKET ANALYTICS\n")
        file.write("LINEAR REGRESSION RESULTS\n")
        file.write("=" * 40 + "\n")

        file.write(f"Records used: {len(df)}\n")
        file.write(f"Training records: {len(X_train)}\n")
        file.write(f"Testing records: {len(X_test)}\n")

        file.write(f"MAE: £{mae:.2f}\n")
        file.write(f"RMSE: £{rmse:.2f}\n")
        file.write(f"R-squared: {r2:.4f}\n")

        file.write("\nTarget: book price in GBP\n")
        file.write(
            "Features: rating, category, stock quantity\n"
        )

    # 14. Create actual-versus-predicted graph
    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.8
    )

    # Reference line: perfect predictions
    minimum = min(
        float(y_test.min()),
        float(np.min(predictions))
    )

    maximum = max(
        float(y_test.max()),
        float(np.max(predictions))
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.title("Actual vs Predicted Book Prices")
    plt.xlabel("Actual Price (£)")
    plt.ylabel("Predicted Price (£)")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_CHART,
        dpi=150
    )

    plt.close()

    # 15. Display a sample of predictions
    print("\nSAMPLE PREDICTIONS")
    print("-" * 40)

    print(
        results_df.head(10).round(2).to_string(index=False)
    )

    print("\nSTEP 4 COMPLETED")
    print(f"Predictions saved to: {OUTPUT_PREDICTIONS}")
    print(f"Metrics saved to: {OUTPUT_RESULTS}")
    print(f"Chart saved to: {OUTPUT_CHART}")

    print(
        "\nNote: these metrics measure performance on the "
        "held-out test data. They may vary if the dataset changes."
    )


if __name__ == "__main__":
    train_price_model()
