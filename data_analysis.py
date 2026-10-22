
import pandas as pd
import matplotlib

# Allows graphs to be saved without opening separate windows
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ==========================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ==========================================

INPUT_FILE = Path("books_cleaned.csv")
CHART_FOLDER = Path("charts")

CHART_FOLDER.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def analyse_books():

    # 1. Load the cleaned dataset
    if not INPUT_FILE.exists():
        print("ERROR: books_cleaned.csv was not found.")
        print("Run Step 2 first.")
        return

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        print("ERROR: The cleaned dataset is empty.")
        return

    # 2. Check that the required columns exist
    required_columns = [
        "title",
        "price_gbp",
        "rating",
        "category",
        "availability"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("Missing columns:", missing_columns)
        return

    # 3. Convert numeric columns
    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    df = df.dropna(subset=["price_gbp"])

    if df.empty:
        print("No valid book prices are available.")
        return

    print("=" * 55)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    # ==========================================
    # PART A: SUMMARY STATISTICS
    # ==========================================

    print("\n1. DATASET OVERVIEW")
    print("-" * 40)

    print(f"Total books analysed: {len(df)}")
    print(f"Number of categories: {df['category'].nunique()}")

    print("\n2. PRICE STATISTICS")
    print("-" * 40)

    print(f"Average price: £{df['price_gbp'].mean():.2f}")
    print(f"Median price: £{df['price_gbp'].median():.2f}")
    print(f"Lowest price: £{df['price_gbp'].min():.2f}")
    print(f"Highest price: £{df['price_gbp'].max():.2f}")
    print(f"Price standard deviation: £{df['price_gbp'].std():.2f}")

    print("\n3. RATING STATISTICS")
    print("-" * 40)

    print(f"Books with ratings: {df['rating'].notna().sum()}")

    if df["rating"].notna().any():
        print(f"Average rating: {df['rating'].mean():.2f} / 5")
    else:
        print("Average rating: unavailable")

    print("\n4. NUMBER OF BOOKS PER CATEGORY")
    print("-" * 40)

    print(
        df["category"]
        .value_counts()
        .to_string()
    )

    print("\n5. STOCK STATUS")
    print("-" * 40)

    print(
        df["availability"]
        .value_counts()
        .to_string()
    )

    # Save numerical summary
    summary = df[["price_gbp", "rating"]].describe()

    summary.to_csv("eda_summary.csv")

    # ==========================================
    # GRAPH 1: PRICE DISTRIBUTION
    # ==========================================

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="price_gbp",
        bins=20,
        kde=True
    )

    plt.title("Distribution of Book Prices")
    plt.xlabel("Book Price (£)")
    plt.ylabel("Number of Books")
    plt.tight_layout()

    plt.savefig(
        CHART_FOLDER / "graph1_price_distribution.png",
        dpi=150
    )

    plt.close()

    # ==========================================
    # GRAPH 2: RATING DISTRIBUTION
    # ==========================================

    rating_data = df.dropna(subset=["rating"]).copy()

    if not rating_data.empty:

        rating_data = rating_data[
            rating_data["rating"].between(1, 5)
        ]

        rating_counts = (
            rating_data["rating"]
            .astype(int)
            .value_counts()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
        )

        plt.figure(figsize=(8, 5))

        sns.barplot(
            x=rating_counts.index,
            y=rating_counts.values
        )

        plt.title("Distribution of Book Ratings")
        plt.xlabel("Star Rating")
        plt.ylabel("Number of Books")
        plt.xticks([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
        plt.tight_layout()

        plt.savefig(
            CHART_FOLDER / "graph2_rating_distribution.png",
            dpi=150
        )

        plt.close()

    else:
        print("\nGraph 2 skipped: no ratings available.")

    # ==========================================
    # GRAPH 3: AVERAGE PRICE BY CATEGORY
    # ==========================================

    category_prices = (
        df.groupby("category", dropna=False)
        .agg(
            average_price=("price_gbp", "mean"),
            number_of_books=("title", "count")
        )
        .sort_values("average_price", ascending=False)
    )

    # Limit the graph to the 10 categories
    # with the most books, for readability.
    top_categories = (
        df["category"]
        .value_counts()
        .head(10)
        .index
    )

    plot_data = category_prices.loc[
        category_prices.index.isin(top_categories)
    ].sort_values("average_price", ascending=True)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=plot_data.reset_index(),
        x="average_price",
        y="category"
    )

    plt.title("Average Book Price by Category")
    plt.xlabel("Average Price (£)")
    plt.ylabel("Book Category")
    plt.tight_layout()

    plt.savefig(
        CHART_FOLDER / "graph3_category_prices.png",
        dpi=150
    )

    plt.close()

    print("\nAVERAGE PRICE BY CATEGORY")
    print("-" * 40)

    print(
        category_prices
        .round(2)
        .to_string()
    )

    # Save category statistics
    category_prices.to_csv("category_summary.csv")

    # ==========================================
    # GRAPH 4: STOCK AVAILABILITY
    # ==========================================

    stock_counts = (
        df["availability"]
        .fillna("Unknown")
        .value_counts()
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=stock_counts.index,
        y=stock_counts.values
    )

    plt.title("Book Stock Availability")
    plt.xlabel("Availability Status")
    plt.ylabel("Number of Books")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        CHART_FOLDER / "graph4_stock_availability.png",
        dpi=150
    )

    plt.close()

    # ==========================================
    # COMPLETE
    # ==========================================

    print("\nSTEP 3 COMPLETED SUCCESSFULLY")
    print("-" * 40)
    print(f"Graphs saved in: {CHART_FOLDER.resolve()}")
    print("Summary saved as: eda_summary.csv")
    print("Category data saved as: category_summary.csv")

    print("\nGenerated chart files:")

    for chart in sorted(CHART_FOLDER.glob("graph*.png")):
        print(f"- {chart.name}")


if __name__ == "__main__":
    analyse_books()
