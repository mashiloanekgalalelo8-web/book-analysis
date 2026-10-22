
import pandas as pd
from pathlib import Path

# ==========================================
# STEP 2: DATA CLEANING
# ==========================================

INPUT_FILE = Path("books_raw.csv")
OUTPUT_FILE = Path("books_cleaned.csv")


def clean_books():
    """Clean the raw book dataset and save the results."""

    # 1. Check whether the raw dataset exists
    if not INPUT_FILE.exists():
        print("ERROR: books_raw.csv was not found.")
        print("Run step1_web_scraping.py first.")
        return

    # 2. Load the raw dataset
    df = pd.read_csv(INPUT_FILE)

    print("=" * 50)
    print("STEP 2: DATA CLEANING")
    print("=" * 50)

    print(f"\nOriginal number of rows: {len(df)}")

    # 3. Display missing values before cleaning
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # 4. Remove spaces around text
    text_columns = [
        "title",
        "category",
        "availability",
        "book_url"
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

            # Treat blank strings as missing
            df[column] = df[column].replace(
                "", pd.NA
            )

    # 5. Convert price and rating to numeric values
    for column in ["price_gbp", "rating"]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # 6. Remove duplicate books
    if "book_url" in df.columns:
        valid_urls = df["book_url"].notna()

        duplicates = df.loc[
            valid_urls
        ].duplicated(
            subset=["book_url"],
            keep="first"
        )

        duplicate_indexes = duplicates[
            duplicates
        ].index

        df = df.drop(index=duplicate_indexes)

    else:
        df = df.drop_duplicates(
            subset=["title", "category"],
            keep="first"
        )

    print(
        "\nRows after duplicate removal: "
        f"{len(df)}"
    )

    # 7. Remove records without a title or price
    required_columns = [
        column
        for column in ["title", "price_gbp"]
        if column in df.columns
    ]

    df = df.dropna(
        subset=required_columns
    )

    # 8. Remove invalid prices
    df = df[df["price_gbp"] >= 0]

    # 9. Keep ratings from 1 to 5, or missing ratings
    if "rating" in df.columns:
        valid_rating = (
            df["rating"].between(1, 5)
            | df["rating"].isna()
        )

        df = df[valid_rating]

    # 10. Standardise category and availability
    for column in ["category", "availability"]:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("Unknown")
                .replace("", "Unknown")
            )

    # 11. Make ratings whole numbers where present
    if "rating" in df.columns:
        df["rating"] = df["rating"].astype("Int64")

    # 12. Keep the dataset within the 200-book limit
    df = df.head(200).reset_index(drop=True)

    # 13. Save the cleaned dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # 14. Display cleaning results
    print("\nCLEANING COMPLETE")
    print("-" * 50)
    print(f"Final number of rows: {len(df)}")
    print(f"Final number of columns: {len(df.columns)}")

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    print("\nCleaned dataset preview:")
    print(df.head(10).to_string(index=False))

    print(f"\nCleaned dataset saved as: {OUTPUT_FILE}")

    print("\nColumn data types:")
    print(df.dtypes)

    return df


if __name__ == "__main__":
    clean_books()
