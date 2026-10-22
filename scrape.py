
import requests
import pandas as pd
import time

from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ==========================================
# STEP 1: WEB SCRAPING
# Maximum of 200 unique books
# ==========================================

BASE_URL = "https://books.toscrape.com/"
MAX_BOOKS = 200

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# Create a session for webpage requests
session = requests.Session()
session.headers.update(HEADERS)


def get_soup(url):
    """Download a webpage and convert it to BeautifulSoup."""
    response = session.get(url, timeout=30)
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def get_rating(book):
    """Convert a star rating into a number from 1 to 5."""
    rating_element = book.select_one("p.star-rating")

    if rating_element is None:
        return None

    rating_classes = rating_element.get("class", [])

    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    for word, number in rating_map.items():
        if word in rating_classes:
            return number

    return None


def scrape_books():
    """Collect a maximum of 200 unique books."""
    books = []
    seen_urls = set()

    # Get the homepage
    soup = get_soup(BASE_URL)

    # Collect category links in displayed order
    category_links = soup.select(
        "div.side_categories ul li ul li a"
    )

    if not category_links:
        raise RuntimeError("No category links found.")

    for category_link in category_links:

        if len(books) >= MAX_BOOKS:
            break

        category = category_link.get_text(strip=True)

        category_url = urljoin(
            BASE_URL,
            category_link.get("href", "")
        )

        next_page = category_url

        while next_page and len(books) < MAX_BOOKS:

            print(
                f"Scraping {category}: "
                f"{len(books)}/{MAX_BOOKS} books collected"
            )

            page = get_soup(next_page)

            book_cards = page.select("article.product_pod")

            for card in book_cards:

                if len(books) >= MAX_BOOKS:
                    break

                title_element = card.select_one("h3 a")
                price_element = card.select_one(
                    "p.price_color"
                )

                if title_element is None or price_element is None:
                    continue

                title = (
                    title_element.get("title")
                    or title_element.get_text(strip=True)
                )

                book_url = urljoin(
                    next_page,
                    title_element.get("href", "")
                )

                # Avoid collecting the same book twice
                if book_url in seen_urls:
                    continue

                seen_urls.add(book_url)

                price_text = price_element.get_text(strip=True)

                # Website prices are displayed in GBP (£)
                try:
                    price = float(
                        price_text.replace("£", "").strip()
                    )
                except ValueError:
                    price = None

                rating = get_rating(card)

                availability_element = card.select_one(
                    "p.instock.availability"
                )

                availability = (
                    availability_element.get_text(
                        " ", strip=True
                    )
                    if availability_element
                    else "Unknown"
                )

                books.append({
                    "title": title,
                    "price_gbp": price,
                    "rating": rating,
                    "category": category,
                    "availability": availability,
                    "book_url": book_url
                })

            # Find the next page in the current category
            next_link = page.select_one("li.next a")

            if next_link and next_link.get("href"):
                next_page = urljoin(
                    next_page,
                    next_link["href"]
                )
            else:
                next_page = None

            time.sleep(0.15)

    # Final safety check: maximum of 200 records
    books = books[:MAX_BOOKS]

    return books


def main():
    print("BOOK MARKET ANALYTICS")
    print("Starting web scraping...")
    print(f"Maximum books: {MAX_BOOKS}")
    print("-" * 40)

    try:
        books = scrape_books()

        if not books:
            print("No books were collected.")
            return

        # Convert the collected books into a DataFrame
        dataframe = pd.DataFrame(books)

        # Save the dataset
        dataframe.to_csv(
            "books_raw.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print("\nSCRAPING COMPLETE")
        print(f"Unique books collected: {len(dataframe)}")
        print("File created: books_raw.csv")

        print("\nFirst five books:")
        print(dataframe.head().to_string(index=False))

        print("\nDataset information:")
        print(dataframe.info())

    except requests.RequestException as error:
        print(f"Website connection error: {error}")

    except Exception as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
