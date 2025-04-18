import requests
from bs4 import BeautifulSoup
import json
import re
import time
from urllib.parse import urlparse, parse_qs

def scrape_assessment_details(url, headers):
    """Scrape individual assessment page to extract attributes and description."""
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Initialize attributes
        remote_testing = "No"
        adaptive_irt = "No"
        duration = ""
        test_type = ""
        description = ""

        # The page structure needs to be inspected.
        # Try to find text blocks or labels containing the required info.

        text = soup.get_text(separator="|", strip=True).lower()

        if "remote testing" in text:
            remote_testing = "Yes"
        if "adaptive" in text or "irt" in text:
            adaptive_irt = "Yes"

        duration_match = re.search(r"(\d{1,3}\s?(minutes|min))", text)
        if duration_match:
            duration = duration_match.group(1)

        test_types = ["cognitive", "personality", "aptitude", "skills", "behavioral", "technical"]
        for tt in test_types:
            if tt in text:
                test_type = tt.capitalize()
                break

        # Extract description - try to find a paragraph or div with description
        # This is heuristic: look for meta description or first paragraph in main content
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            description = meta_desc["content"]
        else:
            # fallback: first paragraph in main content
            main_content = soup.find("main")
            if main_content:
                p = main_content.find("p")
                if p:
                    description = p.get_text(strip=True)

        return remote_testing, adaptive_irt, duration, test_type, description
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return "Unknown", "Unknown", "", "", ""

def is_valid_assessment_url(url):
    """Check if the URL is a valid assessment URL, excluding pagination or filter URLs."""
    parsed = urlparse(url)
    if parsed.query:
        # Exclude URLs with query parameters like start or type which are pagination or filters
        qs = parse_qs(parsed.query)
        if "start" in qs or "type" in qs:
            return False
    return True

def scrape_shl_product_catalog():
    url = "https://www.shl.com/solutions/products/product-catalog/"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SHL-Scraper/1.0)"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    assessments = []

    print("Page title:", soup.title.string if soup.title else "No title")

    main_content = soup.find("main")
    if not main_content:
        main_content = soup.body

    product_links = main_content.find_all("a", href=True)
    product_links = [a for a in product_links if "/solutions/products/" in a['href'] and a.get_text(strip=True)]

    print(f"Found {len(product_links)} product links")

    seen_urls = set()

    # Filter out non-assessment entries by name heuristics
    non_assessment_names = {"products", "product catalog", "solutions", "solutions products"}

    for a in product_links:
        link = a['href']
        if not link.startswith("http"):
            link = "https://www.shl.com" + link
        if link in seen_urls:
            continue
        if not is_valid_assessment_url(link):
            print(f"Skipping URL due to query params: {link}")
            continue
        seen_urls.add(link)

        name = a.get_text(strip=True).lower()
        if name in non_assessment_names or name.isdigit():
            print(f"Skipping name: {name}")
            continue

        # Scrape individual assessment page for details
        remote_testing, adaptive_irt, duration, test_type, description = scrape_assessment_details(link, headers)

        assessment = {
            "name": a.get_text(strip=True),
            "url": link,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
            "duration": duration,
            "test_type": test_type,
            "description": description
        }
        assessments.append(assessment)

        # Be polite and avoid hammering the server
        time.sleep(1)

    with open("shl_assessments.json", "w") as f:
        json.dump(assessments, f, indent=2)

    print(f"Scraped {len(assessments)} assessments and saved to shl_assessments.json")

if __name__ == "__main__":
    scrape_shl_product_catalog()
