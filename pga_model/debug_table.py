"""
Prints the raw structure of a stats page's table so we can see exactly
which column holds what, instead of guessing.

Usage:
    python debug_table.py
"""

from playwright.sync_api import sync_playwright

URL = "https://www.pgatour.com/stats/detail/103"  # GIR% page

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(3000)

    print(f"Page title: {page.title()}")
    print(f"Final URL: {page.url}")

    # Take a screenshot no matter what happens next, so we can see
    # exactly what the browser loaded.
    page.screenshot(path="debug_screenshot.png", full_page=True)
    print("Saved debug_screenshot.png")

    try:
        page.wait_for_selector("table tbody tr", timeout=30000)
        count_1 = len(page.query_selector_all("table tbody tr"))
        print(f"Row count right after selector found: {count_1}")

        page.wait_for_timeout(3000)
        count_2 = len(page.query_selector_all("table tbody tr"))
        print(f"Row count after waiting 3 more seconds: {count_2}")

        page.mouse.wheel(0, 5000)
        page.wait_for_timeout(2000)
        count_3 = len(page.query_selector_all("table tbody tr"))
        print(f"Row count after scrolling: {count_3}")

        print("\n--- Raw HTML of first 3 rows ---")
        rows = page.query_selector_all("table tbody tr")
        for i, row in enumerate(rows[:3]):
            print(f"\nRow {i}:")
            print(row.inner_html()[:800])
    except Exception as e:
        print(f"\nNo table found: {e}")
        print("\n--- First 2000 chars of page content ---")
        print(page.content()[:2000])

    browser.close()
