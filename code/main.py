import csv
import sys
from pathlib import Path

import retrieval
import triage
from tqdm import tqdm

INPUT_FILE = Path(__file__).parent.parent / "support_tickets" / "support_tickets.csv"
OUTPUT_FILE = Path(__file__).parent.parent / "support_tickets" / "output.csv"

OUTPUT_COLUMNS = ["Issue", "Subject", "Company", "Status", "Product Area", "Request Type", "Response", "Justification"]


def process_tickets():

    # loading and building index prior to ticket issuance
    print("Loading retrieval index...")
    index = retrieval.get_index()
    index.build()

    # reading tickets
    rows = list(csv.DictReader(INPUT_FILE.open(encoding="utf-8")))
    print(f"\nProcessing {len(rows)} tickets...\n")

    results = []

    for row in tqdm(rows, desc="Triaging", unit="ticket"):
        issue = (row.get("Issue") or "").strip()
        subject = (row.get("Subject") or "").strip()
        company = (row.get("Company") or "").strip()

        # Case for empty tickets
        if not issue:
            result = {
                "Status": "Escalated",
                "Product Area": "Unknown",
                "Request Type": "invalid",
                "Response": "No issue text was provided.",
                "Justification": "Empty ticket — nothing to triage.",
            }
        else:
            raw = triage.triage_ticket(issue, subject, company)
            result = {
                "Status": raw.get("status", "escalated").capitalize(),
                "Product Area": raw.get("product_area", "Unknown"),
                "Request Type": raw.get("request_type", "product_issue"),
                "Response": raw.get("response", ""),
                "Justification": raw.get("justification", ""),
            }

        results.append({
            "Issue": issue,
            "Subject": subject,
            "Company": company,
            **result,
        })

        # Printing a short summary for eadch ticket processed
        if result["Status"].lower() == "replied":
            status_marker = "+"
        else:
            status_marker = "^"
        print(
            f"  {status_marker} [{result['Status'].upper():9s}]"
            f" [{result['Request Type']:15s}]"
            f" {result['Product Area'][:25]:25s}"
            f" | {(subject or issue)[:50]}"
        )

    # writing results to output.csv
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    process_tickets()