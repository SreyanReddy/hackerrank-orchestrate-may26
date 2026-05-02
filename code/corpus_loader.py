import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

COMPANY_MAP = {
    "hackerrank": "HackerRank",
    "claude": "Claude",
    "visa": "Visa",
}

def parse_frontmatter(text):

    if not text.startswith("---"):
        return {}, text
    
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    
    frontmatter_block = text[3:end].strip()
    body = text[end + 3:].strip()

    metadata = {}
    lines = frontmatter_block.splitlines()

    for line in lines:
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"')
            metadata[key] = value

    return metadata, body


def extract_title(metadata, body, filepath):

    if metadata.get("title"):
        return metadata["title"]

    heading_match = None
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("# "):
            heading_match = line[2:].strip()
            break

    if heading_match:
        return heading_match

    name = filepath.stem
    name = name.replace("-", " ")
    return name


def load_corpus():

    documents = []

    for md_file in DATA_DIR.rglob("*.md"):
        if md_file.name == "index.md":
            continue

        parts = md_file.relative_to(DATA_DIR).parts
        company_key = parts[0].lower()
        company = COMPANY_MAP.get(company_key, company_key.capitalize())

        if len(parts) > 2:
            category = parts[1]
        else:
            category = company_key

        raw_text = md_file.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        metadata, body = parse_frontmatter(raw_text)
        title = extract_title(metadata, body, md_file)
        source_url = metadata.get("source_url", "")

        clean_text = body

        # remove image markdown
        clean_text = re.sub(r"!\[.*?\]\(.*?\)", "", clean_text)

        # remove link markdown 
        clean_text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean_text)

        # remove formatting characters
        clean_text = re.sub(r"[`*_~#>]", "", clean_text)

        # remove whitespace
        clean_text = re.sub(r"\s+", " ", clean_text)

        clean_text = clean_text.strip()
        if len(clean_text) < 50:
            continue

        document = {
            "company": company,
            "category": category,
            "title": title,
            "url": source_url,
            "content": clean_text[:4000],
        }
        documents.append(document)

    print("Loaded", len(documents), "documents from data/")

    return documents


if __name__ == "__main__":

    docs = load_corpus()
    from collections import Counter

    companies = [d["company"] for d in docs]
    counts = Counter(companies)

    print(counts)