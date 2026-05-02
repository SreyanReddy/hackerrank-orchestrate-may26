import json
import re
import litellm

import retrieval

litellm.set_verbose = False

MODEL = "ollama/qwen2.5"  #       Local LLM because API Keys are expensive lol

#       Keywords that automatically trigger escalation before calling the LLM
ESCALATION_KEYWORDS = [
    "fraud", "unauthorized", "stolen", "hacked", "compromised",
    "identity theft", "phishing", "scam", "suspicious transaction",
    "chargeback", "legal action", "lawsuit", "attorney", "court",
    "compliance", "account closure", "permanent ban", "data breach",
    "gdpr", "data deletion", "double charged", "wrong charge",
    "refund not received", "cheating", "plagiarism", "academic dishonesty",
]

SYSTEM_PROMPT = """\
You are a senior support triage specialist handling tickets for three companies: HackerRank, Claude, and Visa.

For every ticket, you must return a JSON object with exactly these five fields:
{
  "status": "replied" | "escalated",
  "product_area": "a specific area like Billing, Account Access, Assessments, API, Card Services etc.",
  "request_type": "product_issue" | "feature_request" | "bug" | "invalid",
  "response": "your response to the user",
  "justification": "your internal reasoning for this decision"
}

Rules you MUST follow:
1.  ESCALATE if the issue involves: fraud, unauthorized charges, account compromise, legal threats,
    data breaches, or situations where you lack sufficient information to help safely.
2.  ESCALATE if the company is irrelevant or unidentifiable and the issue is sensitive.
3.  ESCALATE if you don't have enough information to help safely.
4.  REPLY for clear FAQs, how-to questions, feature requests, and general product issues
    that are addressed in the support documentation.
5.  REPLY with an out-of-scope message if the issue is clearly unrelated to any supported company
    (status: "replied", request_type: "invalid").
6.  Base your response ONLY on the support documentation provided — do not make up or hallucinate policies.
7.  Write responses in a structured, friendly format. Start with "Hi," or "Hello," Use numbered steps when explaining a process.
    For step-by-step instructions use \n to separate steps, not actual newlines.
    Keep the entire response as a single JSON-safe string.
    NEVER include raw HTML tags, code blocks, or double-quote characters inside your response.
    If you need to show a code example, describe it in plain English or use single quotes only.
    Use clear paragraph breaks between sections.
    End with an offer to help further if needed.
    Match the tone of a professional human support agent.
8.  product_area should be a specific category like "Billing", "Account Access",
    "Assessments", "API", "Card Services", "Payments", "Technical Issue", etc.
9.  Always respond in English unless the ticket itself is clearly and entirely written in another language. Do not switch languages based on retrieved documents.
10. CITATIONS belong in the "justification" field ONLY — never in the "response" field.
    The "response" field is what the user sees: write it in plain, friendly English with NO URLs.
    The "justification" field is internal notes: always include the article title and URL here.
    Correct justification format: "Based on 'Pause Subscription' (https://support.hackerrank.com/...) — the article confirms..."
    WRONG: putting any URL or "see article at ..." in the response field.
11. For ESCALATED tickets, keep the response to 2 sentences maximum —
    acknowledge the issue and tell them a specialist will follow up.
12. For vague or unclear tickets, ask one clarifying question rather than
    writing a long response covering every possibility.
13. If the support documentation contains a phone number or contact detail
    relevant to the issue, include it in your response field.
14. For long multi-step processes (more than 5 steps), summarize only the
    key steps briefly. Do NOT reproduce the full article verbatim.
    Keep your response under 300 words.

    I am providing a set of examples so that you can understand what i am trying to state. 
    Again, these are JUST EXAMPLES.

EXAMPLES OF CORRECT RESPONSES
==============================

Example 1 — Simple FAQ (Replied):
Issue: "How long do tests stay active in the system?"
Company: HackerRank
{
  "status": "replied",
  "product_area": "Assessments",
  "request_type": "product_issue",
  "response": "Hi,\n\nTests in HackerRank remain active indefinitely unless a start and end time are set. Without these, tests do not expire automatically.\n\nTo set expiration times, go to the test Settings, select the General section, and update the Start and End date fields. To keep a test active indefinitely, clear these fields by clicking the X icon.\n\nLet me know if you need further help!",
  "justification": "Based on 'Test Expiration and Scheduling' (https://support.hackerrank.com/hc/en-us/articles/...) — the article confirms tests remain active indefinitely without an end date set. Status: replied because this is a documented FAQ with a clear answer."
}

Example 2 — Step-by-step process (Replied):
Issue: "A candidate needs extra time on their assessment. How do I add it and reinvite them?"
Company: HackerRank
{
  "status": "replied",
  "product_area": "Assessments",
  "request_type": "product_issue",
  "response": "Hi,\n\nTo add extra time and reinvite a candidate:\n1. Go to the Tests tab and select the test.\n2. Open the Candidates tab.\n3. Check the box next to the candidate.\n4. Click More > Add Time Accommodation.\n5. Enter the percentage in multiples of five and click Save.\n\nThe candidate can then be reinvited with the updated time. Let me know if you need further assistance!",
  "justification": "Based on 'Adding Time Accommodations for Candidates' (https://support.hackerrank.com/hc/en-us/articles/...) — the article documents the exact steps for time accommodations. Status: replied because this is a documented how-to process."
}

Example 3 — Escalation (Escalated):
Issue: "Site is down and none of the pages are accessible"
Company: None
{
  "status": "escalated",
  "product_area": "Technical Issue",
  "request_type": "bug",
  "response": "Hi, we have received your report and are escalating this to our technical team immediately.",
  "justification": "A full site outage cannot be resolved through documentation and requires immediate human intervention."
}

Example 4 — Out of scope (Replied, invalid):
Issue: "What is the name of the actor in Iron Man?"
Company: None
{
  "status": "replied",
  "product_area": "General",
  "request_type": "invalid",
  "response": "Hi, I'm sorry but this request is outside the scope of our support. We can only assist with HackerRank, Claude, and Visa related issues.",
  "justification": "The ticket has no relation to any of the three supported companies."
}

Output ONLY the JSON object — no markdown, no explanation outside the JSON.
"""


def check_for_escalation(issue, subject):
    combined_text = (issue + " " + subject).lower()
    for keyword in ESCALATION_KEYWORDS:
        if keyword in combined_text:
            return True, f"Contains sensitive keyword: '{keyword}'"
    return False, ""


def format_docs_for_prompt(docs):
    if not docs:
        return "No relevant documentation found."

    formatted = []
    for i, doc in enumerate(docs, 1):
        company = doc.get("company", "Unknown")
        title = doc.get("title", "Untitled")
        url = doc.get("url", "")
        text = doc.get("text", "")

        preview = text[:450]  # Max characters in a doc set to 450 for preview
        formatted.append(
            f"[Article {i}] {company} — {title}\n"
            f"Source: {url}\n"
            f"{preview}"
        )
    return "\n\n".join(formatted)


def parse_llm_response(raw_text):
    # Remove ```json ... ``` from the top
    raw_text = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("`").strip()
    raw_text = raw_text.replace('\r\n', '\\n').replace('\r', '\\n')

    # Find the JSON object in the response
    match = re.search(r"\{[\s\S]+\}", raw_text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # If JSON parsing failed, try to pull out individual fields with regex
    def extract_field(field_name):
        # Allow escaped quotes inside the value with (?:[^"\\]|\\.)*
        m = re.search(rf'"{field_name}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_text)
        return m.group(1) if m else ""

    return {
        "status": extract_field("status") or "escalated",
        "product_area": extract_field("product_area") or "Unknown",
        "request_type": extract_field("request_type") or "product_issue",
        "response": extract_field("response") or "Unable to process this ticket automatically.",
        "justification": extract_field("justification") or "Could not parse LLM output.",
    }


def triage_ticket(issue, subject, company):

    # Step1 - escalation triggers
    should_escalate, escalation_reason = check_for_escalation(issue, subject)

    # Step2 - finding relevant docs
    company_filter = company if company and company.lower() != "none" else None
    query = (subject + " " + issue).strip()
    docs = retrieval.search(query, company=company_filter, top_k=3)

    # Step3 - Search everything if filtering by company does not give enough docs
    if len(docs) < 2 and company_filter:
        extra = retrieval.search(query, company=None, top_k=2)
        seen_urls = {d["url"] for d in docs}
        docs += [d for d in extra if d["url"] not in seen_urls]

    docs_text = format_docs_for_prompt(docs)

    # Step4 - building the prompt with ticket details and retrieved docs
    user_prompt = f"""\
TICKET
======
Company:  {company or "Unknown"}
Subject:  {subject or "(none)"}
Issue:
{issue}

ESCALATION CHECK: {"ESCALATE — " + escalation_reason if should_escalate else "No automatic triggers detected."}

RELEVANT SUPPORT ARTICLES
==========================
{docs_text}

Now analyze this ticket and return the JSON response.
"""

    # Step5 - LLM call
    try:
        response = litellm.completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1500,
            timeout=180,
        )
        raw_output = response.choices[0].message.content or ""
    except Exception as e:
        return {
            "status": "escalated",
            "product_area": "Unknown",
            "request_type": "product_issue",
            "response": "An internal error occurred. This ticket has been escalated for manual review.",
            "justification": f"LLM call failed: {e}",
        }

    result = parse_llm_response(raw_output)

    if should_escalate and result.get("status") != "escalated":
        result["status"] = "escalated"
        result["justification"] = f"Auto-escalated: {escalation_reason}. " + result.get("justification", "")

    # Defaults for when all five fields are not mentioned
    defaults = {
        "status": "escalated",
        "product_area": "Unknown",
        "request_type": "product_issue",
        "response": "Unable to determine a response.",
        "justification": "Missing field filled by fallback.",
    }
    for field, default_value in defaults.items():
        if not result.get(field):
            result[field] = default_value

    return result