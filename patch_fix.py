"""
Run from the project root:  python patch_fix.py
Safe to run twice. Makes a .bak copy of every file it changes.

Fixes:
  1. Gemini answers cut off mid-sentence (adds an output-token limit, handles
     partial responses, retries once if the answer/plan format is missing).
  2. Better answer format: lead-in sentence + numbered list + bold key terms,
     and "general requirements first" for eligibility questions.
  3. Garbled characters like "â€”" in .py and .md files.
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
SKIP = {"venv", ".venv", ".git", "__pycache__", "ugbs_welfare_db", "node_modules"}


def read(p):
    raw = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    return raw.replace("\r\n", "\n"), crlf


def write(p, text, crlf):
    shutil.copy(p, str(p) + ".bak")
    p.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


NEW_GEMINI = '''def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        config.GEMINI_MODEL,
        generation_config={"max_output_tokens": 2048, "temperature": 0.3},
    )
    response = model.generate_content(prompt)
    try:
        print(f"[llm_engine] gemini finish_reason: {response.candidates[0].finish_reason}")
    except Exception:
        pass
    try:
        return response.text.strip()
    except Exception:
        # Partial / blocked responses: keep whatever text parts exist.
        text = ""
        for cand in getattr(response, "candidates", None) or []:
            for part in getattr(cand.content, "parts", None) or []:
                text += getattr(part, "text", "") or ""
        return text.strip()


'''

NEW_FORMAT = (
    "Format the answer in Markdown: one short lead-in sentence, then a numbered\n"
    "list (or bullets) for requirements, steps or documents, with key terms in\n"
    "**bold**. For eligibility questions, list the GENERAL requirements first,\n"
    "then mention specific schemes only if the context covers them. Use up to\n"
    "about 200 words, warm and practical."
)


def patch_llm_engine():
    p = ROOT / "llm_engine.py"
    if not p.exists():
        print("llm_engine.py not found - run this from the project root.")
        return
    text, crlf = read(p)
    orig = text
    report = []

    if "generation_config" not in text:
        new, n = re.subn(r"def _call_gemini\(prompt: str\) -> str:\n.*?(?=def _call_anthropic)",
                         lambda m: NEW_GEMINI, text, flags=re.S)
        text = new
        report.append(("Gemini call (token limit + partial-response handling)", n))
    else:
        report.append(("Gemini call", "already patched"))

    if "max_tokens=500" in text:
        text = text.replace("max_tokens=500", "max_tokens=1200")
        report.append(("Anthropic max_tokens 500 -> 1200", 1))

    pat = re.compile(r"Keep it under 120 words, warm, and\s+practical\.")
    if pat.search(text):
        text = pat.sub(lambda m: NEW_FORMAT, text)
        report.append(("Answer format instructions", len(pat.findall(orig))))
    else:
        report.append(("Answer format instructions", "not found / already patched"))

    old = "raw = _strip_kb_tags(_call_llm(prompt))\n"
    if old in text and "one retry" not in text:
        text = text.replace(
            old,
            old + "        if _PLAN_DELIMITER not in raw:  # one retry if the format was missed\n"
                  "            raw = _strip_kb_tags(_call_llm(prompt))\n", 1)
        report.append(("Retry once when answer/plan format is missing", 1))
    else:
        report.append(("Retry", "not found / already patched"))

    if text != orig:
        write(p, text, crlf)
    for name, res in report:
        print(f"  llm_engine.py - {name}: {res}")


MOJIBAKE = {
    "â€”": "—", "â€“": "–", "â€™": "’", "â€˜": "‘", "â€œ": "“",
    "â€\u009d": "”", "â€¦": "…", "â€¢": "•", "Â·": "·", "Â ": " ",
}


def fix_mojibake():
    total = 0
    for p in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.md")):
        if SKIP & set(p.relative_to(ROOT).parts) or p.name == Path(__file__).name:
            continue
        try:
            text, crlf = read(p)
        except UnicodeDecodeError:
            continue
        new = text
        for bad, good in MOJIBAKE.items():
            new = new.replace(bad, good)
        if new != text:
            write(p, new, crlf)
            total += 1
            print(f"  fixed garbled characters in {p.relative_to(ROOT)}")
    left = [p for p in ROOT.rglob("*.py")
            if not (SKIP & set(p.relative_to(ROOT).parts)) and "â€" in p.read_text("utf-8", "ignore")
            and p.name != Path(__file__).name]
    print(f"  files fixed: {total}; files still containing 'â€': {[str(x.relative_to(ROOT)) for x in left] or 'none'}")


if __name__ == "__main__":
    print("Patching llm_engine.py ...")
    patch_llm_engine()
    print("Fixing garbled characters ...")
    fix_mojibake()
    print("Done. Next: git add -A ; git commit -m \"Fix answer cut-off and formatting\" ; git push")
