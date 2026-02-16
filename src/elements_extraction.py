import re
import elements_hardcode
def normalize_text(text) -> str:
    return re.sub(r"[^\w\s]", " ", text.lower())

def detect_elements(chunk_text, elements):
    found = set()
    text = normalize_text(chunk_text)

    for key, name in elements.items():
        if key in text:
            found.add(name)

    return list(found)