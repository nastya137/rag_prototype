RULE_TRIGGERS = [
    "должен", "должны", "следует", "необходимо",
    "допускается", "не допускается",
    "запрещается",
    "выполняется", "оформляется",
    "указывается", "приводится",
    "печатается", "располагается",
    "нумеруется", "нумеруются",
    "размер", "шрифт", "интервал", "поля",
]

RULE_TYPES = {
    "шрифт": "font",
    "размер": "font_size",
    "пт": "font_size",
    "интервал": "line_spacing",
    "поля": "margins",
    "мм": "margins",
    "нумер": "numbering",
    "располаг": "layout",
    "оформля": "formatting",
}

import elements_extraction
import re

def split_into_sentences(text: str):
    text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 25]

def detect_rules(chunk_text: str):
    rules = []
    sentences = split_into_sentences(chunk_text)

    for sent in sentences:
        sent_low = sent.lower()

        if any(trigger in sent_low for trigger in RULE_TRIGGERS):
            # отсеиваем мусор типа "см. раздел"
            if len(sent_low) < 300:
                rules.append(sent.strip())

    return rules

def detect_rule_type(rule_text: str):
    text = rule_text.lower()
    for key, rule_type in RULE_TYPES.items():
        if key in text:
            return rule_type
    return "other"

def extract_rules_from_chunk(chunk_text, chunk_id, elements):
    raw_rules = detect_rules(chunk_text)
    extracted = []

    for rule in raw_rules:
        rule_type = detect_rule_type(rule)
        elems = elements_extraction.detect_elements(rule, elements)

        extracted.append({
            "rule_id": f"rule_{chunk_id}_{abs(hash(rule)) % 10000}",
            "text": rule,
            "type": rule_type,
            "elements": elems,
            "chunk_id": chunk_id
        })

    return extracted
