import re


ORGAN_NOTES = [
    "амбра", "белый кедр", "бергамот", "ваниль", "ветивер",
    "жасмин", "ирис", "мандарин", "мускус", "пачули", "роза", "сандал",
]


def build_synonym_map(catalog_notes: list[str]) -> dict[str, str]:
    synonym_map = {}
    for on in ORGAN_NOTES:
        pattern = re.compile(
            r"\b" + re.escape(on) + r"\b"
            + r"|"
            + r"\b" + re.escape(on) + r"[аеиоуыя]\b"
        )
        for cn in catalog_notes:
            if cn == on:
                continue
            if pattern.search(cn):
                synonym_map[cn] = on
    return synonym_map


def normalize_notes(notes_series, synonym_map: dict[str, str]):
    return notes_series.map(lambda n: synonym_map.get(n, n))
