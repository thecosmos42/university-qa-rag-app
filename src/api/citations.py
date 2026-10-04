import re


def parse_cited_indices(text: str) -> set[int]:
    """Return every SOURCE number cited in the text, including grouped forms
    such as [SOURCE 1, SOURCE 2] and [SOURCE 1, 3]. Page numbers are ignored."""
    cited: set[int] = set()
    for group in re.findall(r"\[[^\]]*SOURCE[^\]]*\]", text or "", flags=re.IGNORECASE):
        group = re.sub(r"page\s*\d+", "", group, flags=re.IGNORECASE)
        cited.update(int(n) for n in re.findall(r"\d+", group))
    return cited
