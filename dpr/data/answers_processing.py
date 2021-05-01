"""
Due to inconsistency in processing steps of the original DPR package, we need some heuristics to cover more positives.
"""


from functools import partial
from typing import List


def remove_leading_space(token: str, text: str) -> str:
    """
    Example: "Moloka ʻi" -> "Molokaʻi" with token="ʻ"
    """
    assert len(token) == 1
    text = text.replace(" " + token, token)
    return text


def remove_trailing_space(token: str, text: str) -> str:
    """
    Example: "€ 24 million" -> "€24 million" with token="€"
    """
    assert len(token) == 1
    text = text.replace(token + " ", token)
    return text


def remove_leading_and_trailing_spaces(token: str, text: str) -> str:
    """
    Example: "15.8 × 10 ly" -> "15.8×10 ly" with token="×"
    """
    assert len(token) == 1
    text = text.replace(" " + token + " ", token)
    return text


def remove_leading_or_trailing_space(token: str, text: str) -> List[str]:
    """
    Example:
    1) "−273.15 ° C" -> ["−273.15 °C", "−273.15° C]
    2) "90 °" -> ["90 °", "90°"]
    """
    assert len(token) == 1
    texts = []
    texts.append(text.replace(" " + token, token))
    texts.append(text.replace(token + " ", token))
    return texts


def remove_token(token: str, text: str) -> str:
    """
    Example: "Rowan University \u200e" -> "Rowan University  " with token="\u200e"
    """
    assert len(token) == 1
    text = text.replace(token, " ")
    return text


def handle_double_hyphen(text: str) -> List[str]:
    """
    Example: "in 1899 -- 1900" -> ["in 1899 - 1900", "in 1899-1900"]
    """
    texts = []
    texts.append(text.replace("--", "-"))
    texts.append(texts[0].replace(" - ", "-"))
    return texts


def handle_double_quote(text: str) -> str:
    """
    Example: "`` Cups ''" -> "  Cups  "
    """
    if "``" in text and "''" in text:
        return text.replace("``", " ").replace("''", " ")
    return text


handling_functions = [
    partial(remove_leading_space, token="ʻ"),

    partial(remove_trailing_space, token="£"),
    partial(remove_trailing_space, token="€"),
    partial(remove_trailing_space, token="₹"),
    partial(remove_trailing_space, token="−"),  # do not confuse this with "-"

    partial(remove_leading_and_trailing_spaces, token="×"),  # do not confuse this with "x"

    partial(remove_token, token="\u200e"),
    handle_double_quote,

    # These should be the last ones
    partial(remove_leading_or_trailing_space, token="°"),
    handle_double_hyphen,
]


def get_expanded_answer(text: str) -> List[str]:
    processed_texts = text

    for handling_func in handling_functions:
        if isinstance(processed_texts, (list, tuple)):
            processed_texts = [handling_func(text=t) for t in processed_texts]
            if isinstance(processed_texts[0], (list, tuple)):
                processed_texts = sum(processed_texts, [])  # flatten out
        else:
            processed_texts = handling_func(text=processed_texts)

    processed_texts = [t for t in processed_texts if t != text]  # remove duplicates
    processed_texts = list(set(processed_texts))  # remove duplicates
    processed_texts = [" ".join(t.split()) for t in processed_texts]  # remove leading/trailing space
    return processed_texts