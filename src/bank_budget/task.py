"""The task: file one US bank-statement line under one of 17 budget categories.

Every line starts with ``[debit]`` (money out) or ``[credit]`` (money in). The model gets
the same system prompt before and after fine-tuning, the line as the user turn, and
answers with one category name. Everything that defines what the model reads and how its
answer is scored lives here, together with the *shop key* used to keep whole shops out of
training (see ``shop_key`` and :mod:`bank_budget.data.split`).
"""

from __future__ import annotations

import re
from typing import Any

CATEGORIES = (
    "Restaurants",
    "Groceries",
    "Shopping",
    "Transportation",
    "Entertainment",
    "Utilities",
    "Subscription",
    "Healthcare",
    "Insurance",
    "Mortgage",
    "Rent",
    "Travel",
    "Education",
    "Personal Care",
    "Transfer",
    "Income",
    "Fees",
)
NO_CATEGORY = "other"  # an answer that names no category; always wrong
SYSTEM_PROMPT = (
    "You sort lines from a US bank statement into budget categories. Each line starts with "
    "[debit] (money out) or [credit] (money in). Reply with exactly one category from this "
    "list and nothing else: " + ", ".join(CATEGORIES) + "."
)


def messages(line: str, category: str | None = None) -> list[dict[str, str]]:
    """System prompt + the line; with ``category``, the answer as the assistant turn."""
    out = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": line}]
    if category is not None:
        out.append({"role": "assistant", "content": category})
    return out


def chat_record(line: str, category: str) -> dict[str, Any]:
    """One MLX-LM chat training example."""
    return {"messages": messages(line, category)}


def render_generation_prompt(tokenizer: Any, line: str) -> str:
    """The model's own chat template with the answer turn opened (as text)."""
    return str(
        tokenizer.apply_chat_template(messages(line), add_generation_prompt=True, tokenize=False)
    )


_BY_LOWER = {c.lower(): c for c in CATEGORIES}


def parse_category(answer: str) -> str:
    """The category an answer names, or ``other``.

    An answer that *is* a category name (any case, trailing dots/spaces ignored) wins;
    otherwise the category whose name occurs earliest as a whole word.
    """
    s = answer.strip().lower()
    if s.rstrip(". ") in _BY_LOWER:
        return _BY_LOWER[s.rstrip(". ")]
    found: tuple[int, str] | None = None
    for low, name in _BY_LOWER.items():
        m = re.search(r"\b" + re.escape(low) + r"\b", s)
        if m and (found is None or m.start() < found[0]):
            found = (m.start(), name)
    return found[1] if found else NO_CATEGORY


# --------------------------------------------------------------------------- shop key

# Payment wrappers banks and payment processors put in front of the shop name.
_WRAPPERS = (
    r"preapproved payment bill user payment:",
    r"preauthorized withdrawal to",
    r"subscription payment:",
    r"payment to:",
    r"buyer payment:",
    r"google \*",
    r"pypl \*",
    r"preapproved payment:",
    r"express checkout payment:",
    r"withdrawal from",
    r"preauthorized deposit from",
    r"paypal inst xfer",
    r"paypal \*",
    r"pypl\*",
    r"pp\*",
    r"sq \*",
    r"tst\*",
    r"clv\*",
    r"orig co name:",
    r"pos debit",
    r"pos purchase",
    r"debit card purchase",
    r"purchase authorized on \d\d/\d\d",
    r"recurring payment",
    r"ach debit",
    r"ach credit",
)
_WRAPPER = re.compile(r"^(?:" + "|".join(_WRAPPERS) + r")\s*")
_SIGN = re.compile(r"^\[(debit|credit)\]\s*")
_CUT = re.compile(r"[;*#]|\bppd id\b|\bweb id\b|\bentry descr\b")
_DIGIT_HEAD = re.compile(r"\s*[\d-]+[a-z]*\s*[a-z&']*")
_DIGIT_SHOP = re.compile(r"\s*\d+-?[a-z]|\s*\d+ [a-z]{3,}")


def shop_key(line: str, words: int = 2) -> str:
    """The shop a line belongs to, as its first ``words`` lower-case words.

    ``[debit] SQ *MOSSY ROCK BAKERY 0451 BOULDER CO`` → ``mossy rock``: the sign, up to
    three payment wrappers, ids after ``; * #`` / ``ppd id`` / ``web id`` / ``entry descr``
    and everything from the first digit on (store numbers, addresses, dates) are removed —
    except at the start of names that begin with digits (``24 HOUR FITNESS 0042`` →
    ``hour fitness``, not empty). Only letters, ``&``, ``'`` and ``.`` form words.
    """
    s = _SIGN.sub("", line.lower())
    for _ in range(3):
        stripped = _WRAPPER.sub("", s).strip()
        if stripped == s:
            break
        s = stripped
    s = _CUT.split(s)[0]
    m = _DIGIT_HEAD.match(s)
    head = m.group(0) if m and _DIGIT_SHOP.match(s) else ""
    s = head + re.sub(r"\d.*$", "", s[len(head) :])
    found = [w.strip(".'") for w in re.findall(r"[a-z&'.]+", s) if w.strip(".'")]
    return " ".join(found[:words]) or "_empty"


def _first_word(line: str) -> str:
    return shop_key(line, words=1).split(".")[0].replace("'", "").replace("&", "")


def _squashed(line: str, letters: int) -> str:
    return re.sub(r"[^a-z]", "", shop_key(line, words=4))[:letters]


def shop_groups(lines: list[str]) -> list[str]:
    """One group id per line; lines of the same shop share it.

    Two lines are the same shop when their shop keys share the first word, or the first
    6 letters with spaces removed (``TRADERJOES`` / ``TRADER JOES``). This merges somewhat
    too much on purpose: a held-out shop must really be new.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    firsts = []
    for line in lines:
        word, letters = "w:" + _first_word(line), "s:" + _squashed(line, 6)
        firsts.append(word)
        if not letters.endswith(":"):
            parent[find(word)] = find(letters)
    return [find(w) for w in firsts]
