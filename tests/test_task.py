from __future__ import annotations

import pytest

from bank_budget import task


def test_categories_and_system_prompt():
    assert len(task.CATEGORIES) == 17 and len(set(task.CATEGORIES)) == 17
    assert task.SYSTEM_PROMPT.startswith("You sort lines from a US bank statement")
    assert task.SYSTEM_PROMPT.endswith(", ".join(task.CATEGORIES) + ".")
    assert "[debit] (money out) or [credit] (money in)" in task.SYSTEM_PROMPT


def test_messages_and_chat_record():
    m = task.messages("[debit] X")
    assert [x["role"] for x in m] == ["system", "user"] and m[1]["content"] == "[debit] X"
    rec = task.chat_record("[debit] X", "Rent")
    assert rec["messages"][-1] == {"role": "assistant", "content": "Rent"}


@pytest.mark.parametrize(
    ("answer", "category"),
    [
        ("Rent", "Rent"),
        ("rent.", "Rent"),
        ("  Personal Care ", "Personal Care"),
        ("I think Groceries", "Groceries"),
        ("transfer of income", "Transfer"),  # the earliest category name wins
        ("Income/Transfer", "Income"),
        ("subscriptions", "other"),  # whole words only
        ("personal careless", "other"),
        ("", "other"),
    ],
)
def test_parse_category(answer, category):
    assert task.parse_category(answer) == category


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ("[debit] SQ *MOSSY ROCK BAKERY 0451 BOULDER CO", "mossy rock"),
        ("[debit] TST*SILVER LOTUS - FRESNO", "silver lotus"),
        ("[debit] PYPL *SQ *NOVA STUDIO", "nova studio"),  # two wrappers
        ("[credit] QUILLSTONE PAYROLL PPD ID: 7730", "quillstone payroll"),
        ("[debit] PURCHASE AUTHORIZED ON 09/14 CORNER DELI", "corner deli"),
        ("[debit] 24 HOUR FITNESS 0042 RENO NV", "hour fitness"),
        ("[debit] 1234", "_empty"),
    ],
)
def test_shop_key(line, key):
    assert task.shop_key(line) == key


def test_shop_groups_merge_spellings_of_one_shop():
    lines = [
        "[debit] traderjoes #512 austin",
        "[debit] TRADER JOES #88",
        "[debit] SQ *MOSSY ROCK BAKERY 0451",
        "[debit] MOSSY CREEK TAPS",
    ]
    groups = task.shop_groups(lines)
    assert groups[0] == groups[1]  # first 6 letters
    assert groups[2] == groups[3]  # first word
    assert groups[0] != groups[2]
