from typing import Any, TypedDict


class HistoryTemp(TypedDict, total=False):
    """Type for history dictionaries."""

    scores: list[int]
    layout_init: dict[int | str, Any]  # tuple[int, int] | list[tuple[int, int]]]
    layout_final: dict[int | str, Any]  # tuple[int, int] | list[tuple[int, int]]]
