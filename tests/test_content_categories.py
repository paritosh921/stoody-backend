import pytest

from core.content_categories import (
    ensure_content_category_ids_preserved,
    find_content_category,
    normalize_content_categories,
)


def test_categories_are_institution_defined_and_canonicalized():
    categories = normalize_content_categories(
        [
            {"id": "weekly-test", "name": "  Weekly   Test  ", "active": True},
            {"id": "half-yearly", "name": "Half Yearly", "active": False},
        ],
        strict=True,
    )

    assert categories == [
        {"id": "weekly-test", "name": "Weekly Test", "active": True},
        {"id": "half-yearly", "name": "Half Yearly", "active": False},
    ]
    assert find_content_category(categories, "WEEKLY-TEST", require_active=True) == categories[0]


def test_duplicate_names_are_rejected_case_insensitively():
    with pytest.raises(ValueError, match="Duplicate content category name"):
        normalize_content_categories(
            [
                {"id": "weekly", "name": "Weekly Test"},
                {"id": "weekly-2", "name": "weekly test"},
            ],
            strict=True,
        )


def test_archived_or_unknown_category_cannot_be_assigned():
    categories = normalize_content_categories(
        [{"id": "archived", "name": "Old Pattern", "active": False}],
        strict=True,
    )

    with pytest.raises(ValueError, match="archived"):
        find_content_category(categories, "archived", require_active=True)
    with pytest.raises(ValueError, match="not configured"):
        find_content_category(categories, "missing", require_active=True)


def test_existing_category_ids_must_be_archived_instead_of_removed():
    existing = [{"id": "term-one", "name": "Term One", "active": True}]
    archived = [{"id": "term-one", "name": "First Term", "active": False}]

    ensure_content_category_ids_preserved(existing, archived)

    with pytest.raises(ValueError, match="archive them instead"):
        ensure_content_category_ids_preserved(existing, [])
