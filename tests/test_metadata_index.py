from rag.metadata_index import MetadataIndex


def test_single_filter():
    idx = MetadataIndex()

    idx.add("1", {"category": "electronics"})
    idx.add("2", {"category": "fashion"})

    result = idx.filter_ids(
        {"category": "electronics"}
    )

    assert result == {"1"}


def test_multiple_filters():
    idx = MetadataIndex()

    idx.add(
        "1",
        {
            "category": "electronics",
            "brand": "sony",
        },
    )

    idx.add(
        "2",
        {
            "category": "electronics",
            "brand": "samsung",
        },
    )

    result = idx.filter_ids(
        {
            "category": "electronics",
            "brand": "sony",
        }
    )

    assert result == {"1"}


def test_no_match():
    idx = MetadataIndex()

    idx.add(
        "1",
        {"category": "electronics"}
    )

    result = idx.filter_ids(
        {"category": "fashion"}
    )

    assert result == set()
