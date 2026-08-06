store = KeywordStore()

store.upsert(
    payload="Dell laptop with Intel processor",
    metadata={"category": "laptop"}
)

results = store.query("Dell")

assert len(results) > 0