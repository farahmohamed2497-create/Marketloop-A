def count_tokens(messages):
    total = 0

    for msg in messages:
        total += len(msg["content"].split())

    return total
