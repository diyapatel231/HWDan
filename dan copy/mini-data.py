import json
import gzip

data = {
    "mini-train": [
        {"page": "Rouble", "text": "What is this currency of russia"},
        {"page": "Pound", "text": "What is this currency of england"},
        {"page": "Moscow", "text": "What is this capital of russia"},
        {"page": "London", "text": "What is this capital of england"},
        {"page": "Rouble", "text": "What 's russia 's currency"},
        {"page": "Pound", "text": "What 's england 's currency"},
        {"page": "Moscow", "text": "What 's russia 's capital"},
        {"page": "London", "text": "What 's england 's capital"}
    ],
    "mini-dev": [
        {"page": "Rouble", "text": "What currency is used in russia"},
        {"page": "Pound", "text": "What currency is used in england"},
        {"page": "Moscow", "text": "What is the capital and largest city of russia"},
        {"page": "London", "text": "What is the capital and largest city of england"},
        {"page": "Rouble", "text": "What 's the currency in russia"},
        {"page": "Pound", "text": "What 's the currency in england"},
        {"page": "Moscow", "text": "What 's the capital of russia"},
        {"page": "London", "text": "What 's the capital of england"}
    ]
}

# Write mini-train.json.gz as a single JSON array
with gzip.open('mini-train.json.gz', 'wt', encoding='utf-8') as f:
    json.dump(data["mini-train"], f, indent=2)

# Write mini-dev.json.gz as a single JSON array
with gzip.open('mini-dev.json.gz', 'wt', encoding='utf-8') as f:
    json.dump(data["mini-dev"], f, indent=2)

print("Files created successfully!")
print(f"mini-train.json.gz: {len(data['mini-train'])} examples")
print(f"mini-dev.json.gz: {len(data['mini-dev'])} examples")