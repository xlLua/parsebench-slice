import pandas as pd

base = "hf://datasets/llamaindex/ParseBench/"
text = pd.read_json(base + "text_content.jsonl", lines=True)

ocr = text[text["tags"].apply(lambda t: "ocr" in t)]
print(ocr["pdf"].nunique(), "scanned pages")
print(ocr["type"].value_counts())   # which kinds of checks exist for these pages