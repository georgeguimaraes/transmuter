"""Prints the reference values of test/transmuter/text_gliner_test.exs, from gliner 0.2.29 with ONNX Runtime, and writes test/fixtures/gliner_inputs.json.

Usage: uv run --with "gliner==0.2.29" --with "transformers==4.57.1" --with onnxruntime python -I test/support/gliner_references.py

gliner pins its own range of transformers, so this script has its own uv run line.
"""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from gliner.data_processing import UniEncoderSpanProcessor, UniEncoderTokenProcessor
from gliner.data_processing.tokenizer import WordsSplitter
from gliner.model import BaseGLiNER, GLiNER

span_repo = "onnx-community/gliner_small-v2.1"
token_repo = "knowledgator/gliner-pii-small-v1.0"

# The inputs of the model for each tokenizer family: Metaspace (DeBERTa-v3) and ByteLevel
# (ModernBERT). The processor makes them as GLiNER.inference does, without the model: the
# words of the whitespace splitter, the labels without duplicates, the prompt, and the
# tokenizer with is_split_into_words. The word offsets are byte offsets, as in Transmuter.
cases = [
    ("Cristiano Ronaldo was born on 5 February 1985 in Funchal, Madeira.", ["person", "date", "location"]),
    ("José Müller visitó São Paulo y Zürich en 2023.", ["person", "city", "person"]),
    ("The state-of-the-art e-mail_server runs at well-known sites.", ["phone number", "software"]),
    ("東京タワーは東京都港区にあります。Apple发布了iPhone 15。", ["organization", "location"]),
    # DEL is a word without DeBERTa tokens. gliner counts it in text_lengths, and the words
    # after it get lower numbers in words_mask.
    ("Pay\x7f Ana Lima \x7f today.", ["person"]),
]

fixtures = {}
splitter = WordsSplitter("whitespace")
for repo, processor_class in [(span_repo, UniEncoderSpanProcessor), (token_repo, UniEncoderTokenProcessor)]:
    config = GLiNER._config_from_dict(json.load(open(hf_hub_download(repo, "gliner_config.json"))))
    tokenizer = BaseGLiNER._set_tokenizer_spec_tokens(AutoTokenizer.from_pretrained(repo))
    processor = processor_class(config, tokenizer, splitter)
    rows = []
    for text, labels in cases:
        words = list(splitter(text))
        batch = processor.collate_raw_batch([{"tokenized_text": [w for w, _, _ in words], "ner": None}], entity_types=list(dict.fromkeys(labels)))
        inputs = processor.tokenize_inputs(batch["tokens"], batch["classes_to_id"])
        rows.append(
            {
                "text": text,
                "labels": labels,
                "words": [[len(text[:start].encode()), len(text[:end].encode())] for _, start, end in words],
                "input_ids": inputs["input_ids"][0].tolist(),
                "words_mask": inputs["words_mask"][0].tolist(),
                "text_lengths": batch["seq_length"][0].tolist(),
            }
        )
    fixtures[repo] = rows

path = Path(__file__).parent.parent / "fixtures" / "gliner_inputs.json"
path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=1) + "\n")
print("wrote", path)


# The entities of GLiNER.inference with ONNX Runtime, on the files of the tests: the fp16
# export of gliner_small-v2.1 and the fp32 export of gliner-pii-small-v1.0. The texts of
# gliner_small-v2.1 run in one batch, each with its own labels. The script prints
# (start, end, label, text, score) of each entity, with byte offsets, as in Transmuter.
def entities(model, texts, labels):
    for text, row in zip(texts, model.inference(texts, labels, batch_size=len(texts))):
        print([(len(text[: e["start"]].encode()), len(text[: e["end"]].encode()), e["label"], e["text"], round(e["score"], 4)) for e in row])


small = GLiNER.from_pretrained(span_repo, load_onnx_model=True, onnx_model_file="onnx/model_fp16.onnx")
entities(
    small,
    [
        "Cristiano Ronaldo dos Santos Aveiro, born 5 February 1985, is a Portuguese footballer who plays for Al Nassr and the Portugal national team.",
        "José Müller visitó São Paulo y Zürich en 2023.",
        "Cristiano Ronaldo plays for Al Nassr.",
    ],
    [["person", "date", "organization", "country"], ["person", "city"], ["person", "organization"]],
)
del small

pii = GLiNER.from_pretrained(token_repo, load_onnx_model=True, onnx_model_file="onnx/model.onnx")
entities(
    pii,
    ["John Smith lives at 42 Baker Street in London. His email is john.smith@example.com and his phone number is +44 20 7946 0958."],
    [["person", "address", "email", "phone number"]],
)
