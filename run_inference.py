# run_inference.py
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sample_texts import SAMPLE_TEXTS


def load_model(model_dir: str, device: str) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model, max_length: int = 256, top_k: int = 3) -> dict:
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)[0]
    k = min(top_k, probs.numel())
    top_probs, top_ids = torch.topk(probs, k=k)

    id2label = getattr(model.config, "id2label", None) or {}
    top = []
    for p, i in zip(top_probs.tolist(), top_ids.tolist()):
        label = id2label.get(i, str(i))
        top.append((label, p))

    return {"top_k": top, "prediction": top[0][0], "confidence": top[0][1]}


def main():
    parser = argparse.ArgumentParser(description="Run inference on a saved HF text classifier.")
    parser.add_argument("--model_dir", type=str, default="clinical_nlp_model", help="Path to saved model folder.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda. Default: auto")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt mode.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(args.model_dir, device)

    print(f"\nLoaded model from: {args.model_dir}")
    print(f"Device: {device}\n")

    print("=== Example predictions ===")
    for t in SAMPLE_TEXTS:
        out = predict(t, tokenizer, model, max_length=args.max_length, top_k=3)
        top_str = ", ".join([f"{lbl} ({p:.3f})" for lbl, p in out["top_k"]])
        print(f"- {t}")
        print(f"  -> {out['prediction']} | conf={out['confidence']:.3f} | top3: {top_str}\n")

    if args.interactive:
        print("=== Interactive mode (type 'q' to quit) ===")
        while True:
            s = input("> ").strip()
            if s.lower() in {"q", "quit", "exit"}:
                break
            if not s:
                continue
            out = predict(s, tokenizer, model, max_length=args.max_length, top_k=3)
            print(f"Prediction: {out['prediction']} (conf={out['confidence']:.3f})")
            for lbl, p in out["top_k"]:
                print(f"  - {lbl}: {p:.3f}")
            print("")


if __name__ == "__main__":
    main()
