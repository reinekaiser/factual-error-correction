import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Verifier:
    def __init__(
        self,
        model_name_or_path: str = "xlm-roberta-large",
        num_label: int = 3,
        max_len: int = 256,
        device: str = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model_name_or_path = model_name_or_path
        self.num_label = num_label
        self.max_len = max_len

        print(f"[INFO] Loading model: {self.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_label
        )

        self.model.to(self.device)
        self.model.eval()
        print("[INFO] Model loaded successfully!")

    def encode(self, claim, evidence):
        return self.tokenizer(
            claim,
            evidence,
            max_length=self.max_len,
            truncation="longest_first",
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.device)

    @torch.no_grad()
    def predict(self, claim, evidence):
        inputs = self.encode(claim, evidence)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

        return {
            "label": torch.argmax(probs, dim=-1).item(),
            "probs": probs.cpu().numpy().tolist()[0]
        }
