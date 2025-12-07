import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler

class SentenceRetriever:
    def __init__(self, model_name: str, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
        print(f"Model loaded: {model_name} on {self.device}")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # (batch_size, seq_len, hidden_dim)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def retrieve_evidence(self, claim: str, context: list[str], top_k: int = 5):
        if not isinstance(context, list):
            context = [context] if context is not None else []

        sentences = [claim] + context
        encoded = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)

        embeddings = self._mean_pooling(output, encoded['attention_mask'])
        claim_emb = embeddings[0].unsqueeze(0)
        context_embs = embeddings[1:]

        similarities = self.cosine(claim_emb, context_embs).cpu().numpy()
        scaled = MinMaxScaler().fit_transform(similarities.reshape(-1, 1)).flatten()

        ranked = sorted(zip(context, scaled), key=lambda x: x[1], reverse=True)
        top_k = min(top_k, len(ranked))
        top_sentences = [s for s, _ in ranked[:top_k]]
        top_scores = [v for _, v in ranked[:top_k]]

        return top_sentences, top_scores
