from pyngrok import ngrok
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn
import threading
import os
from taoquan.utils.crawler import NewsCrawler
from taoquan.utils.inference_model import Seq2SeqPredictor
from taoquan.utils.sbert import SentenceRetriever
from taoquan.utils.verfier import Verifier
import argparse
from pydantic import BaseModel
from typing import Optional, Literal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Đường dẫn tới model")
    parser.add_argument("--sbert_path", type=str, help="Đường dẫn tới sbert model")
    parser.add_argument("--news_links", type=str, help="Đường dẫn tới news links")
    parser.add_argument("--verifier_path", type=str, help="Đường dẫn tới verify model")

    args = parser.parse_args()

    crawler = NewsCrawler()
    predictor = Seq2SeqPredictor(args.model_path, args.verifier_path)
    retriver = SentenceRetriever(model_name=args.sbert_path)
    verifier = Verifier(model_name_or_path=args.verifier_path)

    class InferenceRequest(BaseModel):
        text: str
        evidence: Optional[str] = None
        mask_strategy: Optional[str] = None

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],            
        allow_credentials=True,
        allow_methods=["*"],   
        allow_headers=["*"],   
    )

    @app.get("/")
    def root():
        return {"message": "Táo quân 2025"}

    def read_links_from_file(file_path=args.news_links):
        path = Path(file_path)
        if not path.exists():
            return []

        data = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) == 2:
                    url, title = parts
                else:
                    # fallback nếu không có title
                    url = parts[0]
                    title = ""

                data.append({"url": url, "title": title})

        return data

    @app.get("/news/list")
    def get_list(
        page: int = Query(1, ge=1, description="Số trang muốn lấy"),
        chunk_size: int = Query(10, ge=1, le=100, description="Số item trên 1 trang")
    ):
        """
        Lấy danh sách {url, title} từ file links.txt theo page & chunk_size
        """
        items = read_links_from_file(file_path=args.news_links)
        total_items = len(items)

        start = (page - 1) * chunk_size
        end = start + chunk_size
        page_items = items[start:end]

        return {
            "total_items": total_items,
            "page": page,
            "chunk_size": chunk_size,
            "data": page_items,
        }

    @app.get("/news/crawl")
    def crawl_single_url(url: str = Query(..., description="URL bài báo cần crawl")):
        article = crawler.crawl_single(url)
        return article

    @app.post("/news/inference")
    def get_inference(req: InferenceRequest):
        text = req.text
        evidence = req.evidence
        mask_strategy = req.mask_strategy
        if evidence and len(evidence.split(".")) > 5:
            evidence = evidence.replace("\n", "")
            evidence = evidence.split(".")
            top_sentences, _ = retriver.retrieve_evidence(text, evidence, top_k = 5)
            evidence = ".".join(top_sentences)
        result = predictor.generate_single(text, evidence, mask_strategy)
        verified_result = verifier.predict(result, evidence)
        label_map = ["Support", "Refute", "NEI"]
        return {"text": text, "generated": result, "label": label_map[verified_result["label"]], "probs": verified_result["probs"][verified_result["label"]]}

    public_url = ngrok.connect(8000)
    print("Public URL:", public_url)

    def run_app():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    thread = threading.Thread(target=run_app)
    thread.start()