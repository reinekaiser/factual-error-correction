from pyngrok import ngrok
from fastapi import FastAPI, Query
import uvicorn
import threading
import os
from taoquan.utils.crawler import AsyncNewsCrawler
from taoquan.utils.inference_model import Seq2SeqPredictor
from taoquan.utils.sbert import SentenceRetriever
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Đường dẫn tới model")
    parser.add_argument("--sbert_path", type=str, help="Đường dẫn tới sbert model")
    parser.add_argument("--news_links", type=str, help="Đường dẫn tới news links")

    args = parser.parse_args()

    crawler = AsyncNewsCrawler()
    predictor = Seq2SeqPredictor(model_name_or_path=args.model_path)
    retriver = SentenceRetriever(model_name=args.sbert_path)

    app = FastAPI()

    @app.get("/")
    def root():
        return {"message": "Táo quân 2025"}

    def read_links_from_file(file_path = args.news_links):
        path = Path(file_path)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
        return urls

    @app.get("/news/list")
    async def get_list(
        page: int = Query(1, ge=1, description="Số trang muốn lấy"),
        chunk_size: int = Query(10, ge=1, le=100, description="Số URL trên 1 trang")
    ):
        """
        Lấy danh sách URL từ file links.txt theo page & chunk_size
        """
        urls = read_links_from_file("links.txt")
        total_urls = len(urls)

        start = (page - 1) * chunk_size
        end = start + chunk_size
        page_urls = urls[start:end]

        return {
            "total_urls": total_urls,
            "page": page,
            "chunk_size": chunk_size,
            "urls": page_urls
        }

    @app.get("/news/crawl")
    async def crawl_single_url(url: str = Query(..., description="URL bài báo cần crawl")):
        article = await crawler.crawl_single(url)
        return article

    @app.get("/news/inference")
    async def get_inference(text: str, evidence=None):
        if evidence and len(evidence.split(".")) > 10:
            top_sentences, _ = retriver(text, evidence, top_k = 10)
            evidence = top_sentences.join(".")
        result = await predictor.generate_single_async(text, evidence)
        return {"text": text, "generated": result}

    public_url = ngrok.connect(8000)
    print("Public URL:", public_url)

    def run_app():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    thread = threading.Thread(target=run_app)
    thread.start()



