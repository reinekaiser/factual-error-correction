from fastapi import FastAPI
import os
from taoquan.utils.crawler import AsyncNewsCrawler
from taoquan.utils.inference_model import Seq2SeqPredictor

app = FastAPI()
crawler = AsyncNewsCrawler()
predictor = Seq2SeqPredictor("./model...")

@app.get("/")
def root():
    return {"message": "Táo quân 2025"}

def read_links_from_file(file_path: str = "links.txt") -> List[str]:
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
    result = await predictor.generate_single_async(text, evidence)
    return {"text": text, "generated": result}

