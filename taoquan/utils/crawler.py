import httpx
import trafilatura

class NewsCrawler:
    def __init__(self):
        pass

    def crawl_single(self, url: str) -> dict:
        """
        Crawl 1 bài báo từ URL (sync) chỉ dùng Trafilatura
        """
        try:
            with httpx.Client(timeout=15) as client:
                r = client.get(url)
                html = r.text

            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                output_format='txt'
            )
            meta = trafilatura.extract_metadata(html)

            return {
                "url": url,
                "title": meta.get("title") if meta else "",
                "author": meta.get("authors") if meta else "",
                "publish_date": meta.get("date") if meta else "",
                "content": text if text else "",
                "error": None if text else "No content extracted"
            }

        except Exception as e:
            return {"url": url, "error": str(e)}
