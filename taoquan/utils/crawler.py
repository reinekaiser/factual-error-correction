import asyncio
import httpx
import trafilatura

class AsyncNewsCrawler:
    def __init__(self):
        pass

    async def crawl_single(self, url: str) -> dict:
        """
        Crawl 1 bài báo từ URL (async) chỉ dùng Trafilatura
        """
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(url)
                html = r.text

            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                output_format='text'
            )
            meta = trafilatura.extract_metadata(html)

            if text:
                return {
                    "url": url,
                    "title": meta.get("title") if meta else "",
                    "author": meta.get("authors") if meta else "",
                    "publish_date": meta.get("date") if meta else "",
                    "content": text
                }
            else:
                return {"url": url, "error": "No content extracted"}

        except Exception as e:
            return {"url": url, "error": str(e)}
