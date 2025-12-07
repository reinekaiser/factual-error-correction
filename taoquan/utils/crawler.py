import httpx
import trafilatura

class AsyncNewsCrawler:
    def __init__(self):
        pass

    def crawl_single(self, url: str) -> dict:
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

            title = meta.title if meta and hasattr(meta, "title") else ""
            author = meta.author if meta and hasattr(meta, "author") else ""
            date = meta.date if meta and hasattr(meta, "date") else ""

            return {
                "url": url,
                "title": title,
                "author": author,
                "publish_date": date,
                "content": text if text else "",
                "error": None if text else "No content extracted"
            }

        except Exception as e:
            return {"url": url, "error": str(e)}
