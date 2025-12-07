import { useEffect, useState } from "react";
import "./index.css";

const BASE_URL = import.meta.env.VITE_BE_BASE_URL || "http://localhost:8000";

export default function App() {
  const [urls, setUrls] = useState([]);
  const [article, setArticle] = useState(null);
  const [content, setContent] = useState("");
  const [claim, setClaim] = useState("");
  const [inference, setInference] = useState(null);

  const [page, setPage] = useState(1);
  const [chunkSize] = useState(10);
  const [totalUrls, setTotalUrls] = useState(0);

  const totalPages = Math.ceil(totalUrls / chunkSize);

  useEffect(() => {
    loadList(page);
  }, []);

  async function loadList(pageNumber = 1) {
    try {
      const res = await fetch(
        `${BASE_URL}/news/list?page=${pageNumber}&chunk_size=${chunkSize}`,
        { headers: { "ngrok-skip-browser-warning": "true" } }
      );
      const data = await res.json();
      setUrls(data.urls || []);
      setTotalUrls(data.total_urls || 0);
      setPage(pageNumber);
    } catch (err) {
      console.error("Error loading list:", err);
    }
  }

  async function loadArticle(url) {
    try {
      const res = await fetch(
        `${BASE_URL}/news/crawl?url=` + encodeURIComponent(url),
        { headers: { "ngrok-skip-browser-warning": "true" } }
      );
      const data = await res.json();
      setArticle(data);
      setContent(data.content || "");
      setInference(null);
    } catch (err) {
      console.error("Error loading article:", err);
    }
  }

  async function sendInference() {
    if (!claim.trim()) return;

    try {
      const res = await fetch(
        `${BASE_URL}/news/inference?text=${encodeURIComponent(
          claim
        )}&evidence=${encodeURIComponent(content)}`,
        { headers: { "ngrok-skip-browser-warning": "true" } }
      );
      const data = await res.json();
      setInference(data);
    } catch (err) {
      console.error("Error sending inference:", err);
    }
  }

  function handleKey(e) {
    if (e.key === "Enter") sendInference();
  }

  return (
    <div className="h-screen flex flex-col font-sans text-gray-800 bg-blue-50">
      {/* Header */}
      <header className="w-full bg-blue-600 text-white p-4 text-2xl font-bold shadow-md">
        Táo quân 2025
      </header>

      {/* Body: Sidebar + Main */}
      <div className="flex flex-1">
        {/* Sidebar */}
        <aside className="w-72 bg-white border-r border-blue-200 shadow-md flex flex-col">
          <h2 className="text-xl font-semibold p-4 border-b border-blue-200 bg-blue-100 text-blue-800">
            Danh sách bài báo
          </h2>
          <div className="flex-1 overflow-y-auto">
            {urls.length === 0 && (
              <p className="p-4 text-blue-400">Đang tải danh sách...</p>
            )}
            {urls.map((u, idx) => (
              <div
                key={idx}
                onClick={() => loadArticle(u)}
                className="p-3 border-b border-blue-100 hover:bg-blue-50 cursor-pointer truncate text-blue-700"
                title={u}
              >
                {u}
              </div>
            ))}
          </div>

          {/* Pagination */}
          <div className="flex justify-center gap-1 mt-2 p-2 flex-wrap">
            <button
              onClick={() => page > 1 && loadList(page - 1)}
              disabled={page === 1}
              className="px-2 py-1 bg-blue-100 text-blue-800 rounded disabled:opacity-50"
            >
              Prev
            </button>

            {Array.from({ length: totalPages }, (_, i) => (
              <button
                key={i}
                onClick={() => loadList(i + 1)}
                className={`px-2 py-1 rounded ${
                  page === i + 1
                    ? "bg-blue-600 text-white"
                    : "bg-blue-50 text-blue-800"
                }`}
              >
                {i + 1}
              </button>
            ))}

            <button
              onClick={() => page < totalPages && loadList(page + 1)}
              disabled={page === totalPages}
              className="px-2 py-1 bg-blue-100 text-blue-800 rounded disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 flex flex-col p-6">
          {/* Article */}
          <div className="flex-1 w-full overflow-y-auto bg-white shadow-inner rounded-t-lg p-6">
            {article ? (
              <>
                <h1 className="text-3xl font-bold mb-4 text-blue-800">
                  {article.title || "Không có tiêu đề"}
                </h1>
                <div className="whitespace-pre-wrap text-lg leading-relaxed text-gray-800">
                  {content}
                </div>
              </>
            ) : (
              <p className="text-blue-400 italic">
                Chọn một URL ở bên trái để xem nội dung...
              </p>
            )}
          </div>

          {/* Claim input */}
          <div className="w-full p-4 border-t bg-blue-50 flex flex-col">
            <input
              value={claim}
              onChange={(e) => setClaim(e.target.value)}
              onKeyDown={handleKey}
              className="w-full p-3 border border-blue-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400 text-lg bg-white text-blue-800 placeholder-blue-400"
              placeholder="Nhập claim rồi nhấn Enter..."
            />

            {inference && (
              <pre className="mt-4 bg-white p-4 rounded-lg text-sm text-blue-800 overflow-x-auto whitespace-pre-wrap shadow-inner border border-blue-100">
                {JSON.stringify(inference, null, 2)}
              </pre>
            )}
          </div>
        </main>
      </div>
    </div>

  );
}
