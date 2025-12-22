import { useEffect, useState } from "react";
import "./index.css";

const BASE_URL = import.meta.env.VITE_BE_BASE_URL || "https://capably-conceptacular-jerilyn.ngrok-free.dev";

export default function App() {
  const [urls, setUrls] = useState([]);
  const [article, setArticle] = useState(null);
  const [content, setContent] = useState("");
  const [claim, setClaim] = useState("");
  const [inference, setInference] = useState(null);
  const [loading, setLoading] = useState(false);
  const [maskStrategy, setMaskStrategy] = useState('heuristic')
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

      setUrls(data.data || []);
      setTotalUrls(data.total_items || 0);
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

    setLoading(true);

    try {
      const res = await fetch(
        `${BASE_URL}/news/inference`,
        {
          method: "POST",
          headers: { "ngrok-skip-browser-warning": "true" } ,
          body: JSON.stringify({
            text: claim,
            evidence: content,
            mask_strategy: maskStrategy,
          }),
        }
      );
      const data = await res.json();
      setInference(data);
    } catch (err) {
      console.error("Error sending inference:", err);
    }

    setLoading(false);
  }

  return (
    <div className="h-screen flex flex-col font-sans text-gray-800 bg-blue-50">
      <header className="w-full bg-blue-600 text-white p-4 text-2xl font-bold shadow-md">
        Táo quân 2025
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Sidebar */}
        <aside className="w-72 bg-white border-r border-blue-200 shadow-md flex flex-col">
          <h2 className="text-xl font-semibold p-4 border-b border-blue-200 bg-blue-100 text-blue-800">
            Danh sách bài báo
          </h2>
          <div className="flex-1 overflow-y-auto min-h-0">
            {urls.length === 0 && (
              <p className="p-4 text-blue-400">Đang tải danh sách...</p>
            )}
            {urls.map((item, idx) => (
              <div
                key={idx}
                onClick={() => loadArticle(item.url)}
                className="p-3 border-b border-blue-100 hover:bg-blue-50 cursor-pointer truncate text-blue-700"
                title={item.url}
              >
                {item.title || "Không có tiêu đề"}
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
        <main className="flex-1 flex flex-col min-h-0">
          <div className="flex-1 overflow-y-auto bg-white p-6 shadow-inner rounded-t-lg min-h-0">
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

          <div className="w-full p-4 border-t bg-blue-50 flex-shrink-0">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium text-blue-700">
                Mask strategy:
              </label>

              <select
                value={maskStrategy}
                onChange={(e) => setMaskStrategy(e.target.value)}
                className="px-3 py-2 border border-blue-200 rounded-lg 
                          bg-white text-blue-800 shadow-sm
                          focus:outline-none focus:ring-2 focus:ring-blue-400"
              >
                <option value="heuristic">Heuristic</option>
                <option value="lime">LIME</option>
              </select>
            </div>
            <div className="flex items-stretch gap-3 mt-3">
              <textarea
                value={claim}
                onChange={(e) => setClaim(e.target.value)}
                className="flex-1 p-3 border border-blue-200 rounded-lg shadow-sm 
                          focus:outline-none focus:ring-2 focus:ring-blue-400 
                          text-lg bg-white text-blue-800 placeholder-blue-400 
                          min-h-[80px]"
                placeholder="Nhập claim..."
              />

              <button
                onClick={sendInference}
                disabled={loading}
                className={`w-[90px] p-3 border border-blue-200 rounded-lg shadow-sm
                            flex items-center justify-center text-white text-base font-semibold
                            ${loading ? "bg-blue-300" : "bg-blue-600 hover:bg-blue-700"}`}
              >
                {loading ? "..." : "Gửi"}
              </button>
            </div>

            {inference && (
              <div className="mt-4 bg-white p-4 rounded-lg text-base text-blue-800 shadow-inner border border-blue-100">
                <p><b>Generated:</b> {inference.generated}</p>
                <p><b>Label:</b> {inference.label}</p>
                <p><b>Probabilities:</b></p>
                <pre className="bg-blue-50 p-2 rounded border border-blue-100 overflow-x-auto">
                  {JSON.stringify(inference.probs, null, 2)}
                </pre>
              </div>
            )}
          </div>

        </main>
      </div>
    </div>
  );
}
