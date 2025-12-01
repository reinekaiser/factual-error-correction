import { useState, useEffect } from "react";

function App() {
  const [links, setLinks] = useState([]);
  const [page, setPage] = useState(1);
  const chunk_size = 10;
  const [totalPages, setTotalPages] = useState(1);

  const [selectedLink, setSelectedLink] = useState("");
  const [evidence, setEvidence] = useState("");
  const [inference, setInference] = useState("");
  const [generated, setGenerated] = useState("");

  useEffect(() => {
    loadLinks(page);
  }, [page]);

  const loadLinks = async (p) => {
    const res = await fetch(`/news/list?page=${p}&chunk_size=${chunk_size}`);
    const data = await res.json();
    setLinks(data.urls);
    setTotalPages(Math.ceil(data.total_urls / chunk_size));
  };

  const selectLink = async (link) => {
    setSelectedLink(link);
    setGenerated("");
    setInference("");

    const res = await fetch(`/news/crawl?url=${encodeURIComponent(link)}`);
    const data = await res.json();
    setEvidence(data.content || data.text || "No content");
  };

  const submitInference = async () => {
    if (!inference) return alert("Please enter inference text");
    const params = new URLSearchParams({ text: inference });
    const res = await fetch(`/news/inference?${params.toString()}`);
    const data = await res.json();
    setGenerated(data.generated);
  };

  return (
    <div className="flex h-screen font-sans">
      {/* Sidebar */}
      <div className="w-64 bg-gray-900 text-white flex flex-col">
        <h1 className="p-4 text-xl font-bold border-b border-gray-700">Articles</h1>
        <div className="flex-1 overflow-y-auto">
          {links.map((link) => (
            <div
              key={link}
              className={`p-3 cursor-pointer hover:bg-gray-700 ${
                selectedLink === link ? "bg-gray-700" : ""
              }`}
              onClick={() => selectLink(link)}
            >
              {link}
            </div>
          ))}
        </div>
        <div className="p-2 border-t border-gray-700 flex justify-between">
          <button onClick={() => setPage((p) => Math.max(1, p - 1))} className="px-2 py-1 bg-gray-800 rounded">Prev</button>
          <span>{page} / {totalPages}</span>
          <button onClick={() => setPage((p) => Math.min(totalPages, p + 1))} className="px-2 py-1 bg-gray-800 rounded">Next</button>
        </div>
      </div>

      {/* Main panel */}
      <div className="flex-1 flex flex-col bg-gray-100">
        <div className="flex-1 p-6 overflow-y-auto bg-white border-b border-gray-300 whitespace-pre-wrap">
          <h2 className="text-lg font-semibold mb-2">Evidence</h2>
          {evidence}
        </div>
        <div className="p-4 bg-gray-200 flex flex-col">
          <textarea
            value={inference}
            onChange={(e) => setInference(e.target.value)}
            placeholder="Enter inference..."
            className="w-full p-2 mb-2 rounded resize-none h-24"
          />
          <button
            onClick={submitInference}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 self-start"
          >
            Submit
          </button>
          {generated && (
            <div className="mt-3 p-3 bg-white rounded shadow whitespace-pre-wrap">
              <strong>Generated:</strong> {generated}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
