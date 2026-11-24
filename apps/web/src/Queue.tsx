import { useCallback, useEffect, useState } from "react";
import { api, ApiError, type DocumentSummary, type QueueItem } from "./api";

const STATUS_LABEL: Record<string, string> = {
  uploaded: "Uploaded",
  ocr_done: "Reading",
  extracted: "Checking",
  validated: "Checking",
  needs_review: "Needs review",
  approved: "Approved",
  indexed: "Approved",
  failed: "Failed",
};

function age(iso: string): string {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)} h ago`;
  return `${Math.round(minutes / 1440)} d ago`;
}

export function Queue({ onOpen, onSignOut }: { onOpen: (id: string) => void; onSignOut: () => void }) {
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [recent, setRecent] = useState<DocumentSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(0);
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    try {
      const [q, docs] = await Promise.all([api.queue(), api.documents()]);
      setQueue(q);
      setRecent(docs.items.slice(-12).reverse());
      setError(null);
      setLoaded(true);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onSignOut();
      else setError(String(e));
    }
  }, [onSignOut]);

  useEffect(() => {
    load();
    const timer = setInterval(load, 5000); // pipeline status changes in the background
    return () => clearInterval(timer);
  }, [load]);

  async function upload(files: FileList | null) {
    if (!files) return;
    setUploading(files.length);
    for (const file of Array.from(files)) {
      try {
        await api.upload(file);
      } catch (e) {
        setError(`${file.name}: ${e}`);
      }
      setUploading((n) => n - 1);
    }
    load();
  }

  return (
    <main className="page">
      <header className="topbar">
        <h1>DocFlow</h1>
        <span className="muted">Review queue</span>
        <span className="spacer" />
        <label className="button">
          {uploading ? `Uploading ${uploading}…` : "Upload documents"}
          <input type="file" multiple accept=".pdf,.png,.jpg,.jpeg" hidden onChange={(e) => upload(e.target.files)} />
        </label>
        <button className="ghost" onClick={onSignOut}>
          Sign out
        </button>
      </header>
      {error && <p className="error">{error}</p>}

      <section className="card">
        <h2>
          Needs review <span className="count">{queue.length}</span>
        </h2>
        {!loaded ? (
          <p className="muted">{error ? "Could not load the queue." : "Loading…"}</p>
        ) : queue.length === 0 ? (
          <p className="muted">Nothing waiting. Every document so far was approved automatically or by a reviewer.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Document</th>
                <th>Waiting</th>
                <th className="num">Fields to check</th>
              </tr>
            </thead>
            <tbody>
              {queue.map((item, i) => (
                <tr key={item.document_id} className="clickable" onClick={() => onOpen(item.document_id)}>
                  <td>
                    {i === 0 && <span className="pill next">next</span>} {item.filename}
                  </td>
                  <td className="muted">{age(item.created_at)}</td>
                  <td className="num">{item.fields_to_review < 0 ? "whole document" : item.fields_to_review}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="card">
        <h2>Recent documents</h2>
        <table>
          <tbody>
            {recent.map((d) => (
              <tr key={d.id} className="clickable" onClick={() => onOpen(d.id)}>
                <td>{d.filename}</td>
                <td>
                  <span className={`pill status-${d.status}`}>{STATUS_LABEL[d.status] ?? d.status}</span>
                </td>
                <td className="muted num">{age(d.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  );
}
