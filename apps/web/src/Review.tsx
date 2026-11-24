import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type DocumentDetail, type Extraction, type Field } from "./api";
import { display, failedChecks, label, nextPending, reviewOrder, suggestion } from "./fields";

const COLORS: Record<string, string> = {
  needs_review: "var(--review)",
  auto_approved: "var(--ok)",
  reviewed: "var(--reviewed)",
};

interface Props {
  documentId: string;
  onClose: () => void;
  onOpen: (id: string) => void;
}

export function Review({ documentId, onClose, onOpen }: Props) {
  const [doc, setDoc] = useState<DocumentDetail | null>(null);
  const [extraction, setExtraction] = useState<Extraction | null>(null);
  const [images, setImages] = useState<Record<number, string>>({});
  const [focused, setFocused] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [zoom, setZoom] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const focusedAt = useRef(Date.now());
  const pageRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef<Record<string, HTMLLIElement | null>>({});

  const order = useMemo(() => (extraction ? reviewOrder(extraction.fields) : []), [extraction]);

  // Load the document, its extraction and presigned page images.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const d = await api.document(documentId);
        if (cancelled) return;
        setDoc(d);
        const urls = await Promise.all(d.pages.map((p) => api.pageImage(documentId, p.page_no)));
        if (!cancelled) setImages(Object.fromEntries(d.pages.map((p, i) => [p.page_no, urls[i].url])));
        if (d.latest_extraction) {
          const ex = await api.extraction(d.latest_extraction.id);
          if (cancelled) return;
          setExtraction(ex);
          const first = reviewOrder(ex.fields)[0] ?? null;
          setFocused(first);
        }
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [documentId]);

  const focus = useCallback((key: string | null) => {
    setFocused(key);
    setEditing(null);
    focusedAt.current = Date.now();
  }, []);

  // Keep the focused field visible in the list and its box visible on the page.
  useEffect(() => {
    if (!focused || !extraction) return;
    rowRefs.current[focused]?.scrollIntoView({ block: "nearest" });
    const field = extraction.fields[focused];
    const page = doc?.pages.find((p) => p.page_no === field?.page);
    const box = pageRef.current;
    if (!field?.bbox || !page?.height || !box) return;
    const img = box.querySelector<HTMLElement>(`[data-page="${page.page_no}"]`);
    if (!img) return;
    const scale = img.clientHeight / page.height;
    const y = img.offsetTop + ((field.bbox[1] + field.bbox[3]) / 2) * scale;
    const x = img.offsetLeft + ((field.bbox[0] + field.bbox[2]) / 2) * (img.clientWidth / (page.width ?? 1));
    box.scrollTo({ top: y - box.clientHeight / 2, left: x - box.clientWidth / 2, behavior: "smooth" });
  }, [focused, extraction, doc, zoom]);

  const seconds = () => (Date.now() - focusedAt.current) / 1000;

  const correct = useCallback(
    async (key: string, action: "accept" | "edit", value?: unknown) => {
      if (!extraction || busy) return;
      setBusy(true);
      try {
        const updated = await api.correct(extraction.id, key, action, value ?? null, seconds());
        setExtraction(updated);
        setError(null);
        focus(nextPending(reviewOrder(updated.fields), updated.fields, key) ?? key);
      } catch (e) {
        setError(String(e));
      } finally {
        setBusy(false);
      }
    },
    [extraction, busy, focus],
  );

  const approve = useCallback(async () => {
    if (!extraction || !doc || busy) return;
    const pending = order.filter((k) => extraction.fields[k].status === "needs_review");
    const issues = extraction.issues.length;
    const parts = [
      pending.length ? `accept the ${pending.length} remaining field(s) as they are` : "",
      issues ? `acknowledge ${issues} document issue(s)` : "",
    ].filter(Boolean);
    if (parts.length && !confirm(`Approve and ${parts.join(" and ")}?`)) return;
    setBusy(true);
    try {
      let current = extraction;
      for (const key of pending) current = await api.correct(current.id, key, "accept", null, 0);
      setExtraction(await api.approve(doc.id, issues > 0));
      const queue = await api.queue();
      const next = queue.find((q) => q.document_id !== doc.id);
      if (next) onOpen(next.document_id);
      else onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, [extraction, doc, order, busy, onOpen, onClose]);

  // Keyboard flow: a reviewer can clear a document without touching the mouse.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!extraction || !focused) return;
      const field = extraction.fields[focused];
      if (editing) {
        if (e.key === "Escape") setEditing(null);
        return; // the input handles Enter itself
      }
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        approve();
      } else if (e.key === "Enter") {
        e.preventDefault();
        correct(focused, "accept");
      } else if (e.key === "e" || e.key === "E") {
        e.preventDefault();
        setDraft(field.value == null ? "" : String(field.value));
        setEditing(focused);
      } else if ((e.key === "s" || e.key === "S") && suggestion(field) != null) {
        e.preventDefault();
        correct(focused, "edit", suggestion(field));
      } else if (e.key === "Tab") {
        e.preventDefault();
        const next = nextPending(order, extraction.fields, focused, e.shiftKey ? -1 : 1);
        const i = order.indexOf(focused);
        focus(next ?? order[(i + (e.shiftKey ? -1 : 1) + order.length) % order.length]);
      } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const i = order.indexOf(focused) + (e.key === "ArrowDown" ? 1 : -1);
        focus(order[Math.max(0, Math.min(order.length - 1, i))]);
      } else if (e.key === "z" || e.key === "Z") {
        setZoom((z) => (z === 1 ? 2 : 1));
      } else if (e.key === "Escape") {
        onClose();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [extraction, focused, editing, order, approve, correct, focus, onClose]);

  if (error && !doc) return <main className="page"><p className="error">{error}</p></main>;
  if (!doc) return <main className="page"><p className="muted">Loading…</p></main>;

  const failed = extraction
    ? order.flatMap((k) => failedChecks(extraction.fields[k]).map((c) => ({ key: k, ...c })))
    : [];
  const total = extraction ? Object.keys(extraction.fields).length : 0;
  const auto = extraction?.counts.auto_approved ?? 0;

  return (
    <main className="review">
      <header className="topbar">
        <button className="ghost" onClick={onClose}>← Queue</button>
        <h1>{doc.filename}</h1>
        <span className={`pill status-${extraction?.status === "approved" ? "approved" : doc.status}`}>
          {extraction?.status === "approved" ? "approved" : doc.status.replace("_", " ")}
        </span>
        <span className="spacer" />
        <span className="muted keys">
          <kbd>Tab</kbd> next · <kbd>Enter</kbd> accept · <kbd>E</kbd> edit · <kbd>S</kbd> use suggestion ·{" "}
          <kbd>Z</kbd> zoom · <kbd>Ctrl</kbd>+<kbd>Enter</kbd> approve
        </span>
        <button onClick={approve} disabled={busy || !extraction || extraction.status === "approved"}>
          Approve document
        </button>
      </header>
      {error && <p className="error">{error}</p>}
      {doc.failure_reason && <p className="error">{doc.failure_reason}</p>}

      <div className="panes">
        <div className="page-pane" ref={pageRef}>
          {doc.pages.map((p) => (
            <div key={p.page_no} className="page-image" data-page={p.page_no} style={{ width: `${zoom * 100}%` }}>
              {images[p.page_no] && <img src={images[p.page_no]} alt={`Page ${p.page_no}`} />}
              {extraction && p.width && p.height && (
                <svg viewBox={`0 0 ${p.width} ${p.height}`} preserveAspectRatio="none">
                  {order
                    .filter((k) => extraction.fields[k].page === p.page_no && extraction.fields[k].bbox)
                    .map((k) => {
                      const f = extraction.fields[k];
                      const [x0, y0, x1, y1] = f.bbox!;
                      return (
                        <rect
                          key={k}
                          x={x0 - 4}
                          y={y0 - 4}
                          width={x1 - x0 + 8}
                          height={y1 - y0 + 8}
                          rx={4}
                          className={k === focused ? "box focused" : "box"}
                          style={{ stroke: COLORS[f.status ?? "auto_approved"] }}
                          onClick={() => focus(k)}
                        >
                          <title>{label(k)}</title>
                        </rect>
                      );
                    })}
                </svg>
              )}
            </div>
          ))}
        </div>

        <aside className="field-pane">
          {!extraction ? (
            <p className="muted">No extraction yet: the document is still in the pipeline.</p>
          ) : (
            <>
              <div className="summary">
                <div>
                  <strong>{auto}</strong> of {total} fields auto-approved
                  <div className="bar">
                    <span style={{ width: `${(auto / Math.max(total, 1)) * 100}%` }} />
                  </div>
                </div>
                <div>
                  <strong>{extraction.counts.needs_review}</strong> to review ·{" "}
                  <strong>{extraction.counts.reviewed}</strong> reviewed
                </div>
              </div>
              {(failed.length > 0 || extraction.issues.length > 0) && (
                <div className="rules">
                  <h3>Failed checks</h3>
                  <ul>
                    {extraction.issues.map((i, n) => (
                      <li key={`i${n}`}>
                        <b>{i.rule}</b> {i.message}
                      </li>
                    ))}
                    {failed.map((c, n) => (
                      <li key={n} className="clickable" onClick={() => focus(c.key)}>
                        <b>{c.rule}</b> {label(c.key)}: {c.message}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <ul className="fields">
                {order.map((k) => (
                  <FieldRow
                    key={k}
                    name={k}
                    field={extraction.fields[k]}
                    focused={k === focused}
                    editing={k === editing}
                    draft={draft}
                    setDraft={setDraft}
                    onFocus={() => focus(k)}
                    onSave={() => correct(k, "edit", draft)}
                    rowRef={(el) => {
                      rowRefs.current[k] = el;
                    }}
                  />
                ))}
              </ul>
            </>
          )}
        </aside>
      </div>
    </main>
  );
}

interface RowProps {
  name: string;
  field: Field;
  focused: boolean;
  editing: boolean;
  draft: string;
  setDraft: (v: string) => void;
  onFocus: () => void;
  onSave: () => void;
  rowRef: (el: HTMLLIElement | null) => void;
}

function FieldRow({ name, field, focused, editing, draft, setDraft, onFocus, onSave, rowRef }: RowProps) {
  const status = field.status ?? "auto_approved";
  const hint = suggestion(field);
  return (
    <li ref={rowRef} className={`field ${status} ${focused ? "focused" : ""}`} onClick={onFocus}>
      <div className="field-head">
        <span className="field-label">{label(name)}</span>
        <span className="confidence" title="calibrated confidence">
          {field.source === "review" ? "reviewed" : `${Math.round(field.confidence * 100)}%`}
        </span>
      </div>
      {editing ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            onSave();
          }}
        >
          <input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)} aria-label={`Edit ${label(name)}`} />
        </form>
      ) : (
        <div className="field-value">{display(field.value)}</div>
      )}
      {failedChecks(field).map((c, i) => (
        <div key={i} className="check">
          {c.message ?? c.rule}
        </div>
      ))}
      {hint != null && status === "needs_review" && (
        <div className="hint">
          Suggestion: <code>{String(hint)}</code> — press <kbd>S</kbd>
        </div>
      )}
      {field.signals.match === "none" && field.source !== "review" && (
        <div className="check">not found on the page</div>
      )}
    </li>
  );
}
