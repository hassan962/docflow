import { useState } from "react";
import { apiKey } from "./api";
import { Queue } from "./Queue";
import { Review } from "./Review";

function KeyForm({ onSaved }: { onSaved: () => void }) {
  const [value, setValue] = useState("");
  return (
    <main className="center">
      <form
        className="card key-form"
        onSubmit={(e) => {
          e.preventDefault();
          apiKey.set(value.trim());
          onSaved();
        }}
      >
        <h1>DocFlow review</h1>
        <p className="muted">Paste your API key. It stays in this browser tab only.</p>
        <input
          autoFocus
          type="password"
          placeholder="dfk_…"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          aria-label="API key"
        />
        <button type="submit" disabled={!value.trim().startsWith("dfk_")}>
          Continue
        </button>
      </form>
    </main>
  );
}

export function App() {
  const [hasKey, setHasKey] = useState(Boolean(apiKey.get()));
  const [documentId, setDocumentId] = useState<string | null>(null);

  if (!hasKey) return <KeyForm onSaved={() => setHasKey(true)} />;
  const signOut = () => {
    apiKey.clear();
    setHasKey(false);
  };
  return documentId ? (
    <Review documentId={documentId} onClose={() => setDocumentId(null)} onOpen={setDocumentId} />
  ) : (
    <Queue onOpen={setDocumentId} onSignOut={signOut} />
  );
}
