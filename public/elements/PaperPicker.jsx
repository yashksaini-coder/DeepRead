// PaperPicker — sidebar UI for picking a bundled paper or uploading a PDF.
//
// Receives:
//   props.papers  — [{ slug, short, year, pages, title, authors, cache_name }, ...]
//   props.loaded  — [string] of paper.cache_name values currently in the library
//
// On click, fires the same Python `load_paper` / `upload_pdf` action callbacks
// that the old in-chat action buttons used. Replaces the markdown "Library" +
// "Classics" sections that used to live in the sidebar.

import { Button } from "@/components/ui/button";
import { useState } from "react";

export default function PaperPicker() {
  const papers = props.papers || [];
  const loadedSet = new Set(props.loaded || []);
  const [busy, setBusy] = useState(null);

  const trigger = async (name, payload) => {
    setBusy(name + ":" + (payload?.slug || ""));
    try {
      // Chainlit injects callAction into the custom-element sandbox; it fires
      // the matching @cl.action_callback on the server.
      await callAction({ name, payload });
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="flex flex-col gap-1 px-1 py-2">
      <div className="text-xs uppercase tracking-wide text-muted-foreground px-2 mb-1">
        Papers
      </div>

      {papers.map((p) => {
        const loaded = loadedSet.has(p.cache_name);
        const key = "load_paper:" + p.slug;
        return (
          <Button
            key={p.slug}
            variant="ghost"
            size="sm"
            disabled={loaded || busy !== null}
            onClick={() => trigger("load_paper", { slug: p.slug })}
            className="justify-start gap-2 h-auto py-2 px-2 font-normal"
            title={`${p.title} — ${p.authors}, ${p.year} (${p.pages} pages)`}
          >
            <span className="text-xs w-3 text-center opacity-60">
              {loaded ? "✓" : "·"}
            </span>
            <span className="flex-1 text-left">
              <span className="font-medium">{p.short}</span>
              <span className="text-muted-foreground"> ({p.year})</span>
            </span>
            <span className="text-xs text-muted-foreground">
              {p.pages}p
            </span>
          </Button>
        );
      })}

      <Button
        variant="ghost"
        size="sm"
        disabled={busy !== null}
        onClick={() => trigger("upload_pdf", {})}
        className="justify-start gap-2 h-auto py-2 px-2 font-normal mt-1 border-t border-border/40 rounded-none"
        title="Upload a PDF, image, or screenshot from disk"
      >
        <span className="text-xs w-3 text-center opacity-60">+</span>
        <span className="flex-1 text-left">Upload your own…</span>
      </Button>
    </div>
  );
}
