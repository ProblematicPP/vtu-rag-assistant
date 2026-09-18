import { useEffect, useRef, useState } from "react";

import { PaperIcon } from "./icons";

const MAX_HEIGHT = 200;

export function Composer({
  busy,
  canDownload,
  onAsk,
  onAttach,
  onDownload,
  draft,
  onDraftUsed,
}: {
  busy: boolean;
  canDownload: boolean;
  onAsk: (question: string) => void;
  onAttach: (file: File) => void;
  onDownload: () => void;
  draft: string;
  onDraftUsed: () => void;
}) {
  const [text, setText] = useState("");
  const box = useRef<HTMLTextAreaElement>(null);
  const picker = useRef<HTMLInputElement>(null);

  // A starter chip or a repeated question drops straight into the composer
  useEffect(() => {
    if (!draft) return;
    setText(draft);
    box.current?.focus();
    onDraftUsed();
  }, [draft, onDraftUsed]);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [text]);

  function send() {
    const question = text.trim();
    if (question.length < 3 || busy) return;
    setText("");
    onAsk(question);
  }

  return (
    <div className="composer">
      <div className="box">
        <textarea
          ref={box}
          value={text}
          rows={1}
          placeholder="Ask anything from your notes"
          aria-label="Your question"
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              send();
            }
          }}
        />
        <div className="tools">
          <button
            type="button"
            className="ghost"
            onClick={() => picker.current?.click()}
            disabled={busy}
          >
            <PaperIcon /> Attach a question paper
          </button>
          <div className="spacer" />
          {canDownload && (
            <button type="button" className="ghost" onClick={onDownload}>
              Download answers as PDF
            </button>
          )}
          <button
            type="button"
            className="send"
            onClick={send}
            disabled={busy || text.trim().length < 3}
            aria-label="Send question"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true" width="16" height="16">
              <path
                d="M12 19V5M5 12l7-7 7 7"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </div>
      <input
        ref={picker}
        type="file"
        accept=".pdf,.png,.jpg,.jpeg,.webp"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file) onAttach(file);
        }}
      />
      <p className="hint">Enter sends · Shift + Enter starts a new line</p>
    </div>
  );
}
