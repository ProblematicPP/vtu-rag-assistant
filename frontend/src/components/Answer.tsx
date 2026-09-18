import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { AskResponse, Figure } from "../types";

function captionFor(figure: Figure) {
  if (figure.caption) return figure.caption;
  return figure.kind === "page"
    ? `Scanned page ${figure.page}`
    : `Diagram to redraw — page ${figure.page}`;
}

/** How the answer was arrived at: where it came from, and what the run did. */
function Provenance({ answer }: { answer: AskResponse }) {
  const notes = answer.notes ?? [];
  const run: string[] = [];
  if (answer.resolved_question) run.push(`answered as “${answer.resolved_question}”`);
  if (answer.in_scope === false) run.push("outside the syllabus, nothing searched");
  if (answer.rewritten_queries?.length) {
    run.push(`searched again as “${answer.rewritten_queries.at(-1)}”`);
  }
  if (answer.search_mode) run.push(`${answer.search_mode} search`);
  if (answer.cached) run.push("from cache");
  else if (answer.latency_ms) run.push(`${(answer.latency_ms / 1000).toFixed(1)}s`);

  if (!notes.length && !run.length) return null;
  return (
    <p className="provenance">
      {notes.length > 0 && <span className="from">From your notes: </span>}
      {notes.map((note, index) => (
        <span key={note.note_id}>
          {index > 0 && <span aria-hidden="true"> · </span>}
          <a
            href={`${note.url}${note.pages.length ? `#page=${note.pages[0]}` : ""}`}
            target="_blank"
            rel="noreferrer"
          >
            {note.subject_code} module {note.module_number}
            {note.pages.length > 0 && `, p. ${note.pages[0]}`}
          </a>
        </span>
      ))}
      {run.length > 0 && (
        <span className="run">
          {notes.length > 0 && <span aria-hidden="true"> · </span>}
          {run.join(" · ")}
        </span>
      )}
    </p>
  );
}

export function Answer({ text, answer }: { text: string; answer?: AskResponse }) {
  return (
    <div className="answer">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
      {(answer?.figures ?? []).map((figure) => (
        <figure key={figure.id}>
          <img src={figure.url} alt={captionFor(figure)} loading="lazy" />
          <figcaption>{captionFor(figure)}</figcaption>
        </figure>
      ))}
      {answer && <Provenance answer={answer} />}
    </div>
  );
}
