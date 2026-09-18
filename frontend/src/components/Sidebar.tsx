import { useEffect, useState } from "react";

import { ALL, indexSummary, listModules, listSubjects } from "../api";
import type { Chat, Filters, Mode, Module, Subject } from "../types";

function IndexNote() {
  const [note, setNote] = useState("Checking what's indexed…");

  useEffect(() => {
    indexSummary().then((summary) => {
      if (!summary.reachable) {
        setNote("Can't reach the API — check that the stack is running.");
      } else if (!summary.notes) {
        setNote("No notes indexed yet. Put PDFs in data/, then run a sync.");
      } else {
        const notes = `${summary.notes} note${summary.notes === 1 ? "" : "s"}`;
        const subjects = `${summary.subjects} subject${summary.subjects === 1 ? "" : "s"}`;
        setNote(
          `${notes} from ${subjects} · ${summary.sections} sections · matching by ${summary.matching}`,
        );
      }
    });
  }, []);

  return <p className="index-note">{note}</p>;
}

export function Sidebar({
  filters,
  onFilters,
  mode,
  onMode,
  chats,
  currentId,
  onOpenChat,
  onNewChat,
  onDeleteChat,
  open,
  onClose,
}: {
  filters: Filters;
  onFilters: (next: Filters) => void;
  mode: Mode;
  onMode: (next: Mode) => void;
  chats: Chat[];
  currentId: string;
  onOpenChat: (id: string) => void;
  onNewChat: () => void;
  onDeleteChat: (id: string) => void;
  open: boolean;
  onClose: () => void;
}) {
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [modules, setModules] = useState<Module[]>([]);

  useEffect(() => {
    let live = true;
    listSubjects(filters.semester)
      .then((found) => live && setSubjects(found))
      .catch(() => live && setSubjects([]));
    return () => {
      live = false;
    };
  }, [filters.semester]);

  useEffect(() => {
    if (filters.subjectCode === ALL) {
      setModules([]);
      return;
    }
    let live = true;
    listModules(filters.subjectCode)
      .then((found) => live && setModules(found))
      .catch(() => live && setModules([]));
    return () => {
      live = false;
    };
  }, [filters.subjectCode]);

  function toggleModule(number: number) {
    const chosen = filters.modules.includes(number)
      ? filters.modules.filter((m) => m !== number)
      : [...filters.modules, number].sort((a, b) => a - b);
    onFilters({ ...filters, modules: chosen });
  }

  return (
    <aside className={open ? "sidebar open" : "sidebar"}>
      <div className="brand">
        <span className="star" aria-hidden="true">
          ✳
        </span>
        ChatVTU
        <button type="button" className="close" onClick={onClose} aria-label="Close menu">
          ✕
        </button>
      </div>

      <button type="button" className="new-chat" onClick={onNewChat}>
        ＋ New question
      </button>

      <p className="side-label">What to search</p>

      <label className="field">
        <span>Semester</span>
        <select
          value={filters.semester}
          onChange={(event) =>
            onFilters({ semester: event.target.value, subjectCode: ALL, modules: [] })
          }
        >
          <option value={ALL}>All semesters</option>
          {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
            <option key={n} value={String(n)}>
              Semester {n}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Subject</span>
        <select
          value={filters.subjectCode}
          onChange={(event) =>
            onFilters({ ...filters, subjectCode: event.target.value, modules: [] })
          }
        >
          <option value={ALL}>All subjects</option>
          {subjects.map((subject) => (
            <option key={subject.code} value={subject.code}>
              {subject.code} · {subject.name}
            </option>
          ))}
        </select>
      </label>

      {modules.length > 0 && (
        <fieldset className="modules">
          <legend>Modules</legend>
          {modules.map((module) => (
            <label key={module.number} className={module.indexed_notes ? "" : "empty"}>
              <input
                type="checkbox"
                checked={filters.modules.includes(module.number)}
                onChange={() => toggleModule(module.number)}
              />
              <span>
                M{module.number}
                {module.title ? ` · ${module.title}` : ""}
                {module.indexed_notes ? "" : " (empty)"}
              </span>
            </label>
          ))}
        </fieldset>
      )}

      <fieldset className="segmented">
        <legend className="side-label">Answering</legend>
        <div className="segments">
          {(["careful", "quick"] as Mode[]).map((option) => (
            <button
              key={option}
              type="button"
              className={mode === option ? "selected" : ""}
              aria-pressed={mode === option}
              onClick={() => onMode(option)}
            >
              {option === "careful" ? "Careful" : "Quick"}
            </button>
          ))}
        </div>
        <p className="side-hint">
          {mode === "careful"
            ? "Checks the excerpts and searches again if they're weak."
            : "One search, one answer. Faster, less thorough."}
        </p>
      </fieldset>

      <p className="side-label">Chats</p>
      <nav className="chats">
        {chats.length === 0 && <p className="side-hint">Nothing yet.</p>}
        {chats.map((chat) => (
          <div key={chat.id} className={chat.id === currentId ? "chat current" : "chat"}>
            <button type="button" onClick={() => onOpenChat(chat.id)} title={chat.title}>
              {chat.title || "New chat"}
            </button>
            <button
              type="button"
              className="remove"
              onClick={() => onDeleteChat(chat.id)}
              aria-label={`Delete ${chat.title}`}
            >
              ✕
            </button>
          </div>
        ))}
      </nav>

      <IndexNote />
    </aside>
  );
}
