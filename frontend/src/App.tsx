import { useCallback, useState } from "react";

import { ALL, answerPaperQuestion, ask, extractPaper, historyFrom, paperPdf } from "./api";
import { Composer } from "./components/Composer";
import { Sidebar } from "./components/Sidebar";
import { Thread } from "./components/Thread";
import type { Filters, Mode, PaperQuestion } from "./types";
import { useChats } from "./useChats";

function said(error: unknown) {
  const detail = error instanceof Error ? error.message : String(error);
  return `That didn't work — ${detail}. Try again in a moment; if it keeps happening, check that the stack is running.`;
}

export default function App() {
  const chat = useChats();
  const [filters, setFilters] = useState<Filters>({
    semester: "3",
    subjectCode: ALL,
    modules: [],
  });
  const [mode, setMode] = useState<Mode>("careful");
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);

  const { append, replace, messages, newId } = chat;

  const onAsk = useCallback(
    async (question: string) => {
      // The thread updates before the request goes out, so the question and a
      // waiting reply are on screen immediately
      const replyId = newId();
      const history = historyFrom(messages);
      append(
        { id: newId(), role: "user", text: question },
        { id: replyId, role: "assistant", text: "Reading your notes", pending: true },
      );
      setBusy(true);
      try {
        const answer = await ask(question, filters, mode, history);
        replace(replyId, { pending: false, text: answer.answer, answer });
      } catch (error) {
        replace(replyId, { pending: false, text: said(error) });
      } finally {
        setBusy(false);
      }
    },
    [append, filters, messages, mode, newId, replace],
  );

  const onAttach = useCallback(
    async (file: File) => {
      const noticeId = newId();
      append(
        { id: newId(), role: "user", text: file.name, attachment: file.name },
        { id: noticeId, role: "assistant", text: "Reading the question paper", pending: true },
      );
      setBusy(true);
      try {
        const paper = await extractPaper(file);
        const questions = paper.questions ?? [];
        if (questions.length === 0) {
          replace(noticeId, {
            pending: false,
            text:
              "I couldn't find any questions in that file. VTU papers number their parts " +
              "like “Q.1 a.” — a straighter, sharper scan usually fixes it.",
          });
          return;
        }

        const marks = questions.reduce((total, q) => total + (q.marks ?? 0), 0);
        const listing = questions
          .map((q) => `**Q${q.number}** ${q.text}${q.marks ? `  ·  ${q.marks} marks` : ""}`)
          .join("\n\n");
        replace(noticeId, {
          pending: false,
          text:
            `Found **${questions.length} questions**${marks ? ` worth ${marks} marks` : ""} in ` +
            `${file.name}. Answering them now.\n\n${listing}`,
        });

        // One at a time, each appearing as soon as it lands
        for (const question of questions) {
          const heading = `Q${question.number} · ${question.text}`;
          const answerId = newId();
          append({
            id: answerId,
            role: "assistant",
            heading,
            text: "Answering",
            pending: true,
          });
          try {
            const answer = await answerPaperQuestion(question, filters);
            replace(answerId, { pending: false, text: answer.answer, answer });
          } catch {
            replace(answerId, {
              pending: false,
              text: "_I couldn't answer this one from the indexed notes._",
            });
          }
        }

        chat.setPaper({ title: paper.title || file.name, questions });
        append({
          id: newId(),
          role: "assistant",
          text:
            "All questions answered. **Download answers as PDF** below the composer gives you " +
            "the whole set to print, diagrams included.",
        });
      } catch (error) {
        replace(noticeId, { pending: false, text: said(error) });
      } finally {
        setBusy(false);
      }
    },
    [append, chat, filters, newId, replace],
  );

  const onDownload = useCallback(async () => {
    if (!chat.paper) return;
    const { title, questions } = chat.paper;
    try {
      const blob = await paperPdf(title, questions as PaperQuestion[], filters);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${title} - answers.pdf`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      append({ id: newId(), role: "assistant", text: said(error) });
    }
  }, [append, chat.paper, filters, newId]);

  return (
    <div className="app">
      <Sidebar
        filters={filters}
        onFilters={setFilters}
        mode={mode}
        onMode={setMode}
        chats={chat.chats}
        currentId={chat.currentId}
        onOpenChat={(id) => {
          chat.openChat(id);
          setMenuOpen(false);
        }}
        onNewChat={() => {
          chat.startNewChat();
          setMenuOpen(false);
        }}
        onDeleteChat={chat.removeChat}
        open={menuOpen}
        onClose={() => setMenuOpen(false)}
      />
      {menuOpen && <div className="scrim" onClick={() => setMenuOpen(false)} />}

      <main className="main">
        <header className="topbar">
          <button
            type="button"
            className="menu"
            onClick={() => setMenuOpen(true)}
            aria-label="Open menu"
          >
            ☰
          </button>
          <span className="scope">
            {filters.subjectCode === ALL
              ? filters.semester === ALL
                ? "Searching every note"
                : `Searching semester ${filters.semester}`
              : `Searching ${filters.subjectCode}${
                  filters.modules.length ? ` · module ${filters.modules.join(", ")}` : ""
                }`}
          </span>
        </header>

        <Thread messages={messages} onStarter={setDraft} />

        <Composer
          busy={busy}
          canDownload={Boolean(chat.paper) && !busy}
          onAsk={onAsk}
          onAttach={onAttach}
          onDownload={onDownload}
          draft={draft}
          onDraftUsed={() => setDraft("")}
        />
      </main>
    </div>
  );
}
