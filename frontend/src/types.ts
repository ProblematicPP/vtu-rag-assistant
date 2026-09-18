export interface Figure {
  id: number;
  url: string;
  page: number;
  kind: string;
  caption: string | null;
}

export interface NoteRef {
  note_id: number;
  title: string;
  subject_code: string;
  module_number: number;
  url: string;
  pages: number[];
}

/** What /ask and /agentic-ask return. Agent-only fields are optional. */
export interface AskResponse {
  question: string;
  resolved_question?: string | null;
  answer: string;
  figures?: Figure[];
  notes?: NoteRef[];
  search_mode?: string | null;
  cached?: boolean;
  latency_ms?: number | null;
  in_scope?: boolean;
  rewritten_queries?: string[];
}

export interface Subject {
  code: string;
  name: string;
  semester: number;
}

export interface Module {
  number: number;
  title: string | null;
  notes: number;
  indexed_notes: number;
}

export interface PaperQuestion {
  number: string;
  text: string;
  marks: number | null;
}

export type Role = "user" | "assistant";

export interface Message {
  id: string;
  role: Role;
  /** Prose for a user message; the answer body for an assistant one. */
  text: string;
  /** Shown above the answer when it belongs to a numbered paper question. */
  heading?: string;
  /** Set while the answer is still being fetched. */
  pending?: boolean;
  /** An attached question paper rather than a typed question. */
  attachment?: string;
  answer?: AskResponse;
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  /** Set once a paper has been answered in this chat, so the PDF stays offered. */
  paper?: { title: string; questions: PaperQuestion[] };
  createdAt: number;
}

export interface Filters {
  semester: string;
  subjectCode: string;
  modules: number[];
}

export type Mode = "careful" | "quick";
