import type {
  AskResponse,
  Filters,
  Message,
  Mode,
  Module,
  PaperQuestion,
  Subject,
} from "./types";

export const ALL = "all";

/** Same origin: nginx (in the container) and Vite (in dev) proxy to the API. */
async function get<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return (await response.json()) as T;
}

/** What a bare status code means to someone waiting on an answer. */
function plainly(status: number) {
  if (status === 503) return "the model or the search index isn't responding yet";
  if (status === 504) return "the answer took too long to come back";
  if (status >= 500) return "the API hit an error answering that";
  if (status === 422) return "the API couldn't make sense of that request";
  return `the API answered ${status}`;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error((data as { detail?: string }).detail || plainly(response.status));
  }
  return data as T;
}

export function searchScope(filters: Filters, branch = "cse", scheme = "2022") {
  return {
    branch,
    scheme,
    ...(filters.semester !== ALL ? { semester: Number(filters.semester) } : {}),
    ...(filters.subjectCode !== ALL ? { subject_code: filters.subjectCode } : {}),
    module_numbers: filters.modules,
  };
}

/**
 * The turns a follow-up can lean on. Paper uploads and answers still being
 * fetched are left out — neither is a question the student asked in words.
 */
export function historyFrom(messages: Message[], limit = 6) {
  const turns: { question: string; answer: string }[] = [];
  messages.forEach((message, index) => {
    if (message.role !== "user" || message.attachment) return;
    const reply = messages[index + 1];
    const answered = reply && reply.role === "assistant" && !reply.pending;
    turns.push({ question: message.text, answer: answered ? reply.text : "" });
  });
  return turns.slice(-limit);
}

export function listSubjects(semester: string) {
  const params = new URLSearchParams({ branch: "cse", scheme: "2022" });
  if (semester !== ALL) params.set("semester", semester);
  return get<Subject[]>(`/api/v1/subjects?${params}`);
}

export function listModules(code: string) {
  return get<{ modules: Module[] }>(`/api/v1/subjects/${code}/modules`).then((d) => d.modules);
}

export function ask(
  question: string,
  filters: Filters,
  mode: Mode,
  history: { question: string; answer: string }[],
): Promise<AskResponse> {
  const endpoint = mode === "careful" ? "agentic-ask" : "ask";
  return post<AskResponse>(`/api/v1/${endpoint}`, {
    question,
    history,
    ...searchScope(filters),
  });
}

export async function extractPaper(file: File) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/v1/papers/extract", { method: "POST", body: form });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "That file couldn't be read");
  return data as { title: string; questions: PaperQuestion[] };
}

export function answerPaperQuestion(question: PaperQuestion, filters: Filters) {
  return post<AskResponse>("/api/v1/ask", {
    question: question.text,
    marks: question.marks,
    ...searchScope(filters),
  });
}

export async function paperPdf(title: string, questions: PaperQuestion[], filters: Filters) {
  const response = await fetch("/api/v1/papers/pdf", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, questions, ...searchScope(filters) }),
  });
  if (!response.ok) throw new Error("The PDF couldn't be built");
  return response.blob();
}

export interface IndexSummary {
  notes: number;
  subjects: number;
  sections: number;
  matching: string;
  reachable: boolean;
}

/**
 * One honest line about what can actually be searched right now.
 *
 * /health answers 503 while any service is degraded, but the body still says
 * which ones — reading it beats reporting the whole API unreachable.
 */
export async function indexSummary(): Promise<IndexSummary> {
  try {
    const [health, notes] = await Promise.all([
      fetch("/health").then(
        (r) => r.json() as Promise<{ services: Record<string, { ok: boolean; chunks?: number }> }>,
      ),
      get<{ status: string; subject_code: string }[]>("/api/v1/notes?limit=500"),
    ]);
    const indexed = notes.filter((n) => n.status === "indexed");
    return {
      notes: indexed.length,
      subjects: new Set(indexed.map((n) => n.subject_code)).size,
      sections: health?.services?.opensearch?.chunks ?? 0,
      matching: health?.services?.embeddings?.ok ? "keyword and meaning" : "keyword only",
      reachable: true,
    };
  } catch {
    return { notes: 0, subjects: 0, sections: 0, matching: "", reachable: false };
  }
}
