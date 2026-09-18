import { useCallback, useEffect, useRef, useState } from "react";

import type { Chat, Message } from "./types";

const STORE = "chatvtu.chats.v1";
const TITLE_CHARS = 48;

function newId() {
  return Math.random().toString(36).slice(2, 10);
}

export function emptyChat(): Chat {
  return { id: newId(), title: "", messages: [], createdAt: Date.now() };
}

function load(): Chat[] {
  try {
    const raw = localStorage.getItem(STORE);
    const chats = raw ? (JSON.parse(raw) as Chat[]) : [];
    // A reload mid-answer would otherwise leave a bubble waiting forever
    return chats.map((chat) => ({
      ...chat,
      messages: chat.messages.map((m) =>
        m.pending ? { ...m, pending: false, text: m.text || "_That answer was interrupted._" } : m,
      ),
    }));
  } catch {
    return [];
  }
}

/**
 * The conversations in the sidebar, one entry per chat rather than per question.
 *
 * Messages are appended the moment a question is asked, so the thread shows the
 * question and a waiting reply straight away; the answer replaces the waiting
 * reply in place when it arrives.
 */
export function useChats() {
  const [chats, setChats] = useState<Chat[]>(load);
  const [currentId, setCurrentId] = useState<string>(() => emptyChat().id);
  // Kept in a ref so callbacks always append to the chat on screen, even when
  // several updates land during one answer
  const currentRef = useRef(currentId);
  currentRef.current = currentId;

  useEffect(() => {
    try {
      localStorage.setItem(STORE, JSON.stringify(chats));
    } catch {
      // Private windows and blocked site data: the session still works, it just
      // won't be there tomorrow
    }
  }, [chats]);

  const current = chats.find((chat) => chat.id === currentId);
  const messages = current?.messages ?? [];

  /** Applies an edit to the open chat, creating it on the first message. */
  const update = useCallback((edit: (chat: Chat) => Chat) => {
    setChats((previous) => {
      const id = currentRef.current;
      const found = previous.some((chat) => chat.id === id);
      const base = found
        ? previous
        : [...previous, { id, title: "", messages: [], createdAt: Date.now() }];
      return base.map((chat) => (chat.id === id ? edit(chat) : chat));
    });
  }, []);

  const append = useCallback(
    (...added: Message[]) =>
      update((chat) => ({
        ...chat,
        title: chat.title || titleFor(added) || chat.title,
        messages: [...chat.messages, ...added],
      })),
    [update],
  );

  /** Replaces one message by id — how a waiting reply becomes the answer. */
  const replace = useCallback(
    (id: string, next: Partial<Message>) =>
      update((chat) => ({
        ...chat,
        messages: chat.messages.map((m) => (m.id === id ? { ...m, ...next } : m)),
      })),
    [update],
  );

  const setPaper = useCallback(
    (paper: Chat["paper"]) => update((chat) => ({ ...chat, paper })),
    [update],
  );

  const startNewChat = useCallback(() => {
    // Not listed until it holds a question: an empty chat is just a blank page
    setChats((previous) => previous.filter((chat) => chat.messages.length > 0));
    setCurrentId(emptyChat().id);
  }, []);

  const removeChat = useCallback(
    (id: string) => {
      setChats((previous) => previous.filter((chat) => chat.id !== id));
      if (id === currentRef.current) setCurrentId(emptyChat().id);
    },
    [setCurrentId],
  );

  return {
    chats: [...chats].filter((chat) => chat.messages.length > 0).reverse(),
    currentId,
    openChat: setCurrentId,
    messages,
    paper: current?.paper,
    append,
    replace,
    setPaper,
    startNewChat,
    removeChat,
    newId,
  };
}

function titleFor(added: Message[]) {
  const first = added.find((m) => m.role === "user");
  if (!first) return "";
  const text = first.attachment || first.text;
  return text.length > TITLE_CHARS ? `${text.slice(0, TITLE_CHARS).trimEnd()}…` : text;
}
