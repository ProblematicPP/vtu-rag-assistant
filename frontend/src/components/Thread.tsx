import { useEffect, useRef } from "react";

import type { Message } from "../types";
import { Answer } from "./Answer";
import { PaperIcon } from "./icons";

const STARTERS = [
  "Explain dual-mode operation with a neat diagram",
  "What is a system call? List its types",
  "Difference between monolithic and microkernel structures",
];

function Waiting({ text }: { text: string }) {
  return (
    <p className="waiting" role="status">
      {text}
      <span className="dots" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
    </p>
  );
}

function Bubble({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="turn user">
        <div className="bubble">
          {message.attachment ? (
            <>
              <PaperIcon /> <strong>{message.attachment}</strong> — answer every question
            </>
          ) : (
            message.text
          )}
        </div>
      </div>
    );
  }
  return (
    <div className="turn assistant">
      {message.heading && <h3 className="q-heading">{message.heading}</h3>}
      {message.pending ? (
        <Waiting text={message.text} />
      ) : (
        <Answer text={message.text} answer={message.answer} />
      )}
    </div>
  );
}

export function Thread({
  messages,
  onStarter,
}: {
  messages: Message[];
  onStarter: (text: string) => void;
}) {
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="thread empty">
        <div className="greeting">
          <span className="star" aria-hidden="true">
            ✳
          </span>
          <h1>What are we revising?</h1>
          <p>Answers come only from the notes you&rsquo;ve added — never from anywhere else.</p>
          <div className="starters">
            {STARTERS.map((text) => (
              <button key={text} type="button" onClick={() => onStarter(text)}>
                {text}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="thread">
      <div className="turns">
        {messages.map((message) => (
          <Bubble key={message.id} message={message} />
        ))}
        <div ref={end} />
      </div>
    </div>
  );
}
