'use client';

import { MessageCircle, Send, Square, Trash2, X } from 'lucide-react';
import {
  type FormEvent,
  type KeyboardEvent,
  useEffect,
  useRef,
  useState,
} from 'react';
import {
  buildChatMessages,
  type ConversationMessage,
} from './build-chat-messages';
import { loadDocumentation } from './load-documentation';
import { requestSiemensCompletion } from './siemens-client';

const chatEnabled = process.env.NEXT_PUBLIC_AI_CHAT_ENABLED !== 'false';

export function PublicAIChat() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [input, setInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const abortController = useRef<AbortController | null>(null);
  const messageEnd = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
      if ((event.metaKey || event.ctrlKey) && event.key === '/') {
        event.preventDefault();
        setOpen(true);
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  useEffect(() => {
    messageEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, error, isLoading]);

  if (!chatEnabled) return null;

  const submit = async (event?: FormEvent) => {
    event?.preventDefault();
    const content = input.trim();
    if (!content || isLoading) return;

    const nextMessages: ConversationMessage[] = [
      ...messages,
      { role: 'user', content },
    ];
    const controller = new AbortController();

    setMessages(nextMessages);
    setInput('');
    setError(null);
    setIsLoading(true);
    abortController.current = controller;

    try {
      const documentation = await loadDocumentation();
      const chatMessages = buildChatMessages(
        documentation,
        nextMessages,
        window.location.href,
      );
      const answer = await requestSiemensCompletion(
        chatMessages,
        controller.signal,
      );

      setMessages([...nextMessages, { role: 'assistant', content: answer }]);
    } catch (caughtError) {
      if (
        !(caughtError instanceof DOMException) ||
        caughtError.name !== 'AbortError'
      ) {
        setError(
          caughtError instanceof Error
            ? caughtError.message
            : 'The AI request failed.',
        );
      }
    } finally {
      abortController.current = null;
      setIsLoading(false);
    }
  };

  const onInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  };

  const clearConversation = () => {
    abortController.current?.abort();
    setMessages([]);
    setInput('');
    setError(null);
  };

  return (
    <>
      {open ? (
        <button
          type="button"
          aria-label="Close AI chat"
          onClick={() => setOpen(false)}
          className="fixed inset-0 z-30 bg-fd-overlay backdrop-blur-xs lg:hidden"
        />
      ) : null}

      {open ? (
        <aside
          aria-label="AI chat"
          className="fixed inset-x-2 inset-y-4 z-30 overflow-hidden rounded-2xl border bg-fd-card text-fd-card-foreground shadow-xl lg:sticky lg:top-0 lg:in-[#nd-docs-layout]:[grid-area:toc] lg:ms-auto lg:h-dvh lg:w-[400px] lg:rounded-none lg:border-y-0 lg:border-e-0 lg:border-s lg:shadow-none 2xl:w-[460px]"
        >
          <div className="flex size-full flex-col p-2 lg:p-3">
            <header className="flex items-center gap-2 rounded-xl border bg-fd-secondary px-3 py-2 text-fd-secondary-foreground shadow-sm">
              <div className="flex-1">
                <p className="text-sm font-medium">Ask the research notebook</p>
                <p className="text-xs text-fd-muted-foreground">
                  Answers use the published documentation as context.
                </p>
              </div>
              <button
                type="button"
                onClick={clearConversation}
                aria-label="Clear conversation"
                title="Clear conversation"
                className="rounded-full p-2 text-fd-muted-foreground transition-colors hover:bg-fd-accent hover:text-fd-accent-foreground"
              >
                <Trash2 className="size-4" />
              </button>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="Close"
                className="rounded-full p-2 text-fd-muted-foreground transition-colors hover:bg-fd-accent hover:text-fd-accent-foreground"
              >
                <X className="size-4" />
              </button>
            </header>

            <div
              aria-live="polite"
              className="flex flex-1 flex-col gap-3 overflow-y-auto px-1 py-4"
            >
              {messages.length === 0 ? (
                <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center text-sm text-fd-muted-foreground/80">
                  <MessageCircle className="size-5" />
                  <p>Ask about the mathematics, methods, or experiments.</p>
                </div>
              ) : null}

              {messages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={
                    message.role === 'user'
                      ? 'ms-8 rounded-xl bg-fd-primary px-3 py-2 text-sm whitespace-pre-wrap text-fd-primary-foreground'
                      : 'me-8 rounded-xl border bg-fd-secondary px-3 py-2 text-sm whitespace-pre-wrap text-fd-secondary-foreground'
                  }
                >
                  {message.content}
                </div>
              ))}

              {isLoading ? (
                <p className="px-2 text-xs text-fd-muted-foreground">
                  Reading the notebook and generating an answer…
                </p>
              ) : null}

              {error ? (
                <div className="rounded-xl border border-fd-error/40 bg-fd-error/10 px-3 py-2 text-sm text-fd-error">
                  {error}
                </div>
              ) : null}
              <div ref={messageEnd} />
            </div>

            <form
              onSubmit={submit}
              className="rounded-xl border bg-fd-secondary text-fd-secondary-foreground shadow-sm"
            >
              <div className="flex items-end gap-2 p-2">
                <textarea
                  rows={2}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={onInputKeyDown}
                  placeholder="Ask a question"
                  aria-label="Question"
                  className="max-h-40 min-h-12 flex-1 resize-y bg-transparent p-2 text-sm placeholder:text-fd-muted-foreground focus-visible:outline-none"
                />
                {isLoading ? (
                  <button
                    type="button"
                    onClick={() => abortController.current?.abort()}
                    aria-label="Abort request"
                    title="Abort request"
                    className="grid size-9 place-items-center rounded-full border text-fd-muted-foreground hover:bg-fd-accent"
                  >
                    <Square className="size-3.5 fill-current" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
                    aria-label="Send"
                    className="grid size-9 place-items-center rounded-full bg-fd-primary text-fd-primary-foreground disabled:opacity-40"
                  >
                    <Send className="size-4" />
                  </button>
                )}
              </div>
            </form>
          </div>
        </aside>
      ) : null}

      <button
        type="button"
        onClick={() => setOpen(true)}
        data-state={open ? 'open' : 'closed'}
        className="fixed bottom-4 end-4 z-20 flex w-24 items-center gap-3 rounded-2xl border bg-fd-secondary px-3 py-2 text-sm text-fd-muted-foreground shadow-lg transition-[translate,opacity,background-color] hover:bg-fd-accent hover:text-fd-accent-foreground data-[state=open]:translate-y-10 data-[state=open]:opacity-0"
      >
        <MessageCircle className="size-4.5" />
        Ask AI
      </button>
    </>
  );
}
