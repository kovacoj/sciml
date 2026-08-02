'use client';

import { MessageCircle, Send, Square, Trash2, X } from 'lucide-react';
import {
  type FormEvent,
  type KeyboardEvent,
  useEffect,
  useRef,
  useState,
} from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import {
  submitQueueRequest,
  type QueueProgress,
} from './github-queue-client';
import type { QueueConversationMessage } from './queue-protocol';

const chatEnabled = process.env.NEXT_PUBLIC_AI_CHAT_ENABLED !== 'false';
const basePath =
  process.env.NEXT_PUBLIC_BASE_PATH?.replace(/\/$/, '') ?? '';

function resolveChatLink(href: string | undefined): string | undefined {
  if (!href?.startsWith('/') || href.startsWith(`${basePath}/`)) return href;
  return `${basePath}${href}`;
}

export function PublicAIChat() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<QueueConversationMessage[]>([]);
  const [input, setInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState<QueueProgress | null>(null);
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

    const nextMessages: QueueConversationMessage[] = [
      ...messages,
      { role: 'user', content },
    ];
    const controller = new AbortController();

    setMessages(nextMessages);
    setInput('');
    setError(null);
    setIsLoading(true);
    setProgress(null);
    abortController.current = controller;

    try {
      const answer = await submitQueueRequest(
        {
          version: 1,
          requestId: crypto.randomUUID(),
          question: content,
          currentPageUrl: window.location.href,
          conversation: nextMessages,
        },
        controller.signal,
        setProgress,
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
      setProgress(null);
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
    setProgress(null);
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
                      : 'prose prose-sm dark:prose-invert me-8 max-w-none rounded-xl border bg-fd-secondary px-3 py-2 text-fd-secondary-foreground'
                  }
                >
                  {message.role === 'assistant' ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                      components={{
                        a: ({ children, href }) => (
                          <a
                            href={resolveChatLink(href)}
                            target="_blank"
                            rel="noreferrer"
                            className="font-medium text-fd-primary underline underline-offset-4"
                          >
                            {children}
                          </a>
                        ),
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  ) : (
                    message.content
                  )}
                </div>
              ))}

              {isLoading ? (
                <p className="px-2 text-xs text-fd-muted-foreground">
                  {progress === 'queued'
                    ? 'Queued in GitHub Actions…'
                    : 'GitHub Actions is generating an answer…'}
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
