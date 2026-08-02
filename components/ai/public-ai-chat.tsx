'use client';
/* eslint-disable react-hooks/exhaustive-deps, react-hooks/purity, react-hooks/set-state-in-effect */

import { MessageCircle, RotateCcw, Send, Square, Trash2, X } from 'lucide-react';
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
  enqueueQueueRequest,
  pollQueueRequest,
  type QueueProgress,
} from './github-queue-client';
import {
  buildRecentContext,
  CHAT_STORAGE_KEY,
  createChatState,
  findPendingMessage,
  parseChatState,
  type StoredMessage,
} from './chat-storage';
import type { QueueConversationMessage } from './queue-protocol';
import {
  extractDocumentationLinkTarget,
  extractNavigationTarget,
  isExplicitNavigationRequest,
  stripNavigationAction,
} from './queue-protocol';

const chatEnabled = process.env.NEXT_PUBLIC_AI_CHAT_ENABLED !== 'false';
const basePath =
  process.env.NEXT_PUBLIC_BASE_PATH?.replace(/\/$/, '') ?? '';
const legacyConversationPrefix = 'sciml-ai-chat-conversation-v2';

function resolveChatLink(href: string | undefined): string | undefined {
  if (!href?.startsWith('/') || href.startsWith(`${basePath}/`)) return href;
  return `${basePath}${href}`;
}

function normalizePath(path: string): string {
  const normalized = path.replace(/\/$/, '');
  return normalized || '/';
}

function isConversation(value: unknown): value is QueueConversationMessage[] {
  return (
    Array.isArray(value) &&
    value.every(
      (message) =>
        typeof message === 'object' &&
        message !== null &&
        ((message as QueueConversationMessage).role === 'user' ||
          (message as QueueConversationMessage).role === 'assistant') &&
        typeof (message as QueueConversationMessage).content === 'string',
    )
  );
}

export function PublicAIChat() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<StoredMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState<QueueProgress | null>(null);
  const [hasRestoredMessages, setHasRestoredMessages] = useState(false);
  const abortController = useRef<AbortController | null>(null);
  const messageViewport = useRef<HTMLDivElement | null>(null);
  const conversationId = useRef('');
  const hasResumedPending = useRef(false);
  const activeRequestId = useRef<string | null>(null);

  const saveState = (nextMessages: StoredMessage[], nextOpen = open) => {
    if (!conversationId.current) return;
    try {
      localStorage.setItem(
        CHAT_STORAGE_KEY,
        JSON.stringify({
          version: 3,
          conversationId: conversationId.current,
          open: nextOpen,
          messages: nextMessages,
        }),
      );
    } catch {
      // Chat remains usable when browser storage is unavailable.
    }
  };

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
    try {
      const currentPath = normalizePath(window.location.pathname);
      const storedState = parseChatState(localStorage.getItem(CHAT_STORAGE_KEY));
      if (storedState) {
        conversationId.current = storedState.conversationId;
        setMessages(storedState.messages);
        setOpen(storedState.open);
      } else {
        const state = createChatState();
        const legacyKeys = Array.from(
          { length: localStorage.length },
          (_, index) => localStorage.key(index),
        ).filter(
          (key): key is string =>
            key?.startsWith(`${legacyConversationPrefix}:`) ?? false,
        );
        const currentLegacyKey = `${legacyConversationPrefix}:${currentPath}`;
        legacyKeys.sort((left) => (left === currentLegacyKey ? -1 : 0));
        let legacyMessages: QueueConversationMessage[] = [];
        for (const key of legacyKeys) {
          const legacyText = localStorage.getItem(key);
          if (!legacyText) continue;
          try {
            const legacy: unknown = JSON.parse(legacyText);
            if (
              isConversation(legacy) &&
              legacy.length > legacyMessages.length
            ) {
              legacyMessages = legacy;
            }
          } catch {
            // Ignore malformed legacy entries while checking other pages.
          }
        }
        state.messages = legacyMessages
          .filter((message) => message.content.trim())
          .map((message) => ({
            ...message,
            id: crypto.randomUUID(),
            createdAt: Date.now(),
            status: 'completed' as const,
          }));
        conversationId.current = state.conversationId;
        setMessages(state.messages);
        setOpen(state.open);
      }

      localStorage.removeItem('sciml-ai-chat-conversation-v1');
      localStorage.removeItem('sciml-ai-chat-open-v1');
      localStorage.removeItem('sciml-ai-chat-handoff-v1');
    } catch {
      // Ignore malformed or unavailable browser storage.
    } finally {
      setHasRestoredMessages(true);
    }
  }, []);

  useEffect(() => {
    if (!hasRestoredMessages) return;
    saveState(messages);
  }, [hasRestoredMessages, messages]);

  useEffect(() => {
    if (!hasRestoredMessages) return;
    saveState(messages, open);
  }, [hasRestoredMessages, open]);

  useEffect(() => {
    const viewport = messageViewport.current;
    if (!viewport) return;

    viewport.scrollTo({
      top: viewport.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages, isLoading]);

  const finishRequest = (
    pending: StoredMessage,
    sourceMessages: StoredMessage[],
    answer: string,
  ) => {
    if (activeRequestId.current !== pending.requestId) return;
    const markerTarget = extractNavigationTarget(answer, basePath);
    const explicitlyRequestedNavigation = isExplicitNavigationRequest(
      pending.content,
    );
    const navigationTarget =
      markerTarget ??
      (explicitlyRequestedNavigation
        ? extractDocumentationLinkTarget(answer, basePath)
        : null);
    const shouldNavigate =
      navigationTarget !== null && explicitlyRequestedNavigation;
    let visibleAnswer = stripNavigationAction(answer);
    if (markerTarget && !shouldNavigate) {
      const link = `[Open the requested documentation page](${markerTarget})`;
      visibleAnswer = visibleAnswer ? `${visibleAnswer}\n\n${link}` : link;
    }
    const completedMessages = sourceMessages.map((message) =>
      message.id === pending.id
        ? { ...message, status: 'completed' as const, error: undefined }
        : message,
    );
    if (visibleAnswer) {
      completedMessages.push({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: visibleAnswer,
        createdAt: Date.now(),
        status: 'completed',
        requestId: pending.requestId,
      });
    }
    setMessages(completedMessages);
    saveState(completedMessages);
    if (navigationTarget && shouldNavigate) {
      window.location.href = new URL(
        navigationTarget,
        window.location.origin,
      ).href;
    }
  };

  const failRequest = (
    pending: StoredMessage,
    sourceMessages: StoredMessage[],
    caughtError: unknown,
  ) => {
    if (activeRequestId.current !== pending.requestId) return;
    const message =
      caughtError instanceof DOMException && caughtError.name === 'AbortError'
        ? 'The request was stopped.'
        : caughtError instanceof Error
          ? caughtError.message
          : 'The AI request failed.';
    const failedMessages = sourceMessages.map((item) =>
      item.id === pending.id
        ? { ...item, status: 'failed' as const, error: message }
        : item,
    );
    setMessages(failedMessages);
    saveState(failedMessages);
  };

  async function resumeRequest(
    pending: StoredMessage,
    sourceMessages: StoredMessage[],
  ) {
    if (!pending.requestId || !pending.submittedAt) return;
    const controller = new AbortController();
    activeRequestId.current = pending.requestId;
    abortController.current = controller;
    setIsLoading(true);
    setProgress('queued');
    setOpen(true);
    try {
      const answer = await pollQueueRequest(
        pending.requestId,
        pending.submittedAt,
        controller.signal,
        setProgress,
      );
      finishRequest(pending, sourceMessages, answer);
    } catch (caughtError) {
      failRequest(pending, sourceMessages, caughtError);
    } finally {
      if (activeRequestId.current === pending.requestId) {
        activeRequestId.current = null;
        abortController.current = null;
      }
      setIsLoading(false);
      setProgress(null);
    }
  }

  useEffect(() => {
    if (!hasRestoredMessages || hasResumedPending.current) return;
    hasResumedPending.current = true;
    const pending = findPendingMessage(messages);
    if (!pending) return;
    if (!pending.submittedAt) {
      setMessages((current) =>
        current.map((message) =>
          message.id === pending.id
            ? {
                ...message,
                status: 'failed',
                error: 'The request was interrupted before it reached the queue.',
              }
            : message,
        ),
      );
      return;
    }
    void resumeRequest(pending, messages);
  }, [hasRestoredMessages]);

  const submitQuestion = async (content: string, retryId?: string) => {
    if (!content || isLoading) return;
    const requestId = crypto.randomUUID();
    const pending: StoredMessage = {
      id: retryId ?? crypto.randomUUID(),
      role: 'user',
      content,
      createdAt: Date.now(),
      status: 'pending',
      requestId,
    };
    const priorMessages = messages.filter((message) => message.id !== retryId);
    const nextMessages = [...priorMessages, pending];
    const controller = new AbortController();
    activeRequestId.current = requestId;

    setMessages(nextMessages);
    saveState(nextMessages);
    setInput('');
    setIsLoading(true);
    setProgress(null);
    abortController.current = controller;

    try {
      const submittedAt = await enqueueQueueRequest(
        {
          version: 2,
          requestId,
          conversationId: conversationId.current,
          question: content,
          currentPageUrl: window.location.href,
          context: { recentMessages: buildRecentContext(priorMessages) },
        },
        controller.signal,
      );
      pending.submittedAt = submittedAt;
      const submittedMessages = nextMessages.map((message) =>
        message.id === pending.id ? { ...pending } : message,
      );
      setMessages(submittedMessages);
      saveState(submittedMessages);
      setProgress('queued');
      const answer = await pollQueueRequest(
        requestId,
        submittedAt,
        controller.signal,
        setProgress,
      );
      finishRequest(pending, submittedMessages, answer);
    } catch (caughtError) {
      failRequest(pending, nextMessages, caughtError);
    } finally {
      if (activeRequestId.current === requestId) {
        activeRequestId.current = null;
        abortController.current = null;
      }
      setIsLoading(false);
      setProgress(null);
    }
  };

  const submit = async (event?: FormEvent) => {
    event?.preventDefault();
    const content = input.trim();
    await submitQuestion(content);
  };

  const onInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  };

  const clearConversation = () => {
    activeRequestId.current = null;
    abortController.current?.abort();
    abortController.current = null;
    setMessages([]);
    try {
      localStorage.removeItem(CHAT_STORAGE_KEY);
    } catch {
      // Chat remains usable when browser storage is unavailable.
    }
    setInput('');
    setProgress(null);
  };

  if (!chatEnabled) return null;

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
          className="fixed inset-x-2 inset-y-4 z-30 overflow-hidden rounded-2xl border bg-fd-card text-fd-card-foreground shadow-xl lg:inset-y-0 lg:end-0 lg:start-auto lg:w-[400px] lg:rounded-none lg:border-y-0 lg:border-e-0 lg:border-s 2xl:w-[460px]"
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
              ref={messageViewport}
              aria-live="polite"
              className="flex flex-1 flex-col gap-3 overflow-y-auto overscroll-contain px-1 py-4"
            >
              {messages.length === 0 ? (
                <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center text-sm text-fd-muted-foreground/80">
                  <MessageCircle className="size-5" />
                  <p>Ask about the mathematics, methods, or experiments.</p>
                </div>
              ) : null}

              {messages.map((message) => (
                <div
                  key={message.id}
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
                        a: ({ children, href }) => {
                          const resolvedHref = resolveChatLink(href);
                          const isExternal =
                            resolvedHref?.startsWith('http://') ||
                            resolvedHref?.startsWith('https://');
                          return (
                            <a
                              href={resolvedHref}
                              target={isExternal ? '_blank' : undefined}
                              rel={isExternal ? 'noreferrer' : undefined}
                              className="font-medium text-fd-primary underline underline-offset-4"
                            >
                              {children}
                            </a>
                          );
                        },
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  ) : (
                    <>
                      {message.content}
                      {message.status === 'failed' ? (
                        <span className="mt-2 flex items-center justify-between gap-2 border-t border-current/20 pt-2 text-xs">
                          <span>{message.error ?? 'Request failed.'}</span>
                          <button
                            type="button"
                            onClick={() =>
                              void submitQuestion(message.content, message.id)
                            }
                            disabled={isLoading}
                            className="inline-flex items-center gap-1 rounded-md border border-current/30 px-2 py-1 font-medium disabled:opacity-50"
                          >
                            <RotateCcw className="size-3" />
                            Retry
                          </button>
                        </span>
                      ) : null}
                    </>
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
              <p className="border-t px-3 py-2 text-[11px] leading-4 text-fd-muted-foreground">
                Conversation history is stored in this browser. Recent context
                is temporarily submitted through a public GitHub issue to
                generate answers.
              </p>
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
