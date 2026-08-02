'use client';

import { MessageCircle, Send, X } from 'lucide-react';
import { useEffect, useState } from 'react';

export function DemoChat() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
      if ((event.metaKey || event.ctrlKey) && event.key === '/') {
        event.preventDefault();
        setOpen(true);
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

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
        <aside className="fixed inset-x-2 inset-y-4 z-30 overflow-hidden rounded-2xl border bg-fd-card text-fd-card-foreground shadow-xl lg:sticky lg:top-0 lg:in-[#nd-docs-layout]:[grid-area:toc] lg:ms-auto lg:h-dvh lg:w-[400px] lg:rounded-none lg:border-y-0 lg:border-e-0 lg:border-s lg:shadow-none 2xl:w-[460px]">
          <div className="flex size-full flex-col p-2 lg:p-3">
            <header className="flex items-start gap-2 rounded-xl border bg-fd-secondary text-fd-secondary-foreground shadow-sm">
              <div className="flex-1 px-3 py-2">
                <p className="mb-2 text-sm font-medium">AI Chat</p>
                <p className="text-xs text-fd-muted-foreground">
                  AI can be inaccurate, please verify the answers.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="Close"
                className="m-1 rounded-full p-2 text-fd-muted-foreground transition-colors hover:bg-fd-accent hover:text-fd-accent-foreground"
              >
                <X className="size-4" />
              </button>
            </header>

            <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center text-sm text-fd-muted-foreground/80">
              <MessageCircle className="size-5 fill-current" strokeWidth={0} />
              <p>Start a new chat below.</p>
              <p className="max-w-64 text-xs">
                Preview only. Connect the custom provider to enable responses.
              </p>
            </div>

            <div className="rounded-xl border bg-fd-secondary text-fd-secondary-foreground shadow-sm">
              <div className="flex items-start pe-2">
                <textarea
                  disabled
                  rows={1}
                  placeholder="Ask a question"
                  className="min-h-12 flex-1 resize-none bg-transparent p-3 text-sm placeholder:text-fd-muted-foreground focus-visible:outline-none disabled:cursor-not-allowed"
                />
                <button
                  type="button"
                  disabled
                  aria-label="Send"
                  className="mt-2 grid size-8 place-items-center rounded-full bg-fd-primary text-fd-primary-foreground opacity-50"
                >
                  <Send className="size-4" />
                </button>
              </div>
            </div>
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
