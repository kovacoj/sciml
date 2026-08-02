'use client';

import { Bot, MessageCircle, Send, X } from 'lucide-react';
import { useState } from 'react';

export function DemoChat() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="fixed bottom-5 right-5 z-40 flex items-center gap-2 rounded-full border bg-fd-primary px-4 py-3 text-sm font-medium text-fd-primary-foreground shadow-lg transition-transform hover:scale-[1.03]"
        aria-label="Open research assistant preview"
      >
        <MessageCircle className="size-4" />
        Ask AI
      </button>

      {open ? (
        <div className="fixed inset-0 z-50 flex items-end justify-end bg-black/20 p-3 backdrop-blur-[2px] sm:p-6">
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby="demo-chat-title"
            className="flex h-[min(38rem,85vh)] w-full max-w-md flex-col overflow-hidden rounded-2xl border bg-fd-background shadow-2xl"
          >
            <header className="flex items-center gap-3 border-b px-4 py-3">
              <div className="grid size-9 place-items-center rounded-full bg-fd-primary text-fd-primary-foreground">
                <Bot className="size-5" />
              </div>
              <div className="min-w-0 flex-1">
                <h2 id="demo-chat-title" className="font-semibold">Research Assistant</h2>
                <p className="text-xs text-fd-muted-foreground">Interface preview</p>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-md p-2 text-fd-muted-foreground hover:bg-fd-accent hover:text-fd-accent-foreground"
                aria-label="Close research assistant preview"
              >
                <X className="size-4" />
              </button>
            </header>

            <div className="flex-1 space-y-4 overflow-y-auto p-4">
              <div className="max-w-[85%] rounded-2xl rounded-tl-sm bg-fd-muted px-4 py-3 text-sm">
                Ask about the notebook, compare experiment notes, or find a mathematical definition.
              </div>
              <div className="ml-auto max-w-[85%] rounded-2xl rounded-tr-sm bg-fd-primary px-4 py-3 text-sm text-fd-primary-foreground">
                Summarize the eigenvalue experiment.
              </div>
              <div className="max-w-[85%] rounded-2xl rounded-tl-sm border px-4 py-3 text-sm">
                A custom OpenAI-compatible provider has not been connected yet. This panel currently demonstrates the intended chat experience only.
              </div>
            </div>

            <footer className="border-t p-3">
              <div className="flex items-center gap-2 rounded-xl border bg-fd-muted/40 p-2">
                <input
                  disabled
                  placeholder="Connect a provider to start chatting"
                  className="min-w-0 flex-1 bg-transparent px-2 text-sm outline-none disabled:cursor-not-allowed"
                />
                <button
                  type="button"
                  disabled
                  className="grid size-9 place-items-center rounded-lg bg-fd-primary text-fd-primary-foreground opacity-50"
                  aria-label="Send message"
                >
                  <Send className="size-4" />
                </button>
              </div>
              <p className="mt-2 text-center text-xs text-fd-muted-foreground">Demo only. No messages leave your browser.</p>
            </footer>
          </section>
        </div>
      ) : null}
    </>
  );
}
