type MarimoNotebookProps = {
  slug: string;
  title: string;
  height?: number;
};

export function MarimoNotebook({
  slug,
  title,
  height = 860,
}: MarimoNotebookProps) {
  const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
  const normalizedSlug = slug.replace(/^\/+|\/+$/g, '');
  const source = `${basePath}/notebooks/${normalizedSlug}/`;

  return (
    <section className="my-8">
      <div className="overflow-hidden rounded-xl border bg-fd-background">
        <iframe
          src={source}
          title={title}
          loading="lazy"
          allow="clipboard-read; clipboard-write; fullscreen"
          sandbox="allow-scripts allow-same-origin allow-downloads allow-forms allow-popups"
          style={{
            display: 'block',
            width: '100%',
            height: `${height}px`,
            border: 0,
          }}
        />
      </div>

      <p className="mt-2 text-sm text-fd-muted-foreground">
        <a
          href={source}
          target="_blank"
          rel="noreferrer"
          className="underline underline-offset-4"
        >
          Open the interactive notebook in a separate page
        </a>
      </p>
    </section>
  );
}
