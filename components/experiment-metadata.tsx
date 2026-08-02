interface ExperimentMetadataProps {
  status?: string;
  lastVerified?: string;
  sourceCommit?: string;
  environment?: string;
  randomSeed?: string;
  reproduce?: string;
  reproducibility?: string;
}

export function ExperimentMetadata(props: ExperimentMetadataProps) {
  const rows = [
    ['Status', props.status],
    ['Last verified', props.lastVerified],
    ['Source commit', props.sourceCommit],
    ['Environment', props.environment],
    ['Random seed', props.randomSeed],
    ['Reproduce', props.reproduce],
    ['Reproducibility', props.reproducibility],
  ].filter((row): row is [string, string] => Boolean(row[1]));

  if (rows.length === 0) return null;

  return (
    <dl className="my-4 grid grid-cols-[max-content_minmax(0,1fr)] gap-x-4 gap-y-1 rounded-lg border bg-fd-secondary/40 px-3 py-2 text-sm">
      {rows.map(([label, value]) => (
        <div key={label} className="contents">
          <dt className="text-fd-muted-foreground">{label}</dt>
          <dd className="min-w-0 font-medium break-words">
            {label === 'Source commit' || label === 'Reproduce' ? (
              <code>{value}</code>
            ) : (
              value
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}
