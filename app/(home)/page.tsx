import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="mx-auto flex max-w-3xl flex-1 flex-col justify-center px-6 py-20">
      <p className="mb-3 text-sm font-medium text-fd-muted-foreground">Research Notebook</p>
      <h1 className="mb-5 text-4xl font-bold tracking-tight sm:text-5xl">
        Scientific ML &amp; Numerical PDEs
      </h1>
      <p className="mb-8 max-w-2xl text-lg text-fd-muted-foreground">
        Notes for ongoing computational experiments, mathematical theory, reproducibility,
        and research planning.
      </p>
      <Link href="/docs" className="w-fit rounded-lg bg-fd-primary px-5 py-3 font-medium text-fd-primary-foreground">
        Open the notebook
      </Link>
    </div>
  );
}
