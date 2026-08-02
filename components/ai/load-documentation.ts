let cachedDocumentation: string | null = null;

function getBasePath(): string {
  return process.env.NEXT_PUBLIC_BASE_PATH?.replace(/\/$/, '') ?? '';
}

export async function loadDocumentation(): Promise<string> {
  if (cachedDocumentation !== null) {
    return cachedDocumentation;
  }

  const response = await fetch(`${getBasePath()}/llms-full.txt`, {
    cache: 'force-cache',
  });

  if (!response.ok) {
    throw new Error(
      `Unable to load documentation: HTTP ${response.status}.`,
    );
  }

  cachedDocumentation = await response.text();
  return cachedDocumentation;
}
