import { source } from '@/lib/source';

export async function getLLMText(
  page: (typeof source)['$inferPage'],
): Promise<string> {
  const processed = await page.data.getText('processed');
  const basePath =
    process.env.NEXT_PUBLIC_BASE_PATH?.replace(/\/$/, '') ?? '';

  return [
    `# ${page.data.title}`,
    `URL: ${basePath}${page.url}`,
    '',
    processed,
  ].join('\n');
}
