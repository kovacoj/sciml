import {
  encodeQueueRequest,
  pollForQueueResponse,
  type IssueComment,
  type QueueRequest,
} from './queue-protocol';

export type QueueProgress = 'queued' | 'running' | 'completed';

interface QueueConfiguration {
  token: string;
  owner: string;
  repository: string;
  issue: number;
  timeoutMs: number;
  pollIntervalMs: number;
}

function getConfiguration(): QueueConfiguration {
  const token = process.env.NEXT_PUBLIC_GITHUB_QUEUE_TOKEN;
  const owner = process.env.NEXT_PUBLIC_GITHUB_QUEUE_OWNER;
  const repository = process.env.NEXT_PUBLIC_GITHUB_QUEUE_REPO;
  const issue = Number(process.env.NEXT_PUBLIC_GITHUB_QUEUE_ISSUE);
  const timeoutMs = Number(
    process.env.NEXT_PUBLIC_GITHUB_QUEUE_TIMEOUT_MS ?? 180_000,
  );
  const pollIntervalMs = Number(
    process.env.NEXT_PUBLIC_GITHUB_QUEUE_POLL_INTERVAL_MS ?? 3_000,
  );

  if (!token || !owner || !repository || !Number.isInteger(issue)) {
    throw new Error('The GitHub documentation queue is not configured.');
  }

  return {
    token,
    owner,
    repository,
    issue,
    timeoutMs,
    pollIntervalMs,
  };
}

async function wait(milliseconds: number, signal: AbortSignal): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const timer = window.setTimeout(resolve, milliseconds);
    signal.addEventListener(
      'abort',
      () => {
        window.clearTimeout(timer);
        reject(new DOMException('The request was aborted.', 'AbortError'));
      },
      { once: true },
    );
  });
}

async function githubRequest(
  url: string,
  token: string,
  init: RequestInit,
): Promise<Response> {
  const response = await fetch(url, {
    ...init,
    mode: 'cors',
    credentials: 'omit',
    cache: 'no-store',
    headers: {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
      'X-GitHub-Api-Version': '2022-11-28',
      ...init.headers,
    },
  });

  if (!response.ok) {
    const details = (await response.text()).slice(0, 500);
    throw new Error(
      `GitHub queue request failed with HTTP ${response.status}. ${details}`,
    );
  }

  return response;
}

function commentsUrl(config: QueueConfiguration): string {
  return `https://api.github.com/repos/${config.owner}/${config.repository}/issues/${config.issue}/comments`;
}

export async function enqueueQueueRequest(
  request: QueueRequest,
  signal: AbortSignal,
): Promise<string> {
  const config = getConfiguration();
  const submittedAt = new Date().toISOString();

  await githubRequest(commentsUrl(config), config.token, {
    method: 'POST',
    signal,
    body: JSON.stringify({ body: encodeQueueRequest(request) }),
  });
  return submittedAt;
}

export async function pollQueueRequest(
  requestId: string,
  submittedAt: string,
  signal: AbortSignal,
  onProgress: (progress: QueueProgress) => void,
): Promise<string> {
  const config = getConfiguration();
  return pollForQueueResponse({
    requestId,
    timeoutMs: config.timeoutMs,
    pollIntervalMs: config.pollIntervalMs,
    signal,
    onProgress,
    waitForNext: wait,
    fetchComments: async () => {
      const response = await githubRequest(
        `${commentsUrl(config)}?per_page=100&since=${encodeURIComponent(submittedAt)}`,
        config.token,
        { method: 'GET', signal },
      );
      return (await response.json()) as IssueComment[];
    },
  });
}
