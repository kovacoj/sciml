export type ChatRole = 'system' | 'user' | 'assistant';

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

interface SiemensResponse {
  choices?: Array<{
    message?: {
      role?: string;
      content?: string;
    };
  }>;
}

function getRequiredConfiguration(): {
  apiKey: string;
  baseUrl: string;
  model: string;
} {
  const apiKey = process.env.NEXT_PUBLIC_SIEMENS_API_KEY;
  const baseUrl =
    process.env.NEXT_PUBLIC_SIEMENS_BASE_URL ??
    'https://api.siemens.com/llm/v1';
  const model =
    process.env.NEXT_PUBLIC_SIEMENS_MODEL ?? 'qwen-3.6-27b';

  if (!apiKey) {
    throw new Error('The Siemens API key was not included in this build.');
  }

  return {
    apiKey,
    baseUrl: baseUrl.replace(/\/$/, ''),
    model,
  };
}

export async function requestSiemensCompletion(
  messages: ChatMessage[],
  signal?: AbortSignal,
): Promise<string> {
  const { apiKey, baseUrl, model } = getRequiredConfiguration();
  let response: Response;

  try {
    response = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      mode: 'cors',
      credentials: 'omit',
      cache: 'no-store',
      signal,
      headers: {
        Accept: 'application/json',
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages,
        max_tokens: 1000,
        temperature: 0.2,
        stream: false,
        chat_template_kwargs: {
          enable_thinking: false,
        },
      }),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw error;
    }

    throw new Error(
      'The browser could not call the Siemens API. The Siemens endpoint may not permit cross-origin browser requests from this GitHub Pages site.',
    );
  }

  if (!response.ok) {
    const responseText = await response.text();

    throw new Error(
      [
        `Siemens LLM request failed with HTTP ${response.status}.`,
        responseText.slice(0, 500),
      ].join(' '),
    );
  }

  const data = (await response.json()) as SiemensResponse;
  const content = data.choices?.[0]?.message?.content;

  if (typeof content !== 'string' || content.trim() === '') {
    throw new Error('The Siemens LLM returned an empty or invalid response.');
  }

  return content.trim();
}
