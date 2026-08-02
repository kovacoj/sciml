import type { ChatMessage } from './siemens-client';

export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

export function buildChatMessages(
  documentation: string,
  conversation: ConversationMessage[],
  currentPageUrl: string,
): ChatMessage[] {
  const systemPrompt = [
    'You are an AI assistant for a mathematical research documentation website.',
    '',
    'Use the supplied documentation as the primary reference.',
    'You may add general mathematical or programming background,',
    'but distinguish it from statements made in the documentation.',
    '',
    'Do not claim that an experiment was performed unless the',
    'documentation explicitly says that it was performed.',
    '',
    `Current page: ${currentPageUrl}`,
    '',
    'DOCUMENTATION:',
    documentation,
  ].join('\n');

  return [{ role: 'system', content: systemPrompt }, ...conversation];
}
