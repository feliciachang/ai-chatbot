// app/api/chat/route.ts
import { NextRequest } from 'next/server';
import { streamText, convertToModelMessages, type UIMessage } from 'ai';
import { modalReasoningModel } from '../../../lib/modal-provider';

export async function POST(req: NextRequest) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = await streamText({
    model: modalReasoningModel,
    messages: convertToModelMessages(messages),
  });

  return result.toUIMessageStreamResponse();
}
