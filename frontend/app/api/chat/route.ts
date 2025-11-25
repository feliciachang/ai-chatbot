// app/api/chat/route.ts
import { NextRequest } from 'next/server';
import { streamText, convertToModelMessages, type UIMessage } from 'ai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { wrapLanguageModel, extractReasoningMiddleware } from 'ai';

export const modalProvider = createOpenAICompatible({
    name: 'modal',
    baseURL: 'https://feliciachang--example-vllm-inference-serve.modal.run/v1',
  });

export const modalReasoningModel = wrapLanguageModel({
  model: modalProvider('Qwen/Qwen3-8B-FP8'),
  middleware: [
    extractReasoningMiddleware({
      tagName: 'think',      
      separator: '\n\n',     
    }),
  ],
});

export async function POST(req: NextRequest) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = await streamText({
    model: modalReasoningModel,
    messages: convertToModelMessages(messages),
  });

  return result.toUIMessageStreamResponse();
}
