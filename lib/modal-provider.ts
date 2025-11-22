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
      tagName: 'think',      // tell it which XML tag to treat as reasoning
      separator: '\n\n',     // how to join reasoning chunks (optional)
      // startWithReasoning: true, // optional, only if model sometimes omits the opening <think>
    }),
  ],
});