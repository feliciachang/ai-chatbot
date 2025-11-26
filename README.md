# How to build a Qwen chatbot with Vercel AI SDK and Modal

It only took me a day to deploy this swanky chatbot running a Qwen-3B model using Vercel and Modal:

https://github.com/user-attachments/assets/1f185fe5-62e1-4eca-893e-8789988165ff


The following tutorial will walk through the frontend and backend for a Qwen chatbot in three simple steps:
1. Deploying the Qwen-3B model on Modal
3. Using Vercel's AI SDK to connect to Modal
4. Building a chat interface with Vercel's AI Elements

Let's start with some project scaffolding:
```bash
mkdir my-chatbot 
cd my-chatbot
mkdir backend
cd backend
```

# 1. Deploying the Qwen-3B model on Modal

The following code snippet is all you need to run LLM inference with Qwen-3B and vLLM and expose and endpoint to serve requests to the model. [There's a great tutorial](https://modal.com/docs/examples/vllm_inference) in the Modal docs that walks through the technical concepts in this code snippet, if you'd like to learn more. 

Paste the following code in a python file named `vllm-inference.py`.

``` python 
import json
from typing import Any

import aiohttp
import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

MODEL_NAME = "Qwen/Qwen3-8B-FP8"
MODEL_REVISION = "220b46e3b2180893580a4454f21f22d3ebb187d3"  # avoid nasty surprises when repos update!

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

FAST_BOOT = True

app = modal.App("example-vllm-inference")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)
```

To deploy the API on Modal, just run: 
 
``` bash
python -m  modal deploy vllm-inference.py
```

If you haven't installed Modal's CLI, create an account on [modal.com](modal.com). 
``` bash
# Optional steps if you run into python versioning issues
uv venv --python 3.11 .venv --prompt modal
source .venv/bin/activate
python3.11 -m ensurepip --upgrade

# Install Modal and connect your account
python -m pip install modal
modal setup
```

Once it is deployed, you’ll see a URL appear in the command line, something like https://your-workspace-name--example-vllm-inference-serve.modal.run.

# 2. Using Vercel's AI SDK to connect to Modal

Now onto the frontend! Start by creating a NextJS app using the defaults.

```bash
cd ..
pnpm create next-app@latest frontend
cd frontend
```

Then install the [OpenAI Compatible provider](https://ai-sdk.dev/providers/openai-compatible-providers#openai-compatible-providers) from Vercel's AI SDK, which we will use to connect to the Qwen-3B model running on Modal

```bash
pnpm add ai @ai-sdk/openai-compatible
```

In the `app` folder, create a `/chat` route by creating a `app/api/chat/route.ts` file with the following nested folders. Then paste the following code:

```ts
import { NextRequest } from 'next/server';
import { streamText, convertToModelMessages, type UIMessage } from 'ai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { wrapLanguageModel, extractReasoningMiddleware } from 'ai';

export const modalProvider = createOpenAICompatible({
    name: 'modal',
    baseURL: 'https://YOUR-MODAL-WORKSPACE--example-vllm-inference-serve.modal.run/v1',
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
```

Make sure to to change the parameters in the BaseURL to match the URL outputted from the command line in the earlier step. It should looke something like `https://your-workspace-name--example-vllm-inference-serve.modal.run`. We want to access the `/v1` endpoint.

# Building a chat interface with Vercel's AI Elements

Then, using Vercel's [AI Elements](https://ai-sdk.dev/elements/examples/chatbot), we can use out-of-the-box UI elements to create a chat interface that uses the NextJS's custom model provider to call Modal.

Install AI Elements and the AI SDK Dependencies:

```bash
npx ai-elements@latest
pnpm add @ai-sdk/react zod
```

In your `app/page.tsx`, replace the code with the file below.

```tsx
'use client';
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import {
  Message,
  MessageContent,
  MessageResponse,
  MessageActions,
  MessageAction,
} from '@/components/ai-elements/message';
import {
  PromptInput,
  PromptInputAttachment,
  PromptInputAttachments,
  PromptInputBody,
  PromptInputHeader,
  type PromptInputMessage,
  PromptInputSelect,
  PromptInputSelectContent,
  PromptInputSelectItem,
  PromptInputSelectTrigger,
  PromptInputSelectValue,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputFooter,
  PromptInputTools,
} from '@/components/ai-elements/prompt-input';
import { useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { CopyIcon, RefreshCcwIcon } from 'lucide-react';
import {
  Source,
  Sources,
  SourcesContent,
  SourcesTrigger,
} from '@/components/ai-elements/sources';
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from '@/components/ai-elements/reasoning';
import { Loader } from '@/components/ai-elements/loader';
const models = [
  {
    name: 'Qwen3',
    value: 'Qwen/Qwen3-8B-FP8',
  },
];
import { DefaultChatTransport } from 'ai';
const ChatBotDemo = () => {
  const [input, setInput] = useState('');
  const [model, setModel] = useState<string>(models[0].value);
  const { messages, sendMessage, status, regenerate } = useChat({
    transport: new DefaultChatTransport({
      api: '/api/chat',
    }),
  });
  const handleSubmit = (message: PromptInputMessage) => {
    const hasText = Boolean(message.text);
    const hasAttachments = Boolean(message.files?.length);
    if (!(hasText || hasAttachments)) {
      return;
    }
    sendMessage(
      { 
        text: message.text || 'Sent with attachments',
        files: message.files 
      },
      {
        body: {
          model: model,
        },
      },
    );
    setInput('');
  };


  return (
    <div className="max-w-4xl mx-auto p-6 relative size-full h-screen">
      <div className="flex flex-col h-full">
        <Conversation className="h-full">
          <ConversationContent>
            {messages.map((message) => (
              <div key={message.id}>
                {message.role === 'assistant' && message.parts.filter((part) => part.type === 'source-url').length > 0 && (
                  <Sources>
                    <SourcesTrigger
                      count={
                        message.parts.filter(
                          (part) => part.type === 'source-url',
                        ).length
                      }
                    />
                    {message.parts.filter((part) => part.type === 'source-url').map((part, i) => (
                      <SourcesContent key={`${message.id}-${i}`}>
                        <Source
                          key={`${message.id}-${i}`}
                          href={part.url}
                          title={part.url}
                        />
                      </SourcesContent>
                    ))}
                  </Sources>
                )}
                {message.parts.map((part, i) => {
                  switch (part.type) {
                    case 'text':
                      return (
                        <Message key={`${message.id}-${i}`} from={message.role}>
                          <MessageContent>
                            <MessageResponse>
                              {part.text}
                            </MessageResponse>
                          </MessageContent>
                          {message.role === 'assistant' && i === messages.length - 1 && (
                            <MessageActions>
                              <MessageAction
                                onClick={() => regenerate()}
                                label="Retry"
                              >
                                <RefreshCcwIcon className="size-3" />
                              </MessageAction>
                              <MessageAction
                                onClick={() =>
                                  navigator.clipboard.writeText(part.text)
                                }
                                label="Copy"
                              >
                                <CopyIcon className="size-3" />
                              </MessageAction>
                            </MessageActions>
                          )}
                        </Message>
                      );
                    case 'reasoning':
                      return (
                        <Reasoning
                          key={`${message.id}-${i}`}
                          className="w-full"
                          isStreaming={status === 'streaming' && i === message.parts.length - 1 && message.id === messages.at(-1)?.id}
                        >
                          <ReasoningTrigger />
                          <ReasoningContent>{part.text}</ReasoningContent>
                        </Reasoning>
                      );
                    default:
                      return null;
                  }
                })}
              </div>
            ))}
            {status === 'submitted' && <Loader />}
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>
        <PromptInput onSubmit={handleSubmit} className="mt-4" globalDrop multiple>
          <PromptInputHeader>
            <PromptInputAttachments>
              {(attachment) => <PromptInputAttachment data={attachment} />}
            </PromptInputAttachments>
          </PromptInputHeader>
          <PromptInputBody>
            <PromptInputTextarea
              onChange={(e) => setInput(e.target.value)}
              value={input}
            />
          </PromptInputBody>
          <PromptInputFooter>
            <PromptInputTools>
              <PromptInputSelect
                onValueChange={(value) => {
                  setModel(value);
                }}
                value={model}
              >
                <PromptInputSelectTrigger>
                  <PromptInputSelectValue />
                </PromptInputSelectTrigger>
                <PromptInputSelectContent>
                  {models.map((model) => (
                    <PromptInputSelectItem key={model.value} value={model.value}>
                      {model.name}
                    </PromptInputSelectItem>
                  ))}
                </PromptInputSelectContent>
              </PromptInputSelect>
            </PromptInputTools>
            <PromptInputSubmit disabled={!input && !status} status={status} />
          </PromptInputFooter>
        </PromptInput>
      </div>
    </div>
  );
};
export default ChatBotDemo;
```

Now, you can play with a fully-fledged chatbot running the Qwen-3B model by running `npm run dev`.

```bash
npm run dev
```