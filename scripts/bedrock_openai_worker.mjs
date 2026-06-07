import { createInterface } from "node:readline";
import { BedrockOpenAI } from "openai";

const clients = new Map();

function getClient(region) {
  if (!clients.has(region)) {
    clients.set(region, new BedrockOpenAI({ awsRegion: region }));
  }
  return clients.get(region);
}

function outputText(response) {
  if (typeof response.output_text === "string") {
    return response.output_text;
  }
  const chunks = [];
  for (const item of response.output || []) {
    for (const part of item.content || []) {
      if (typeof part.text === "string") {
        chunks.push(part.text);
      } else if (typeof part.refusal === "string") {
        chunks.push(part.refusal);
      }
    }
  }
  return chunks.join("");
}

function reasoningText(response) {
  const chunks = [];
  for (const item of response.output || []) {
    if (item.type === "reasoning") {
      for (const part of item.summary || []) {
        if (typeof part.text === "string") {
          chunks.push(part.text);
        }
      }
    }
  }
  return chunks.join("") || null;
}

function usage(response) {
  const u = response.usage || {};
  return {
    input_tokens: u.input_tokens ?? null,
    output_tokens: u.output_tokens ?? null
  };
}

async function handle(req) {
  const client = getClient(req.awsRegion);
  const body = {
    model: req.model,
    input: req.input
  };
  if (req.maxOutputTokens != null) {
    body.max_output_tokens = req.maxOutputTokens;
  }
  if (req.reasoningEffort) {
    body.reasoning = { effort: req.reasoningEffort };
  }
  if (req.temperature != null) {
    body.temperature = req.temperature;
  }
  const response = await client.responses.create(body);
  return {
    id: req.id,
    content: outputText(response),
    reasoning: reasoningText(response),
    finish_reason: response.status ?? null,
    usage: usage(response)
  };
}

const rl = createInterface({
  input: process.stdin,
  crlfDelay: Infinity
});

rl.on("line", async (line) => {
  let req;
  try {
    req = JSON.parse(line);
    const result = await handle(req);
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (err) {
    process.stdout.write(`${JSON.stringify({
      id: req?.id ?? null,
      error: err?.stack || err?.message || String(err)
    })}\n`);
  }
});
