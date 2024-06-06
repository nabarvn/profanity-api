import { Hono } from "hono";
import { cors } from "hono/cors";
import { Index } from "@upstash/vector";
import { PROFANITY_THRESHOLD, WHITELIST } from "./config";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const semanticSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 25,
  separators: [" "],

  // for context
  chunkOverlap: 9,
});

type Env = {
  UPSTASH_VECTOR_REST_URL: string;
  UPSTASH_VECTOR_REST_TOKEN: string;
};

const app = new Hono<{ Bindings: Env }>();

// to make requests from anywhere
app.use(cors());

app.post("/", async (c) => {
  if (c.req.header("Content-Type") !== "application/json") {
    return c.json({ error: "JSON body expected." }, { status: 406 });
  }

  try {
    const index = new Index({
      url: c.env.UPSTASH_VECTOR_REST_URL,
      token: c.env.UPSTASH_VECTOR_REST_TOKEN,

      // disable needed for cf worker deployment
      cache: false,
    });

    const body = await c.req.json();

    let { message } = body as { message: string };

    if (!message) {
      return c.json(
        { error: "Message argument is required." },
        { status: 400 }
      );
    }

    if (message.split(/\s/).length < 2) {
      return c.json(
        {
          error: "Please enter a longer text, at least 2 words.",
        },
        { status: 400 }
      );
    }

    // this is because of the cloudflare worker sub-request limit
    if (message.split(/\s/).length > 35 || message.length > 1000) {
      return c.json(
        {
          error:
            "Due to temporary cloudflare limits, a message can only be up to 35 words or 1000 characters.",
        },
        { status: 400 }
      );
    }

    // filtering out the whitelisted words
    message = message
      .split(/\s/)
      .filter((word) => !WHITELIST.includes(word.toLowerCase()))
      .join(" ");

    // taking the message and splitting it into chunks
    const [wordChunks, semanticChunks] = await Promise.all([
      splitTextIntoWords(message),
      splitTextIntoSemantics(message),
    ]);

    // sets do not allow duplicate entries
    const flaggedFor = new Set<{ score: number; text: string }>();

    // checking for profanity
    const vectorRes = await Promise.all([
      ...wordChunks.map(async (wordChunk) => {
        // returns `topK` number of vectors
        const [vector] = await index.query({
          topK: 1,
          data: wordChunk,
          includeMetadata: true,
        });

        // set the profanity threshold to high value for a close word match
        if (vector && vector.score > 0.95) {
          flaggedFor.add({
            text: vector.metadata?.text as string,
            score: vector.score,
          });
        }

        return { score: 0 };
      }),

      ...semanticChunks.map(async (semanticChunk) => {
        // returns `topK` number of vectors
        const [vector] = await index.query({
          topK: 1,
          data: semanticChunk,
          includeMetadata: true,
        });

        // set the profanity threshold to a lower value for multiple words
        if (vector && vector.score > PROFANITY_THRESHOLD) {
          flaggedFor.add({
            text: vector.metadata?.text as string,
            score: vector.score,
          });
        }

        return vector;
      }),
    ]);

    if (flaggedFor.size > 0) {
      // accessing the entry with highest profanity
      const mostProfaneWord = Array.from(flaggedFor).sort((a, b) =>
        a.score > b.score ? -1 : 1
      )[0];

      return c.json({
        isProfanity: true,
        score: mostProfaneWord.score,
        flaggedFor: mostProfaneWord.text,
      });
    } else {
      const sorted = vectorRes.sort((a, b) => (a.score > b.score ? -1 : 1))[0];

      return c.json({
        isProfanity: false,
        score: sorted.score,
      });
    }
  } catch (err) {
    console.error(err);

    return c.json(
      { error: "Something went wrong.", err: JSON.stringify(err) },
      { status: 500 }
    );
  }
});

function splitTextIntoWords(text: string): string[] {
  return text.split(/\s/);
}

async function splitTextIntoSemantics(text: string): Promise<string[]> {
  // no semantics for single words
  if (text.split(/\s/).length === 1) return [];

  const documents = await semanticSplitter.createDocuments([text]);
  const chunks = documents.map((chunk) => chunk.pageContent);

  return chunks;
}

export default app;
