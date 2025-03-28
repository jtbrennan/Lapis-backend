import express from "express";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const port = process.env.PORT || 4000;

app.use(express.json());

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  organization: process.env.OPENAI_ORG_ID
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.Index("lapis-01");

// Text chunking function
const chunkText = (text, chunkSize = 1000, chunkOverlap = 200) => {
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    // Find the end of the chunk
    let end = start + chunkSize;
    
    // Ensure we don't cut words in half
    if (end < text.length) {
      // Look backwards to find the last sentence break or word boundary
      while (end > start && !/\s/.test(text[end])) {
        end--;
      }
    } else {
      end = text.length;
    }

    // Extract the chunk
    chunks.push(text.slice(start, end).trim());

    // Move start to create overlap
    start = end - chunkOverlap;
  }

  return chunks;
};

app.post("/embedding", async (req, res) => {
  const { text, id, title, teamId, organizationId } = req.body;

  // Validate required fields
  if (!text || !id || !title || !teamId || !organizationId) {
    return res.status(400).json({
      error: "Text, ID, title, teamId, and organizationId are required.",
      missingFields: {
        text: !text,
        id: !id,
        title: !title,
        teamId: !teamId,
        organizationId: !organizationId
      }
    });
  }

  try {
    // Chunk the text
    const chunks = chunkText(text);

    // Process each chunk
    const embeddingPromises = chunks.map(async (chunk, index) => {
      // Generate embedding for the chunk
      const response = await openai.embeddings.create({
        model: "text-embedding-ada-002",
        input: chunk
      });

      const embedding = response.data[0].embedding;

      // Upsert chunk with unique ID and metadata
      return index.upsert([{
        id: `${id}_chunk_${index}`,
        values: embedding,
        metadata: {
          text: chunk,
          title: title,
          teamId: teamId,
          organizationId: organizationId,
          chunkIndex: index,
          totalChunks: chunks.length,
          createdAt: new Date().toISOString()
        }
      }]);
    });

    // Wait for all chunks to be processed
    await Promise.all(embeddingPromises);

    res.json({
      message: "Document processed and embedded successfully!",
      details: {
        id,
        title,
        teamId,
        organizationId,
        chunkCount: chunks.length
      }
    });
  } catch (error) {
    console.error("An error occurred:", error);
    res.status(500).json({
      error: "Failed to generate or store embeddings.",
      details: error.message
    });
  }
});

app.get("/", (req, res) => {
  res.send("Embedding Service is running");
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});

export default app;