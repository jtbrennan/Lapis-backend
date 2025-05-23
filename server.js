import express from "express";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
import cors from "cors";

dotenv.config();

const app = express();
const port = process.env.PORT || 4000;

app.use(express.json());
app.use(cors());

app.get("/", (req, res) => {
  res.send("working");
});

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  organization: process.env.OPENAI_ORG_ID
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.Index("lapis-01");

app.post("/embedding", async (req, res) => {
  const { text, id, documentId, title, teamId, organizationId } = req.body;

  console.log("Received request to /embedding");
  console.log("Request body:", req.body);

  // Validate required fields
  if (!text || !id || !title || !teamId || !organizationId || !documentId) {
    console.log("Missing required fields in request");
    return res.status(400).json({ 
      error: "Text, ID, documentId, title, teamId, and organizationId are required.",
      missingFields: {
        text: !text,
        id: !id,
        documentId: !documentId,
        title: !title,
        teamId: !teamId,
        organizationId: !organizationId
      }
    });
  }

  try {
    console.log("Generating embedding using OpenAI...");
    const response = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: text
    });

    const embedding = response.data[0].embedding;
    console.log("Generated embedding:", embedding);

    // Upsert with metadata
    console.log("Upserting embedding into Pinecone...");
    await index.upsert([
      {
        id: id,
        values: embedding,
        metadata: {
          text: text,
          documentId: documentId, // Explicitly use the documentId from the request
          title: title,
          teamId: teamId,
          organizationId: organizationId,
          createdAt: new Date().toISOString()
        }
      }
    ]);

    console.log("Embedding upserted successfully!");
    res.json({ 
      message: "Embedding stored successfully!", 
      details: {
        id,
        documentId,
        title,
        teamId,
        organizationId
      }
    });

  } catch (error) {
    console.error("An error occurred:", error);
    res.status(500).json({ 
      error: "Failed to generate or store embedding.",
      details: error.message 
    });
  }
});

app.post("/search", async (req, res) => {
  const { query, teamId, organizationId, topK = 5 } = req.body;

  // Validate required fields
  if (!query || !teamId || !organizationId) {
    return res.status(400).json({
      error: "Query, teamId, and organizationId are required.",
      missingFields: {
        query: !query,
        teamId: !teamId,
        organizationId: !organizationId
      }
    });
  }

  try {
    console.log("Generating embedding for search query...");
    // Generate embedding for the search query
    const response = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: query
    });

    const queryEmbedding = response.data[0].embedding;

    console.log("Querying Pinecone for similar embeddings...");
    // Query Pinecone with the embedding
    const results = await index.query({
      topK: Number(topK),
      vector: queryEmbedding,
      filter: {
        teamId: { "$eq": teamId },
        organizationId: { "$eq": organizationId }
      },
      includeMetadata: true
    });

    console.log("Search results:", results);
    
    // Format the results for the client
    const formattedResults = results.matches.map(match => ({
      id: match.id,
      score: match.score,
      text: match.metadata?.text,
      documentId: match.metadata?.documentId,
      title: match.metadata?.title,
      teamId: match.metadata?.teamId,
      organizationId: match.metadata?.organizationId,
      createdAt: match.metadata?.createdAt
    }));

    res.json({
      query: query,
      results: formattedResults
    });

  } catch (error) {
    console.error("Search error:", error);
    res.status(500).json({
      error: "Failed to perform search.",
      details: error.message
    });
  }
});


app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});