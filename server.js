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
  const { text, id, title, teamId, organizationId } = req.body;

  console.log("Received request to /embedding");
  console.log("Request body:", req.body);

  // Validate required fields
  if (!text || !id || !title || !teamId || !organizationId) {
    console.log("Missing required fields in request");
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
          documentId: documentId, 
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
  const { 
    query, 
    topK = 5, 
    namespace = "default", 
    filter = {},
    includeMetadata = true,
    includeValues = false
  } = req.body;

  try {
    // Generate embedding for the query
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: query
    });

    const queryVector = embeddingResponse.data[0].embedding;

    // Perform hybrid search
    const searchResponse = await index.namespace(namespace).query({
      topK,
      vector: queryVector,
      filter,
      includeMetadata,
      includeValues
    });

    res.json({
      message: "Search completed successfully",
      results: searchResponse.matches || []
    });

  } catch (error) {
    console.error("Search error:", error);
    res.status(500).json({ 
      error: "Failed to perform search",
      details: error.message 
    });
  }
});

// Test route for search
app.get("/test-search", async (req, res) => {
  try {
    const testQuery = "Example search query";
    
    // Generate embedding for test query
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: testQuery
    });

    const queryVector = embeddingResponse.data[0].embedding;

    // Perform test search
    const searchResponse = await index.namespace("default").query({
      topK: 5,
      vector: queryVector,
      includeMetadata: true
    });

    res.json({
      testQuery,
      results: searchResponse.matches || []
    });
  } catch (error) {
    console.error("Test search error:", error);
    res.status(500).json({ 
      error: "Failed to perform test search",
      details: error.message 
    });
  }
});


app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});