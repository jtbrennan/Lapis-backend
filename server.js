import express from "express";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const port = process.env.PORT || 4000;

app.use(express.json());

const cors = require("cors");
app.use(cors({ origin: "http://localhost:3000" }));

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

// Original embedding endpoint
app.post("/embedding", async (req, res) => {
  try {
    const describeIndexResponse = await pinecone.describeIndex("lapis-01");
    console.log("Index Info:", describeIndexResponse);
  } catch (error) {
    console.error("Error fetching index details:", error);
  }
      
  const { text, id } = req.body;
  console.log("Received request to /embedding");
  console.log("Request body:", req.body);
      
  if (!text || !id) {
    console.log("Missing text or id in request");
    return res.status(400).json({ error: "Text and ID are required." });
  }
    
  try {
    console.log("Generating embedding using OpenAI...");
    const response = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: text
    });
      
    console.log("OpenAI embedding response:", response);
      
    const embedding = response.data[0].embedding;
    console.log("Generated embedding:", embedding);
      
    console.log("Upserting embedding into Pinecone...");
    await index.upsert([{ id: id, values: embedding }]);
    console.log("Embedding upserted successfully!");
      
    res.json({ message: "Embedding stored successfully!", id });
  } catch (error) {
    console.error("An error occurred:", error);
    res.status(500).json({ error: "Failed to generate or store embedding." });
  }
});

// Function to chunk text into smaller pieces
function chunkText(text, maxChunkSize = 1000, overlapSize = 100) {
  const chunks = [];
  
  // Simple chunking by character count
  if (text.length <= maxChunkSize) {
    chunks.push(text);
    return chunks;
  }
  
  let startIndex = 0;
  
  while (startIndex < text.length) {
    let endIndex = startIndex + maxChunkSize;
    
    // If we're not at the end of the text, try to find a good breakpoint
    if (endIndex < text.length) {
      // Look for a period, question mark, or exclamation point followed by a space
      const periodIndex = text.indexOf('. ', endIndex - 200);
      const questionIndex = text.indexOf('? ', endIndex - 200);
      const exclamationIndex = text.indexOf('! ', endIndex - 200);
      const newlineIndex = text.indexOf('\n', endIndex - 200);
      
      // Find the closest breakpoint that's before our max size
      const possibleBreaks = [periodIndex, questionIndex, exclamationIndex, newlineIndex]
        .filter(index => index > startIndex && index < endIndex);
      
      if (possibleBreaks.length > 0) {
        endIndex = Math.max(...possibleBreaks) + 1; // +1 to include the period
      }
    } else {
      endIndex = text.length;
    }
    
    chunks.push(text.substring(startIndex, endIndex));
    
    // Move start index forward, accounting for overlap
    startIndex = endIndex - overlapSize > startIndex ? endIndex - overlapSize : startIndex + 1;
  }
  
  return chunks;
}

// New endpoint for chunking and embedding text
app.post("/chunk-and-embed", async (req, res) => {
  const { text, documentId, metadata = {} } = req.body;
  
  console.log("Received request to /chunk-and-embed");
  console.log("Document ID:", documentId);
  
  if (!text || !documentId) {
    console.log("Missing text or documentId in request");
    return res.status(400).json({ error: "Text and documentId are required." });
  }
  
  try {
    // Step 1: Chunk the text
    console.log("Chunking text...");
    const chunks = chunkText(text);
    console.log(`Created ${chunks.length} chunks`);
    
    // Step 2: Generate embeddings and store them in Pinecone
    const results = [];
    
    for (let i = 0; i < chunks.length; i++) {
      const chunkId = `${documentId}-chunk-${i}`;
      const chunk = chunks[i];
      
      // Generate embedding
      console.log(`Generating embedding for chunk ${i+1}/${chunks.length}...`);
      const response = await openai.embeddings.create({
        model: "text-embedding-ada-002",
        input: chunk
      });
      
      const embedding = response.data[0].embedding;
      
      // Store in Pinecone with metadata
      await index.upsert([{ 
        id: chunkId, 
        values: embedding,
        metadata: {
          ...metadata,
          documentId: documentId,
          chunkIndex: i,
          chunkTotal: chunks.length,
          textPreview: chunk.substring(0, 100) + "..."
        }
      }]);
      
      results.push({
        chunkId,
        chunkIndex: i,
        chunkSize: chunk.length
      });
    }
    
    console.log("All chunks processed and stored successfully!");
    res.json({ 
      message: "Document chunked and embedded successfully!",
      documentId,
      chunks: results
    });
    
  } catch (error) {
    console.error("An error occurred:", error);
    res.status(500).json({ error: "Failed to process or store document chunks." });
  }
});

// Endpoint to search for answers
app.post("/search", async (req, res) => {
  const { query, topK = 3 } = req.body;
  
  if (!query) {
    return res.status(400).json({ error: "Query is required." });
  }
  
  try {
    // Generate embedding for the query
    console.log("Generating embedding for search query...");
    const response = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: query
    });
    
    const queryEmbedding = response.data[0].embedding;
    
    // Search Pinecone
    console.log("Searching Pinecone for similar documents...");
    const searchResults = await index.query({
      vector: queryEmbedding,
      topK: topK,
      includeMetadata: true,
      includeValues: false
    });
    
    // Generate answer using OpenAI
    if (searchResults.matches && searchResults.matches.length > 0) {
      // Collect the relevant chunks
      const relevantChunks = searchResults.matches.map(match => {
        return {
          text: match.metadata.textPreview,
          score: match.score
        };
      });
      
      // Use OpenAI to generate an answer based on the chunks
      const contextText = relevantChunks.map(chunk => chunk.text).join("\n\n");
      
      const completionResponse = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: "You are a helpful assistant that answers questions based on the provided context. If the answer cannot be found in the context, say so."
          },
          {
            role: "user",
            content: `Context information:\n${contextText}\n\nQuestion: ${query}\n\nAnswer the question based on the context information.`
          }
        ]
      });
      
      const answer = completionResponse.choices[0].message.content;
      
      res.json({
        query,
        answer,
        sources: relevantChunks,
        rawMatches: searchResults.matches
      });
    } else {
      res.json({
        query,
        answer: "I couldn't find any relevant information to answer your question.",
        sources: [],
        rawMatches: []
      });
    }
  } catch (error) {
    console.error("An error occurred during search:", error);
    res.status(500).json({ error: "Failed to search or generate answer." });
  }
});

// Endpoint to ingest data from various sources
app.post("/ingest", async (req, res) => {
  const { source, documents, sourceType } = req.body;
  
  if (!source || !documents || !Array.isArray(documents)) {
    return res.status(400).json({ error: "Source and documents array are required." });
  }
  
  try {
    console.log(`Ingesting ${documents.length} documents from ${source}...`);
    
    const results = [];
    
    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      
      // Create a unique document ID
      const documentId = `${sourceType}-${source}-${doc.id || i}`;
      
      // Process the document through the chunking and embedding pipeline
      const chunks = chunkText(doc.content);
      
      for (let j = 0; j < chunks.length; j++) {
        const chunkId = `${documentId}-chunk-${j}`;
        const chunk = chunks[j];
        
        // Generate embedding
        const response = await openai.embeddings.create({
          model: "text-embedding-ada-002",
          input: chunk
        });
        
        const embedding = response.data[0].embedding;
        
        // Store in Pinecone with metadata
        await index.upsert([{ 
          id: chunkId, 
          values: embedding,
          metadata: {
            source: source,
            sourceType: sourceType,
            documentId: documentId,
            title: doc.title || "Untitled",
            url: doc.url || "",
            lastUpdated: doc.lastUpdated || new Date().toISOString(),
            chunkIndex: j,
            chunkTotal: chunks.length,
            textPreview: chunk.substring(0, 100) + "..."
          }
        }]);
      }
      
      results.push({
        documentId,
        title: doc.title || "Untitled",
        chunkCount: chunks.length
      });
    }
    
    console.log("All documents processed and stored successfully!");
    res.json({ 
      message: `Successfully ingested ${documents.length} documents from ${source}`,
      results
    });
    
  } catch (error) {
    console.error("An error occurred during ingestion:", error);
    res.status(500).json({ error: "Failed to ingest documents." });
  }
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});