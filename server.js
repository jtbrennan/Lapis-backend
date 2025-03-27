import express from "express";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const port = process.env.PORT || 4000;


app.use(express.json());

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

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});