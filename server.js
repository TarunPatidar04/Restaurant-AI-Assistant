import { config } from "dotenv";
config();
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import path from "path";

import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import { DynamicStructuredTool } from "@langchain/core/tools";

const app = express();
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

const _dirname = path.resolve();

const getMenuTool = new DynamicStructuredTool({
  name: "getMenu",
  description: `Returns the final answer for today's menu for the given
category (breakfast, lunch, or dinner). Use this tool to answer the user's
menu question directly.`,
  schema: z.object({
    category: z
      .string()
      .describe("Type of food. Example: breakfast, lunch, dinner"),
  }),
  func: async ({ category }) => {
    const menus = {
      breakfast: "Aloo Paratha, Poha, Masala Chai",
      lunch: "Paneer Butter Masala, Dal Fry, Jeera Rice, Roti",
      dinner: "Veg Biryani, Raita, Salad, Gulab Jamun",
    };
    return menus[category.toLowerCase()] || "No menu found for that category.";
  },
});

const model = new ChatGoogleGenerativeAI({
  model: "models/gemini-2.5-flash",
  maxOutputTokens: 2048,
  temperature: 0.7,
  apiKey: process.env.GOOGLE_API_KEY,
});

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant that uses tools when needed."],
  ["human", "{input}"],
  ["ai", "{agent_scratchpad}"],
]);

const agent = await createToolCallingAgent({
  llm: model,
  tools: [getMenuTool],
  prompt: prompt,
});

const executor = await AgentExecutor.fromAgentAndTools({
  agent,
  tools: [getMenuTool],
  maxIterations: 3,
  verbose: true,
  returnIntermediateSteps: true,
});

app.get("/", (req, res) => {
  res.sendFile(path.join(_dirname, "public", "index.html"));
});

app.post("/api/chat", async (req, res) => {
  try {
    const { input } = req.body;
    // console.log("input", input);
    const result = await executor.invoke({ input });
    // console.log("result : ", result);
    const data = result?.intermediateSteps[0]?.observation;
    // console.log("data : ", data);
    if (result.output && data) {
      return res.json({ output: data });
    } else if (data != null) {
      return res.json({ output: data });
    }
    res.status(500).json({ output: "Agent could not find a valid answer" });
  } catch (error) {
    console.log("error POST : ", error);
  }
});

app.listen(3000, () => {
  console.log("Server running on http://localhost:3000");
});
