import express from "express";
import fs from "fs";
import path from "path";
import OpenAI from "openai";
import axios from"axios";
import multer from 'multer';
import bodyParser from 'body-parser'; 

const app = express();
app.use(bodyParser.json());

const port = 3000;
const constant={
    CHAT_GPT:"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",
    LLAMA_KEY:"mCBkyECy6J1iUNsYHXvyv1LS2AFVeDCG"
}
const openai = new OpenAI({
  apiKey:  constant.CHAT_GPT,
});
var options = {
    method: "POST",
    headers: {
      Accept: "*/*",
      "User-Agent": "Thunder Client (https://www.thunderclient.com)",
      Authorization: `Bearer ${constant.LLAMA_KEY}`,
      "Content-Type": "application/json",
    },
  };

// Set up multer for handling file uploads
const storage = multer.memoryStorage(); // You can change this to disk storage if needed
const upload = multer({ storage: storage });
// with chat gpt
app.post("/text-to-voice", async (req, res) => {
  try {
    const { text } = req.body;
    const mp3 = await openai.audio.speech.create({
      model: "tts-1",
      voice: "shimmer",
      input: text,
    });

    const buffer = Buffer.from(await mp3.arrayBuffer());
    const speechFile = path.resolve("./speech.mp3");
    await fs.promises.writeFile(speechFile, buffer);

    res.send("Text converted to voice successfully.");
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).send("Internal Server Error");
  }
});
app.post("/summery", async (req,res) => {
  try {
    const { text } = req.body;

    const completion = await openai.chat.completions.create({
      messages: [
        {
          role: "system",
          content: `Give it in a summary with points.${text}. give the repsone in the pointer format with head line
        `,
        },
      ],
      model: "gpt-3.5-turbo",
    });

    res.send(completion.choices[0].message.content);
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).send("Internal Server Error");
  }
});

app.post("/text", async (req,res) => {
  try {
    const { text } = req.body;

    const completion = await openai.chat.completions.create({
      messages: [
        {
          role: "system",
          content: text,
        },
      ],
      model: "gpt-3.5-turbo",
    });

    res.send(completion.choices[0].message.content);
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).send("Internal Server Error");
  }
});

app.post("/voice-to-text", upload.single('mp3file'),async (req,res) => {
  try {
    const buffer = req.file.buffer;
    const translation = await openai.audio.translations.create({
      file: fs.createReadStream("speech.mp3"),
      model: "whisper-1",
    });

    res.send(translation.text);
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).send("Internal Server Error");
  }
});

// with llama

app.post('/llama-chat', (req,res)=> {
    const { text } = req.body;

  options = {
    ...options,
    url: "https://api.deepinfra.com/v1/inference/meta-llama/Llama-2-70b-chat-hf",
   data: {
    text  
    },
  };
  axios
    .request(options)
    .then(function (response) {
      res.send(response.data);
    })
    .catch(function (error) {
        console.log(error,"==error")
      res.send(error.response.data);
    });
})

// Start the Express server
app.listen(port, () => {
    console.log(`API listening at http://localhost:${port}`);
  });