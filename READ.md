#Install dependencies: npm install
#Start the server: npm start


# Text and Voice API

This API provides various functionalities for text and voice processing, utilizing OpenAI and Llama services. It supports text-to-voice conversion, text summarization, text-based conversations with GPT-3.5 Turbo, voice-to-text translation, and chat operations with the Llama API.

## Endpoints

### Text-to-Voice Conversion

Converts text to voice using OpenAI's TTS model.

- **Endpoint:** `/text-to-voice`
- **Method:** POST
- **Request Body:**
  ```json
  {
    "text": "Your text here..."
  }


### Text Summarization
Summarizes text using OpenAI's GPT-3.5 Turbo.

**Endpoint:** `/summery`
**Method:** POST
**Request Body:**
  ```json
  {
    "text": "Your text here..."
  }


### Text-Based Conversations

Generates responses for text-based conversations using GPT-3.5 Turbo.
  **Endpoint:** `/text`
  **Method:** POST
  **Request Body:**
  ```json
  {
    "text": "Your text here..."
  }

### Voice-to-Text Translation
Translates voice (MP3 file) to text using OpenAI's Whisper-1.

**Endpoint:** `/voice-to-text`
Method: POST
Request Body: (Use multipart/form-data to upload an MP3 file named mp3file)
Response:
Translated text or an error message.

### Chat Operations with Llama
Performs chat operations using the Llama API.
Endpoint: /llama-chat
Method: POST
Request Body:
{
  "text": "Your text for Llama chat here..."
}