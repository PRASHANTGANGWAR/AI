import requests
import time
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize  # Add this line
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizerFast, EncoderDecoderModel, pipeline, TFAutoModelForSeq2SeqLM
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from summarizer import Summarizer

# Add these lines to download NLTK stopwords
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')


def query_huggingface_api(text):
    API_URL = "https://api-inference.huggingface.co/models/slauw87/bart_summarisation"
    headers = {"Authorization": "Bearer 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"}
    payload = {"inputs": text}

    while True:
        response = requests.post(API_URL, headers=headers, json=payload)
        data = response.json()
        
        if "error" in data and "currently loading" in data["error"]:
            print("Model is still loading. Waiting...")
            time.sleep(10)
        else:
            return data

def generate_summary_bart_base(text):
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    model_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)
    summary = model_pipeline(text)
    return summary[0]['summary_text']

def generate_summary_text_summarization(text):
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def generate_summary_cnn_daily_mail(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization')
    model = EncoderDecoderModel.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization').to(device)
    
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    output = model.generate(input_ids, attention_mask=attention_mask)

    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_summary_hugging_face_api(text):
    output_api = query_huggingface_api(text)
    summary_from_api = output_api[0]['summary_text']
    return summary_from_api

def generate_summary_lsa(text, sentences_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=sentences_count)
    return " ".join([str(sentence) for sentence in summary])

def generate_summary_frequency_based(text, num_sentences=3):
    sentences = sent_tokenize(text)
    frequency = {}
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    
    for word in words:
        word = word.lower()
        if word not in stop_words:
            if word in frequency:
                frequency[word] += 1
            else:
                frequency[word] = 1

    sentence_scores = {}
    for sentence in sentences:
        for word, freq in frequency.items():
            if word in sentence.lower():
                if sentence in sentence_scores:
                    sentence_scores[sentence] += freq
                else:
                    sentence_scores[sentence] = freq

    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    return " ".join(summary_sentences)

def main():
    text = "Machine learning algorithms are trained to find relationships and patterns in data. They use historical data as input to make predictions, classify information, cluster data points, reduce dimensionality and even help generate new content, as demonstrated by new ML-fueled applications such as ChatGPT, Dall-E 2 and GitHub Copilot.Machine learning is widely applicable across many industries. While machine learning is a powerful tool for solving problems, improving business operations and automating tasks, it's also a complex and challenging technology, requiring deep expertise and significant resources. Choosing the right algorithm for a task calls for a strong grasp of mathematics and statistics. Training machine learning algorithms often involves large amounts of good quality data to produce accurate results. The results themselves can be difficult to understand -- particularly the outcomes produced by complex algorithms, such as the deep learning neural networks patterned after the human brain. And ML models can be costly to run and tune.TechTarget's guide to machine learning is a primer on this important field of computer science, further explaining what machine learning is, how to do it and how it is applied in business. You'll find information on the various types of machine learning algorithms, the challenges and best practices associated with developing and deploying ML models, and what the future holds for machine learning. Throughout the guide, there are hyperlinks to related articles that cover the topics in greater depth.Machine learning has played a progressively central role in human society since its beginnings in the mid-20th century, when AI pioneers like Walter Pitts, Warren McCulloch, Alan Turing and John von Neumann laid the groundwork for computation. The training of machines to learn from data and improve over time has enabled organizations to automate routine tasks that were previously done by humans -- in principle, freeing us up for more creative and strategic work."

    # Query Hugging Face API
    summary_from_api = generate_summary_hugging_face_api(text)
    print("-----------------Hugging Face API Output---------------")
    print("Hugging Face API Output:", summary_from_api)

    # BART Base Summary
    summary_bart_base = generate_summary_bart_base(text)
    print("-----------------BART Base Summary---------------")
    print("BART Base Summary:", summary_bart_base)

    # Text Summarization Summary
    summary_text_summarization = generate_summary_text_summarization(text)
    print("-----------------Text Summarization Summary---------------")
    print("Text Summarization Summary:", summary_text_summarization)

    # CNN Daily Mail Summary
    summary_cnn_daily_mail = generate_summary_cnn_daily_mail(text)
    print("-----------------CNN Daily Mail Summary------------")
    print("CNN Daily Mail Summary:", summary_cnn_daily_mail)

    # LSA Summary
    summary_lsa = generate_summary_lsa(text)
    print("-----------------LSA Summary---------------")
    print("LSA Summary:", summary_lsa)

    # Frequency-Based Summary
    summary_frequency_based = generate_summary_frequency_based(text)
    print("-----------------Frequency-Based Summary---------------")
    print("Frequency-Based Summary:", summary_frequency_based)

if __name__ == "__main__":
    main()
