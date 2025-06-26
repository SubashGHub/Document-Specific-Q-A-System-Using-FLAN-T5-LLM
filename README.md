
# Doc-Specific Q&A System using FLAN-T5(LLM) + RAG



                                                                                                                                
## 📌 Overview

An end-to-end Question Answering (QA) system purpose-built for querying domain-specific documents. Fine-tuned and deployed using a lightweight, efficient stack. This project focuses on extracting grounded answers from **ATM software requirement documentation** using **Retrieval-Augmented Generation (RAG)**.

                                                                                                                                
## ✨ Features
- ✅ Fine-tuned [`flan-t5-base`](https://huggingface.co/google/flan-t5-base) on domain-specific QA pairs
- ✅ Lightweight local inference for resource-constrained environments
- ✅ Semantic retrieval using [`SentenceTransformers`](https://www.sbert.net/) + [`FAISS`](https://github.com/facebookresearch/faiss)
- ✅ Answer re-ranking via [`CrossEncoder`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)
- ✅ Exposed via a `FastAPI` endpoint for real-time querying
- ✅ Retrieval-Augmented Generation (RAG) to ensure contextually grounded answers
- ✅ Logged queries, answers, and context chunks for continuous improvement
## 🏋️‍♂️ Training

- Created custom QA pairs from ATM domain documentation (`train_qa.json`)
- Fine-tuned `flan-t5-base` using `Seq2SeqTrainer`
- Tracked training with loss curves and evaluation logs
## Run Locally

Clone the project

```bash
  git clone https://github.com/SubashGHub/Document-Specific-Q-A-System-Using-FLAN-T5-LLM.git
```
Create Virtual Environment 
```bash
python -m venv venv
venv\Scripts\activate

```
Install dependencies

```bash
  pip install -r requirements.txt
```
Go to the project directory

```bash
  cd ./app_ui
```

Run *FastAPI* Server

```bash
  api_ask:uv_app --port 8000

```

Test the API

```bash
 curl "http://localhost:8000/ask?query=What message is shows after PIN change?"

```

## How It Works

- **Input**: User asks a natural language question.
- **Semantic Search**: FAISS retrieves relevant document chunks using sentence embeddings.
- **Re-ranking**: Chunks are scored by a CrossEncoder to prioritize relevance.
- **Prompting**: Top N chunks are formatted into a prompt and fed into the FLAN-T5 model.
- **Output**: A grounded, accurate answer is generated based on real documentation.

## 📈 Example Query

- **Input**:  "*What message is shows after PIN change?*"
- **Output**: "Your PIN has been successfully changed. Please do not share your PIN with anyone."

## 💻 Tech Stack



| Component             | Description                                        |
|----------------------|----------------------------------------------------|
| 🤖 FLAN-T5           | Lightweight generative model for QA tasks         |
| 🧠 SentenceTransformers | For dense embedding of document and questions  |
| 🗃️ FAISS              | Fast similarity search engine                      |
| 🔁 CrossEncoder        | Improves relevance via context re-ranking         |
| 🚀 FastAPI             | Lightweight web API for serving inference         |
| 📚 Langchain           | For chaining and prompt templating (optional)     |
| 🧪 Transformers        | Model loading, training, and inference             |


## License



[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
