import json
from datetime import datetime
from fastapi import FastAPI, Query
from transformers import AutoTokenizer, T5ForConditionalGeneration
from langchain.llms.base import LLM
from typing import Optional, List, ClassVar
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vecstore import doc_emb
from utils import path_utils


# ---- Local LLM Wrapper ----
class FlanT5LLM(LLM):
    model_name: str = "google/flan-t5-base"
    fine_tuned: ClassVar[str] = path_utils.get_path('ft_model_path')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_tokenizer", AutoTokenizer.from_pretrained(self.fine_tuned))
        object.__setattr__(self, "_model", T5ForConditionalGeneration.from_pretrained(self.fine_tuned))

    @property
    def _llm_type(self) -> str:
        return "flan-t5"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Make prompt explicit for instruction-tuned model
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,  # Enable sampling
            top_k=50,  # Top-k sampling
            top_p=0.95,  # Nucleus sampling
            temperature=0.8  # Controls creativity
        )
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return response


llm = FlanT5LLM()

# Save query, context, answer
def save_out_json(query, retrieved_chunks, answer):
    currenttime = datetime.now().replace(microsecond=0).isoformat()
    data_to_save = {
        "timestamp": currenttime,
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "Generated answer": answer
    }
    with open('../data/query_answers_.json', 'a', encoding='utf-8') as f:
        f.write(json.dumps(data_to_save, ensure_ascii=False, indent=3))
        f.write(',')
        f.write("\n")

uv_app = FastAPI()

# ---- API Endpoint ----
@uv_app.get("/ask")
def ask_question(query: str = Query(description="Your question about the ATM requirements")):

    query_emb = doc_emb.model.encode([query], convert_to_numpy=True)
    query_emb = doc_emb.normalize(query_emb)  # Normalize query

    distances, indices = doc_emb.index.search(query_emb, 5)
    retrieved_texts = [doc_emb.meta['texts'][idx] for idx in indices[0]]

    # ---------- Re-rank ----------
    rerank_inputs = [[query, chunk] for chunk in retrieved_texts]
    scores = doc_emb.reranker_model.predict(rerank_inputs)

    top_k = 3
    ranked_chunks = [chunk for _, chunk in sorted(zip(scores, retrieved_texts), reverse=True)]
    context = [txt for txt in ranked_chunks[:top_k]]

    # Combine context and query for LLM input
    prompt = (f"You're a helpful assistant specialized in domine specific Q&A."
            f"Based on the following context, answer the question clearly and in a conversational tone."
            # f"Instruction: Answer the question truthfully based only on the provided context."
                f"context: {context}"
                f"Question: {query}"
                f"Answer: "
             )

    # Generate answer
    answer = llm(prompt)

    # Save all info
    save_out_json(query, context, answer)

    return {"answer": answer}