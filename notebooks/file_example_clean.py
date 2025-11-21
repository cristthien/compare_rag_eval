import os
os.system("pip install ragas datasets langchain langchain-community langchain-openai")
import os
os.system("pip install deepeval pandas matplotlib seaborn scipy")

# Install packages nếu cần (uncomment)
# !pip install ragas datasets langchain langchain-community langchain-openai
# !pip install deepeval pandas matplotlib seaborn scipy

import os, sys, warnings, json, pandas as pd, numpy as np
from typing import Dict, List, Any
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.abspath('.'))

# Load .env và kiểm tra API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Bạn chưa đặt OPENAI_API_KEY trong file .env")

print("✅ Setup complete!")
# Load test cases
with open('../data/testcase/factual_testcase/results.json', 'r', encoding='utf-8-sig') as f:
    results = json.load(f)

print(f"📊 Loaded {len(results)} test cases\n")

# Display first test case
print("Example test case:")
for  testcase in results[:2]:
    print(f"ID: {testcase['question_id']}")
    print(f"Question: {testcase['question']}")


import os
os.system(" pip install langchain_openai")

# Setup RAGAS with Ollama (local LLM)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize Ollama LLM for RAGAS
ollama_llm = ChatOpenAI(model="gpt-4o-mini")
ollama_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

ragas_llm = LangchainLLMWrapper(ollama_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)
import json
from ragas import evaluate
from datasets import Dataset

file_path = "../data/testcase/factual_testcase/results.json"

with open(file_path, "r", encoding="utf-8-sig") as f:
    results = json.load(f)

# Lọc những trường hợp có answer hợp lệ
valid_results = [r for r in results if r.get("answer")]

# Chuẩn hoá contexts: RAGAS yêu cầu contexts là list[str], nhưng giữ cả title
def normalize_contexts(ctx_list):
    if not ctx_list:
        return []
    # Trả về list[str] theo format [title:{title}|content:{content}]
    return [f"[title:{c.get('title','')}|content:{c.get('content','')}]" for c in ctx_list]


ragas_data = {
    "question": [r["question"] for r in valid_results],
    "answer": [r["answer"] for r in valid_results],
    "contexts": [normalize_contexts(r["contexts"]) for r in valid_results],
    "ground_truth": [r.get("ground_truth", "") for r in valid_results],  # 🔥 thêm ground truth
}

ragas_dataset = Dataset.from_dict(ragas_data)


# In theo từng câu hỏi kèm content, answer và ground_truth
for i, r in enumerate(valid_results):
    print(f"Câu hỏi: {r['question']}\n")
    
    print("Nội dung:")
    for c in normalize_contexts(r["contexts"]):
        print(f"- {c}\n")  # xuống dòng sau mỗi content
    
    print(f"Answer: {r['answer']}\n")
    print(f"Ground truth: {r.get('ground_truth', '')}\n")
    
    print("="*50 + "\n")  # phân cách giữa các câu hỏi
    
# ---------------------------------------------------
# 2️⃣ Import metrics async
# ---------------------------------------------------

from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness, ContextRelevance, ContextRecall
import pandas as pd

metrics_dict = {
    "AnswerCorrectness": AnswerCorrectness(llm=ragas_llm),
    "AnswerRelevancy": AnswerRelevancy(embeddings=ragas_embeddings),
    "Faithfulness": Faithfulness(llm=ragas_llm),
    "ContextRelevance": ContextRelevance(llm=ragas_llm),
    "ContextRecall": ContextRecall()
}


# ---------------------------------------------------
# 2️⃣ Import các metric async-compatible
# ---------------------------------------------------

from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness, ContextRelevance, ContextRecall
import pandas as pd

metrics_dict = {
    "AnswerCorrectness": AnswerCorrectness(llm=ragas_llm),
    "AnswerRelevancy": AnswerRelevancy(embeddings=ragas_embeddings),
    "Faithfulness": Faithfulness(llm=ragas_llm),
    "ContextRelevance": ContextRelevance(llm=ragas_llm),
    "ContextRecall": ContextRecall()
}

summary_scores = {}

for name, metric in metrics_dict.items():
    print(f"\n🚀 Running metric: {name} ...")
    
    result = evaluate(
        dataset=ragas_dataset,
        metrics=[metric],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        batch_size=5,
        show_progress=True
    )
    
    score = list(result._scores_dict.values())[0]
    summary_scores[name] = score

df_scores = pd.DataFrame([summary_scores])
print("\n✅ Summary scores preview:")
print(df_scores)

import numpy as np

# Tạo dict để lưu trung bình
avg_scores = {}

for col in df_scores.columns:
    avg_scores[col] = np.mean(df_scores[col][0])  # df_scores[col][0] là list score

# Xuất kết quả
print("\n✅ Average score for each metric:")
for metric, avg in avg_scores.items():
    print(f"{metric}: {avg:.4f}")

