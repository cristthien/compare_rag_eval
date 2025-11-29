#!/usr/bin/env python
# coding: utf-8

# # RAG Evaluation Without Ground Truth — Experiments
# 
# **Mục tiêu:** Thử nghiệm các framework evaluation không cần `expected_answer` và `ground_truth_context`
# 
# ## Vấn đề
# 
# Hệ thống hiện tại phụ thuộc vào:
# - ❌ `expected_answer` — không có sẵn trong thực tế
# - ❌ `ground_truth_context` — không có vì RAG tự động retrieve
# 
# ## Giải pháp
# 
# Test 3 frameworks:
# 1. **DeepEval** — LLM-as-Judge (no ground truth mode)
# 2. **RAGAS** — RAG-specific metrics
# 3. **OpenRAG-Eval** (Optional) — Research approach
# 
# ## Workflow
# 
# 1. Load test cases
# 2. Generate answers from RAG API
# 3. Evaluate with each framework
# 4. Meta-evaluate: So sánh consistency
# 5. Human-in-the-loop validation
# 
# ---

# ## 📦 Setup & Installation
# 
# Cài đặt các thư viện cần thiết

# In[2]:


get_ipython().system('pip install ragas datasets langchain langchain-community langchain-openai')
get_ipython().system('pip install deepeval pandas matplotlib seaborn scipy')


# In[3]:


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


# ## 1️⃣ Load Test Cases
# 
# Load và explore test cases từ `data/testcases.json`

# In[4]:


# Load test cases
with open('../data/testcase/factual_testcase/results.json', 'r', encoding='utf-8-sig') as f:
    results = json.load(f)

print(f"📊 Loaded {len(results)} test cases\n")

# Display first test case
print("Example test case:")
for  testcase in results[:2]:
    print(f"ID: {testcase['question_id']}")
    print(f"Question: {testcase['question']}")



# ## 3️⃣ RAGAS Evaluation (No Ground Truth)
# 
# RAGAS có các metrics không cần `expected_answer`:
# - **Faithfulness** — Answer có faithful với retrieved context không?
# - **Answer Relevancy** — Answer có relevant với question không?
# 
# Cả 2 metrics này đều reference-free!

# In[5]:


get_ipython().system(' pip install langchain_openai')


# In[6]:


# Setup RAGAS with Ollama (local LLM)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize Ollama LLM for RAGAS
ollama_llm = ChatOpenAI(model="gpt-4o-mini")
ollama_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

ragas_llm = LangchainLLMWrapper(ollama_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)


# In[7]:


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



# In[8]:


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



# In[9]:


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


# In[12]:


import numpy as np

# Tạo dict để lưu trung bình
avg_scores = {}

for col in df_scores.columns:
    avg_scores[col] = np.mean(df_scores[col][0])  # df_scores[col][0] là list score

# Xuất kết quả
print("\n✅ Average score for each metric:")
for metric, avg in avg_scores.items():
    print(f"{metric}: {avg:.4f}")


# In[15]:


import pandas as pd
import csv

# Xuất df_scores với UTF-8 và quoting
df_scores.to_csv(
    "summary_scores.csv",      # tên file xuất ra
    index=False,               # không xuất cột index
    encoding="utf-8-sig",      # utf-8-sig giúp Excel đọc đúng tiếng Việt
    quoting=csv.QUOTE_ALL      # đặt tất cả giá trị trong dấu ngoặc kép
)

print("✅ df_scores đã được lưu vào summary_scores.csv với UTF-8 và quoting")


# In[17]:


import pandas as pd
import ast

# Đọc CSV hiện tại
df = pd.read_csv("summary_scores.csv")

# Hàm chuyển chuỗi list thành list Python
def parse_list(s):
    if pd.isna(s):
        return []
    s = s.replace("np.float64(", "").replace(")", "")
    return ast.literal_eval(s)

# Chuyển tất cả các cột từ chuỗi list sang list Python
for col in df.columns:
    df[col] = df[col].apply(parse_list)

# Giả sử tất cả các list trong 1 hàng đầu tiên, flatten mọi cột
num_rows = len(df.iloc[0,0])  # số phần tử trong list đầu tiên
data = {col: df[col].iloc[0] for col in df.columns}

# Tạo DataFrame mới với mỗi phần tử 1 hàng
df_flat = pd.DataFrame(data)

# Lưu CSV mới, mỗi phần tử 1 hàng
df_flat.to_csv("summary_scores.csv", index=False, encoding="utf-8-sig")

print("✅ Đã xuất CSV dạng list từ trên xuống cho tất cả các cột")

