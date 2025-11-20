import json
import os
from utils.api_client import RAGAPIClient
# Initialize API client


def load_questions(file_path):
    """Load questions from a JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_rag_answers(questions, api_client):
    """Generate RAG answers for a list of questions"""
    rag_responses = []
    for i, tc in enumerate(questions):
        question = tc.get('question', '')
        print(f"[{i}] Generating answer for: {question[:50]}...")
        
        try:
            response = api_client.ask_question(question)
            formatted = api_client.format_response(response)
            
            rag_responses.append({
                'question_id': tc.get('id', i),
                'ground_truth': tc.get('ground_truth', ''),
                'question': question,
                'answer': formatted.get('answer', ''),
                'contexts': formatted.get('sources', []),
                'processing_time': formatted.get('processing_time', 0)
            })
            
            print(f"  ✅ Got answer ({len(formatted.get('answer',''))} chars)")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            rag_responses.append({
                'question_id': tc.get('id', i),
                'question': question,
                'answer': '',
                'contexts': [],
                'error': str(e)
            })
    
    print(f"\n✅ Generated {len([r for r in rag_responses if r.get('answer')])} answers")
    return rag_responses

def save_results(rag_responses, output_path):
    """Save RAG responses to a JSON file"""
    with open(output_path, "w", encoding="utf-8-sig") as f:
        json.dump(rag_responses, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(rag_responses)} responses to {output_path}")

def process_testcases(root_folder, api_client):
    """Process all questions.json in subfolders and save results.json in the same folder"""
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file == "questions.json":
                input_path = os.path.join(subdir, file)
                output_path = os.path.join(subdir, "results.json")  # lưu ngay cùng folder
                
                print(f"\n📂 Processing file: {input_path}")
                questions = load_questions(input_path)
                responses = generate_rag_answers(questions, api_client)
                save_results(responses, output_path)
api_client = RAGAPIClient(base_url="http://localhost:8000")

root_folder = "/Users/admin/Documents/School/compare_rag_eval/data/testcase"
print("Looking for testcases in:", os.path.abspath(root_folder))

for subdir, dirs, files in os.walk(root_folder):
    for file in files:
        if file == "questions.json":
            print("  ✅ Found file:", os.path.join(subdir, file))

process_testcases(root_folder, api_client)