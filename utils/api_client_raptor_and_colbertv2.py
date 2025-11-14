"""
API Client for existing RAG service
"""
import requests
from typing import Dict, Any, List, Optional
from loguru import logger
from .utils import strip_markdown


class RAGAPIClient:
    """
    Client to interact with existing RAG API service
    """
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of RAG service (default: http://localhost:8000/api/v1)
        """
        self.base_url = base_url.rstrip('/')
        # Ensure base_url ends with /api/v1
        if not self.base_url.endswith('/api/v1'):
            self.base_url = f"{self.base_url}/api/v1"
        
        self.ask_endpoint = f"{self.base_url}/query"
        logger.info(f"Initialized RAG API Client: {self.base_url}")
    
    def ask_question(
        self, 
        question: str,
    ) -> Dict[str, Any]:
        """
        Ask a question to the RAG service
        
        Args:
            question: Question to ask
            conversation_id: Optional conversation ID for context
        Returns:
            API response dictionary with answer, sources, suggestions, etc.
        """
        payload = {"query": question}
        
        try:
            logger.info(f"Asking question: {question[:50]}...")
            response = requests.post(
                self.ask_endpoint,
                json=payload,
                timeout=50
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Got response in {data.get('metadata', {}).get('processing_time', 0):.2f}s")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "",
                "sources": []
            }
    
    def check_health(self) -> bool:
        """
        Check if API service is available
        
        Returns:
            True if service is healthy, False otherwise
        """
        from urllib.parse import urlsplit
        parts = urlsplit(self.base_url)
        server_root = f"{parts.scheme}://{parts.netloc}"
        try:
            # Try a simple request
            response = requests.get(f"{server_root}", timeout=5)
            return response.status_code == 200
        except:
            # If /health doesn't exist, try the main endpoint
            try:
                response = requests.post(
                    self.ask_endpoint,
                    json={"query": "test"},
                    timeout=5
                )
                return response.status_code in [200, 400, 422]  # Service exists
            except:
                return False
    
    def format_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format API response for evaluation
        
        Args:
            response: Raw API response
            
        Returns:
            Formatted response with standard fields
        """
        if not response.get('success', False):
            return {
                'answer': '',
                'context': '',
                'sources': [],
                'error': response.get('error', 'Unknown error')
            }
        
        # Extract answer
        answer = response.get('answer', '')
        
        # Extract context as array - each source's content becomes one element
        sources = response.get('sources', [])
        context_array = []
        for source in sources:
            content = source.get('content', '')
            stripped_content = strip_markdown(content)
            if stripped_content:  # Only add non-empty content
                context_array.append(stripped_content)
        
        return {
            'question': response.get('original_question', ''),
            'answer': answer,
            'context': context_array,  # Array of content strings from sources
            'sources': sources,  # Keep full source metadata
            'conversation_id': response.get('conversation_id', ''),
            'suggestions': response.get('suggestions', []),
            'metadata': response.get('metadata', {}),
            'processing_time': response.get('metadata', {}).get('processing_time', 0)
        }
    
    def ask_question_with_strategy(
        self,
        question: str,
        strategy: str = "traditional",
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ask a question using a specific testing strategy/variation
        
        Args:
            question: Question to ask
            strategy: Strategy/variation to use (e.g., "traditional", "self-rag", "custom-v1")
            conversation_id: Optional conversation ID for context
            
        Returns:
            API response dictionary
        """
        # Construct endpoint dynamically based on strategy/variation name
        endpoint = f"{self.base_url}/testing/strategy/{strategy}"
        
        payload = {"question": question}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        try:
            logger.info(f"Asking question with '{strategy}' variation: {question[:50]}...")
            response = requests.post(
                endpoint,
                json=payload,
                timeout=50
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Got response from '{strategy}' in {data.get('metadata', {}).get('processing_time', 0):.2f}s")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for '{strategy}': {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "",
                "sources": []
            }


def generate_test_data_from_api(
    testcases_file: str = "data/testcases.json",
    output_file: str = "data/testcases_with_generated.json",
    api_url: str = "http://localhost:8000"
) -> None:
    """
    Generate answers for test cases using the RAG API
    
    Args:
        testcases_file: Path to test cases JSON
        output_file: Path to save test cases with generated answers
        api_url: Base URL of RAG API
    """
    from .utils import load_json, save_json
    from tqdm import tqdm
    
    logger.info(f"Loading test cases from {testcases_file}")
    testcases = load_json(testcases_file)
    
    # Initialize API client
    client = RAGAPIClient(base_url=api_url)
    
    # Check if service is available
    if not client.check_health():
        logger.error(f"RAG API service not available at {api_url}")
        logger.error("Please make sure the service is running")
        return
    
    logger.info(f"Generating answers for {len(testcases)} test cases")
    
    for testcase in tqdm(testcases, desc="Generating answers"):
        question = testcase.get('question', '')
        
        try:
            # Call API
            response = client.ask_question(question)
            formatted = client.format_response(response)
            
            # Add generated data to testcase
            testcase['actual_output'] = formatted['answer']
            testcase['generated_answer'] = formatted['answer']
            testcase['generated_context'] = formatted['context']
            testcase['api_sources'] = formatted['sources']
            testcase['api_metadata'] = formatted['metadata']
            testcase['processing_time'] = formatted['processing_time']
            
            logger.info(f"Generated answer for {testcase.get('id', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error generating answer for {testcase.get('id', 'unknown')}: {e}")
            testcase['actual_output'] = ''
            testcase['error'] = str(e)
    
    # Save results
    logger.info(f"Saving results to {output_file}")
    save_json(testcases, output_file)
    logger.info("Done!")


if __name__ == "__main__":
    # Test the API client
    client = RAGAPIClient()
    
    print("Testing RAG API Client...")
    print(f"Service health: {client.check_health()}")
    
    # Test question
    response = client.ask_question("Điều kiện để học chương trình CNTT Nhật")
    formatted = client.format_response(response)
    
    print(f"\nQuestion: {formatted.get('question', 'N/A')}")
    print(f"Answer: {formatted.get('answer', 'N/A')[:200]}...")
    print(f"Sources: {len(formatted.get('sources', []))}")
    print(f"Processing time: {formatted.get('processing_time', 0):.2f}s")
