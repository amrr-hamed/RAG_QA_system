 
import os
import requests
from langchain.prompts import PromptTemplate
from typing import Dict, List, Tuple

class QAChain:
    def __init__(self):
        """Initialize the QA chain."""
        self.prompt_template = PromptTemplate.from_template(
            """
            حلل الوثائق التالية و افهمها جيدا ثم اجب عن السؤال التالي:
            
            السؤال: {question}

            الوثائق المتاحة: {context}

            الإجابة:
            """
        )

    def call_gemini_api(self, prompt: str) -> str:
        """Call the Gemini API with the given prompt."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                response_data = response.json()
                if 'candidates' in response_data and len(response_data['candidates']) > 0:
                    return response_data['candidates'][0]['content']['parts'][0]['text']
                return ""
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")
                
        except Exception as e:
            raise Exception(f"Error calling Gemini API: {str(e)}")

    def process_query(self, retriever, query: str) -> Tuple[str, List]:
        """Process a query and return the answer and source documents."""
        try:
            # Retrieve relevant documents
            retrieved_docs = retriever.get_relevant_documents(query)
            
            # Create context from retrieved documents
            context = " ".join(doc.page_content for doc in retrieved_docs)
            
            # Format prompt and get answer
            prompt = self.prompt_template.format(question=query, context=context)
            answer = self.call_gemini_api(prompt)
            
            return answer, retrieved_docs

        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")