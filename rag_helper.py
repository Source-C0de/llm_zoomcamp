import os
from dotenv import load_dotenv

load_dotenv()
# Contains all our rag functions and its functionality
INSTRUCTION = '''
You are a helpful assistant. Answer the QUESTION using only the information provided in the CONTEXT.

Use the context to find relevant information and provide accurate answers.
If answer is not found in the context,
respond with "I don't know"

'''

PROMPT_TEMPLATE = '''
QUESTION: {question}
CONTEXT: {context}
'''.strip()
class RAGHelper:
    
    def __init__(
        self,
        index,
        client,
        instructions = INSTRUCTION,
        prompt = PROMPT_TEMPLATE,
        filename = 'filename',
        model = os.environ["model"]
    ):
        self.index = index
        self.model = model
        self.instruction = instructions
        self.filename = filename
        self.prompt_template = prompt
        self.client = client
    
        
    def search(self, query , num_result = 5):
        # boost_dict = {"content": 2.0},
        # filter_dict = {'filename': self.filename}
        return self.index.search(
            query,
            num_results = num_result,
        )
        
    def build_context(self, search_result):
        context = ""
        for doc in search_result:
            context += f"""
            FILE: {doc['filename']}{doc['content']}
        """ 
        return context
            
            
            
    def build_prompt(self,query, search_result):
        context = self.build_context(search_result)
        return self.prompt_template.format(
            question=query,
            context = context
        )
        
    def llm(self, prompt):
        input_messages = [
            {
                "role": 'system', 
                'content': self.instruction
            },
            {
                "role": 'user',
                'content': prompt
            }
        ]
        response = self.client.responses.create(
            model = self.model,
            input = input_messages
        )
        return response
    
    def rag(self, query):
        search_result = self.search(query)
        prompt =  self.build_prompt(query, search_result)
        
        result = self.llm(prompt)
        answer = result.output_text
        input_tokens = result.usage.input_tokens
        
        return answer, input_tokens
        
        
        


