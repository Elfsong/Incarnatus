# Author: Du Mingzhe (dumingzhex@gmail.com)
# Date: 2024/03/09

from openai import OpenAI

class Generation(object):
    def __init__(self) -> None:
        pass
    
    def generate(self):
        raise NotImplementedError("Don't call the interface directly.")
    
    
class LLMGeneration(Generation):
    def __init__(self, model_name="gpt-3.5-turbo") -> None:
        super().__init__()
        self.model_name = model_name
        self.llm_client = OpenAI()
        
    def prompt_list_generate(self, query, history, web_results, personal_results):
        prompt_list = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
        return prompt_list
        
    def generate(self, query, history=None, web_results=None, personal_results=None):
        prompt_list = self.prompt_list_generate(query, history, web_results, personal_results)
        
        response = self.llm_client.chat.completions.create(
            model = self.model_name,
            messages = prompt_list
        )
        return response.choices[0].message.content