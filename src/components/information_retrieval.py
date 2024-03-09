# Author: Du Mingzhe (dumingzhex@gmail.com)
# Date: 2024/03/09

import utilities
from typing import Any
from openai import OpenAI

class InformationRetrieval(object):
    def __init__(self) -> None:
        pass
        
    def create(self) -> Any:
        raise NotImplementedError("Don't call the interface directly.")
    
    def read(self) -> Any:
        raise NotImplementedError("Don't call the interface directly.")
    
    def update(self) -> Any:
        raise NotImplementedError("Don't call the interface directly.")
    
    def delete(self) -> Any:
        raise NotImplementedError("Don't call the interface directly.")

class EmbeddingModel(object):
    def __init__(self, model_name='text-embedding-3-small') -> None:
        self.client_token = None
        self.model_name = model_name
        self.embedding_client = OpenAI()
    
    def get_embedding(self, text):
        response = self.embedding_client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return response.data[0].embedding
    
class PersonalIndex(InformationRetrieval):
    def __init__(self) -> None:
        super().__init__()
        self.client_token = None
        self.index_client = None
        
    def create(self, data) -> Any:        
        instances = list()
        
        for instance in data['instances']:
            instances += {
                "id": "vec1", 
                "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
                "metadata": {"genre": "drama"},
            }
            
        self.index_client.upsert(
            vectors=instances,
            namespace= "default"
        )
        
    
    