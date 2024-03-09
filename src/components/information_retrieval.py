# Author: Du Mingzhe (dumingzhex@gmail.com)
# Date: 2024/03/09

import utilities
from typing import Any
from openai import OpenAI
from pinecone import Pinecone

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
    
    def query(self) -> Any:
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
        self.embedding_client = EmbeddingModel()
        self.index_client = Pinecone(api_key="********-****-****-****-************")
        self.index = self.index_client.Index("mingzhe")
        
    def create(self, data, namespace='default') -> Any:        
        instances = list()
        
        for instance in data['instances']:
            instances += {
                "id": "vec1", 
                "values": self.embedding_client.get_embedding(instance['content']), 
                "metadata": instance['metadata'],
            }
            
        self.index.upsert(
            vectors = instances,
            namespace = namespace
        )
    
    def query(self, data, top_k=3, filter={}, namespace='default'):
        results = self.index.query(
            namespace = namespace,
            vector = self.embedding_client.get_embedding(data),
            top_k = top_k,
            include_values = True,
            include_metadata = True,
            filter = filter,
        )
        return results
        
    
    