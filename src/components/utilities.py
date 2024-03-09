# Author: Du Mingzhe (dumingzhex@gmail.com)
# Date: 2024/03/09

import os

class KeyKeeper(object):
    def __init__(self) -> None:
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        
    def validation(self):
        assert self.openai_key, "OpenAI API Key is empty!"