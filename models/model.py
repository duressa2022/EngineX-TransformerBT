

class TranslationRequest:
    def __init__(self,sen:str):
        self.sentence = sen
    @staticmethod
    def from_dict(data:dict)->'TranslationRequest':
        if "sentence" not in data or not isinstance(data["sentence"],str):
            raise ValueError("Sentence has to be provided")
        return TranslationRequest(data["sentence"])
    
class TranslationResponse:
    def __init__(self,translation:str):
        self.translation = translation
    
    def to_dict(self):
        return {
            "translation":self.translation
        }



