
from transformers import pipeline
import torch

class NERModel:
    def __init__(self, model_path="dbmdz/bert-large-cased-finetuned-conll03-english"):
        """Initialize NER model"""
        self.pipeline = pipeline(
            "ner",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )

    def extract_entities(self, text):
        """Extract entities from text"""
        entities = self.pipeline(text)
        return [
            {
                "text": ent["word"],
                "label": ent["entity_group"],
                "score": ent["score"]
            }
            for ent in entities
        ]

    def extract_entities_batch(self, texts):
        """Extract entities from multiple texts"""
        return [self.extract_entities(text) for text in texts]

# Example usage
if __name__ == "__main__":
    ner_model = NERModel()
    text = "Apple Inc. is based in Cupertino, California."
    entities = ner_model.extract_entities(text)
    print(f"Text: {text}")
    print("Entities:", entities)
