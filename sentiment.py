from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
model_path = "./model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained('./tokenizer')

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()

    label_map = {0: 'negative', 1: 'positive'}
    return label_map[predicted_class_id]

# Example usage
if __name__ == "__main__":
    example_text = "خیلی راضی نبودم."
    sentiment = classify_sentiment(example_text)
    print(f"Sentiment: {sentiment}")