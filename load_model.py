from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
import torch

config = BertConfig.from_json_file('release/config.json')
model = BertForSequenceClassification(config)
model.load_state_dict(torch.load('release/pytorch_model.bin'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_predicted_label(input_text):

    input_encoding = tokenizer(input_text, truncation=True, padding=True, return_tensors='pt')
    input_ids = input_encoding['input_ids'].to(device)
    attention_mask = input_encoding['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
    predicted_label = predicted_labels.item()
    return predicted_label

