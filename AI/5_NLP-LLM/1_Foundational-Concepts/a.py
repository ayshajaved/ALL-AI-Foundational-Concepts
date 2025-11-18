from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

class BERTForTextClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.pooler_output  # [CLS] token representation
        x = self.dropout(cls_embedding)
        logits = self.classifier(x)
        return logits

# Example usage:
# Prepare inputs with BertTokenizer and create input_ids, attention_mask tensors
# Initialize model: model = BERTForTextClassification('bert-base-uncased', num_classes=2)
# Pass inputs: logits = model(input_ids, attention_mask)
# Use CrossEntropyLoss during training to compute loss and optimize

