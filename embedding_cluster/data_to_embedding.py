import torch

def data_embedding(data_batch, model, tokenizer, device):
    # process the whole batch
    inputs = tokenizer(data_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        encoder_outputs = model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
    return embeddings

if __name__ == '__main__':
    #example
    nlq = "What is the average salary?"
    sql = "SELECT AVG(salary) FROM employees"
    combined_input = nlq + " " + sql
    embeddings = data_embedding(combined_input)