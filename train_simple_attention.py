import torch
import torch.optim as optim
from transformers import AutoTokenizer
from simple_attention_model import SimpleAttentionModel, SimpleAttentionConfig
from dataset import get_dataset, MyCollator
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse


def train_simple_attention(data_path, save_path, epochs=5, batch_size=8, lr=1e-4):
    """Train the simple attention model on CoT data"""
    
    # Setup tokenizer (using GPT-2 tokenizer as base)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    config = SimpleAttentionConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,  # Smaller for faster training
        num_attention_heads=8,
        max_position_embeddings=512
    )
    model = SimpleAttentionModel(config)
    
    # Load dataset
    dataset = get_dataset(data_path, tokenizer, max_size=10000)  # Small dataset for testing
    
    # Process dataset
    train_data = []
    for i in range(len(dataset)):
        try:
            processed = process_for_training(dataset[i], tokenizer=tokenizer)
            if processed is not None:
                train_data.append(processed)
        except:
            continue
    
    # Create dataloader
    collator = MyCollator(tokenizer, label_pad_token_id=-100)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator)
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    print(f"Training simple attention model on {len(train_data)} samples for {epochs} epochs")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
            
            # Show predictions every 50 batches
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    logits = outputs.logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    labels_cpu = labels.cpu().numpy()[0]
                    
                    # Find where answer starts (first non -100 label after question)
                    answer_start_idx = None
                    for i, label in enumerate(labels_cpu):
                        if label != -100:
                            answer_start_idx = i
                            break
                    
                    if answer_start_idx is not None:
                        # Show predictions and actual for answer portion only
                        answer_length = min(10, len(labels_cpu) - answer_start_idx)
                        predicted_answer = predicted_ids[0][answer_start_idx:answer_start_idx + answer_length]
                        
                        # Get actual answer tokens for comparison
                        actual_answer = labels_cpu[answer_start_idx:answer_start_idx + answer_length]
                        actual_answer = [token for token in actual_answer if token != -100]  # Remove padding tokens
                        
                        print(f"Predicted: {tokenizer.decode(predicted_answer, skip_special_tokens=True)}")
                        print(f"Actual:    {tokenizer.decode(actual_answer, skip_special_tokens=True)}")
                    else:
                        print("No answer tokens found in this batch")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"Simple attention model saved to {save_path}")
    return model


def process_for_training(sample, tokenizer):
    """Process a sample for training"""
    # Separate question and answer
    question = tokenizer.decode(sample["question_tokenized"])
    answer = tokenizer.decode(sample["answer_tokenized"])
    answer = (answer[4:])

    # Tokenize question and answer separately (no special tokens including eos)
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    
    # Remove any end-of-text tokens
    eos_token_id = tokenizer.eos_token_id
    question_tokens = [t for t in question_tokens if t != eos_token_id]
    answer_tokens = [t for t in answer_tokens if t != eos_token_id]
    
    # Convert to plain Python integers to avoid overflow issues
    question_tokens = list(question_tokens)
    answer_tokens = list(answer_tokens)
    
    # Skip if either is empty
    if not question_tokens or not answer_tokens:
        return None
    
    # Combine for input
    full_tokens = question_tokens + answer_tokens
    
    # Create labels: -100 for question tokens (ignored in loss), actual tokens for answer
    labels = [-100] * len(question_tokens) + list(answer_tokens)
    
    return {
        "input_ids": full_tokens,
        "labels": labels,  # Predict only answer tokens
        "attention_mask": [1] * len(full_tokens)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/gsm_train.json", help="Path to training data")
    parser.add_argument("--save_path", default="simple_attention_checkpoint", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    train_simple_attention(
        data_path=args.data_path,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )