# Coconut single question reasoning visualization with training
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer
from simple_attention_model import SimpleAttentionModel, SimpleAttentionConfig
from single_layer_transformer import SingleLayerTransformer, SingleLayerTransformerConfig
from coconut import Coconut
from dataset import get_dataset, get_cot_latent_dataset, MyCollator
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse


def create_question_with_latent_tokens(question, num_latent_tokens, tokenizer, start_id, latent_id, end_id):
    """Create input with specified number of latent tokens"""
    
    # Encode question
    question_tokens = tokenizer.encode(question + "\n", add_special_tokens=True)
    
    # Add latent tokens
    latent_sequence = [start_id] + [latent_id] * num_latent_tokens + [end_id]
    
    # Combine
    full_sequence = question_tokens + latent_sequence
    
    return torch.tensor(full_sequence).unsqueeze(0)


def extract_hidden_states_during_forward(coconut_model, input_ids, max_steps=10):
    """Extract hidden states at each reasoning step"""
    
    # Create attention mask and position ids
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(0, input_ids.shape[1]).unsqueeze(0)
    labels = input_ids.clone()  # Dummy labels
    
    # Track hidden states manually by replicating Coconut forward logic
    hidden_states_per_step = []
    
    # Get initial embeddings
    inputs_embeds = coconut_model.embedding(input_ids)
    
    # Find latent token positions
    latent_indices = (input_ids == coconut_model.latent_token_id).nonzero()
    
    if len(latent_indices) == 0:
        print("No latent tokens found!")
        return []
    
    latent_positions = [idx[1].item() for idx in latent_indices if idx[0] == 0]
    print(f"Latent token positions: {latent_positions}")
    
    # Process each latent token step by step
    current_embeds = inputs_embeds.clone()
    
    for step in range(min(len(latent_positions), max_steps)):
        print(f"Processing reasoning step {step + 1}")
        
        # Get the position of the current latent token
        latent_pos = latent_positions[step]
        
        # Forward pass through base model up to this position
        with torch.no_grad():
            outputs = coconut_model.base_causallm(
                inputs_embeds=current_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True
            )
            
            # Get hidden states
            hidden_states = outputs.hidden_states[-1]  # Last layer: (1, seq_len, hidden_size)
            
            # Store the hidden state at the latent position
            if latent_pos < hidden_states.shape[1]:
                latent_hidden = hidden_states[0, latent_pos, :].cpu().numpy()
                hidden_states_per_step.append({
                    'step': step + 1,
                    'hidden_state': latent_hidden,
                    'position': latent_pos
                })
                
                # Update the latent token embedding with the computed hidden state
                # Use the hidden state from the previous position (reasoning pattern)
                if latent_pos > 0:
                    prev_hidden = hidden_states[0, latent_pos - 1, :]
                    current_embeds[0, latent_pos, :] = prev_hidden
    
    return hidden_states_per_step


def visualize_reasoning_progression(hidden_states_data, question, save_suffix=""):
    """Create 2D visualization of reasoning progression"""
    
    if len(hidden_states_data) < 2:
        print("Need at least 2 reasoning steps for visualization")
        return
    
    # Extract hidden states
    hidden_states = np.array([step['hidden_state'] for step in hidden_states_data])
    step_numbers = [step['step'] for step in hidden_states_data]
    
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Steps: {step_numbers}")
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    hidden_states_2d = pca.fit_transform(hidden_states)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Coconut Reasoning Progression\nQuestion: {question[:50]}...', fontsize=14)
    
    # Plot 1: Reasoning trajectory
    colors = plt.cm.viridis(np.linspace(0, 1, len(hidden_states_2d)))
    
    axes[0, 0].scatter(hidden_states_2d[:, 0], hidden_states_2d[:, 1], 
                      c=colors, s=100, alpha=0.8)
    
    # Add step numbers
    for i, (x, y) in enumerate(hidden_states_2d):
        axes[0, 0].annotate(f'{i+1}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=12, fontweight='bold')
    
    # Draw trajectory arrows
    for i in range(len(hidden_states_2d) - 1):
        axes[0, 0].arrow(hidden_states_2d[i, 0], hidden_states_2d[i, 1],
                        hidden_states_2d[i+1, 0] - hidden_states_2d[i, 0],
                        hidden_states_2d[i+1, 1] - hidden_states_2d[i, 1],
                        head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.6)
    
    axes[0, 0].set_title('Reasoning Trajectory in Latent Space')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Distance from starting point
    start_point = hidden_states_2d[0]
    distances = [np.linalg.norm(point - start_point) for point in hidden_states_2d]
    
    axes[0, 1].plot(step_numbers, distances, 'bo-', linewidth=2, markersize=8)
    axes[0, 1].set_title('Distance from Initial State')
    axes[0, 1].set_xlabel('Reasoning Step')
    axes[0, 1].set_ylabel('Euclidean Distance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: PC components over time
    axes[1, 0].plot(step_numbers, hidden_states_2d[:, 0], 'ro-', label='PC1', linewidth=2)
    axes[1, 0].plot(step_numbers, hidden_states_2d[:, 1], 'bo-', label='PC2', linewidth=2)
    axes[1, 0].set_title('Principal Components Evolution')
    axes[1, 0].set_xlabel('Reasoning Step')
    axes[1, 0].set_ylabel('PC Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Step-to-step movement
    step_movements = []
    for i in range(1, len(hidden_states_2d)):
        movement = np.linalg.norm(hidden_states_2d[i] - hidden_states_2d[i-1])
        step_movements.append(movement)
    
    if step_movements:
        axes[1, 1].bar(range(2, len(step_movements) + 2), step_movements, alpha=0.7)
        axes[1, 1].set_title('Movement Between Steps')
        axes[1, 1].set_xlabel('Step Transition')
        axes[1, 1].set_ylabel('Distance Moved')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('visualization_results', exist_ok=True)
    filename = f'coconut_reasoning{save_suffix}.png'
    filepath = os.path.join('visualization_results', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {filepath}")
    
    plt.show()
    
    # Print statistics
    print("\n=== Reasoning Analysis ===")
    print(f"Question: {question}")
    print(f"Number of reasoning steps: {len(hidden_states_data)}")
    print(f"Hidden state dimensions: {hidden_states.shape[1]}")
    print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
    print(f"Total reasoning trajectory distance: {distances[-1]:.3f}")
    if step_movements:
        print(f"Average step movement: {np.mean(step_movements):.3f}")
        print(f"Largest single step: {np.max(step_movements):.3f}")


def train_coconut_model(coconut_model, tokenizer, latent_id, start_id, end_id, epochs=3):
    """Train the Coconut model on GSM data"""
    
    print("Training Coconut model...")
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coconut_model.to(device)
    optimizer = optim.AdamW(coconut_model.parameters(), lr=1e-4)
    
    # Load training data
    base_dataset = get_dataset("data/gsm_train.json", tokenizer, max_size=100)
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    
    coconut_model.train()
    
    for epoch in range(epochs):
        print(f"\n=== Training Epoch {epoch+1}/{epochs} ===")
        
        # Check if special tokens are in vocabulary
        print(f"Latent token in vocab: {latent_id != tokenizer.unk_token_id}")
        print(f"Start token in vocab: {start_id != tokenizer.unk_token_id}")
        print(f"End token in vocab: {end_id != tokenizer.unk_token_id}")
        
        # Create simple training data with latent tokens
        modified_samples = []
        for sample in base_dataset:
            question = tokenizer.decode(sample['question_tokenized'])
            answer = tokenizer.decode(sample['answer_tokenized'])
            
            # Tokenize each part separately
            question_tokens = tokenizer.encode(question, add_special_tokens=False)
            latent_tokens = tokenizer.encode(" <|start-latent|> <|latent|> <|latent|> <|latent|> <|end-latent|>", add_special_tokens=False)
            answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
            
            # Combine all tokens
            tokens = question_tokens + latent_tokens + answer_tokens
            
            # Create labels: -100 for question+latent (ignore), actual tokens for answer (predict)
            # Add extra -100 at the end to account for Coconut's sequence expansion
            labels = [-100] * (len(question_tokens) + len(latent_tokens)) + answer_tokens + [-100]
            
            modified_samples.append({
                'input_ids': tokens,
                'labels': labels,  # Only predict answer tokens
                'attention_mask': [1] * len(tokens)
            })
        
        dataset = modified_samples
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator)
        
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= 20:  # Limit batches for quick training
                break
                
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}
            
            # Forward pass
            try:
                outputs = coconut_model(**batch)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = epoch_loss / min(len(dataloader), 20)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    coconut_model.eval()
    print("Training complete!")


def evaluate_coconut_model(coconut_model, tokenizer, latent_id, start_id, end_id, num_samples=50):
    """Evaluate the trained Coconut model on GSM validation set"""
    
    print("\n" + "="*50)
    print("EVALUATING COCONUT MODEL")
    print("="*50)
    
    # Load validation dataset
    val_dataset = get_dataset("data/gsm_valid.json", tokenizer, max_size=num_samples)
    
    coconut_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coconut_model.to(device)
    
    correct = 0
    total = 0
    
    for i, sample in enumerate(val_dataset):
        if i >= num_samples:
            break
            
        try:
            # Get question and expected answer
            question = tokenizer.decode(sample['question_tokenized']).strip()
            expected_answer = tokenizer.decode(sample['answer_tokenized']).strip()
            
            # Create input with latent tokens
            num_reasoning_steps = 4
            input_ids = create_question_with_latent_tokens(
                question, num_reasoning_steps, tokenizer, start_id, latent_id, end_id
            ).to(device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = coconut_model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=50,
                    output_embedding=False
                )
                
                # Decode generated response
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract answer from generated text
                # Look for patterns like "### 42" or numbers at the end
                import re
                generated_answer = ""
                
                # Try to find ### pattern first
                hash_match = re.search(r'###\s*(\d+)', generated_text)
                if hash_match:
                    generated_answer = hash_match.group(1)
                else:
                    # Look for numbers at the end
                    num_match = re.search(r'(\d+)\s*$', generated_text.strip())
                    if num_match:
                        generated_answer = num_match.group(1)
                
                # Extract expected answer number
                expected_num = ""
                exp_hash_match = re.search(r'###\s*(\d+)', expected_answer)
                if exp_hash_match:
                    expected_num = exp_hash_match.group(1)
                else:
                    exp_num_match = re.search(r'(\d+)', expected_answer)
                    if exp_num_match:
                        expected_num = exp_num_match.group(1)
                
                # Check if correct
                is_correct = generated_answer == expected_num and generated_answer != ""
                if is_correct:
                    correct += 1
                
                total += 1
                
                # Print progress every 10 samples
                if i % 10 == 0 or i < 5:
                    print(f"\nSample {i+1}:")
                    print(f"Question: {question[:80]}...")
                    print(f"Expected: {expected_num}")
                    print(f"Generated: {generated_answer}")
                    print(f"Correct: {is_correct}")
                    print(f"Full generated: {generated_text[-100:]}")
                
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            continue
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Samples evaluated: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    return accuracy


def visualize_random_evaluation_questions(coconut_model, tokenizer, latent_id, start_id, end_id, num_questions=10):
    """Visualize reasoning progression for 10 random questions from validation set"""
    
    print("\n" + "="*50)
    print("VISUALIZING RANDOM EVALUATION QUESTIONS")
    print("="*50)
    
    import random
    
    # Load validation dataset
    val_dataset = get_dataset("data/gsm_valid.json", tokenizer, max_size=100)
    
    # Randomly sample questions
    random_indices = random.sample(range(len(val_dataset)), min(num_questions, len(val_dataset)))
    
    coconut_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coconut_model.to(device)
    
    successful_visualizations = 0
    
    for i, idx in enumerate(random_indices):
        try:
            sample = val_dataset[idx]
            question = tokenizer.decode(sample['question_tokenized']).strip()
            expected_answer = tokenizer.decode(sample['answer_tokenized']).strip()
            
            print(f"\n--- Visualizing Question {i+1}/10 (Dataset index: {idx}) ---")
            print(f"Question: {question}")
            print(f"Expected: {expected_answer}")
            
            # Create input with latent tokens for visualization
            num_reasoning_steps = 6
            input_ids = create_question_with_latent_tokens(
                question, num_reasoning_steps, tokenizer, start_id, latent_id, end_id
            )
            
            print(f"Input shape: {input_ids.shape}")
            
            # Extract hidden states during reasoning
            hidden_states_data = extract_hidden_states_during_forward(
                coconut_model, input_ids, max_steps=num_reasoning_steps
            )
            
            if len(hidden_states_data) > 1:
                print(f"Successfully extracted {len(hidden_states_data)} reasoning steps")
                
                # Create visualization with unique suffix
                save_suffix = f"_question_{i+1}_idx_{idx}"
                visualize_reasoning_progression(hidden_states_data, question, save_suffix)
                successful_visualizations += 1
                
                # Also generate answer to see model performance
                input_ids_gpu = input_ids.to(device)
                with torch.no_grad():
                    generated_ids = coconut_model.generate(
                        input_ids=input_ids_gpu,
                        attention_mask=torch.ones_like(input_ids_gpu),
                        max_new_tokens=30,
                        output_embedding=False
                    )
                    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    print(f"Generated: {generated_text[-50:]}")  # Show last 50 chars
                
            else:
                print("Failed to extract reasoning steps for this question")
                
        except Exception as e:
            print(f"Error visualizing question {i+1}: {e}")
            continue
    
    print(f"\n=== VISUALIZATION SUMMARY ===")
    print(f"Successfully visualized: {successful_visualizations}/{num_questions} questions")
    print(f"Visualizations saved in: visualization_results/")


def main(checkpoint_path="simple_attention_checkpoint"):
    """Main function to train then visualize Coconut reasoning"""
    
    print(f"Setting up Coconut model from checkpoint: {checkpoint_path}")
    
    # Load pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Add special tokens to tokenizer if not present
    special_tokens = ["<|latent|>", "<|start-latent|>", "<|end-latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # Load config and create model - try both config types
    config = None
    model = None
    
    # Try to load config first to determine model type
    config = None
    for config_class in [SimpleAttentionConfig, SingleLayerTransformerConfig]:
        try:
            config = config_class.from_pretrained(checkpoint_path)
            break
        except Exception:
            continue
    
    if config is None:
        raise RuntimeError(f"Could not load config from {checkpoint_path}")
    
    # Choose model class based on config model_type
    if config.model_type == "single_layer_transformer":
        model = SingleLayerTransformer(config)
    elif config.model_type == "simple_attention":
        model = SimpleAttentionModel(config)
    else:
        raise RuntimeError(f"Unsupported model type: {config.model_type}")
    
    print(f"Successfully loaded {config.model_type} model from {checkpoint_path}")
    
    state_dict = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    
    # Manually resize embeddings for new tokens
    old_vocab_size = config.vocab_size
    new_vocab_size = len(tokenizer)
    
    if new_vocab_size > old_vocab_size:
        # Create new embedding layers with expanded vocabulary
        old_wte = model.wte.weight.data
        old_lm_head = model.lm_head.weight.data
        
        # Resize word token embeddings
        model.wte = nn.Embedding(new_vocab_size, config.hidden_size)
        model.wte.weight.data[:old_vocab_size] = old_wte
        # Initialize new token embeddings with small random values
        nn.init.normal_(model.wte.weight.data[old_vocab_size:], std=0.02)
        
        # Resize language model head
        model.lm_head = nn.Linear(config.hidden_size, new_vocab_size, bias=False)
        model.lm_head.weight.data[:old_vocab_size] = old_lm_head
        nn.init.normal_(model.lm_head.weight.data[old_vocab_size:], std=0.02)
        
        # Update config
        config.vocab_size = new_vocab_size
    
    # Get special token IDs
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    print(f"Special tokens: latent={latent_id}, start={start_id}, end={end_id}")
    
    # Create Coconut model
    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    
    # Train the model first
    train_coconut_model(coconut_model, tokenizer, latent_id, start_id, end_id, epochs=3)
    
    # Evaluate on validation set
    evaluate_coconut_model(coconut_model, tokenizer, latent_id, start_id, end_id, num_samples=20)
    
    # Visualize 10 random questions from evaluation
    visualize_random_evaluation_questions(coconut_model, tokenizer, latent_id, start_id, end_id, num_questions=10)
    
    print("\nReasoning visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coconut single question visualization with training")
    parser.add_argument("--mode", choices=["simple", "transformer"], default="simple",
                       help="Model mode: 'simple' for simple_attention_checkpoint, 'transformer' for transformer_checkpoint (default: simple)")
    parser.add_argument("--checkpoint", 
                       help="Custom path to checkpoint directory (overrides --mode)")
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.mode == "transformer":
        checkpoint_path = "transformer_checkpoint"
    else:
        checkpoint_path = "simple_attention_checkpoint"
    
    main(checkpoint_path)