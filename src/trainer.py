import json
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    # Removed IntervalStrategy import as it's not needed with string arguments
)
import logging
from pathlib import Path
import re
from typing import Dict, Tuple
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*attention_mask.*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTFWriteupTrainer:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.output_dir = "./trained-ctf-model"
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with special tokens"""
        logger.info(f"🔧 Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # FIX 1: Set pad_token to a unique token, not eos_token
        self.tokenizer.pad_token = "<|pad|>"
        
        # Add special tokens for CTF writeups
        special_tokens = {
            "additional_special_tokens": [
                "<|challenge|>", "<|writeup|>", "<|solution_end|>",
                "<|step|>", "<|flag|>", "<|code|>", "<|pad|>"
            ]
        }
            
        # Add special tokens
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Resize token embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        logger.info("✅ Model and tokenizer loaded successfully")
        
    def create_training_prompt(self, challenge_data: Dict) -> Tuple[str, str]:
        """Create training prompt pairs - FIXED"""
        
        # FIX 2: Create a cleaner format that's easier for the model to learn
        input_prompt = f"<|challenge|>{challenge_data['title']} | {challenge_data['category']} | {challenge_data['difficulty']} | {challenge_data['points']} points\n\n{challenge_data['description']}<|writeup|>"

        # Target output (what we want the model to generate)
        target_output = self.format_writeup_output(challenge_data)
        
        return input_prompt, target_output
    
    def format_writeup_output(self, data: Dict) -> str:
        """Format the target writeup output - SIMPLIFIED"""
        
        content = data['full_content']
        
        # FIX 3: Simpler, more consistent format
        formatted_output = f"""
## Solution

{self._extract_clean_solution(content)}

## Flag
{data['flag']}
<|solution_end|>"""
        
        return formatted_output.strip()
    
    def _extract_clean_solution(self, content: str) -> str:
        """Extract a clean, focused solution"""
        
        # Remove markdown headers and excessive formatting
        content = re.sub(r'^#{1,6}\s*', '', content, flags=re.MULTILINE)
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Filter for substantial paragraphs with solution content
        solution_paragraphs = []
        solution_keywords = ['step', 'first', 'then', 'next', 'run', 'execute', 'payload', 'exploit', 'solve']
        
        for para in paragraphs:
            if (len(para) > 30 and 
                any(keyword in para.lower() for keyword in solution_keywords) and
                not para.startswith('```')):  # Skip code blocks for now
                solution_paragraphs.append(para)
                
        # Return top 3 solution paragraphs or fallback
        if solution_paragraphs:
            return '\n\n'.join(solution_paragraphs[:3])
        else:
            # Fallback: return first few meaningful paragraphs
            meaningful = [p for p in paragraphs if len(p) > 50][:2]
            return '\n\n'.join(meaningful) if meaningful else content[:500]
    
    def prepare_dataset(self, data_path: str) -> DatasetDict:
        """Prepare dataset for training - FIXED"""
        logger.info(f"📊 Loading dataset from {data_path}")
        
        # Load raw data
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Create training pairs
        training_data = []
        successful_pairs = 0
        
        for item in raw_data:
            try:
                input_prompt, target_output = self.create_training_prompt(item)
                
                # FIX 4: Better quality checks
                if (len(target_output) > 50 and 
                    len(input_prompt) > 30 and
                    item.get('flag', '') and
                    len(input_prompt + target_output) < 800):  # Prevent overly long examples
                    
                    training_data.append({
                        'text': input_prompt + target_output,  # FIX 5: Single text field
                        'category': item['category']
                    })
                    successful_pairs += 1
                    
            except Exception as e:
                logger.debug(f"Skipping item due to error: {e}")
                continue
        
        logger.info(f"✅ Created {successful_pairs} high-quality training examples")
        
        if successful_pairs < 10:
            logger.warning("⚠️  Very few training examples! Consider getting more data.")
        
        # Create dataset
        df = pd.DataFrame(training_data)
        dataset = Dataset.from_pandas(df)
        
        dataset_dict = DatasetDict()

        # --- Data splitting logic (from previous response) ---
        if successful_pairs >= 3: # Minimum for train, validation, test (at least 1 each)
            train_temp_split = dataset.train_test_split(test_size=0.2, seed=42)
            dataset_dict['train'] = train_temp_split['train']

            if len(train_temp_split['test']) >= 2:
                val_test_split = train_temp_split['test'].train_test_split(test_size=0.5, seed=42)
                dataset_dict['validation'] = val_test_split['train']
                dataset_dict['test'] = val_test_split['test']
            else:
                logger.warning("Not enough samples for a separate test set. Using remaining data for validation only.")
                dataset_dict['validation'] = train_temp_split['test']
                dataset_dict['test'] = Dataset.from_dict({'text': [], 'category': []})
        elif successful_pairs >= 2:
            logger.warning("Not enough samples for train, validation, and test split. Splitting into train and validation only.")
            train_val_split = dataset.train_test_split(test_size=0.5, seed=42)
            dataset_dict['train'] = train_val_split['train']
            dataset_dict['validation'] = train_val_split['test']
            dataset_dict['test'] = Dataset.from_dict({'text': [], 'category': []})
        else:
            logger.error("Extremely few training examples. Cannot create validation or test sets.")
            dataset_dict['train'] = dataset
            dataset_dict['validation'] = Dataset.from_dict({'text': [], 'category': []})
            dataset_dict['test'] = Dataset.from_dict({'text': [], 'category': []})
        # --- End data splitting logic ---
        
        # FIX 6: Improved tokenization function
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',  # Consistent padding
                max_length=512,
                return_attention_mask=True,  # FIX 7: Always return attention mask
                return_tensors=None
            )
            
            # Set labels (for causal LM, labels = input_ids)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Only map existing splits
        for split_name in dataset_dict.keys():
            if len(dataset_dict[split_name]) > 0: # Only tokenize if the split is not empty
                dataset_dict[split_name] = dataset_dict[split_name].map(
                    tokenize_function,
                    batched=True,
                    remove_columns=['text', 'category']  # Remove original columns
                )
            else:
                 # If a split is empty, ensure it's still a Dataset object, even if empty
                 dataset_dict[split_name] = Dataset.from_dict({'input_ids': [], 'attention_mask': [], 'labels': []})
        
        return dataset_dict
    
    def train(self, tokenized_datasets: DatasetDict):
        """Train the model - FIXED"""
        logger.info("🚀 Starting model training...")
        
        # FIX 8: Better training arguments
        # Updated parameters for TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,  # Slightly larger batch
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            logging_steps=5,
            # --- FIX: Changed 'evaluation_strategy' to 'eval_strategy' and used string values ---
            eval_strategy="steps", # Corrected argument name and value type
            save_strategy="steps",     # Keep as steps (already using correct string type)
            # --- End FIX ---
            eval_steps=50,
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            learning_rate=5e-5,  # Standard learning rate
            weight_decay=0.01,
            lr_scheduler_type="cosine",  # Better scheduler
            report_to="none",
            save_total_limit=3,
            max_grad_norm=1.0,
            gradient_checkpointing=True,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
        )
        
        # FIX 9: Proper data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            # Conditionally provide eval_dataset
            eval_dataset=tokenized_datasets['validation'] if len(tokenized_datasets['validation']) > 0 else None,
            tokenizer=self.tokenizer,  # Keep using tokenizer parameter
            data_collator=data_collator,
        )
        
        # Train
        logger.info("📈 Training in progress...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"✅ Training completed! Model saved to {self.output_dir}")
        return trainer
    
    def test_model(self, test_prompt: str):
        """Test the trained model - FIXED"""
        logger.info("🧪 Testing trained model...")
        
        # FIX 10: Proper input preparation with attention mask
        inputs = self.tokenizer(
            test_prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            self.model = self.model.cuda()
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],  # FIX 11: Pass attention mask
                max_new_tokens=200,  # Limit new tokens
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (skip input)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print("\n" + "="*60)
        print("🎯 GENERATED WRITEUP TEST:")
        print("="*60)
        print("INPUT:", test_prompt)
        print("\nGENERATED:")
        print(generated)
        print("="*60)

def main():
    """Main training function"""
    
    # Configuration
    DATA_PATH = "ctf_writeups_dataset.json"
    
    # Check if dataset exists
    if not Path(DATA_PATH).exists():
        logger.error(f"❌ Dataset not found: {DATA_PATH}")
        logger.error("Please run 'python src/data_collector.py' first!")
        return
    
    # Initialize trainer
    trainer = CTFWriteupTrainer()
    
    try:
        # Setup
        trainer.setup_model_and_tokenizer()
        
        # Prepare data
        tokenized_datasets = trainer.prepare_dataset(DATA_PATH)
        
        logger.info(f"\n📊 Dataset sizes:")
        logger.info(f"  Train: {len(tokenized_datasets['train'])}")
        logger.info(f"  Validation: {len(tokenized_datasets['validation'])}")
        logger.info(f"  Test: {len(tokenized_datasets['test'])}")
        
        # Train
        # Only train if there's a non-empty training set
        if len(tokenized_datasets['train']) > 0:
            trainer_obj = trainer.train(tokenized_datasets)
            
            # Test with sample prompt
            test_prompt = "<|challenge|>SQL Injection Login Bypass | Web | Easy | 100 points\n\nCan you bypass the login form at [http://challenge.com/login](http://challenge.com/login)? The application uses basic SQL authentication with no input validation.<|writeup|>"
            
            trainer.test_model(test_prompt)
            
            logger.info("\n🎉 Training completed successfully!")
            logger.info(f"Model saved to: {trainer.output_dir}")
        else:
            logger.error("No training data available. Skipping training and testing.")
            
    except Exception as e:
        logger.error(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
