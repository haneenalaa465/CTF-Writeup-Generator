#!/usr/bin/env python3
"""
CTF Writeup Model Trainer
Fine-tunes microsoft/DialoGPT-medium on CTF writeup data
"""

import json
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import logging
from pathlib import Path
import re
from typing import Dict, Tuple

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
        
        # Add special tokens for CTF writeups
        special_tokens = {
            "additional_special_tokens": [
                "<|challenge|>", "<|writeup|>", "<|solution_end|>",
                "<|step|>", "<|flag|>", "<|code|>"
            ]
        }
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Add special tokens
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load model
        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
        # Resize token embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info("✅ Model and tokenizer loaded successfully")
        
    def create_training_prompt(self, challenge_data: Dict) -> Tuple[str, str]:
        """Create training prompt pairs"""
        
        # Input prompt (what we give to the model)
        input_prompt = f"""<|challenge|>Challenge: {challenge_data['title']}
Category: {challenge_data['category']}
Difficulty: {challenge_data['difficulty']}
Points: {challenge_data['points']}

Description: {challenge_data['description']}

Write a detailed CTF writeup explaining how to solve this challenge step by step.<|writeup|>"""

        # Target output (what we want the model to generate)
        target_output = self.format_writeup_output(challenge_data)
        
        return input_prompt, target_output
    
    def format_writeup_output(self, data: Dict) -> str:
        """Format the target writeup output"""
        
        content = data['full_content']
        
        # Extract solution steps from content
        solution_sections = self._extract_solution_sections(content)
        
        # Create formatted output
        formatted_output = f"""
# {data['title']} - CTF Writeup

## Challenge Overview
**Category:** {data['category']}
**Difficulty:** {data['difficulty']}
**Points:** {data['points']}

{data['description']}

## Solution

"""
        
        # Add solution steps
        if solution_sections:
            for i, section in enumerate(solution_sections[:4], 1):  # Max 4 sections
                formatted_output += f"### Step {i}\n{section.strip()}\n\n"
        else:
            # Fallback: extract meaningful content
            formatted_output += self._extract_fallback_solution(content)
        
        # Add flag
        formatted_output += f"""
## Flag
```
{data['flag']}
```
<|solution_end|>"""
        
        return formatted_output.strip()
    
    def _extract_solution_sections(self, content: str) -> list:
        """Extract solution sections from writeup content"""
        
        # Look for sections with solution-related headings
        section_patterns = [
            r'(#{1,3}\s*(?:step|solution|approach|method|analysis|exploit).+?(?=\n#{1,3}|\Z))',
            r'(#{1,3}\s*\d+\..+?(?=\n#{1,3}|\Z))',  # Numbered sections
            r'(\*\*(?:step|solution|approach)\*\*.+?(?=\n\*\*|\Z))',  # Bold sections
        ]
        
        sections = []
        for pattern in section_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            sections.extend(matches)
            
        return sections[:4]  # Limit to 4 sections
    
    def _extract_fallback_solution(self, content: str) -> str:
        """Extract fallback solution if no clear sections found"""
        
        # Look for paragraphs that seem solution-related
        paragraphs = content.split('\n\n')
        solution_paragraphs = []
        
        solution_keywords = ['solution', 'solve', 'exploit', 'payload', 'attack', 'vulnerability']
        
        for para in paragraphs:
            if (len(para.strip()) > 50 and 
                any(keyword in para.lower() for keyword in solution_keywords)):
                solution_paragraphs.append(para.strip())
                
        if solution_paragraphs:
            return '\n\n'.join(solution_paragraphs[:3])  # Max 3 paragraphs
        else:
            # Last resort: use first few meaningful lines
            lines = [line.strip() for line in content.split('\n') 
                    if len(line.strip()) > 30]
            return '\n\n'.join(lines[:5])
    
    def prepare_dataset(self, data_path: str) -> DatasetDict:
        """Prepare dataset for training"""
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
                
                # Quality check
                if (len(target_output) > 100 and 
                    len(input_prompt) > 50 and
                    'flag' in target_output.lower()):
                    
                    training_data.append({
                        'input': input_prompt,
                        'output': target_output,
                        'category': item['category']
                    })
                    successful_pairs += 1
                    
            except Exception as e:
                logger.debug(f"Skipping item due to error: {e}")
                continue
        
        logger.info(f"✅ Created {successful_pairs} high-quality training examples")
        
        # Create dataset
        df = pd.DataFrame(training_data)
        dataset = Dataset.from_pandas(df)
        
        # Split: 80% train, 15% validation, 5% test
        train_test = dataset.train_test_split(test_size=0.2, seed=42)
        val_test = train_test['test'].train_test_split(test_size=0.25, seed=42)
        
        dataset_dict = DatasetDict({
            'train': train_test['train'],
            'validation': val_test['train'],
            'test': val_test['test']
        })
        
        # Tokenize
        def tokenize_function(examples):
            # Combine input and output for causal LM
            full_texts = []
            for i in range(len(examples['input'])):
                full_text = examples['input'][i] + examples['output'][i]
                full_texts.append(full_text)
            
            tokenized = self.tokenizer(
                full_texts,
                truncation=True,
                padding=False,
                max_length=1024,  # Adjust based on GPU memory
                return_overflowing_tokens=False,
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        tokenized_datasets = dataset_dict.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset_dict['train'].column_names
        )
        
        return tokenized_datasets
    
    def train(self, tokenized_datasets: DatasetDict):
        """Train the model"""
        logger.info("🚀 Starting model training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,  # Small batch size for stability
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,  # Effective batch size = 2*8 = 16
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            learning_rate=5e-5,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            report_to="none",  # Disable wandb for simplicity
            save_total_limit=2,  # Only keep 2 checkpoints
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM (not masked)
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=self.tokenizer,
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
        """Test the trained model with a sample prompt"""
        logger.info("🧪 Testing trained model...")
        
        inputs = self.tokenizer.encode(test_prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            self.model = self.model.cuda()
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n" + "="*60)
        print("🎯 GENERATED WRITEUP TEST:")
        print("="*60)
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
        trainer_obj = trainer.train(tokenized_datasets)
        
        # Test with sample prompt
        test_prompt = """<|challenge|>Challenge: SQL Injection Login Bypass
Category: Web
Difficulty: Easy
Points: 100

Description: Can you bypass the login form at http://challenge.com/login? The application uses basic SQL authentication with no input validation.

Write a detailed CTF writeup explaining how to solve this challenge step by step.<|writeup|>"""
        
        trainer.test_model(test_prompt)
        
        logger.info("\n🎉 Training completed successfully!")
        logger.info(f"Model saved to: {trainer.output_dir}")
        logger.info("\nYou can now generate writeups using:")
        logger.info("python src/generator.py")
        
    except Exception as e:
        logger.error(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
