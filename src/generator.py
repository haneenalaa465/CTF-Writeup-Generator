#!/usr/bin/env python3
"""
CTF Writeup Generator
Uses trained model to generate writeups from challenge descriptions
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTFGenerator:
    def __init__(self, model_path: str = "./trained-ctf-model"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        try:
            logger.info(f"🔧 Loading trained model from {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def generate(self, 
                title: str, 
                category: str, 
                difficulty: str,
                description: str,
                points: str = "100",
                max_length: int = 1024,
                temperature: float = 0.7) -> str:
        """
        Generate a CTF writeup
        
        Args:
            title: Challenge title
            category: Challenge category (Web, Binary Exploitation, etc.)
            difficulty: Challenge difficulty (Easy, Medium, Hard)
            description: Challenge description
            points: Points value (default: 100)
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            
        Returns:
            Generated writeup text
        """
        prompt = self._create_prompt(title, category, difficulty, description, points)
        
        logger.info(f"🎯 Generating writeup for: {title}")
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.encode("<|solution_end|>")[0] if "<|solution_end|>" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        writeup = generated[len(prompt):].strip()
        
        logger.info("✅ Writeup generated successfully")
        return writeup
    
    def _create_prompt(self, title: str, category: str, difficulty: str, description: str, points: str) -> str:
        """Create the input prompt for the model"""
        return f"""<|challenge|>Challenge: {title}
Category: {category}
Difficulty: {difficulty}
Points: {points}

Description: {description}

Write a detailed CTF writeup explaining how to solve this challenge step by step.<|writeup|>"""

    def generate_multiple(self, challenges: list, **kwargs) -> list:
        """Generate writeups for multiple challenges"""
        writeups = []
        
        for i, challenge in enumerate(challenges, 1):
            logger.info(f"🔄 Processing challenge {i}/{len(challenges)}")
            
            writeup = self.generate(
                title=challenge['title'],
                category=challenge['category'],
                difficulty=challenge['difficulty'],
                description=challenge['description'],
                points=challenge.get('points', '100'),
                **kwargs
            )
            
            writeups.append({
                'challenge': challenge,
                'writeup': writeup
            })
        
        return writeups

def main():
    """Example usage"""
    # Load generator
    generator = CTFGenerator()
    
    # Example challenge
    writeup = generator.generate(
        title="SQL Injection Login Bypass",
        category="Web",
        difficulty="Easy",
        description="Can you bypass the login form at http://challenge.com/login? The application uses basic SQL authentication with no input validation."
    )
    
    print("\n" + "="*60)
    print("GENERATED WRITEUP:")
    print("="*60)
    print(writeup)
    print("="*60)

if __name__ == "__main__":
    main()
