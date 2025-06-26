#!/usr/bin/env python3
"""
Example usage of the CTF Writeup Generator
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.generator import CTFGenerator

def main():
    # Initialize generator
    generator = CTFGenerator("./trained-ctf-model")
    
    # Example challenges
    challenges = [
        {
            "title": "Buffer Overflow Challenge",
            "category": "Binary Exploitation",
            "difficulty": "Medium",
            "description": "Exploit this vulnerable C program to get shell access. The binary has basic stack protection disabled and uses gets() function.",
            "points": "250"
        },
        {
            "title": "XSS Reflected",
            "category": "Web",
            "difficulty": "Easy", 
            "description": "Find and exploit a reflected XSS vulnerability in the search functionality at http://challenge.com/search",
            "points": "100"
        },
        {
            "title": "Caesar Cipher Variant",
            "category": "Cryptography",
            "difficulty": "Easy",
            "description": "Decode this message: 'Khoor Zruog'. The cipher uses a simple substitution method.",
            "points": "50"
        }
    ]
    
    print("🎯 CTF Writeup Generator Examples")
    print("=" * 50)
    
    for i, challenge in enumerate(challenges, 1):
        print(f"\n📝 Example {i}: {challenge['title']}")
        print("-" * 30)
        
        writeup = generator.generate(
            title=challenge['title'],
            category=challenge['category'], 
            difficulty=challenge['difficulty'],
            description=challenge['description'],
            points=challenge['points']
        )
        
        print(writeup)
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
