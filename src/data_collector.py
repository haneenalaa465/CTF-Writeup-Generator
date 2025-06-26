#!/usr/bin/env python3
"""
CTF Writeup Data Collector
Scrapes high-quality CTF writeups from GitHub repositories
"""

import requests
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTFDataCollector:
    def __init__(self, github_token: Optional[str] = None):
        self.writeups_data = []
        self.github_token = github_token
        
        # Top CTF writeup repositories
        self.repositories = [
            "sajjadium/ctf-writeups",
            "rexdotsh/ctf-writeups",
            "p4-team/ctf",
            "ctfs/write-ups-2016",
            "ctfs/write-ups-2017",
            "TFNS/writeups",
            "TeamGreyFang/CTF-Writeups",
            "shiltemann/CTF-writeups-public"
        ]
        
    def collect_all(self) -> List[Dict]:
        """Main method to collect all writeups"""
        logger.info("🚀 Starting CTF writeup collection...")
        
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
            
        for repo in self.repositories:
            logger.info(f"📥 Collecting from {repo}...")
            try:
                self._scrape_repository(repo, headers)
                time.sleep(2)  # Rate limiting
            except Exception as e:
                logger.error(f"❌ Error with {repo}: {e}")
                continue
                
        logger.info(f"✅ Total collected: {len(self.writeups_data)} writeups")
        return self._clean_and_deduplicate()
        
    def _scrape_repository(self, repo: str, headers: dict):
        """Scrape a single repository"""
        # Try main branch first, then master
        for branch in ['main', 'master']:
            api_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
            response = requests.get(api_url, headers=headers)
            
            if response.status_code == 200:
                tree = response.json()
                markdown_files = [item for item in tree['tree'] 
                               if item['path'].endswith('.md') and item['type'] == 'blob']
                
                logger.info(f"  Found {len(markdown_files)} markdown files")
                
                for file_info in markdown_files[:50]:  # Limit per repo
                    try:
                        file_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_info['path']}"
                        file_response = requests.get(file_url)
                        
                        if file_response.status_code == 200:
                            content = file_response.text
                            parsed = self._parse_writeup_content(content, file_info['path'], repo)
                            if parsed:
                                self.writeups_data.append(parsed)
                                
                    except Exception as e:
                        continue
                break
                
    def _parse_writeup_content(self, content: str, filepath: str, repo: str) -> Optional[Dict]:
        """Parse CTF writeup content to extract structured information"""
        
        # Skip files that are too short
        if len(content) < 300:
            return None
            
        # Extract title
        title_match = re.search(r'^#\s+(.+?)$', content, re.MULTILINE)
        if not title_match:
            title_match = re.search(r'title[:\s]+(.+?)(?:\n|$)', content, re.IGNORECASE)
        
        title = title_match.group(1).strip() if title_match else Path(filepath).stem
        
        # Extract category from filepath or content
        category = self._extract_category(filepath, content)
        
        # Extract difficulty
        difficulty_match = re.search(r'difficulty[:\s]+(.+?)(?:\n|$)', content, re.IGNORECASE)
        difficulty = difficulty_match.group(1).strip() if difficulty_match else "Medium"
        
        # Extract points
        points_match = re.search(r'points?[:\s]+(\d+)', content, re.IGNORECASE)
        points = points_match.group(1) if points_match else "100"
        
        # Extract description
        description = self._extract_description(content)
        
        # Extract flag
        flag = self._extract_flag(content)
        
        # Quality checks
        if (len(description) < 20 or 
            not flag or 
            len(content) < 500 or
            'writeup' not in content.lower()):
            return None
            
        return {
            'title': title,
            'category': category,
            'difficulty': difficulty,
            'points': points,
            'description': description,
            'full_content': content,
            'flag': flag,
            'source_repo': repo,
            'source_file': filepath
        }
    
    def _extract_category(self, filepath: str, content: str) -> str:
        """Extract challenge category"""
        # Try to find category in content first
        category_patterns = [
            r'category[:\s]+(.+?)(?:\n|$)',
            r'tags?[:\s]+(.+?)(?:\n|$)',
            r'\*\*category\*\*[:\s]+(.+?)(?:\n|$)',
        ]
        
        for pattern in category_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip().split(',')[0]
                
        # Fallback to filepath analysis
        filepath_lower = filepath.lower()
        if any(word in filepath_lower for word in ['web', 'http', 'sql']):
            return "Web"
        elif any(word in filepath_lower for word in ['pwn', 'binary', 'exploit']):
            return "Binary Exploitation"
        elif any(word in filepath_lower for word in ['crypto', 'cipher']):
            return "Cryptography"
        elif any(word in filepath_lower for word in ['forensics', 'stego']):
            return "Forensics"
        elif any(word in filepath_lower for word in ['reverse', 'rev']):
            return "Reverse Engineering"
        else:
            return "General"
    
    def _extract_description(self, content: str) -> str:
        """Extract challenge description"""
        desc_patterns = [
            r'(?:description|challenge)[:\s\n]+(.+?)(?:\n\n|\n#+|solution|approach)',
            r'## Description\s*\n(.+?)(?:\n#+|\n\n)',
            r'### Description\s*\n(.+?)(?:\n#+|\n\n)'
        ]
        
        for pattern in desc_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:500]
                
        return ""
    
    def _extract_flag(self, content: str) -> str:
        """Extract flag from content"""
        flag_patterns = [
            r'flag\{[^}]+\}',
            r'CTF\{[^}]+\}',
            r'FLAG\{[^}]+\}',
            r'[A-Z]+\{[^}]+\}'
        ]
        
        for pattern in flag_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(0)
                
        return ""
    
    def _clean_and_deduplicate(self) -> List[Dict]:
        """Remove duplicates and return clean dataset"""
        seen = set()
        unique_writeups = []
        
        for writeup in self.writeups_data:
            identifier = (writeup['title'].lower(), writeup['description'][:100].lower())
            if identifier not in seen:
                seen.add(identifier)
                unique_writeups.append(writeup)
        
        return unique_writeups
    
    def save_dataset(self, writeups: List[Dict], filename: str = "ctf_writeups_dataset.json"):
        """Save dataset to JSON file"""
        logger.info(f"📊 Dataset Statistics:")
        logger.info(f"  Total writeups: {len(writeups)}")
        
        # Category distribution
        categories = {}
        for writeup in writeups:
            cat = writeup['category']
            categories[cat] = categories.get(cat, 0) + 1
            
        for cat, count in sorted(categories.items()):
            logger.info(f"  {cat}: {count}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(writeups, f, indent=2, ensure_ascii=False)
            
        logger.info(f"💾 Dataset saved to {filename}")
        return writeups

def main():
    """Main function"""
    collector = CTFDataCollector()
    
    logger.info("🚀 Starting CTF writeup data collection...")
    logger.info("This may take 10-15 minutes depending on your connection...")
    
    writeups = collector.collect_all()
    collector.save_dataset(writeups)
    
    logger.info(f"✅ Data collection complete!")
    logger.info(f"Ready to proceed with fine-tuning using {len(writeups)} writeups")

if __name__ == "__main__":
    main()
