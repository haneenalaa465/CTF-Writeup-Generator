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
        
        # Expanded list of Top CTF writeup repositories
        self.repositories = [
            "sajjadium/ctf-writeups",
            "rexdotsh/ctf-writeups",
            "p4-team/ctf",
            "ctfs/write-ups-2016",
            "ctfs/write-ups-2017",
            "ctfs/write-ups-2018", # Added more years
            "ctfs/write-ups-2019",
            "ctfs/write-ups-2020",
            "ctfs/write-ups-2021",
            "ctfs/write-ups-2022",
            "ctfs/write-ups-2023",
            "TFNS/writeups",
            "TeamGreyFang/CTF-Writeups",
            "shiltemann/CTF-writeups-public",
            "pwning/public-writeup", # Another large collection
            "CTFLearn/writeups",    # Educational platform writeups
            "Knightsec-CTF/Writeups",
            "InternodeCTF/writeups",
            "VulnHub/ctf-writeups", # VulnHub VM writeups
            "RPISEC/MBE-Labs", # More specific binary exploitation
            "datajerk/ctf-writeups",
            "0x90/writeups",
            "HKUST-SecTeam/CTF-Writeups",
            "csc-hack/CTF-Writeups",
            "s-rah/ctf-writeups",
            "Knightsec-CTF/Writeups"
        ]
        
    def collect_all(self) -> List[Dict]:
        """Main method to collect all writeups"""
        logger.info("🚀 Starting CTF writeup collection...")
        
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
            logger.info("Using GitHub token for increased rate limits.")
        else:
            logger.warning("No GitHub token provided. Rate limits might be hit quickly.")
            
        for repo in self.repositories:
            logger.info(f"📥 Collecting from {repo}...")
            try:
                self._scrape_repository(repo, headers)
                # Be careful with rate limiting, especially without a token.
                # GitHub API allows 5000 requests/hour with token, 60 without.
                # A 2-second sleep might still be too fast if you hit many files/repos.
                # We'll rely on catching exceptions for rate limits.
                time.sleep(1) # Reduced sleep slightly, but a token is best.
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Network/API Error with {repo}: {e}")
                if "403" in str(e) and "rate limit" in str(e).lower():
                    logger.error("GitHub API rate limit hit. Consider using a personal access token.")
                    break # Stop collecting if rate limit is hit
                continue
            except Exception as e:
                logger.error(f"❌ General Error with {repo}: {e}")
                continue
                
        logger.info(f"✅ Total collected: {len(self.writeups_data)} writeups")
        return self._clean_and_deduplicate()
        
    def _scrape_repository(self, repo: str, headers: dict):
        """Scrape a single repository"""
        # Try main branch first, then master
        for branch in ['main', 'master', 'dev', 'gh-pages']: # Added more common branch names
            api_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
            response = requests.get(api_url, headers=headers)
            
            if response.status_code == 200:
                tree = response.json()
                markdown_files = [item for item in tree['tree'] 
                                 if item['path'].lower().endswith('.md') and item['type'] == 'blob'
                                 and not any(x in item['path'].lower() for x in ['_posts', 'template', 'readme.md', '.github'])] # Exclude common non-writeup files
                
                logger.info(f"  Found {len(markdown_files)} markdown files (after basic filtering)")
                
                # --- IMPORTANT CHANGE: REMOVE OR INCREASE LIMIT ---
                # Changed from [:50] to [:500] (or remove for all if you have a token)
                # If you have a GitHub token and want ALL files, remove `[:500]`
                # If you don't have a token, stick to a reasonable limit like 100-200.
                files_to_process = markdown_files # Process all files by default if not limiting
                # files_to_process = markdown_files[:500] # Example: process up to 500 files per repo

                # Iterate through files in reverse order to potentially get newer or more relevant ones first (optional)
                # for file_info in reversed(files_to_process):
                for file_info in files_to_process:
                    # Introduce a small delay for each file if not using a token, or if processing many files
                    # This is crucial if not using a token to avoid hitting minute-level rate limits
                    if not self.github_token:
                        time.sleep(0.1) # Small delay per file for unauthenticated requests

                    try:
                        file_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_info['path']}"
                        file_response = requests.get(file_url, headers=headers) # Pass headers for consistency
                        
                        if file_response.status_code == 200:
                            content = file_response.text
                            parsed = self._parse_writeup_content(content, file_info['path'], repo)
                            if parsed:
                                self.writeups_data.append(parsed)
                                
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"❗ Failed to fetch {file_info['path']} from {repo}: {e}")
                        if "403" in str(e) and "rate limit" in str(e).lower():
                            logger.error("GitHub API rate limit hit while fetching file. Stopping current repo.")
                            return # Stop processing files for this repo
                        continue
                    except Exception as e:
                        logger.debug(f"Skipping file {file_info['path']} due to parsing error: {e}")
                        continue
                break # Break from branch loop if a valid branch was found and processed
                
    def _parse_writeup_content(self, content: str, filepath: str, repo: str) -> Optional[Dict]:
        """Parse CTF writeup content to extract structured information"""
        
        # Relaxed minimum content length slightly
        if len(content) < 200: # Changed from 300 to 200
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
        
        # --- Relaxed Quality Checks ---
        # Removed 'writeup' not in content.lower() as it might be too strict
        # Relaxed description length, but it's still important for good input.
        if (len(description) < 10 or # Changed from 20 to 10
            not flag or 
            len(content) < 300): # Changed from 500 to 300
            # logger.debug(f"Skipping {filepath} due to quality check failure: desc_len={len(description)}, flag_found={bool(flag)}, content_len={len(content)}")
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
                # Take the first word/tag, clean it up
                cat = match.group(1).strip().split(',')[0].strip()
                # Simple cleanup for common patterns like "web exploitation" -> "Web"
                if "web" in cat.lower(): return "Web"
                if "binary" in cat.lower() or "pwn" in cat.lower(): return "Binary Exploitation"
                if "crypto" in cat.lower(): return "Cryptography"
                if "forensics" in cat.lower(): return "Forensics"
                if "reverse" in cat.lower() or "rev" in cat.lower(): return "Reverse Engineering"
                if "misc" in cat.lower(): return "Miscellaneous"
                return cat # Return as-is if no specific mapping
                
        # Fallback to filepath analysis
        filepath_lower = filepath.lower()
        if any(word in filepath_lower for word in ['web', 'http', 'sql']):
            return "Web"
        elif any(word in filepath_lower for word in ['pwn', 'binary', 'exploit', 'reversing', 'rev']): # Added rev
            return "Binary Exploitation"
        elif any(word in filepath_lower for word in ['crypto', 'cipher']):
            return "Cryptography"
        elif any(word in filepath_lower for word in ['forensics', 'stego']):
            return "Forensics"
        elif any(word in filepath_lower for word in ['reverse', 'rev']):
            return "Reverse Engineering"
        elif any(word in filepath_lower for word in ['misc', 'osint', 'blockchain']): # Added misc, osint, blockchain
            return "Miscellaneous"
        else:
            return "General"
    
    def _extract_description(self, content: str) -> str:
        """Extract challenge description"""
        desc_patterns = [
            r'(?:description|challenge)[:\s\n]+(.+?)(?:\n\n|\n#+|solution|approach|---)', # Added --- for markdown front matter
            r'## Description\s*\n(.+?)(?:\n#+|\n\n|---)',
            r'### Description\s*\n(.+?)(?:\n#+|\n\n|---)',
            r'#+\s*(?:Challenge|Problem)\s*\n(.+?)(?:\n#+|\n\n|---)' # New pattern for "Challenge" or "Problem" heading
        ]
        
        for pattern in desc_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                # Clean up markdown formatting from the description
                clean_desc = re.sub(r'[`*]', '', match.group(1)).strip() # Remove backticks and asterisks
                return clean_desc[:500] # Still limit length to avoid overly long descriptions
                
        return ""
    
    def _extract_flag(self, content: str) -> str:
        """Extract flag from content"""
        flag_patterns = [
            r'[Cc][Tt][Ff]\{[^}]+\}', # Case-insensitive CTF{}
            r'[Ff][Ll][Aa][Gg]\{[^}]+\}', # Case-insensitive FLAG{}
            r'[A-Z0-9_]{3,20}\{[\w\d_@\-]+\}', # More generic pattern for common flag formats (e.g., TEAMNAME{flag})
            r'[\w\d]{32}', # Common for MD5/SHA256 hashes if flags are sometimes raw hashes
            r'^[a-fA-F0-9]{32}$' # Strict MD5 pattern
        ]
        
        for pattern in flag_patterns:
            match = re.search(pattern, content) # Removed re.IGNORECASE for more precise flag matching, can add back if needed
            if match:
                return match.group(0)
                
        return ""
    
    def _clean_and_deduplicate(self) -> List[Dict]:
        """Remove duplicates and return clean dataset"""
        seen = set()
        unique_writeups = []
        
        for writeup in self.writeups_data:
            # Use a more robust identifier for deduplication
            # Combination of title, first 100 chars of description, and category
            identifier = (
                writeup['title'].lower(), 
                writeup['description'][:100].lower(),
                writeup['category'].lower()
            )
            if identifier not in seen:
                seen.add(identifier)
                unique_writeups.append(writeup)
            else:
                logger.debug(f"Skipping duplicate: {writeup['title']}")
        
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
    # It's highly recommended to use a GitHub Personal Access Token
    # for significantly higher API rate limits (5000 requests/hour vs 60/hour).
    # You can set it as an environment variable or pass it directly.
    # Example: GITHUB_TOKEN="ghp_YOUR_TOKEN_HERE" python src/data_collector.py
    github_token = os.getenv("GITHUB_TOKEN") # Import os at the top if using this
    collector = CTFDataCollector(github_token=github_token)
    
    logger.info("🚀 Starting CTF writeup data collection...")
    logger.info("This may take a while depending on the number of repositories and rate limits.")
    
    writeups = collector.collect_all()
    collector.save_dataset(writeups)
    
    logger.info(f"✅ Data collection complete!")
    logger.info(f"Ready to proceed with fine-tuning using {len(writeups)} writeups")

if __name__ == "__main__":
    # Add os import for GITHUB_TOKEN
    import os 
    main()
