import os
import requests
import PyPDF2
import io
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import asyncio
import aiohttp
from urllib.parse import urlparse
import hashlib
from bs4 import BeautifulSoup
import re

@dataclass
class DocumentChunk:
    """Data class for document chunks"""
    text: str
    metadata: Dict
    source: str
    page_number: Optional[int] = None
    section: Optional[str] = None

class DocumentProcessor:
    """Process and extract content from research papers"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(cache_dir, exist_ok=True)
        
    async def process_paper(self, paper_url: str, title: str) -> List[DocumentChunk]:
        """
        Process a research paper and extract text chunks
        
        Args:
            paper_url: URL of the paper
            title: Title of the paper
            
        Returns:
            List of DocumentChunk objects
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(paper_url)
            cached_content = self._get_from_cache(cache_key)
            
            if cached_content:
                return cached_content
            
            # Download and process document
            content = await self._download_document(paper_url)
            if not content:
                return []
            
            # Extract text based on file type
            if paper_url.lower().endswith('.pdf'):
                chunks = self._process_pdf(content, title, paper_url)
            else:
                chunks = self._process_html(content, title, paper_url)
            
            # Cache the results
            self._save_to_cache(cache_key, chunks)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing paper {paper_url}: {e}")
            return []
    
    async def _download_document(self, url: str) -> Optional[bytes]:
        """Download document from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        self.logger.warning(f"Failed to download {url}: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {e}")
            return None
    
    def _process_pdf(self, content: bytes, title: str, source: str) -> List[DocumentChunk]:
        """Process PDF content and extract text chunks"""
        chunks = []
        
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        # Clean and chunk the text
                        cleaned_text = self._clean_text(text)
                        page_chunks = self._chunk_text(cleaned_text, max_chunk_size=1000)
                        
                        for i, chunk in enumerate(page_chunks):
                            chunks.append(DocumentChunk(
                                text=chunk,
                                metadata={
                                    'title': title,
                                    'page': page_num + 1,
                                    'chunk_id': f"page_{page_num + 1}_chunk_{i}",
                                    'total_pages': len(pdf_reader.pages)
                                },
                                source=source,
                                page_number=page_num + 1
                            ))
                            
                except Exception as e:
                    self.logger.warning(f"Error processing page {page_num}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            
        return chunks
    
    def _process_html(self, content: bytes, title: str, source: str) -> List[DocumentChunk]:
        """Process HTML content and extract text chunks"""
        chunks = []
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text from different sections
            sections = self._extract_sections(soup)
            
            for section_name, section_text in sections.items():
                if section_text.strip():
                    cleaned_text = self._clean_text(section_text)
                    section_chunks = self._chunk_text(cleaned_text, max_chunk_size=1000)
                    
                    for i, chunk in enumerate(section_chunks):
                        chunks.append(DocumentChunk(
                            text=chunk,
                            metadata={
                                'title': title,
                                'section': section_name,
                                'chunk_id': f"{section_name}_chunk_{i}"
                            },
                            source=source,
                            section=section_name
                        ))
                        
        except Exception as e:
            self.logger.error(f"Error processing HTML: {e}")
            
        return chunks
    
    def _extract_sections(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract different sections from HTML"""
        sections = {}
        
        # Try to find abstract
        abstract = soup.find('div', {'class': re.compile(r'abstract', re.I)})
        if abstract:
            sections['abstract'] = abstract.get_text()
        
        # Try to find introduction
        intro = soup.find('div', {'class': re.compile(r'introduction', re.I)})
        if intro:
            sections['introduction'] = intro.get_text()
        
        # Extract main content
        main_content = soup.find('div', {'class': re.compile(r'content|main|body', re.I)})
        if main_content:
            sections['main_content'] = main_content.get_text()
        else:
            # Fallback to body text
            sections['main_content'] = soup.get_text()
        
        return sections
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters and artifacts
        text = re.sub(r'[^\w\s\.,;:!?\-\(\)]', '', text)
        
        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(cleaned_lines)
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into chunks of specified size"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _generate_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[DocumentChunk]]:
        """Get processed document from cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                import json
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return [DocumentChunk(**chunk) for chunk in data]
            except Exception as e:
                self.logger.warning(f"Error loading from cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, chunks: List[DocumentChunk]):
        """Save processed document to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            import json
            data = [chunk.__dict__ for chunk in chunks]
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {e}")