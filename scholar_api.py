import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from serpapi import GoogleSearch
import asyncio

@dataclass
class ScholarResult:
    """Data class for Google Scholar search results"""
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int]
    citation_count: int
    pdf_url: Optional[str]
    scholar_url: str
    venue: Optional[str]
    
class GoogleScholarAPI:
    """Google Scholar API client using SerpAPI"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

    async def search_papers(self, query: str, max_results: int = 10) -> List[ScholarResult]:
        """
        Search for academic papers using SerpAPI Google Scholar
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of ScholarResult objects
        """
        try:
            if not self.api_key:
                self.logger.error("SerpAPI key is missing")
                return []

            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": self.api_key,
                "num": min(max_results, 10)
            }

            # Run SerpAPI search in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None, 
                lambda: GoogleSearch(params).get_dict()
            )
            
            return self._parse_serpapi_results(search_results)

        except Exception as e:
            self.logger.error(f"Error searching papers: {e}")
            return []

    def _parse_serpapi_results(self, data: Dict) -> List[ScholarResult]:
        """Parse SerpAPI results"""
        results = []
        
        # SerpAPI returns results in 'organic_results' field for Google Scholar
        organic_results = data.get('organic_results', [])
        
        for item in organic_results:
            try:
                # Extract authors from the publication info
                publication_info = item.get('publication_info', {})
                authors = publication_info.get('authors', [])
                
                # Extract PDF link from resources
                pdf_url = None
                resources = item.get('resources', [])
                for resource in resources:
                    if resource.get('file_format') == 'PDF':
                        pdf_url = resource.get('link')
                        break
                
                result = ScholarResult(
                    title=item.get('title', 'No title'),
                    authors=authors if isinstance(authors, list) else [authors] if authors else [],
                    abstract=item.get('snippet', ''),
                    year=publication_info.get('year', None),
                    citation_count=item.get('inline_links', {}).get('cited_by', {}).get('total', 0),
                    pdf_url=pdf_url,
                    scholar_url=item.get('link', ''),
                    venue=publication_info.get('summary', None)
                )
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Error parsing result: {e}")
                continue
                
        return results

    def refine_query(self, user_query: str) -> str:
        """
        Refine user query for better academic search results
        
        Args:
            user_query: Raw user query
            
        Returns:
            Refined query optimized for academic search
        """
        # Add academic keywords and filters
        academic_keywords = [
            'research', 'study', 'analysis', 'survey', 'review',
            'methodology', 'algorithm', 'model', 'framework'
        ]
        
        # Check if query already contains academic terms
        query_lower = user_query.lower()
        has_academic_terms = any(keyword in query_lower for keyword in academic_keywords)
        
        if not has_academic_terms:
            # Add relevant academic context
            refined_query = f"{user_query} research study"
        else:
            refined_query = user_query
            
        return refined_query
