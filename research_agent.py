from typing import TypedDict, List, Dict, Optional, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import asyncio
import logging
from datetime import datetime

from scholar_api import GoogleScholarAPI, ScholarResult
from document_processor import DocumentProcessor
from vector_store import VectorStore
from config import config
from gemini_client import GeminiClient, ChatPromptTemplate

class ResearchState(TypedDict):
    """State for the research agent"""
    original_query: str
    refined_query: str
    search_results: List[ScholarResult]
    processed_documents: List[str]
    context_chunks: List[str]
    final_answer: str
    error: Optional[str]
    step: str

class QueryRefinementTool:
    """Tool for refining user queries for academic search"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def refine_query(self, query: str) -> str:
        """Refine the query using LLM"""
        prompt = ChatPromptTemplate.from_template("""
        You are an expert at converting user questions into effective academic search queries.
        
        User question: {query}
        
        Convert this into a focused academic search query that would find relevant research papers.
        Consider:
        - Key technical terms and concepts
        - Relevant research areas
        - Alternative terminology
        - Specific methodologies or approaches
        
        Return only the refined search query, nothing else.
        """)
        
        messages = prompt.format_messages(query=query)
        response = self.llm.invoke(messages)
        return response.content.strip()

class PaperAnalysisTool:
    """Tool for analyzing research papers and extracting relevant information"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_paper(self, paper_content: str, query: str) -> str:
        """Analyze paper content for relevance to query"""
        prompt = ChatPromptTemplate.from_template("""
        You are analyzing a research paper for relevance to a user query.
        
        User Query: {query}
        
        Paper Content: {paper_content}
        
        Analyze this paper content and:
        1. Determine its relevance to the user query (score 1-10)
        2. Extract key findings, methodologies, and conclusions
        3. Identify specific sections that address the user's question
        
        Format your response as:
        RELEVANCE: [score]
        KEY_FINDINGS: [bullet points]
        RELEVANT_SECTIONS: [quotes with context]
        """)
        
        messages = prompt.format_messages(query=query, paper_content=paper_content[:4000])
        response = self.llm.invoke(messages)
        return response.content

class ResearchAgent:
    """LangGraph-based research agent for academic papers using Gemini"""
    
    def __init__(self):
        # Initialize Gemini client
        self.llm = GeminiClient(
            api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        
        self.scholar_api = GoogleScholarAPI(config.SERPAPI_KEY)
        self.document_processor = DocumentProcessor(config.CACHE_DIR)
        self.vector_store = VectorStore(config.VECTOR_DB_PATH, config.EMBEDDING_MODEL)
        
        self.logger = logging.getLogger(__name__)
        
        # Test Gemini connection
        if not self.llm.test_connection():
            self.logger.error("Failed to connect to Gemini API")
            raise ConnectionError("Unable to connect to Gemini API. Please check your API key.")
        
        model_info = self.llm.get_model_info()
        self.logger.info(f"Connected to Gemini model: {model_info.get('display_name', config.GEMINI_MODEL)}")
        
        # Initialize tools
        self.query_refinement_tool = QueryRefinementTool(self.llm)
        self.paper_analysis_tool = PaperAnalysisTool(self.llm)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("refine_query", self._refine_query)
        workflow.add_node("search_papers", self._search_papers)
        workflow.add_node("process_documents", self._process_documents)
        workflow.add_node("extract_context", self._extract_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("error_handler", self._error_handler)
        
        # Add edges
        workflow.set_entry_point("refine_query")
        workflow.add_edge("refine_query", "search_papers")
        workflow.add_edge("search_papers", "process_documents")
        workflow.add_edge("process_documents", "extract_context")
        workflow.add_edge("extract_context", "generate_answer")
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    async def _refine_query(self, state: ResearchState) -> ResearchState:
        """Refine the user query for better search results"""
        try:
            self.logger.info(f"Refining query: {state['original_query']}")
            
            refined_query = self.query_refinement_tool.refine_query(state['original_query'])
            
            state['refined_query'] = refined_query
            state['step'] = "query_refined"
            
            self.logger.info(f"Refined query: {refined_query}")
            
        except Exception as e:
            self.logger.error(f"Error refining query: {e}")
            state['error'] = str(e)
            state['step'] = "error"
        
        return state
    
    async def _search_papers(self, state: ResearchState) -> ResearchState:
        """Search for relevant papers using Google Scholar API"""
        try:
            self.logger.info(f"Searching papers for: {state['refined_query']}")
            
            # Use refined query if available, otherwise original
            search_query = state.get('refined_query', state['original_query'])
            
            # Search for papers
            results = await self.scholar_api.search_papers(search_query, config.MAX_RESULTS)
            
            state['search_results'] = results
            state['step'] = "papers_found"
            
            self.logger.info(f"Found {len(results)} papers")
            
        except Exception as e:
            self.logger.error(f"Error searching papers: {e}")
            state['error'] = str(e)
            state['step'] = "error"
        
        return state
    
    async def _process_documents(self, state: ResearchState) -> ResearchState:
        """Process found papers and extract content"""
        try:
            self.logger.info("Processing documents")
            
            processed_papers = []
            
            for paper in state['search_results']:
                if paper.pdf_url:
                    # Process the paper
                    chunks = await self.document_processor.process_paper(
                        paper.pdf_url, 
                        paper.title
                    )
                    
                    if chunks:
                        # Add to vector store
                        success = self.vector_store.add_documents(chunks)
                        if success:
                            processed_papers.append(paper.title)
                            self.logger.info(f"Processed paper: {paper.title}")
            
            state['processed_documents'] = processed_papers
            state['step'] = "documents_processed"
            
        except Exception as e:
            self.logger.error(f"Error processing documents: {e}")
            state['error'] = str(e)
            state['step'] = "error"
        
        return state
    
    async def _extract_context(self, state: ResearchState) -> ResearchState:
        """Extract relevant context from processed documents"""
        try:
            self.logger.info("Extracting context")
            
            # Search for relevant chunks
            search_results = self.vector_store.hybrid_search(
                state['original_query'], 
                k=10
            )
            
            # Extract and rank context
            context_chunks = []
            for doc, metadata, score in search_results:
                if score > 0.3:  # Threshold for relevance
                    context_chunks.append({
                        'text': doc,
                        'metadata': metadata,
                        'score': score
                    })
            
            state['context_chunks'] = context_chunks
            state['step'] = "context_extracted"
            
            self.logger.info(f"Extracted {len(context_chunks)} relevant chunks")
            
        except Exception as e:
            self.logger.error(f"Error extracting context: {e}")
            state['error'] = str(e)
            state['step'] = "error"
        
        return state
    
    async def _generate_answer(self, state: ResearchState) -> ResearchState:
        """Generate final answer based on extracted context"""
        try:
            self.logger.info("Generating answer")
            
            # Prepare context for LLM
            context_text = "\n\n".join([
                f"Source: {chunk['metadata'].get('title', 'Unknown')}\n{chunk['text']}"
                for chunk in state['context_chunks'][:5]  # Use top 5 chunks
            ])
            
            # Generate answer using LLM
            prompt = ChatPromptTemplate.from_template("""
            You are a research assistant helping users understand academic literature.
            
            User Question: {query}
            
            Context from Research Papers:
            {context}
            
            Based on the provided context from research papers, provide a comprehensive answer to the user's question.
            
            Guidelines:
            1. Directly address the user's question
            2. Cite specific papers and findings
            3. Mention key methodologies and results
            4. Note any limitations or conflicting findings
            5. Provide actionable insights when possible
            
            If the context doesn't contain enough information to answer the question, clearly state this.
            
            Format your answer with clear sections and proper citations.
            """)
            
            messages = prompt.format_messages(
                query=state['original_query'],
                context=context_text
            )
            
            response = self.llm.invoke(messages)
            
            state['final_answer'] = response.content
            state['step'] = "answer_generated"
            
            self.logger.info("Answer generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            state['error'] = str(e)
            state['step'] = "error"
        
        return state
    
    async def _error_handler(self, state: ResearchState) -> ResearchState:
        """Handle errors in the workflow"""
        self.logger.error(f"Error in workflow: {state.get('error', 'Unknown error')}")
        
        state['final_answer'] = f"I encountered an error while processing your request: {state.get('error', 'Unknown error')}. Please try again or rephrase your question."
        
        return state
    
    async def research(self, query: str) -> Dict:
        """
        Main research function
        
        Args:
            query: User's research query
            
        Returns:
            Dictionary with research results
        """
        try:
            # Initialize state
            initial_state = ResearchState(
                original_query=query,
                refined_query="",
                search_results=[],
                processed_documents=[],
                context_chunks=[],
                final_answer="",
                error=None,
                step="initialized"
            )
            
            # Run the workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Prepare response
            response = {
                'query': query,
                'answer': final_state.get('final_answer', 'No answer generated'),
                'sources': [
                    {
                        'title': result.title,
                        'authors': result.authors,
                        'year': result.year,
                        'citation_count': result.citation_count,
                        'url': result.scholar_url
                    }
                    for result in final_state.get('search_results', [])
                ],
                'processed_papers': final_state.get('processed_documents', []),
                'context_chunks_count': len(final_state.get('context_chunks', [])),
                'status': 'success' if not final_state.get('error') else 'error',
                'error': final_state.get('error'),
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in research workflow: {e}")
            return {
                'query': query,
                'answer': f"I encountered an error while processing your request: {str(e)}",
                'sources': [],
                'processed_papers': [],
                'context_chunks_count': 0,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_vector_store_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return self.vector_store.get_collection_stats()
    
    def clear_vector_store(self) -> bool:
        """Clear the vector store"""
        return self.vector_store.clear_collection()