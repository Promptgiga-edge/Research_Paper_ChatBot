"""
LLM Evaluation Framework
This module provides comprehensive evaluation to ensure LLM uses only vector-stored data
and doesn't access external sources during inference.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

# LangChain imports for evaluation
from langchain.evaluation import load_evaluator
from langchain.evaluation.criteria import CriteriaEvalChain
from langchain.evaluation.string_distance import StringDistanceEvalChain
from langchain.evaluation.embedding_distance import EmbeddingDistanceEvalChain
from langchain_community.llms import OpenAI
from langchain.schema import Document

# For network monitoring
import requests
import urllib3
from unittest import mock

# Custom imports (adjust based on your project structure)
from vector_store import VectorStore
from research_agent import ResearchAgent
from gemini_client import GeminiClient


@dataclass
class EvaluationResult:
    """Data class to store evaluation results"""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    timestamp: datetime
    execution_time: float


class NetworkMonitor:
    """Monitor network calls to ensure no external data access"""
    
    def __init__(self):
        self.network_calls = []
        self.blocked_domains = []
        
    def start_monitoring(self):
        """Start monitoring network calls"""
        self.network_calls = []
        
        # Mock requests.get, requests.post, etc.
        original_get = requests.get
        original_post = requests.post
        
        def mock_get(*args, **kwargs):
            self.network_calls.append(f"GET: {args[0] if args else 'Unknown URL'}")
            # Allow only localhost/internal calls
            if args and not self._is_allowed_url(args[0]):
                raise Exception(f"Blocked external network call: {args[0]}")
            return original_get(*args, **kwargs)
            
        def mock_post(*args, **kwargs):
            self.network_calls.append(f"POST: {args[0] if args else 'Unknown URL'}")
            if args and not self._is_allowed_url(args[0]):
                raise Exception(f"Blocked external network call: {args[0]}")
            return original_post(*args, **kwargs)
        
        requests.get = mock_get
        requests.post = mock_post
        
    def _is_allowed_url(self, url: str) -> bool:
        """Check if URL is allowed (internal/localhost only)"""
        allowed_patterns = [
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
            'file://',
            # Add your vector store URL patterns here
        ]
        return any(pattern in url.lower() for pattern in allowed_patterns)
    
    def get_network_calls(self) -> List[str]:
        """Get list of network calls made"""
        return self.network_calls.copy()


class VectorStoreEvaluator:
    """Evaluate if LLM uses only vector store data"""
    
    def __init__(self, vector_store: VectorStore, llm_client):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.network_monitor = NetworkMonitor()
        self.evaluation_results = []
        
    def create_test_dataset(self) -> List[Dict[str, Any]]:
        """Create test dataset with known vector store content and external knowledge"""
        test_cases = [
            {
                "query": "What information do you have about machine learning?",
                "type": "vector_content",
                "expected_source": "vector_store",
                "description": "Query about content that should be in vector store"
            },
            {
                "query": "What's the current weather in New York?",
                "type": "external_knowledge", 
                "expected_source": "none",
                "description": "Query requiring external API that should be blocked"
            },
            {
                "query": "Tell me about the latest news from today",
                "type": "real_time_data",
                "expected_source": "none", 
                "description": "Query requiring real-time data not in vector store"
            },
            {
                "query": "What is 2+2?",
                "type": "general_knowledge",
                "expected_source": "internal",
                "description": "Basic knowledge that doesn't require external sources"
            }
        ]
        
        # Add queries based on actual vector store content
        vector_samples = self._sample_vector_content()
        for sample in vector_samples:
            test_cases.append({
                "query": f"Tell me about {sample['topic']}",
                "type": "vector_content",
                "expected_source": "vector_store",
                "description": f"Query about known vector content: {sample['topic']}",
                "ground_truth": sample['content']
            })
            
        return test_cases
    
    def _sample_vector_content(self) -> List[Dict[str, Any]]:
        """Sample some content from vector store for testing"""
        try:
            # This depends on your vector store implementation
            # Adjust based on your VectorStore class methods
            samples = []
            
            # Example: get some random documents from vector store
            if hasattr(self.vector_store, 'get_all_documents'):
                docs = self.vector_store.get_all_documents()[:5]  # Get first 5
                for doc in docs:
                    samples.append({
                        'topic': doc.metadata.get('title', 'Unknown'),
                        'content': doc.page_content[:200]  # First 200 chars
                    })
            
            return samples
        except Exception as e:
            logging.warning(f"Could not sample vector content: {e}")
            return []
    
    async def evaluate_vector_isolation(self) -> List[EvaluationResult]:
        """Main evaluation method to test vector store isolation"""
        results = []
        test_cases = self.create_test_dataset()
        
        for test_case in test_cases:
            result = await self._evaluate_single_query(test_case)
            results.append(result)
            
        return results
    
    async def _evaluate_single_query(self, test_case: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single query"""
        start_time = time.time()
        
        # Start network monitoring
        self.network_monitor.start_monitoring()
        
        try:
            # Execute query
            response = await self._execute_query(test_case["query"])
            
            # Check for network calls
            network_calls = self.network_monitor.get_network_calls()
            
            # Evaluate response
            evaluation_score = await self._evaluate_response(test_case, response, network_calls)
            
            execution_time = time.time() - start_time
            
            return EvaluationResult(
                test_name=test_case["description"],
                passed=evaluation_score["passed"],
                score=evaluation_score["score"],
                details={
                    "query": test_case["query"],
                    "response": response,
                    "network_calls": network_calls,
                    "evaluation": evaluation_score,
                    "test_type": test_case["type"]
                },
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return EvaluationResult(
                test_name=test_case["description"],
                passed=False,
                score=0.0,
                details={
                    "error": str(e),
                    "query": test_case["query"],
                    "test_type": test_case["type"]
                },
                timestamp=datetime.now(),
                execution_time=execution_time
            )
    
    async def _execute_query(self, query: str) -> str:
        """Execute query using your LLM client"""
        try:
            # Adjust this based on your actual LLM client interface
            if hasattr(self.llm_client, 'query_async'):
                response = await self.llm_client.query_async(query)
            elif hasattr(self.llm_client, 'query'):
                response = self.llm_client.query(query)
            else:
                # Fallback - adjust based on your implementation
                response = str(self.llm_client.generate(query))
                
            return response
        except Exception as e:
            raise Exception(f"Failed to execute query: {e}")
    
    async def _evaluate_response(self, test_case: Dict[str, Any], response: str, network_calls: List[str]) -> Dict[str, Any]:
        """Evaluate the response based on test case expectations"""
        evaluation = {
            "passed": True,
            "score": 1.0,
            "reasons": []
        }
        
        # Check network isolation
        if test_case["type"] in ["external_knowledge", "real_time_data"]:
            if network_calls:
                evaluation["passed"] = False
                evaluation["score"] -= 0.5
                evaluation["reasons"].append(f"Unexpected network calls: {network_calls}")
            
            # For external knowledge queries, response should indicate limitation
            if not self._response_indicates_limitation(response):
                evaluation["passed"] = False
                evaluation["score"] -= 0.3
                evaluation["reasons"].append("Response doesn't indicate knowledge limitation")
        
        # Check vector content usage
        if test_case["type"] == "vector_content" and "ground_truth" in test_case:
            similarity_score = await self._calculate_content_similarity(
                response, test_case["ground_truth"]
            )
            if similarity_score < 0.3:  # Threshold for similarity
                evaluation["score"] -= 0.4
                evaluation["reasons"].append(f"Low similarity to vector content: {similarity_score}")
        
        # Final score adjustment
        evaluation["score"] = max(0.0, evaluation["score"])
        
        return evaluation
    
    def _response_indicates_limitation(self, response: str) -> bool:
        """Check if response indicates knowledge limitations"""
        limitation_indicators = [
            "i don't have access",
            "i cannot access",
            "i don't have current",
            "i cannot provide current",
            "based on my training data",
            "i don't have real-time",
            "i cannot browse",
            "i don't have the ability to access"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in limitation_indicators)
    
    async def _calculate_content_similarity(self, response: str, ground_truth: str) -> float:
        """Calculate similarity between response and ground truth using embeddings"""
        try:
            # Use LangChain's embedding distance evaluator
            evaluator = EmbeddingDistanceEvalChain()
            result = evaluator.evaluate_strings(
                prediction=response,
                reference=ground_truth
            )
            return 1.0 - result["score"]  # Convert distance to similarity
        except Exception as e:
            logging.warning(f"Could not calculate embedding similarity: {e}")
            # Fallback to simple text similarity
            return self._simple_text_similarity(response, ground_truth)
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity as fallback"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class LangChainEvaluator:
    """Use LangChain's evaluation tools for comprehensive testing"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    async def evaluate_with_criteria(self, queries_and_responses: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Evaluate responses using LangChain criteria"""
        
        # Define custom criteria for vector store isolation
        criteria = {
            "vector_isolation": "Does the response only use information that could reasonably be stored in a vector database, without accessing external real-time data or making API calls?",
            "knowledge_limitation": "Does the response appropriately indicate limitations when asked about information not available in the training data?",
            "factual_accuracy": "Is the response factually accurate based on the given context?",
            "relevance": "Is the response relevant to the query?"
        }
        
        results = []
        
        for query, response in queries_and_responses:
            try:
                # Use LangChain's criteria evaluator
                evaluator = load_evaluator("criteria", criteria=criteria)
                eval_result = evaluator.evaluate_strings(
                    input=query,
                    prediction=response
                )
                
                results.append({
                    "query": query,
                    "response": response,
                    "evaluation": eval_result,
                    "timestamp": datetime.now()
                })
                
            except Exception as e:
                results.append({
                    "query": query,
                    "response": response,
                    "evaluation": {"error": str(e)},
                    "timestamp": datetime.now()
                })
        
        return results
    
    async def evaluate_string_distance(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate using string distance metrics"""
        try:
            evaluator = StringDistanceEvalChain()
            
            scores = []
            for pred, ref in zip(predictions, references):
                result = evaluator.evaluate_strings(prediction=pred, reference=ref)
                scores.append(result["score"])
            
            return {
                "average_distance": sum(scores) / len(scores) if scores else 0.0,
                "individual_scores": scores
            }
            
        except Exception as e:
            return {"error": str(e)}


class EvaluationReporter:
    """Generate comprehensive evaluation reports"""
    
    def __init__(self):
        self.report_data = {}
        
    def generate_report(self, evaluation_results: List[EvaluationResult], 
                       langchain_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        report = {
            "summary": self._generate_summary(evaluation_results),
            "detailed_results": self._format_detailed_results(evaluation_results),
            "langchain_evaluation": langchain_results,
            "recommendations": self._generate_recommendations(evaluation_results),
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _generate_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not results:
            return {"error": "No evaluation results"}
            
        passed_tests = sum(1 for r in results if r.passed)
        total_tests = len(results)
        average_score = sum(r.score for r in results) / total_tests
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": passed_tests / total_tests,
            "average_score": average_score,
            "total_execution_time": sum(r.execution_time for r in results)
        }
    
    def _format_detailed_results(self, results: List[EvaluationResult]) -> List[Dict[str, Any]]:
        """Format detailed results for reporting"""
        return [
            {
                "test_name": r.test_name,
                "passed": r.passed,
                "score": r.score,
                "execution_time": r.execution_time,
                "details": r.details,
                "timestamp": r.timestamp.isoformat()
            }
            for r in results
        ]
    
    def _generate_recommendations(self, results: List[EvaluationResult]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Analyze failed tests
        failed_tests = [r for r in results if not r.passed]
        
        if failed_tests:
            network_issues = [r for r in failed_tests if "network_calls" in r.details]
            if network_issues:
                recommendations.append(
                    "Consider implementing stricter network isolation to prevent external API calls"
                )
            
            similarity_issues = [r for r in failed_tests if "similarity" in str(r.details)]
            if similarity_issues:
                recommendations.append(
                    "Improve vector store content coverage or retrieval accuracy"
                )
        
        if not recommendations:
            recommendations.append("All tests passed - vector store isolation is working correctly")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save report to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)


# Main evaluation function
async def run_comprehensive_evaluation(vector_store: VectorStore, llm_client) -> Dict[str, Any]:
    """Run comprehensive evaluation of LLM vector store isolation"""
    
    print("Starting LLM Vector Store Isolation Evaluation...")
    
    # Initialize evaluators
    vector_evaluator = VectorStoreEvaluator(vector_store, llm_client)
    langchain_evaluator = LangChainEvaluator(llm_client)
    reporter = EvaluationReporter()
    
    # Run vector isolation evaluation
    print("Running vector isolation tests...")
    vector_results = await vector_evaluator.evaluate_vector_isolation()
    
    # Prepare data for LangChain evaluation
    queries_and_responses = []
    for result in vector_results:
        if "query" in result.details and "response" in result.details:
            queries_and_responses.append((
                result.details["query"],
                result.details["response"]
            ))
    
    # Run LangChain evaluation
    print("Running LangChain evaluation...")
    langchain_results = await langchain_evaluator.evaluate_with_criteria(queries_and_responses)
    
    # Generate comprehensive report
    print("Generating evaluation report...")
    report = reporter.generate_report(vector_results, langchain_results)
    
    # Save report
    report_filepath = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    reporter.save_report(report, report_filepath)
    
    print(f"Evaluation complete! Report saved to: {report_filepath}")
    print(f"Summary: {report['summary']}")
    
    return report


# Example usage
if __name__ == "__main__":
    # This is an example of how to use the evaluation framework
    # Adjust imports and initialization based on your actual implementation
    
    async def main():
        # Initialize your components
        # vector_store = VectorStore()  # Your vector store instance
        # llm_client = GeminiClient()   # Your LLM client instance
        
        # Run evaluation
        # report = await run_comprehensive_evaluation(vector_store, llm_client)
        
        print("Evaluation framework ready!")
        print("To use:")
        print("1. Initialize your VectorStore and LLM client")
        print("2. Call run_comprehensive_evaluation(vector_store, llm_client)")
        print("3. Review the generated report")
    
    asyncio.run(main())
