"""
LLM Client Module

This module provides a wrapper around LLM calls with built-in evaluation.
"""
import os
import logging
import asyncio
import json
from typing import Dict, Any, Optional, Callable, Awaitable
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class LLMClient:
    """Client for making LLM calls with built-in evaluation."""
    
    def __init__(self, llm, evaluator=None):
        """Initialize the LLM client.
        
        Args:
            llm: The language model instance to use
            evaluator: Optional evaluator function (async)
        """
        self.llm = llm
        self.evaluator = evaluator
        self._executor = ThreadPoolExecutor(max_workers=5)
        self.evaluation_results = []
    
    async def _run_evaluation(self, inputs: Dict, outputs: Dict, reference: Optional[Dict] = None) -> Dict:
        """Run evaluation in the background."""
        if not self.evaluator:
            return {}
            
        try:
            # Run the evaluation in a thread pool
            loop = asyncio.get_event_loop()
            eval_result = await loop.run_in_executor(
                self._executor,
                lambda: self.evaluator(inputs, outputs, reference or {})
            )
            
            # Store the evaluation result
            eval_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'inputs': inputs,
                'outputs': outputs,
                'evaluation': eval_result
            }
            self.evaluation_results.append(eval_data)
            return eval_result
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {'error': str(e)}
    
    async def generate_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response asynchronously with evaluation.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional arguments for the LLM
            
        Returns:
            Dict containing response and evaluation results
        """
        try:
            # Import the HumanMessage class here to avoid circular imports
            from langchain_core.messages import HumanMessage
            
            # Create a proper message object
            message = HumanMessage(content=prompt)
            
            # Generate the response
            response = await self.llm.agenerate([[message]])
            response_text = response.generations[0][0].text
            
            # Prepare evaluation data
            inputs = {"question": prompt}
            outputs = {"answer": response_text}
            
            # Run evaluation in background
            eval_task = asyncio.create_task(
                self._run_evaluation(inputs, outputs)
            )
            
            return {
                'response': response_text,
                'evaluation': await eval_task if eval_task else None
            }
            
        except Exception as e:
            logger.error(f"Error in generate_async: {e}")
            raise
    
    def generate_sync(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response synchronously with evaluation.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional arguments for the LLM
            
        Returns:
            Dict containing response and evaluation results
        """
        try:
            # Generate the response
            response = self.llm.generate([[{"role": "user", "content": prompt}]])
            response_text = response.generations[0][0].text
            
            # Prepare evaluation data
            inputs = {"question": prompt}
            outputs = {"answer": response_text}
            
            # Run evaluation in background if we're in an async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, schedule the evaluation
                    asyncio.create_task(self._run_evaluation(inputs, outputs))
                else:
                    # Otherwise run it synchronously
                    loop.run_until_complete(self._run_evaluation(inputs, outputs))
            except RuntimeError:
                # No event loop, run in thread
                self._executor.submit(
                    lambda: asyncio.run(self._run_evaluation(inputs, outputs))
                )
            
            return {
                'response': response_text,
                'evaluation': None  # Evaluation runs in background
            }
            
        except Exception as e:
            logger.error(f"Error in generate_sync: {e}")
            raise
    
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Get aggregated evaluation metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.evaluation_results:
            return {}
            
        # Calculate basic metrics
        total = len(self.evaluation_results)
        successful = sum(1 for r in self.evaluation_results if not r['evaluation'].get('error'))
        error_rate = (total - successful) / total if total > 0 else 0
        
        # Calculate average scores if available
        scores = [
            r['evaluation'].get('score', 0) 
            for r in self.evaluation_results 
            if not r['evaluation'].get('error')
        ]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            'total_responses': total,
            'successful_evaluations': successful,
            'error_rate': error_rate,
            'average_score': avg_score,
            'last_evaluated': self.evaluation_results[-1]['timestamp'] if self.evaluation_results else None
        }
    
    def save_evaluations(self, filepath: str) -> None:
        """Save evaluation results to a JSON file.
        
        Args:
            filepath: Path to save the evaluation results
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2)
            logger.info(f"Saved {len(self.evaluation_results)} evaluation results to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save evaluations: {e}")
            raise
    
    def __del__(self):
        """Clean up resources."""
        self._executor.shutdown(wait=False)


def with_evaluation(eval_func: Callable):
    """Decorator to add evaluation to any LLM call.
    
    Args:
        eval_func: Function that takes (inputs, outputs, reference) and returns evaluation
        
    Returns:
        Decorated function with evaluation
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the prompt from args or kwargs
            prompt = kwargs.get('prompt') or (args[0] if args else None)
            if not prompt:
                raise ValueError("Prompt not found in arguments")
            
            # Call the original function
            result = await func(*args, **kwargs)
            
            # Run evaluation in background
            if asyncio.iscoroutinefunction(eval_func):
                asyncio.create_task(eval_func(
                    inputs={"question": prompt}, 
                    outputs={"answer": result}
                ))
            
            return result
        return wrapper
    return decorator
