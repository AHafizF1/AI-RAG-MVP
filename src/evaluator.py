"""
LLM Response Evaluation Module

This module provides functionality to evaluate LLM responses using LangSmith and Gemini.
It integrates with the existing evaluate.py functionality.
"""
import os
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable
from functools import wraps
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the evaluation functions from evaluate.py
def import_evaluation_functions():
    """Dynamically import functions from evaluate.py."""
    try:
        spec = importlib.util.spec_from_file_location(
            "evaluate_module", 
            os.path.join(os.path.dirname(__file__), "..", "evaluate.py")
        )
        evaluate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluate_module)
        return evaluate_module
    except Exception as e:
        logger.error(f"Failed to import evaluation functions: {e}")
        raise

class ResponseEvaluator:
    """Handles evaluation of LLM responses using the existing evaluate.py module."""
    
    def __init__(self):
        """Initialize the response evaluator with the evaluation module."""
        try:
            self.eval_module = import_evaluation_functions()
            logger.info("Successfully imported evaluation functions")
        except Exception as e:
            logger.error(f"Failed to initialize ResponseEvaluator: {e}")
            raise
    
    async def evaluate_response(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        reference: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Evaluate a response asynchronously.
        
        Args:
            inputs: Dictionary containing the input to the LLM (e.g., {"question": "..."})
            outputs: Dictionary containing the LLM's output (e.g., {"answer": "..."})
            reference: Optional reference output for comparison
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Use the imported evaluation function
            eval_result = await self.eval_module.evaluate_response_quality_async(
                inputs=inputs,
                outputs=outputs,
                reference_outputs=reference or {}
            )
            
            # Process the evaluation result
            return {
                'success': True,
                'scores': {
                    'overall': eval_result.get('overall_score', 0),
                    'correctness': eval_result.get('correctness', 0),
                    'relevance': eval_result.get('relevance', 0)
                },
                'details': eval_result,
                'passed': eval_result.get('passed', False)
            }
            
        except Exception as e:
            logger.error(f"Error in evaluate_response: {e}")
            return {
                'success': False,
                'error': str(e),
                'passed': False
            }
    
    def __call__(self, *args, **kwargs):
        """Synchronous wrapper for the async evaluate_response method."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, schedule the evaluation
                future = asyncio.run_coroutine_threadsafe(
                    self.evaluate_response(*args, **kwargs),
                    loop
                )
                return future.result()
            else:
                # Otherwise run it in the current event loop
                return loop.run_until_complete(
                    self.evaluate_response(*args, **kwargs)
                )
        except RuntimeError:
            # No event loop, create a new one
            return asyncio.run(self.evaluate_response(*args, **kwargs))
        except Exception as e:
            logger.error(f"Error in synchronous evaluation: {e}")
            return {
                'success': False, 
                'error': str(e),
                'passed': False
            }

def with_evaluation(evaluator=None):
    """Decorator to add evaluation to any LLM call.
    
    Args:
        evaluator: Optional evaluator instance. If None, a new one will be created.
        
    Returns:
        Decorated function with evaluation
    """
    if evaluator is None:
        evaluator = ResponseEvaluator()
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get the prompt and response
            prompt = kwargs.get('prompt') or (args[0] if args else None)
            if not prompt:
                raise ValueError("Prompt not found in arguments")
            
            # Call the original function
            result = await func(*args, **kwargs)
            
            # Run evaluation in background
            asyncio.create_task(
                evaluator.evaluate_response(
                    inputs={"question": prompt}, 
                    outputs={"answer": result}
                )
            )
            
            return result
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get the prompt and response
            prompt = kwargs.get('prompt') or (args[0] if args else None)
            if not prompt:
                raise ValueError("Prompt not found in arguments")
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Run evaluation in background
            asyncio.create_task(
                evaluator.evaluate_response(
                    inputs={"question": prompt}, 
                    outputs={"answer": result}
                )
            )
            
            return result
            
        # Return the appropriate wrapper
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator
