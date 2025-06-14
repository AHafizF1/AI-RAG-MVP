"""
Test script for the evaluation pipeline.

This script demonstrates how to use the evaluation system with sample prompts.
"""
import pytest
import asyncio
from typing import Dict, Any, List, Optional
from src.llm_client import LLMClient
from evaluator import ResponseEvaluator
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test cases
TEST_CASES = [
    {
        "prompt": "What are the key safety guidelines for operating the CNC machine?",
        "expected_keywords": ["safety", "guidelines", "CNC", "machine", "protective"],
        "system_prompt": "You are a helpful assistant that provides accurate information about manufacturing processes."
    },
    {
        "prompt": "How do I request expedited processing for order #12345?",
        "expected_keywords": ["expedite", "order", "process", "request"],
        "system_prompt": "You are a helpful assistant that provides information about order processing."
    },
    {
        "prompt": "What's the maintenance schedule for the injection molding machine?",
        "expected_keywords": ["maintenance", "schedule", "injection", "molding"],
        "system_prompt": "You are a helpful assistant that provides information about equipment maintenance."
    }
]

# Fixture for creating a test LLM client
@pytest.fixture
def llm_client():
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        temperature=0.3,
        max_output_tokens=2048,
        convert_system_message_to_human=True
    )
    return LLMClient(llm=llm)

# Fixture for creating an evaluator
@pytest.fixture
def evaluator():
    return ResponseEvaluator()

@pytest.mark.asyncio
async def test_evaluation(evaluator: ResponseEvaluator, llm_client: LLMClient):
    """Test the evaluation pipeline with sample prompts."""
    print("\n🚀 Starting evaluation test...")
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n🔍 Test Case {i}: {test_case['prompt']}")
        print("-" * 80)
        
        try:
            # Create a proper message format for the LLM
            messages = [
                ("system", test_case.get("system_prompt", "You are a helpful assistant.")),
                ("human", test_case["prompt"])
            ]
            
            # Generate response from LLM
            print("🤖 Generating response...")
            response = await llm_client.generate_async(
                prompt=test_case["prompt"]
            )
            
            # Get the generated text from the response
            generated_text = response.get('response', '').strip()
            if not generated_text:
                logger.warning(f"Empty response received for prompt: {test_case['prompt']}")
                continue
                
            print(f"\n📝 Response: {generated_text[:200]}...")
            
            # Run evaluation
            print("\n📊 Running evaluation...")
            try:
                eval_result = await evaluator.evaluate_response(
                    inputs={"question": test_case["prompt"]},
                    outputs={"answer": generated_text}
                )
                
                # Print evaluation results
                print("\n✅ Evaluation Results:")
                for metric, value in eval_result.items():
                    if isinstance(value, (int, float)):
                        print(f"  - {metric}: {value:.2f}")
                    elif isinstance(value, dict):
                        print(f"  - {metric}:")
                        for k, v in value.items():
                            print(f"    - {k}: {v}")
                    else:
                        print(f"  - {metric}: {value}")
                        
            except Exception as e:
                logger.error(f"Error during evaluation: {str(e)}")
                print(f"\n⚠️  Evaluation failed: {str(e)}")
            
            # Check for expected keywords
            if generated_text:  # Only check keywords if we have a response
                missing_keywords = [
                    kw for kw in test_case["expected_keywords"] 
                    if kw.lower() not in generated_text.lower()
                ]
                
                if missing_keywords:
                    print(f"\n⚠️  Missing expected keywords: {', '.join(missing_keywords)}")
                else:
                    print("\n✅ All expected keywords found in response")
            
                # Add assertions
                assert isinstance(generated_text, str), "Response should be a string"
                assert len(generated_text) > 0, "Response should not be empty"
                assert isinstance(eval_result, dict), "Evaluation result should be a dictionary"
            
        except Exception as e:
            logger.error(f"Error in test case {i}: {str(e)}")
            print(f"\n❌ Test case failed: {str(e)}")
            raise
        
        # Verify the response contains some content
        assert len(generated_text) > 10, "Response should contain meaningful content"
