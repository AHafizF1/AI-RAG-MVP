"""
Evaluation module for the Manufacturing Agent MVP.

This module contains evaluation logic using LangSmith and Gemini.
"""
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# LangSmith and Gemini imports
try:
    from langsmith import Client, wrappers
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import (
        CONCISENESS_PROMPT,
        CORRECTNESS_PROMPT,
        HALLUCINATION_PROMPT
    )
    import google.generativeai as genai
    
    # Initialize Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Initialize LangSmith client
    client = Client()
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    raise

def setup_gemini_llm(eval_mode: bool = False):
    """Set up and return the Gemini LLM.
    
    Args:
        eval_mode: If True, use evaluation-specific model settings
        
    Returns:
        Configured ChatGoogleGenerativeAI instance
    """
    try:
        if eval_mode:
            model_name = os.getenv("EVAL_GEMINI_MODEL", "gemini-1.5-flash-8b")
            temperature = float(os.getenv("EVAL_GEMINI_TEMPERATURE", 0.3))
            max_tokens = int(os.getenv("EVAL_GEMINI_MAX_TOKENS", 4096))
        else:
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
            temperature = float(os.getenv("GEMINI_TEMPERATURE", 0.7))
            max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", 2048))
            
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            convert_system_message_to_human=True
        )
        logger.info(f"Initialized Gemini LLM in {'evaluation' if eval_mode else 'production'} mode")
        logger.info(f"Model: {model_name}, Temperature: {temperature}, Max Tokens: {max_tokens}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Gemini LLM: {e}")
        raise

def create_evaluation_dataset():
    """Create a sample dataset for evaluation or return existing one."""
    dataset_name = "Manufacturing_QA_Evaluation"
    try:
        # Try to get the existing dataset first
        try:
            dataset = client.read_dataset(dataset_name=dataset_name)
            logger.info(f"Using existing evaluation dataset: {dataset_name}")
            return dataset
        except:
            # If dataset doesn't exist, create it
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Dataset for evaluating manufacturing-related QA performance"
            )

            # Define evaluation examples
            examples = [
                {
                    "inputs": {"question": "What is the standard lead time for custom parts?"},
                    "outputs": {"answer": "The standard lead time for custom parts is 4-6 weeks."},
                },
                {
                    "inputs": {"question": "How do I request a quality inspection?"},
                    "outputs": {"answer": "Submit a quality inspection request through the portal with the required documentation."},
                },
            ]

            # Add examples to the dataset
            client.create_examples(dataset_id=dataset.id, examples=examples)
            logger.info(f"Created new evaluation dataset: {dataset_name}")
            return dataset
    except Exception as e:
        logger.error(f"Failed to create or access evaluation dataset: {e}")
        raise

def create_qa_chain(llm):
    """Create a QA chain using Gemini."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful manufacturing assistant. Answer the question based on the context."),
        ("human", "{question}")
    ])
    
    return {
        "question": RunnablePassthrough()
    } | prompt | llm | StrOutputParser()

def create_evaluators():
    """Create multiple evaluators for different aspects of the response."""
    evaluators = {}
    
    # Initialize the evaluation LLM
    eval_llm = setup_gemini_llm(eval_mode=True)
    
    # Helper function to create evaluator prompts
    def create_evaluator_prompt(criteria: str, description: str) -> str:
        return f"""You are an expert evaluator. Your task is to evaluate the response based on {description}.
        
        Question: {{inputs}}
        
        Response to evaluate: {{outputs}}
        
        Reference answer (if available): {{reference_outputs}}
        
        Evaluation Criteria:
        {criteria}
        
        Please provide:
        1. A score between 0 and 1, where 1 is perfect and 0 is completely wrong
        2. A brief explanation of your score
        
        Format your response as a JSON object with 'score' and 'explanation' keys.
        """
    
    # Correctness Evaluator
    evaluators['correctness'] = {
        'prompt': create_evaluator_prompt(
            "1. Is the response factually accurate?\n2. Does it correctly answer the question?",
            "the factual accuracy of the response compared to the reference answer"
        ),
        'llm': eval_llm
    }
    
    # Conciseness Evaluator
    evaluators['conciseness'] = {
        'prompt': create_evaluator_prompt(
            "1. Is the response concise and to the point?\n2. Does it avoid unnecessary details?",
            "the conciseness of the response"
        ),
        'llm': eval_llm
    }
    
    # Hallucination Evaluator
    evaluators['hallucination'] = {
        'prompt': create_evaluator_prompt(
            "1. Does the response contain any made-up or unsubstantiated information?\n2. Is the information verifiable?",
            "whether the response contains hallucinations or made-up information"
        ),
        'llm': eval_llm
    }
    
    # Manufacturing Context Evaluator
    evaluators['manufacturing_context'] = {
        'prompt': create_evaluator_prompt(
            "1. Is the response technically accurate for manufacturing?\n"
            "2. Is it relevant to manufacturing processes?\n"
            "3. Is it practically applicable in a factory setting?\n"
            "4. Does it consider safety aspects?",
            "the technical accuracy and relevance of the response in a manufacturing context"
        ),
        'llm': eval_llm
    }
    
    return evaluators

async def evaluate_single_criteria(evaluator: Dict, eval_input: Dict) -> Dict:
    """Run a single evaluation criteria."""
    try:
        # Ensure eval_input is not None and has required fields
        if not eval_input or 'inputs' not in eval_input or 'outputs' not in eval_input:
            return {
                'score': 0.0,
                'passed': False,
                'comment': 'Invalid evaluation input: missing required fields',
                'error': True
            }
            
        # Format the prompt with the input values
        prompt = evaluator['prompt'].format(
            inputs=eval_input.get('inputs', ''),
            outputs=eval_input.get('outputs', ''),
            reference_outputs=eval_input.get('reference_outputs', '')
        )
        
        # Get the LLM response
        response = await evaluator['llm'].ainvoke([("user", prompt)])
        
        # Parse the response (expecting JSON)
        try:
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
                
            # Try to extract JSON from the response
            import re
            import json
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                # If no JSON found, try to extract a score from the text
                score_match = re.search(r'(?i)score[\s:]*([0-9.]+)', content)
                score = float(score_match.group(1)) if score_match else 0.5
                result = {
                    'score': min(1.0, max(0.0, score / 1.0)),  # Ensure score is between 0 and 1
                    'explanation': content[:500]  # Truncate long explanations
                }
                
            return {
                'score': float(result.get('score', 0.5)),
                'passed': result.get('score', 0) >= 0.7,  # 0.7 threshold for passing
                'comment': result.get('explanation', 'No explanation provided')
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse evaluation response: {e}")
            return {
                'score': 0.5,
                'passed': False,
                'comment': f'Failed to parse evaluation response: {str(e)[:200]}',
                'error': True
            }
            
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        return {
            'score': 0.0,
            'passed': False,
            'comment': f'Evaluation error: {str(e)[:200]}',
            'error': True
        }

async def evaluate_response_quality_async(inputs: Dict, outputs: Dict, reference_outputs: Optional[Dict] = None) -> Dict:
    """Asynchronously evaluate the quality of a response using multiple evaluators.
    
    Args:
        inputs: Dictionary containing the input question/context
        outputs: Dictionary containing the model's response
        reference_outputs: Optional dictionary containing reference/expected output
        
    Returns:
        Dictionary containing evaluation results
    """
    try:
        # Get all evaluators
        evaluators = create_evaluators()
        results = {}
        
        # Prepare common evaluation input with proper defaults
        eval_input = {
            'inputs': inputs.get('question', '') if isinstance(inputs, dict) else str(inputs or ''),
            'outputs': outputs.get('answer', '') if isinstance(outputs, dict) else str(outputs or ''),
            'reference_outputs': reference_outputs.get('answer', '') if (reference_outputs and isinstance(reference_outputs, dict)) else str(reference_outputs or ''),
            'question': inputs.get('question', '') if isinstance(inputs, dict) else str(inputs or ''),
            'answer': outputs.get('answer', '') if isinstance(outputs, dict) else str(outputs or '')
        }
        
        # Run all evaluations concurrently
        import asyncio
        tasks = []
        for eval_name, evaluator in evaluators.items():
            task = asyncio.create_task(evaluate_single_criteria(evaluator, eval_input))
            tasks.append((eval_name, task))
        
        # Wait for all evaluations to complete
        for eval_name, task in tasks:
            results[eval_name] = await task
        
        # Calculate overall score
        scores = [r['score'] for r in results.values() if 'score' in r]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            'success': 1.0 if all(r.get('passed', False) for r in results.values()) else 0.0,
            'scores': {
                'overall': overall_score,
                'correctness': results.get('correctness', {}).get('score', 0.0),
                'relevance': results.get('manufacturing_context', {}).get('score', 0.0)
            },
            'details': {
                'overall_score': overall_score,
                'passed': all(r.get('passed', False) for r in results.values()),
                'error': False,
                'scores': {
                    'overall': overall_score,
                    'correctness': results.get('correctness', {}).get('score', 0.0),
                    'relevance': results.get('manufacturing_context', {}).get('score', 0.0)
                },
                'details': {
                    'error': False,
                    'message': 'Evaluation completed successfully'
                }
            },
            'passed': 1.0 if all(r.get('passed', False) for r in results.values()) else 0.0
        }
        
    except NameError as e:
        # Handle case where 'inputs' is not defined
        error_msg = "Missing required 'inputs' parameter in evaluation"
        logger.error(error_msg)
        return {
            'overall_score': 0.0,
            'passed': False,
            'error': True,
            'error_message': error_msg,
            'scores': {
                'overall': 0.0,
                'correctness': 0.0,
                'relevance': 0.0
            },
            'details': {
                'error': True,
                'message': error_msg
            }
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in evaluate_response_quality: {error_msg}")
        return {
            'overall_score': 0.0,
            'passed': False,
            'error': True,
            'error_message': f'Evaluation error: {error_msg[:500]}',
            'scores': {
                'overall': 0.0,
                'correctness': 0.0,
                'relevance': 0.0
            },
            'details': {
                'error': True,
                'message': error_msg[:500]
            }
        }

def evaluate_response_quality(inputs: Dict, outputs: Dict, reference_outputs: Optional[Dict] = None) -> Dict:
    """Synchronous wrapper for the async evaluation function.
    
    Args:
        inputs: Dictionary containing the input question/context
        outputs: Dictionary containing the model's response
        reference_outputs: Optional dictionary containing reference/expected output
        
    Returns:
        Dictionary containing evaluation results
    """
    import asyncio
    return asyncio.run(evaluate_response_quality_async(inputs, outputs, reference_outputs))

def evaluate_qa_performance():
    """Run the evaluation pipeline."""
    try:
        # Setup Gemini LLM - using the main model for generation
        llm = setup_gemini_llm(eval_mode=False)
        logger.info("Using main model for response generation")
        
        # Log evaluation model info
        eval_llm = setup_gemini_llm(eval_mode=True)
        logger.info(f"Using evaluation model: {os.getenv('EVAL_GEMINI_MODEL', 'gemini-1.5-flash-8b')}")
        
        # Create evaluation dataset
        dataset = create_evaluation_dataset()
        
        # Define the target function for evaluation
        def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
            try:
                if not isinstance(inputs, dict) or 'question' not in inputs:
                    logger.error(f"Invalid input format: {inputs}")
                    return {"answer": "Error: Invalid input format - expected dict with 'question' key"}
                    
                chain = create_qa_chain(llm)
                response = chain.invoke({"question": str(inputs["question"])})
                
                # Ensure response is in the correct format
                if isinstance(response, dict) and 'answer' in response:
                    return {"answer": str(response['answer'])}
                elif isinstance(response, str):
                    return {"answer": response}
                else:
                    return {"answer": str(response)}
                    
            except Exception as e:
                error_msg = f"Error in target function: {str(e)}"
                logger.error(error_msg)
                return {"answer": error_msg}
        
        # Run the evaluation with our custom evaluator
        logger.info("Starting evaluation...")
        experiment_results = client.evaluate(
            target,
            data=dataset.name,
            evaluators=[evaluate_response_quality],
            experiment_prefix="manufacturing-qa-eval",
            max_concurrency=2,
        )
        
        # Log and return results
        logger.info(f"Evaluation completed. Results: {experiment_results}")
        
        # Print summary of results
        print("\n" + "="*80)
        print(f"{' MANUFACTURING AGENT EVALUATION SUMMARY ':=^80}")
        print("="*80)
        
        total_questions = len(experiment_results)
        category_scores = {
            'correctness': [],
            'conciseness': [],
            'hallucination': [],
            'manufacturing_context': []
        }
        
        for i, result in enumerate(experiment_results, 1):
            try:
                eval_result = result.get('results', {})
                
                print(f"\n{'='*35} Question {i} {'='*35}")
                
                # Safely get input
                try:
                    question = result.get('inputs', {}).get('question', 'No question provided')
                    if not isinstance(question, str):
                        question = str(question)
                    print(f"\n📌 Input:")
                    print(f"   {question}")
                except Exception as e:
                    print(f"\n⚠️ Error getting input: {str(e)[:200]}")
                
                # Safely get output
                try:
                    output = result.get('outputs', {}).get('answer', 'No output')
                    if not isinstance(output, str):
                        output = str(output)
                    print(f"\n🤖 Output:")
                    print(f"   {output}")
                except Exception as e:
                    print(f"\n⚠️ Error getting output: {str(e)[:200]}")
                
                # Safely get reference output
                try:
                    ref_output = result.get('reference_outputs', {}).get('answer', 'No reference answer')
                    if not isinstance(ref_output, str):
                        ref_output = str(ref_output)
                    print(f"\n✅ Expected:")
                    print(f"   {ref_output}")
                except Exception as e:
                    print(f"\n⚠️ Error getting reference output: {str(e)[:200]}")
                
                # Print evaluation results for each category
                print("\n📊 Evaluation Results:")
                evaluations = {}
                
                # Handle different possible evaluation result formats
                if isinstance(eval_result, dict):
                    if 'evaluations' in eval_result:
                        evaluations = eval_result.get('evaluations', {})
                    else:
                        evaluations = eval_result
                
                for category in category_scores.keys():
                    if category in evaluations:
                        eval_data = evaluations[category]
                        if isinstance(eval_data, dict):
                            score = float(eval_data.get('score', 0))
                            passed = bool(eval_data.get('passed', score >= 0.7))
                            comment = str(eval_data.get('comment', 'No comment'))
                            
                            # Track scores for final summary
                            category_scores[category].append(score)
                            
                            print(f"\n  {category.upper()}:")
                            print(f"  {'⭐' * int(score * 5):<10} {score:.2f}/1.0")
                            print(f"  Status: {'✅ PASSED' if passed else '❌ FAILED'}")
                            if comment and comment.lower() != 'no comment':
                                print(f"  💬 {comment[:200]}" + ('...' if len(comment) > 200 else ''))
            
            except Exception as e:
                logger.error(f"Error processing evaluation result {i}: {str(e)[:200]}")
                continue
        
        # Calculate and print final statistics
        print("\n" + "="*80)
        print(f"{' FINAL EVALUATION METRICS ':=^80}")
        print("="*80)
        
        # Print scores by category
        print("\n📈 AVERAGE SCORES BY CATEGORY:")
        for category, scores in category_scores.items():
            if scores:  # Only print if we have scores for this category
                avg_score = sum(scores) / len(scores)
                print(f"  {category.replace('_', ' ').title()}: {avg_score*100:.1f}%")
        
        # Calculate and print overall statistics
        all_scores = [score for scores in category_scores.values() for score in scores]
        overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
        
        print(f"\n🏆 OVERALL SCORE: {overall_avg*100:.1f}%")
        print(f"📋 Total Questions Evaluated: {total_questions}")
        print("="*80)
        
        # Print recommendations for improvement
        print("\n🔍 RECOMMENDATIONS FOR IMPROVEMENT:")
        has_issues = False
        for category, scores in category_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score < 0.7:  # Highlight areas needing improvement
                    has_issues = True
                    print(f"  • Consider improving {category.replace('_', ' ')} (current: {avg_score*100:.1f}%)")
        
        if not has_issues and any(category_scores.values()):
            print("  • Great job! All evaluation metrics are above the 70% threshold.")
        
        print("\n" + "="*80)
        
        return experiment_results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    evaluate_qa_performance()
