#!/usr/bin/env python3
"""
Test script for the GAIA agent using real API keys.
This script simulates GAIA benchmark questions and helps debug/improve the agent.
"""

import os
import sys
import json
import tempfile
from typing import List, Dict, Any, Optional
import traceback
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Import our agent
from core_agent import GAIAAgent

# Simulation of GAIA benchmark questions
SAMPLE_QUESTIONS = [
    {
        "task_id": "task_001",
        "question": "What is the capital of France?",
        "expected_answer": "Paris",
        "has_file": False,
        "file_content": None
    },
    {
        "task_id": "task_002",
        "question": "What is the square root of 144?",
        "expected_answer": "12",
        "has_file": False,
        "file_content": None
    },
    {
        "task_id": "task_003",
        "question": "If a train travels at 60 miles per hour, how far will it travel in 2.5 hours?",
        "expected_answer": "150 miles",
        "has_file": False,
        "file_content": None
    },
    {
        "task_id": "task_004", 
        "question": ".rewsna eht sa 'thgir' drow eht etirw ,tfel fo etisoppo eht si tahW",
        "expected_answer": "right",
        "has_file": False,
        "file_content": None
    },
    {
        "task_id": "task_005",
        "question": "Analyze the data in the attached CSV file and tell me the total sales for the month of January.",
        "expected_answer": "$10,250.75",
        "has_file": True,
        "file_content": """Date,Product,Quantity,Price,Total
2023-01-05,Widget A,10,25.99,259.90
2023-01-12,Widget B,5,45.50,227.50
2023-01-15,Widget C,20,50.25,1005.00
2023-01-20,Widget A,15,25.99,389.85
2023-01-25,Widget B,8,45.50,364.00
2023-01-28,Widget D,100,80.04,8004.50"""
    },
    {
        "task_id": "task_006",
        "question": "I'm making a grocery list for my mom, but she's a picky eater. She only eats foods that don't contain the letter 'e'. List 5 common fruits and vegetables she can eat.",
        "expected_answer": "Banana, Kiwi, Corn, Fig, Taro",
        "has_file": False,
        "file_content": None
    },
    {
        "task_id": "task_007",
        "question": "How many studio albums were published by Mercedes Sosa between 1972 and 1985?",
        "expected_answer": "12",
        "has_file": False,
        "file_content": None
    },
    {
        "task_id": "task_008",
        "question": "In the video https://www.youtube.com/watch?v=L1vXC1KMRd0, what color is primarily associated with the main character?",
        "expected_answer": "Blue",
        "has_file": False,
        "file_content": None
    }
]

def initialize_agent():
    """Initialize the GAIAAgent with appropriate API keys."""
    print("Initializing GAIAAgent with API keys...")
    
    # Try X.AI first (xAI) with the correct API endpoint
    if os.getenv("XAI_API_KEY"):
        print("Using X.AI API key")
        try:
            agent = GAIAAgent(
                model_type="OpenAIServerModel",
                model_id="grok-3-latest",  # Use the X.AI model
                api_key=os.getenv("XAI_API_KEY"),
                api_base="https://api.x.ai/v1",  # Correct X.AI endpoint
                temperature=0.2,
                executor_type="local",
                verbose=True,
                system_prompt_suffix=additional_system_prompt  # Add our hints
            )
            print("Using OpenAIServerModel with X.AI API")
            return agent
        except Exception as e:
            print(f"Error initializing with X.AI API: {e}")
            traceback.print_exc()
    
    # Then try OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI API key")
        try:
            model_id = os.getenv("AGENT_MODEL_ID", "gpt-4o")
            agent = GAIAAgent(
                model_type="OpenAIServerModel",
                model_id=model_id,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.2,
                executor_type="local",
                verbose=True
            )
            print(f"Using OpenAIServerModel with model_id: {model_id}")
            return agent
        except Exception as e:
            print(f"Error initializing with OpenAI API: {e}")
            traceback.print_exc()
    
    # Last resort, try Hugging Face
    if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("Using Hugging Face API token")
        try:
            # Use a smaller model that might work within free tier
            model_id = "tiiuae/falcon-7b-instruct"  # Try a smaller model that might be within free tier
            agent = GAIAAgent(
                model_type="HfApiModel",
                model_id=model_id,
                api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                temperature=0.2,
                executor_type="local",
                verbose=True
            )
            print(f"Using HfApiModel with model_id: {model_id}")
            return agent
        except Exception as e:
            print(f"Error initializing with Hugging Face API: {e}")
            traceback.print_exc()
    
    print("ERROR: No valid API keys found in environment. Please set one of the following:")
    print("- XAI_API_KEY (for X.AI)")
    print("- OPENAI_API_KEY")
    print("- HUGGINGFACEHUB_API_TOKEN")
    return None

def save_test_file(task_id: str, content: str) -> str:
    """Save a test file to a temporary location."""
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"test_file_{task_id}.csv")
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    return file_path

def run_tests():
    """Run tests using the GAIAAgent with API keys."""
    agent = initialize_agent()
    
    if not agent:
        print("Failed to initialize agent. Exiting.")
        return
    
    results = []
    correct_count = 0
    total_count = len(SAMPLE_QUESTIONS)
    
    for idx, question_data in enumerate(SAMPLE_QUESTIONS):
        task_id = question_data["task_id"]
        question = question_data["question"]
        expected = question_data["expected_answer"]
        
        print(f"\n{'='*80}")
        print(f"Question {idx+1}/{total_count}: {question}")
        print(f"Expected: {expected}")
        
        # Process any attached file
        file_path = None
        if question_data["has_file"] and question_data["file_content"]:
            file_path = save_test_file(task_id, question_data["file_content"])
            print(f"Created test file: {file_path}")
        
        # Get answer from agent
        try:
            answer = agent.answer_question(question, file_path)
            print(f"Agent answer: {answer}")
            
            # Check if answer matches expected
            is_correct = answer.lower() == expected.lower()
            if is_correct:
                correct_count += 1
                print(f"✅ CORRECT")
            else:
                print(f"❌ INCORRECT - Expected: {expected}")
            
            results.append({
                "task_id": task_id,
                "question": question,
                "expected": expected,
                "answer": answer,
                "is_correct": is_correct
            })
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error processing question: {e}\n{error_details}")
            results.append({
                "task_id": task_id,
                "question": question,
                "expected": expected,
                "answer": f"ERROR: {str(e)}",
                "is_correct": False
            })
    
    # Print summary
    accuracy = (correct_count / total_count) * 100
    print(f"\n{'='*80}")
    print(f"Test Results: {correct_count}/{total_count} correct ({accuracy:.1f}%)")
    
    return results


if __name__ == "__main__":
    print("Running tests for GAIA agent with API keys...")
    
    # Print environment information
    print("\nEnvironment information:")
    print(f"XAI_API_KEY set: {'Yes' if os.getenv('XAI_API_KEY') else 'No'}")
    print(f"OPENAI_API_KEY set: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    print(f"HUGGINGFACEHUB_API_TOKEN set: {'Yes' if os.getenv('HUGGINGFACEHUB_API_TOKEN') else 'No'}")
    print(f"AGENT_MODEL_TYPE: {os.getenv('AGENT_MODEL_TYPE', 'OpenAIServerModel')} (default: OpenAIServerModel)")
    print(f"AGENT_MODEL_ID: {os.getenv('AGENT_MODEL_ID', 'gpt-4o')} (default: gpt-4o)")
    
    results = run_tests()
    
    # Save results to a file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to test_results.json") 