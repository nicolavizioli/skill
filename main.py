import os
import tempfile
import gradio as gr
import pandas as pd
import traceback
from core_agent import GAIAAgent
from api_integration import GAIAApiClient

# Constants
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

def save_task_file(file_content, task_id):
    """
    Save a task file to a temporary location
    """
    if not file_content:
        return None
    
    # Create a temporary file
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"gaia_task_{task_id}.txt")
    
    # Write content to the file
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    print(f"File saved to {file_path}")
    return file_path

def get_agent_configuration():
    """
    Get the agent configuration based on environment variables
    """
    # Default configuration
    config = {
        "model_type": "OpenAIServerModel",  # Default to OpenAIServerModel
        "model_id": "gpt-4o",  # Default model for OpenAI
        "temperature": 0.2,
        "executor_type": "local",
        "verbose": False,
        "provider": "hf-inference",  # For InferenceClientModel
        "timeout": 120        # For InferenceClientModel
    }
    
    # Check for xAI API key and base URL
    xai_api_key = os.getenv("XAI_API_KEY")
    xai_api_base = os.getenv("XAI_API_BASE")
    
    # If we have xAI credentials, use them
    if xai_api_key:
        config["api_key"] = xai_api_key
        if xai_api_base:
            config["api_base"] = xai_api_base
            # Use a model that works well with xAI
            config["model_id"] = "mixtral-8x7b-32768"
    
    # Override with environment variables if present
    if os.getenv("AGENT_MODEL_TYPE"):
        config["model_type"] = os.getenv("AGENT_MODEL_TYPE")
    
    if os.getenv("AGENT_MODEL_ID"):
        config["model_id"] = os.getenv("AGENT_MODEL_ID")
    
    if os.getenv("AGENT_TEMPERATURE"):
        config["temperature"] = float(os.getenv("AGENT_TEMPERATURE"))
    
    if os.getenv("AGENT_EXECUTOR_TYPE"):
        config["executor_type"] = os.getenv("AGENT_EXECUTOR_TYPE")
    
    if os.getenv("AGENT_VERBOSE") is not None:
        config["verbose"] = os.getenv("AGENT_VERBOSE").lower() == "true"
    
    if os.getenv("AGENT_API_BASE"):
        config["api_base"] = os.getenv("AGENT_API_BASE")
    
    # InferenceClientModel specific settings
    if os.getenv("AGENT_PROVIDER"):
        config["provider"] = os.getenv("AGENT_PROVIDER")
    
    if os.getenv("AGENT_TIMEOUT"):
        config["timeout"] = int(os.getenv("AGENT_TIMEOUT"))
    
    return config

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the GAIAAgent on them, submits all answers,
    and displays the results.
    """
    # Check for user login
    if not profile:
        return "Please Login to Hugging Face with the button.", None
    
    username = profile.username
    print(f"User logged in: {username}")
    
    # Get SPACE_ID for code link
    space_id = os.getenv("SPACE_ID")
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    
    # Initialize API client
    api_client = GAIAApiClient(DEFAULT_API_URL)
    
    # Initialize Agent with configuration
    try:
        agent_config = get_agent_configuration()
        print(f"Using agent configuration: {agent_config}")
        
        agent = GAIAAgent(**agent_config)
        print("Agent initialized successfully")
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error initializing agent: {e}\n{error_details}")
        return f"Error initializing agent: {e}", None
    
    # Fetch questions
    try:
        questions_data = api_client.get_questions()
        if not questions_data:
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error fetching questions: {e}\n{error_details}")
        return f"Error fetching questions: {e}", None
    
    # Run agent on questions
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    
    # Progress tracking
    total_questions = len(questions_data)
    completed = 0
    failed = 0
    
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        
        try:
            # Update progress
            completed += 1
            print(f"Processing question {completed}/{total_questions}: Task ID {task_id}")
            
            # Check if the question has an associated file
            file_path = None
            try:
                file_content = api_client.get_file(task_id)
                print(f"Downloaded file for task {task_id}")
                file_path = save_task_file(file_content, task_id)
            except Exception as file_e:
                print(f"No file found for task {task_id} or error: {file_e}")
            
            # Run the agent to get the answer
            submitted_answer = agent.answer_question(question_text, file_path)
            
            # Add to results
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": submitted_answer
            })
        except Exception as e:
            # Update error count
            failed += 1
            error_details = traceback.format_exc()
            print(f"Error running agent on task {task_id}: {e}\n{error_details}")
            
            # Add error to results
            error_msg = f"AGENT ERROR: {e}"
            answers_payload.append({"task_id": task_id, "submitted_answer": error_msg})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": error_msg
            })
    
    # Print summary
    print(f"\nProcessing complete: {completed} questions processed, {failed} failures")
    
    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)
    
    # Submit answers
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload
    }
    
    print(f"Submitting {len(answers_payload)} answers for username '{username}'...")
    
    try:
        result_data = api_client.submit_answers(
            username.strip(),
            agent_code,
            answers_payload
        )
        
        # Calculate success rate
        correct_count = result_data.get('correct_count', 0)
        total_attempted = result_data.get('total_attempted', len(answers_payload))
        success_rate = (correct_count / total_attempted) * 100 if total_attempted > 0 else 0
        
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({correct_count}/{total_attempted} correct, {success_rate:.1f}% success rate)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        
        print("Submission successful.")
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        error_details = traceback.format_exc()
        status_message = f"Submission Failed: {e}\n{error_details}"
        print(status_message)
        return status_message, pd.DataFrame(results_log)

# Build Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        
        1. Log in to your Hugging Face account using the button below.
        2. Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.
        
        **Configuration:**
        
        You can configure the agent by setting these environment variables:
        - `AGENT_MODEL_TYPE`: Model type (HfApiModel, InferenceClientModel, LiteLLMModel, OpenAIServerModel)
        - `AGENT_MODEL_ID`: Model ID
        - `AGENT_TEMPERATURE`: Temperature for generation (0.0-1.0)
        - `AGENT_EXECUTOR_TYPE`: Type of executor ('local' or 'e2b')
        - `AGENT_VERBOSE`: Enable verbose logging (true/false)
        - `AGENT_API_BASE`: Base URL for API calls (for OpenAIServerModel)
        
        **xAI Support:**
        - `XAI_API_KEY`: Your xAI API key
        - `XAI_API_BASE`: Base URL for xAI API (default: https://api.groq.com/openai/v1)
        - When using xAI, set AGENT_MODEL_TYPE=OpenAIServerModel and AGENT_MODEL_ID=mixtral-8x7b-32768
        
        **InferenceClientModel specific settings:**
        - `AGENT_PROVIDER`: Provider for InferenceClientModel (e.g., "hf-inference") 
        - `AGENT_TIMEOUT`: Timeout in seconds for API calls
        """
    )
    
    gr.LoginButton()
    
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
    
    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    
    # Check for environment variables
    config = get_agent_configuration()
    print(f"Agent configuration: {config}")
    
    # Run the Gradio app
    demo.launch(debug=True, share=False)