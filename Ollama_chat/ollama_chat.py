import ollama
import os

class OllamaClient:
    """Ollama client wrapper for chat and prompt generation."""

    def __init__(self, host: str = None, model: str = None):
        """Initialize the Ollama client.

        Args:
            host: Ollama server host URL. Defaults to OLLAMA_HOST env var or localhost.
            model: Model name to use. Defaults to "gpt-oss:120b".
        """
        # --- Configuration ---
        # Default to local Ollama. Change via environment variable if needed.
        self.OLLAMA_HOST = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.MODEL_NAME = model or "gpt-oss:120b"  # Using the model confirmed in your list

        # Initialize the client
        self.client = ollama.Client(host=self.OLLAMA_HOST)
        models_info = self.client.list()
        self.available_models = [getattr(m, 'model', getattr(m, 'name', 'unknown')) for m in models_info.get('models', [])]
        if self.MODEL_NAME not in self.available_models:
            print(f"Warning: {self.MODEL_NAME} not found. Using {self.available_models[0]} instead.")
            self.MODEL_NAME = self.available_models[0]

    def print_available_models(self):
        """Print available models on the Ollama server."""
        print(f"Available models: {self.available_models}")

    def run_single_prompt(self, prompt: str, model: str = None):
        """Function 1: Sends a single prompt and prints the response."""
        # print(f"\n--- Single Prompt to {model} ---")
        if model is None:
            model = self.MODEL_NAME
        try:
            response = self.client.generate(model=model, prompt=prompt)
            # print(f"Response: {response['response']}")
            return response['response']
        except Exception as e:
            print(f"Error: {e}")
            return None

    def start_chat_session(self, model: str = None):
        """Function 2: Starts an interactive conversation loop."""
        if model is None:
            model = self.MODEL_NAME
        print(f"\n--- Starting Chat Session with {model} ---")
        print("Type 'exit' to quit or 'clear' to reset history.")

        messages = []

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'exit':
                print("Exiting chat...")
                break

            if user_input.lower() == 'clear':
                messages = []
                print("Chat history cleared.")
                continue

            if not user_input:
                continue

            messages.append({'role': 'user', 'content': user_input})

            print(f"Assistant: ", end="", flush=True)
            try:
                full_response = ""
                # Using stream=True for a better terminal experience
                for chunk in self.client.chat(model=model, messages=messages, stream=True):
                    content = chunk['message']['content']
                    print(content, end="", flush=True)
                    full_response += content

                print()  # New line after stream ends
                messages.append({'role': 'assistant', 'content': full_response})

            except Exception as e:
                print(f"\nError during chat: {e}")


if __name__ == "__main__":
    # Create Ollama client instance
    ollama_client = OllamaClient()

    # Check connection and models first
    print(f"Connecting to Ollama at {ollama_client.OLLAMA_HOST}...")
    try:
        ollama_client.print_available_models()

        # 1. Test single prompt
        ollama_client.run_single_prompt("Write a one-sentence greeting in chinese")

        # 2. Test interactive chat
        ollama_client.start_chat_session()

    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")