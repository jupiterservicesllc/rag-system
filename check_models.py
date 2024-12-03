from gpt4all import GPT4All

def list_available_models():
    """List all available models from GPT4All"""
    try:
        # Get model list
        models = GPT4All.list_models()
        
        print("\nAvailable GPT4All Models:")
        print("-" * 50)
        
        for i, model in enumerate(models, 1):
            name = model.get('filename', 'Unknown')
            size = model.get('filesize', 'Unknown size')
            print(f"{i}. {name}")
            print(f"   Size: {size} bytes")
            print(f"   URL: {model.get('url', 'No URL')}")
            print()
            
    except Exception as e:
        print(f"Error fetching models: {e}")

if __name__ == "__main__":
    list_available_models()