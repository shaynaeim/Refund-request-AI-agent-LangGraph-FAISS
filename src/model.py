"""OpenAI model integration for the refund processing system."""

import os
import yaml
from typing import Dict, Any
from langchain.chat_models import init_chat_model


class OpenAIModel:
    """Handles OpenAI model initialization and configuration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration file."""
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            # Create default config if it doesn't exist
            self._create_default_config()
            
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        return config
    
    def _create_default_config(self):
        """Create a default configuration file."""
        default_config = {
            'openai': {
                'model_name': 'gpt-4o-mini',
                'temperature': 0.1,
                'api_key': 'your_openai_api_key_here'
            }
        }
        
        with open(self.config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False, indent=2)
            
        print(f"Default configuration created at: {self.config_path}")
        print("Please update the OpenAI API key in the config file.")
    
    def get_model(self):
        """Get the configured OpenAI model."""
        credentials = self.config.get('credentials', {})
        openai_config = credentials.get('openai', {})
        model_name = openai_config.get('model_name', 'gpt-4o-mini')
        temperature = openai_config.get('temperature', 0.1)
        api_key = openai_config.get('api_key')
        
        # Validate API key
        if not api_key or api_key == 'your_openai_api_key_here':
            raise ValueError(
                "OpenAI API key not configured. Please update the 'api_key' in config.yaml"
            )
            
        # Set environment variable
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Map model names to OpenAI format
        openai_models = {
            'gpt-4o': 'openai:gpt-4o',
            'gpt-4o-mini': 'openai:gpt-4o-mini',
            'gpt-4-turbo': 'openai:gpt-4-turbo',
            'gpt-3.5-turbo': 'openai:gpt-3.5-turbo'
        }
        
        model_id = openai_models.get(model_name, f"openai:{model_name}")
        
        return init_chat_model(
            model=model_id,
            temperature=temperature
        )
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate the OpenAI configuration."""
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if openai config exists
        if 'openai' not in self.config:
            results['valid'] = False
            results['errors'].append("Missing 'openai' section in config")
            return results
            
        openai_config = self.config['openai']
        model_name = openai_config.get('model_name')
        api_key = openai_config.get('api_key')
        
        # Validate API key
        if not api_key or api_key == 'your_openai_api_key_here':
            results['valid'] = False
            results['errors'].append("OpenAI API key not configured")
        
        # Validate model name
        available_models = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo']
        if model_name not in available_models:
            results['warnings'].append(f"Model '{model_name}' not in known models: {available_models}")
        
        # Validate temperature
        temperature = openai_config.get('temperature', 0.1)
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            results['warnings'].append("Temperature should be a number between 0 and 2")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured model."""
        openai_config = self.config.get('openai', {})
        return {
            'provider': 'OpenAI',
            'model_name': openai_config.get('model_name', 'gpt-4o-mini'),
            'temperature': openai_config.get('temperature', 0.1),
            'configured': bool(openai_config.get('api_key') and 
                             openai_config.get('api_key') != 'your_openai_api_key_here')
        }


def get_openai_model(config_path: str = "config.yaml"):
    """Convenience function to get configured OpenAI model."""
    model_handler = OpenAIModel(config_path)
    return model_handler.get_model()
