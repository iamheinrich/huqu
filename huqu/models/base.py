from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional

class BaseModel(ABC):
    """Base class for all models in the pipeline."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt to generate a response for
            
        Returns:
            The generated response as a string
        """
        pass

    @abstractmethod
    def setup(self) -> None:
        """Initialize the model and its dependencies."""
        pass