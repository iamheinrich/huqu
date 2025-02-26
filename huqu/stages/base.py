from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import yaml
from ..models.base import BaseModel


class PipelineStage(ABC):
    """Base class for pipeline stages."""
    
    def __init__(self, models: Optional[Dict[str, BaseModel]] = None):
        """Initialize stage with required models."""
        self.models = models
        
        # Load config from project root
        config_path = Path(__file__).parent.parent.parent / "pipeline_config.yaml"
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """Process input data and return output."""
        pass
