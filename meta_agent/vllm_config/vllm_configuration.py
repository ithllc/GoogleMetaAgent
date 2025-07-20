import os
import time
import json
import random
import subprocess
import threading
import queue
import logging
import requests
import socket
import sys
import platform
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
vllm_logger = logging.getLogger('vllm_config')

VLLM_CONFIG = {
    "model": "gemma-3",
    "gpu_memory_utilization": 0.95,
    "max_model_len": 8192,
    "base_port": 8000,
}
