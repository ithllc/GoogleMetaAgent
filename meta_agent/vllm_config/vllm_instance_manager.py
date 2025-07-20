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
from .vllm_configuration import VLLM_CONFIG, vllm_logger

class VLLMInstanceManager:
    """Manages VLLM instances for multiple GPUs with load balancing and request queuing."""
    
    def __init__(
        self,
        model_paths: Dict[str, str],
        base_port: int = 8000,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.95,
        max_queue_length: int = 100,
        health_check_interval: int = 10,
        server_startup_timeout: int = 600,
        debug_mode: bool = False
    ):
        """
        Initialize the VLLM instance manager.
        
        Args:
            model_paths: Dictionary mapping model names to their local file paths
            base_port: Starting port number for VLLM instances
            max_model_len: Maximum sequence length for models
            gpu_memory_utilization: GPU memory utilization (0-1)
            max_queue_length: Maximum number of requests in the queue
            health_check_interval: Interval (seconds) for health checks
            server_startup_timeout: Timeout (seconds) for server startup
            debug_mode: Enable more verbose logging for troubleshooting
        """
        self.model_paths = model_paths
        self.base_port = base_port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.server_startup_timeout = server_startup_timeout
        self.debug_mode = debug_mode
        
        # Verify basic environment and dependencies
        if debug_mode:
            self._check_environment()
            self._verify_ports_available()
            self._verify_model_paths()
        
        # Auto-detect max_model_len from config if not provided
        if max_model_len is None:
            detected_max_len = self._detect_model_max_length()
            if detected_max_len:
                vllm_logger.info(f"Auto-detected max_model_len: {detected_max_len} from model config")
                self.max_model_len = detected_max_len
            else:
                vllm_logger.warning("Could not auto-detect max_model_len, using default of 8192")
                self.max_model_len = 8192
        else:
            self.max_model_len = max_model_len
        
        # Detect GPUs
        self.gpu_ids = self._detect_gpus()
        self.num_gpus = len(self.gpu_ids)
        vllm_logger.info(f"Detected {self.num_gpus} GPUs: {self.gpu_ids}")
        
        # Instance tracking
        self.instances = []
        self.instance_health = {}
        self.instance_load = {}  # Track load per instance
        
        # Initialize request queue
        self.request_queue = queue.Queue(maxsize=max_queue_length)
        self.queue_lock = threading.Lock()
        
        # Health monitoring
        self.health_check_interval = health_check_interval
        self.stop_health_monitor = threading.Event()
        
        # Queue processing
        self.stop_queue_processor = threading.Event()

    def detect_gpus(self):
        """Legacy method for backward compatibility."""
        return len(self._detect_gpus())

    def configure_vllm_server(self):
        """Legacy method for backward compatibility."""
        vllm_logger.info(f"Configuring VLLM server with config: {VLLM_CONFIG}")
        return self.start_instances()
    
    def _check_environment(self) -> None:
        """Verify that the necessary dependencies are available."""
        vllm_logger.info("Checking environment and dependencies...")
        
        # Check Python version
        python_version = platform.python_version()
        vllm_logger.info(f"Python version: {python_version}")
        
        # Try to import vllm
        try:
            import vllm
            vllm_logger.info(f"vLLM version: {vllm.__version__}")
        except ImportError as e:
            vllm_logger.error(f"Failed to import vLLM: {e}")
        except AttributeError:
            vllm_logger.info("vLLM is installed, but version info not available")
        
        # Check for CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            vllm_logger.info(f"CUDA available: {cuda_available}")
            if cuda_available:
                cuda_version = torch.version.cuda
                vllm_logger.info(f"CUDA version: {cuda_version}")
                vllm_logger.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
        except ImportError as e:
            vllm_logger.error(f"Failed to import torch for CUDA check: {e}")
        
        # Run nvidia-smi for more information if available
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                check=True
            )
            vllm_logger.info("nvidia-smi output:\n" + result.stdout)
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            vllm_logger.warning(f"Failed to run nvidia-smi: {e}")
    
    def _verify_ports_available(self) -> None:
        """Check if the base port and subsequent ports are available."""
        vllm_logger.info(f"Checking if base port {self.base_port} and subsequent ports are available...")
        
        # Check base port
        if not self._is_port_available(self.base_port):
            vllm_logger.warning(f"Base port {self.base_port} is already in use!")
        
        # Check next few ports (assuming we might use a few based on number of models)
        for i in range(1, len(self.model_paths) * 2):
            port = self.base_port + i
            if not self._is_port_available(port):
                vllm_logger.warning(f"Port {port} is already in use!")
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result != 0
        except:
            return False
    
    def _verify_model_paths(self) -> None:
        """Verify that model paths exist and are accessible."""
        vllm_logger.info("Verifying model paths...")
        
        for model_name, model_path in self.model_paths.items():
            if not os.path.exists(model_path):
                vllm_logger.error(f"Model path does not exist: {model_path}")
                continue
            
            # Check if it's a directory
            if os.path.isdir(model_path):
                vllm_logger.info(f"Model {model_name} path is a directory: {model_path}")
                # Look for config.json
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    vllm_logger.info(f"Found config.json for model {model_name}")
                else:
                    vllm_logger.warning(f"Could not find config.json for model {model_name}")
            # Check if it's a safetensors file
            elif model_path.endswith('.safetensors'):
                vllm_logger.info(f"Model {model_name} path is a safetensors file: {model_path}")
                # Look for config.json in same directory
                model_dir = os.path.dirname(model_path)
                config_path = os.path.join(model_dir, "config.json")
                if os.path.exists(config_path):
                    vllm_logger.info(f"Found config.json for model {model_name}")
                else:
                    vllm_logger.warning(f"Could not find config.json for model {model_name}")
            else:
                vllm_logger.info(f"Model {model_name} path: {model_path}")
    
    def _detect_model_max_length(self) -> Optional[int]:
        """
        Detect the maximum context length from the model's config.json file.
        
        Returns:
            The detected maximum context length, or None if it could not be detected.
        """
        # Use the first model to detect max context length
        if not self.model_paths:
            return None
        
        first_model_path = next(iter(self.model_paths.values()))
        config_path = os.path.join(first_model_path, "config.json")
        
        # If the model path points to .safetensors files, look in the same directory
        if not os.path.exists(config_path) and first_model_path.endswith('.safetensors'):
            model_dir = os.path.dirname(first_model_path)
            config_path = os.path.join(model_dir, "config.json")
            
            # If still not found, try to find any config.json in the directory
            if not os.path.exists(config_path):
                config_files = glob.glob(os.path.join(model_dir, "*.json"))
                if config_files:
                    config_path = config_files[0]
        
        if not os.path.exists(config_path):
            vllm_logger.warning(f"Could not find config.json for model at {first_model_path}")
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if self.debug_mode:
                vllm_logger.info(f"Model config keys: {', '.join(config.keys())}")
                
            # Check various fields that might indicate max context length
            context_length = None
            
            if 'max_position_embeddings' in config:
                context_length = config['max_position_embeddings']
                if self.debug_mode:
                    vllm_logger.info(f"Found max_position_embeddings: {context_length}")
            elif 'n_positions' in config:
                context_length = config['n_positions']
                if self.debug_mode:
                    vllm_logger.info(f"Found n_positions: {context_length}")
            elif 'context_length' in config:
                context_length = config['context_length']
                if self.debug_mode:
                    vllm_logger.info(f"Found context_length: {context_length}")
            
            return context_length
            
        except Exception as e:
            vllm_logger.error(f"Error reading model config: {e}")
            return None
    
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs using nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            gpu_ids = [int(gpu_id.strip()) for gpu_id in result.stdout.splitlines()]
            return gpu_ids
        except (subprocess.SubprocessError, FileNotFoundError):
            vllm_logger.warning("Failed to detect GPUs using nvidia-smi. Assuming single GPU available.")
            return [0]  # Assume at least one GPU for Cloud Run
    
    def start_instances(self) -> bool:
        """Configure and start all VLLM instances."""
        instance_configs = self._configure_instances()
        if not instance_configs:
            return False
            
        vllm_logger.info(f"Starting {len(instance_configs)} VLLM instances...")
        
        for config in instance_configs:
            process = self._start_instance(
                gpu_ids=config["gpu_ids"],
                model_name=config["model_name"],
                model_path=config["model_path"],
                port=config["port"],
                tensor_parallel_size=config["tensor_parallel_size"]
            )
            
            instance_info = {
                "process": process,
                "config": config,
                "url": f"http://localhost:{config['port']}",
                "healthy": False
            }
            
            self.instances.append(instance_info)
            self.instance_load[config["port"]] = 0
            
        # Wait for servers to be ready
        ready_count = 0
        for instance in self.instances:
            url = instance["url"]
            if self._check_server_ready(url, self.server_startup_timeout):
                instance["healthy"] = True
                ready_count += 1
                
        vllm_logger.info(f"{ready_count}/{len(self.instances)} instances ready")
        return ready_count > 0
    
    def _configure_instances(self) -> List[Dict[str, Any]]:
        """Configure VLLM instances based on available GPUs."""
        if self.num_gpus == 0:
            vllm_logger.error("No GPUs detected, cannot configure VLLM instances")
            return []
            
        instances = []
        
        # For Cloud Run, we typically have 1 GPU, so use all for a single instance per model
        for idx, (model_name, model_path) in enumerate(self.model_paths.items()):
            instances.append({
                "model_name": model_name,
                "model_path": model_path,
                "gpu_ids": ",".join(map(str, self.gpu_ids)),
                "port": self.base_port + idx,
                "tensor_parallel_size": len(self.gpu_ids)
            })
        
        return instances
    
    def _start_instance(
        self,
        gpu_ids: str,
        model_name: str,
        model_path: str,
        port: int,
        tensor_parallel_size: int
    ) -> subprocess.Popen:
        """Start a VLLM instance with the specified configuration."""
        command = f"""
        CUDA_VISIBLE_DEVICES={gpu_ids} python -m vllm.entrypoints.openai.api_server \
        --served-model-name {model_name} \
        --model {model_path} \
        --port {port} \
        --tensor-parallel-size {tensor_parallel_size} \
        --max-model-len {self.max_model_len} \
        --gpu-memory-utilization {self.gpu_memory_utilization}
        """
        
        vllm_logger.info(f"Starting VLLM instance: {model_name} on GPUs {gpu_ids}, port {port}")
        
        if self.debug_mode:
            vllm_logger.info(f"Launch command: {command}")
        
        # Start the process without blocking
        process = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return process
    
    def _check_server_ready(self, api_url: str, timeout: int = 600) -> bool:
        """Check if a VLLM server is ready to accept requests."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(api_url + "/health", timeout=5)
                if response.status_code == 200:
                    vllm_logger.info(f"Server is ready: {api_url}")
                    return True
            except requests.exceptions.RequestException as e:
                if self.debug_mode:
                    vllm_logger.info(f"Health check error for {api_url}: {str(e)}")
                pass
            
            vllm_logger.info(f"Waiting for {api_url} to be ready... ({int(time.time() - start_time)}s/{timeout}s)")
            time.sleep(5)
        
        vllm_logger.error(f"Timeout: Server {api_url} did not start in time.")
        return False


class VLLMLoadBalancer:
    """Provides an API that load balances requests to VLLM instances."""
    
    def __init__(self, instance_manager: VLLMInstanceManager):
        """
        Initialize the load balancer.
        
        Args:
            instance_manager: VLLMInstanceManager instance
        """
        self.instance_manager = instance_manager
