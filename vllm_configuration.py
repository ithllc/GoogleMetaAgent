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
            if (detected_max_len):
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
            vllm_logger.warning("Failed to detect GPUs using nvidia-smi. Assuming no GPUs available.")
            return []
    
    def _configure_instances(self) -> List[Dict[str, Any]]:
        """Configure VLLM instances based on available GPUs."""
        if self.num_gpus == 0:
            vllm_logger.error("No GPUs detected, cannot configure VLLM instances")
            return []
            
        instances = []
        
        # If we have 1-2 GPUs, use all for a single instance per model
        if self.num_gpus <= 2:
            for idx, (model_name, model_path) in enumerate(self.model_paths.items()):
                instances.append({
                    "model_name": model_name,
                    "model_path": model_path,
                    "gpu_ids": ",".join(map(str, self.gpu_ids)),
                    "port": self.base_port + idx,
                    "tensor_parallel_size": len(self.gpu_ids)
                })
        # If we have more GPUs, distribute them among instances
        else:
            # Calculate GPUs per instance (use 2 GPUs per instance if possible)
            gpus_per_instance = min(2, max(1, self.num_gpus // len(self.model_paths)))
            
            gpu_idx = 0
            for idx, (model_name, model_path) in enumerate(self.model_paths.items()):
                # For each model, create multiple instances if there are enough GPUs
                remaining_gpus = self.num_gpus - gpu_idx
                instances_for_model = max(1, remaining_gpus // gpus_per_instance)
                
                for i in range(instances_for_model):
                    if gpu_idx + gpus_per_instance <= self.num_gpus:
                        instance_gpu_ids = self.gpu_ids[gpu_idx:gpu_idx+gpus_per_instance]
                        gpu_idx += gpus_per_instance
                        
                        instances.append({
                            "model_name": model_name,
                            "model_path": model_path,
                            "gpu_ids": ",".join(map(str, instance_gpu_ids)),
                            "port": self.base_port + len(instances),
                            "tensor_parallel_size": len(instance_gpu_ids)
                        })
        
        return instances
    
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
            
        # Start health monitoring
        threading.Thread(
            target=self._health_monitor_thread,
            daemon=True
        ).start()
        
        # Start queue processor
        threading.Thread(
            target=self._queue_processor_thread,
            daemon=True
        ).start()
        
        # Wait for servers to be ready
        ready_count = 0
        for instance in self.instances:
            url = instance["url"]
            if self._check_server_ready(url, self.server_startup_timeout):
                instance["healthy"] = True
                ready_count += 1
                
        vllm_logger.info(f"{ready_count}/{len(self.instances)} instances ready")
        return ready_count > 0
    
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
        
        # Start threads to read stdout and stderr
        def log_output(stream, prefix):
            for line in iter(stream.readline, ''):
                vllm_logger.info(f"{prefix}: {line.strip()}")
                
        if self.debug_mode:
            threading.Thread(
                target=log_output,
                args=(process.stdout, f"VLLM-{port}-OUT"),
                daemon=True
            ).start()
            
            threading.Thread(
                target=log_output,
                args=(process.stderr, f"VLLM-{port}-ERR"),
                daemon=True
            ).start()
        
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
            
            # If the process is no longer running, check its output
            for instance in self.instances:
                if instance["url"] == api_url and instance["process"].poll() is not None:
                    vllm_logger.error(f"Server process for {api_url} has exited prematurely with code {instance['process'].returncode}")
                    stdout, stderr = instance["process"].communicate()
                    vllm_logger.error(f"STDOUT: {stdout}")
                    vllm_logger.error(f"STDERR: {stderr}")
                    return False
        
        vllm_logger.error(f"Timeout: Server {api_url} did not start in time.")
        
        # If debug mode is enabled, try to capture any output from the process
        for instance in self.instances:
            if instance["url"] == api_url:
                if instance["process"].poll() is None:
                    vllm_logger.warning(f"Process for {api_url} is still running but not responding to health checks")
                else:
                    vllm_logger.error(f"Process for {api_url} exited with code {instance["process"].returncode}")
                
                # Try to capture any output
                stdout, stderr = instance["process"].communicate(timeout=5)
                vllm_logger.error(f"STDOUT: {stdout}")
                vllm_logger.error(f"STDERR: {stderr}")
                break
                
        return False
    
    def _health_monitor_thread(self):
        """Monitor the health of all VLLM instances."""
        while not self.stop_health_monitor.is_set():
            for instance in self.instances:
                url = instance["url"]
                try:
                    response = requests.get(url + "/health", timeout=2)
                    healthy = response.status_code == 200
                except:
                    healthy = False
                
                # Update health status
                instance["healthy"] = healthy
            
            time.sleep(self.health_check_interval)
    
    def _queue_processor_thread(self):
        """Process requests from the queue and forward them to VLLM instances."""
        while not self.stop_queue_processor.is_set():
            try:
                # Get a request from the queue
                request_data = self.request_queue.get(timeout=1)
                if request_data is None:
                    continue
                    
                endpoint, json_data, callback = request_data
                
                # Select a healthy instance using load balancing
                instance = self._select_instance(json_data.get("model"))
                
                if instance is None:
                    # No healthy instances available
                    callback({"error": "No healthy VLLM instances available"}, 503)
                    continue
                
                # Update load counter
                port = instance["config"]["port"]
                with self.queue_lock:
                    self.instance_load[port] += 1
                
                try:
                    # Forward the request to the VLLM instance
                    url = f"{instance['url']}/{endpoint.lstrip('/')}"
                    response = requests.post(url, json=json_data, timeout=60)
                    callback(response.json(), response.status_code)
                except Exception as e:
                    vllm_logger.error(f"Error forwarding request: {str(e)}")
                    callback({"error": str(e)}, 500)
                finally:
                    # Update load counter
                    with self.queue_lock:
                        self.instance_load[port] = max(0, self.instance_load[port] - 1)
                    
                    # Mark task as done
                    self.request_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                vllm_logger.error(f"Error in queue processor: {str(e)}")
    
    def _select_instance(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Select a healthy instance based on load balancing."""
        healthy_instances = [
            instance for instance in self.instances
            if instance["healthy"] and (model_name is None or instance["config"]["model_name"] == model_name)
        ]
        
        if not healthy_instances:
            return None
            
        # Find instances with lowest load
        with self.queue_lock:
            load_values = [self.instance_load[instance["config"]["port"]] for instance in healthy_instances]
            min_load = min(load_values)
            least_loaded_instances = [
                instance for idx, instance in enumerate(healthy_instances)
                if load_values[idx] == min_load
            ]
        
        # Randomly select from least loaded instances
        return random.choice(least_loaded_instances)
    
    def enqueue_request(
        self,
        endpoint: str,
        json_data: Dict[str, Any],
        callback: callable
    ) -> bool:
        """
        Add a request to the queue.
        
        Args:
            endpoint: API endpoint (e.g., 'v1/completions')
            json_data: Request JSON payload
            callback: Function to call with the response
            
        Returns:
            True if the request was queued, False if the queue is full
        """
        try:
            # Immediately handle if we have less items than instances
            if self.request_queue.qsize() < len(self.instances):
                instance = self._select_instance(json_data.get("model"))
                if instance:
                    # Attempt to bypass queue for faster service
                    port = instance["config"]["port"]
                    with self.queue_lock:
                        self.instance_load[port] += 1
                        
                    try:
                        url = f"{instance['url']}/{endpoint.lstrip('/')}"
                        response = requests.post(url, json=json_data, timeout=60)
                        callback(response.json(), response.status_code)
                        return True
                    except Exception as e:
                        vllm_logger.error(f"Error in direct request: {str(e)}")
                    finally:
                        with self.queue_lock:
                            self.instance_load[port] = max(0, self.instance_load[port] - 1)
            
            # Add to queue as fallback
            self.request_queue.put_nowait((endpoint, json_data, callback))
            return True
        except queue.Full:
            vllm_logger.error("Request queue is full, rejecting request")
            return False
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get information about the queue and instance status."""
        return {
            "queue_size": self.request_queue.qsize(),
            "queue_max_size": self.request_queue.maxsize,
            "instances": [{
                "model": instance["config"]["model_name"],
                "url": instance["url"],
                "gpus": instance["config"]["gpu_ids"],
                "healthy": instance["healthy"],
                "load": self.instance_load[instance["config"]["port"]]
            } for instance in self.instances]
        }
    
    def shutdown(self):
        """Shut down all VLLM instances and threads."""
        vllm_logger.info("Shutting down VLLM instance manager...")
        
        # Stop worker threads
        self.stop_health_monitor.set()
        self.stop_queue_processor.set()
        
        # Terminate all processes
        for instance in self.instances:
            process = instance["process"]
            if process:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
        vllm_logger.info("VLLM instance manager shutdown complete")
    
    def dump_debug_info(self) -> Dict[str, Any]:
        """Collect and return debug information."""
        debug_info = {
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            "configuration": {
                "model_paths": self.model_paths,
                "base_port": self.base_port,
                "max_model_len": self.max_model_len,
                "gpu_ids": self.gpu_ids,
                "num_gpus": self.num_gpus,
            },
            "instances": []
        }
        
        # Add info about each instance
        for instance in self.instances:
            instance_info = {
                "model_name": instance["config"]["model_name"],
                "url": instance["url"],
                "gpus": instance["config"]["gpu_ids"],
                "port": instance["config"]["port"],
                "healthy": instance["healthy"],
                "process_running": instance["process"].poll() is None,
            }
            
            if instance["process"].poll() is not None:
                instance_info["exit_code"] = instance["process"].returncode
                
            debug_info["instances"].append(instance_info)
        
        # Try to get port status
        for port in range(self.base_port, self.base_port + len(self.instances) + 1):
            debug_info.setdefault("ports", {})[port] = self._is_port_available(port)
        
        return debug_info


class VLLMLoadBalancer:
    """Provides an API that load balances requests to VLLM instances."""
    
    def __init__(self, instance_manager: VLLMInstanceManager):
        """
        Initialize the load balancer.
        
        Args:
            instance_manager: VLLMInstanceManager instance
        """
        self.instance_manager = instance_manager
    
    def completions(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a completions request."""
        # Ensure tool_choice has valid value or set to "none"
        if "tool_choice" in json_data and json_data["tool_choice"] not in ["auto", "none"] and not isinstance(json_data["tool_choice"], dict):
            json_data["tool_choice"] = "none"
                    
        result = {}
        completion_event = threading.Event()
        
        def callback(response, status_code):
            nonlocal result
            result = response
            completion_event.set()
        
        enqueued = self.instance_manager.enqueue_request(
            "v1/completions",
            json_data,
            callback
        )
        
        if not enqueued:
            return {"error": "Server is at capacity, request rejected"}, 429
        
        # Wait for the request to complete
        completion_event.wait()
        return result
    
    def chat_completions(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a chat completions request."""
        # Ensure tool_choice has valid value or set to "none"
        if "tool_choice" in json_data and json_data["tool_choice"] not in ["auto", "none"] and not isinstance(json_data["tool_choice"], dict):
            json_data["tool_choice"] = "none"
            
        result = {}
        completion_event = threading.Event()
        
        def callback(response, status_code):
            nonlocal result
            result = response
            completion_event.set()
        
        enqueued = self.instance_manager.enqueue_request(
            "v1/chat/completions",
            json_data,
            callback
        )
        
        if not enqueued:
            return {"error": "Server is at capacity, request rejected"}, 429
        
        # Wait for the request to complete
        completion_event.wait()
        return result
    
    def embeddings(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an embeddings request."""
        result = {}
        completion_event = threading.Event()
        
        def callback(response, status_code):
            nonlocal result
            result = response
            completion_event.set()
        
        enqueued = self.instance_manager.enqueue_request(
            "v1/embeddings",
            json_data,
            callback
        )
        
        if not enqueued:
            return {"error": "Server is at capacity, request rejected"}, 429
        
        # Wait for the request to complete
        completion_event.wait()
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status information."""
        return self.instance_manager.get_queue_info()

# Example usage
def setup_vllm_with_load_balancing(model_paths, base_port=8000, max_model_len=None, debug_mode=False, timeout=600):
    """
    Set up VLLM with load balancing and queuing.
    
    Args:
        model_paths: Dictionary mapping model names to paths
        base_port: Starting port for VLLM instances
        max_model_len: Maximum sequence length for models, auto-detected if None
        debug_mode: Enable more verbose logging for troubleshooting
        timeout: Server startup timeout in seconds
    
    Returns:
        VLLMLoadBalancer instance
    """
    # Create instance manager
    instance_manager = VLLMInstanceManager(
        model_paths=model_paths,
        base_port=base_port,
        max_model_len=max_model_len,  # Pass None to enable auto-detection
        gpu_memory_utilization=0.95,
        max_queue_length=100,
        debug_mode=debug_mode,
        server_startup_timeout=timeout
    )
    
    # Start instances
    success = instance_manager.start_instances()
    if not success:
        vllm_logger.error("Failed to start VLLM instances")
        # Dump debug info if in debug mode
        if debug_mode:
            debug_info = instance_manager.dump_debug_info()
            vllm_logger.error(f"Debug information: {json.dumps(debug_info, indent=2)}")
        return None
    
    # Create load balancer
    load_balancer = VLLMLoadBalancer(instance_manager)
    return load_balancer


if __name__ == "__main__":
    # Example usage with added debug mode
    model_paths = {
        "phi4-mini-instruct": "/path/to/dspy_model",
        "phi4-mini-instruct-2": "/path/to/pydanticai_model"
    }
    
    # Enable debug mode and increase timeout for troubleshooting
    load_balancer = setup_vllm_with_load_balancing(
        model_paths, 
        debug_mode=True,
        timeout=1200  # Increased timeout to 20 minutes
    )
    
    if load_balancer:
        print("VLLM load balancer started successfully")
        
        # Example completion request
        completion = load_balancer.completions({
            "model": "phi4-mini-instruct",
            "prompt": "Hello, world!",
            "max_tokens": 100
        })
        
        print(json.dumps(completion, indent=2))
        
        # Print status
        print(json.dumps(load_balancer.get_status(), indent=2))
        
        # Keep running (in a real app, you'd have a web server here)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
            load_balancer.instance_manager.shutdown()
    else:
        print("Failed to start VLLM load balancer")
