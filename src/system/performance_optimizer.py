"""
System performance optimizer for the quantitative trading system.
Optimizes system performance and resource usage.
"""

import os
import sys
import time
import psutil
import gc
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.config.settings import config
from src.utils.logging_utils import get_logger
from src.monitoring.system_health import SystemHealthMonitor


class PerformanceOptimizer:
    """
    Optimizes system performance and resource usage.
    Implements memory management, parallel processing, and resource monitoring.
    """
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.logger = get_logger("PerformanceOptimizer")
        self.health_monitor = SystemHealthMonitor()
        self.optimization_active = False
        self.optimization_thread = None
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage by cleaning up unused resources.
        
        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting memory usage optimization")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        
        # Collect garbage
        gc.collect()
        
        # Force Python to release memory to OS if possible
        if hasattr(gc, 'collect'):
            gc.collect()
        
        end_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        end_time = time.time()
        
        memory_saved = start_memory - end_memory
        
        results = {
            'start_memory_mb': start_memory,
            'end_memory_mb': end_memory,
            'memory_saved_mb': memory_saved,
            'duration_seconds': end_time - start_time
        }
        
        self.logger.info(f"Memory optimization completed: {memory_saved:.2f} MB saved")
        
        return results
    
    def optimize_disk_usage(self, cleanup_old_logs: bool = True, 
                           cleanup_temp_files: bool = True) -> Dict[str, Any]:
        """
        Optimize disk usage by cleaning up temporary and old files.
        
        Args:
            cleanup_old_logs: Whether to clean up old log files
            cleanup_temp_files: Whether to clean up temporary files
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting disk usage optimization")
        
        start_time = time.time()
        start_disk = psutil.disk_usage('.').used / (1024 * 1024 * 1024)  # GB
        
        files_removed = 0
        bytes_freed = 0
        
        # Clean up temporary files
        if cleanup_temp_files:
            temp_dir = "data/temp"
            if os.path.exists(temp_dir):
                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            file_size = os.path.getsize(file_path)
                            os.unlink(file_path)
                            files_removed += 1
                            bytes_freed += file_size
                    except Exception as e:
                        self.logger.error(f"Error removing file {file_path}: {e}")
        
        # Clean up old log files
        if cleanup_old_logs:
            log_dir = config.logging.log_directory
            if os.path.exists(log_dir):
                # Keep only the last 7 days of logs
                cutoff_time = time.time() - (7 * 24 * 60 * 60)
                for filename in os.listdir(log_dir):
                    if filename.endswith('.log') or filename.endswith('.jsonl'):
                        file_path = os.path.join(log_dir, filename)
                        try:
                            if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                                file_size = os.path.getsize(file_path)
                                os.unlink(file_path)
                                files_removed += 1
                                bytes_freed += file_size
                        except Exception as e:
                            self.logger.error(f"Error removing log file {file_path}: {e}")
        
        end_disk = psutil.disk_usage('.').used / (1024 * 1024 * 1024)  # GB
        end_time = time.time()
        
        results = {
            'start_disk_gb': start_disk,
            'end_disk_gb': end_disk,
            'disk_saved_gb': start_disk - end_disk,
            'files_removed': files_removed,
            'bytes_freed': bytes_freed,
            'duration_seconds': end_time - start_time
        }
        
        self.logger.info(f"Disk optimization completed: {files_removed} files removed, "
                        f"{bytes_freed / (1024 * 1024):.2f} MB freed")
        
        return results
    
    def optimize_data_access(self) -> Dict[str, Any]:
        """
        Optimize data access patterns for better performance.
        
        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting data access optimization")
        
        start_time = time.time()
        
        # Create index files for faster data access
        data_dir = "data/stocks"
        index_created = 0
        
        if os.path.exists(data_dir):
            # Create a simple index of available stock data
            stock_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            
            index_data = {
                'timestamp': datetime.now().isoformat(),
                'total_files': len(stock_files),
                'stocks': []
            }
            
            for stock_file in stock_files:
                symbol = stock_file.replace('.csv', '')
                file_path = os.path.join(data_dir, stock_file)
                file_size = os.path.getsize(file_path)
                mod_time = os.path.getmtime(file_path)
                
                index_data['stocks'].append({
                    'symbol': symbol,
                    'file_path': file_path,
                    'file_size': file_size,
                    'last_modified': datetime.fromtimestamp(mod_time).isoformat()
                })
            
            # Write index file
            import json
            with open(os.path.join("data", "metadata.json"), 'w') as f:
                json.dump(index_data, f, indent=2)
            
            index_created = 1
        
        end_time = time.time()
        
        results = {
            'index_files_created': index_created,
            'duration_seconds': end_time - start_time
        }
        
        self.logger.info(f"Data access optimization completed: {index_created} index files created")
        
        return results
    
    def optimize_model_inference(self) -> Dict[str, Any]:
        """
        Optimize model inference performance.
        
        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting model inference optimization")
        
        start_time = time.time()
        
        # Check if TensorFlow is available
        try:
            import tensorflow as tf
            tf_available = True
        except ImportError:
            tf_available = False
        
        optimizations_applied = []
        
        if tf_available:
            # Enable TensorFlow optimizations
            try:
                # Enable mixed precision
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                optimizations_applied.append("mixed_precision")
                
                # Enable XLA compilation
                tf.config.optimizer.set_jit(True)
                optimizations_applied.append("xla_compilation")
                
                # Optimize for CPU if no GPU is available
                gpus = tf.config.list_physical_devices('GPU')
                if not gpus:
                    tf.config.threading.set_inter_op_parallelism_threads(2)
                    tf.config.threading.set_intra_op_parallelism_threads(psutil.cpu_count())
                    optimizations_applied.append("cpu_threading")
            except Exception as e:
                self.logger.error(f"Error applying TensorFlow optimizations: {e}")
        
        end_time = time.time()
        
        results = {
            'tensorflow_available': tf_available,
            'optimizations_applied': optimizations_applied,
            'duration_seconds': end_time - start_time
        }
        
        self.logger.info(f"Model inference optimization completed: {len(optimizations_applied)} optimizations applied")
        
        return results
    
    def start_continuous_optimization(self, interval_minutes: int = 30):
        """
        Start continuous optimization in background thread.
        
        Args:
            interval_minutes: Interval between optimization runs in minutes
        """
        if self.optimization_active:
            self.logger.warning("Continuous optimization already active")
            return
        
        self.optimization_active = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            args=(interval_minutes,),
            daemon=True
        )
        self.optimization_thread.start()
        
        self.logger.info(f"Started continuous optimization with {interval_minutes} minute interval")
    
    def stop_continuous_optimization(self):
        """Stop continuous optimization."""
        if not self.optimization_active:
            return
        
        self.optimization_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        
        self.logger.info("Stopped continuous optimization")
    
    def _optimization_loop(self, interval_minutes: int):
        """
        Main optimization loop.
        
        Args:
            interval_minutes: Interval between optimization runs in minutes
        """
        while self.optimization_active:
            try:
                # Get current system metrics
                metrics = self.health_monitor.get_performance_metrics()
                
                # Run optimizations based on system state
                if metrics.memory_usage > 80:
                    self.logger.warning(f"High memory usage detected: {metrics.memory_usage:.1f}%")
                    self.optimize_memory_usage()
                
                if metrics.disk_usage > 85:
                    self.logger.warning(f"High disk usage detected: {metrics.disk_usage:.1f}%")
                    self.optimize_disk_usage()
                
                # Run periodic optimizations
                if (int(metrics.uptime_seconds / 60) % (interval_minutes * 3)) == 0:
                    self.optimize_data_access()
                
                # Sleep until next interval
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                time.sleep(interval_minutes * 60)
    
    def run_full_optimization(self) -> Dict[str, Any]:
        """
        Run all optimization strategies.
        
        Returns:
            Dictionary with combined optimization results
        """
        self.logger.info("Starting full system optimization")
        
        start_time = time.time()
        
        # Run all optimizations
        memory_results = self.optimize_memory_usage()
        disk_results = self.optimize_disk_usage()
        data_results = self.optimize_data_access()
        model_results = self.optimize_model_inference()
        
        end_time = time.time()
        
        results = {
            'memory_optimization': memory_results,
            'disk_optimization': disk_results,
            'data_access_optimization': data_results,
            'model_inference_optimization': model_results,
            'total_duration_seconds': end_time - start_time
        }
        
        self.logger.info(f"Full system optimization completed in {end_time - start_time:.2f} seconds")
        
        return results


def optimize_system() -> Dict[str, Any]:
    """
    Run system optimization and return results.
    
    Returns:
        Dictionary with optimization results
    """
    logger = get_logger("SystemOptimization")
    logger.info("Starting system optimization")
    
    optimizer = PerformanceOptimizer()
    results = optimizer.run_full_optimization()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="System Performance Optimizer")
    parser.add_argument("--continuous", action="store_true", 
                       help="Run continuous optimization")
    parser.add_argument("--interval", type=int, default=30,
                       help="Interval between optimization runs (minutes)")
    
    args = parser.parse_args()
    
    if args.continuous:
        optimizer = PerformanceOptimizer()
        optimizer.start_continuous_optimization(args.interval)
        
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            optimizer.stop_continuous_optimization()
    else:
        results = optimize_system()
        print(f"Optimization completed with results: {results}")