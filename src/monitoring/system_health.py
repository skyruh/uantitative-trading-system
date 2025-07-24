"""
System health monitoring for the quantitative trading system.
Provides startup validation, dependency checking, performance monitoring,
and alerting capabilities.
"""

import os
import sys
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import config
from src.utils.logging_utils import (
    get_logger, 
    log_system_health, 
    log_performance_alert,
    log_error_with_context
)


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    metrics: Dict[str, Any]
    timestamp: datetime
    duration_ms: float


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int
    uptime_seconds: float


@dataclass
class AlertRule:
    """Configuration for performance alerts."""
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: str  # 'warning', 'critical'
    message_template: str
    cooldown_minutes: int = 5


class SystemHealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self):
        self.logger = get_logger("SystemHealth")
        self.start_time = datetime.now()
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_alert_times = {}
        
        # Default alert rules
        self.alert_rules = [
            AlertRule("cpu_usage", 80.0, "gt", "warning", 
                     "High CPU usage: {value:.1f}%", 5),
            AlertRule("cpu_usage", 95.0, "gt", "critical", 
                     "Critical CPU usage: {value:.1f}%", 2),
            AlertRule("memory_usage", 85.0, "gt", "warning", 
                     "High memory usage: {value:.1f}%", 5),
            AlertRule("memory_usage", 95.0, "gt", "critical", 
                     "Critical memory usage: {value:.1f}%", 2),
            AlertRule("disk_usage", 90.0, "gt", "warning", 
                     "High disk usage: {value:.1f}%", 10),
            AlertRule("disk_usage", 98.0, "gt", "critical", 
                     "Critical disk usage: {value:.1f}%", 5),
        ]
        
        # Performance tracking
        self.performance_history = []
        self.max_history_size = 1000
    
    def validate_system_startup(self) -> List[HealthCheckResult]:
        """Validate system dependencies and configuration at startup."""
        self.logger.info("Starting system validation...")
        results = []
        
        # Check Python version
        results.append(self._check_python_version())
        
        # Check required directories
        results.append(self._check_directories())
        
        # Check configuration
        results.append(self._check_configuration())
        
        # Check required packages
        results.append(self._check_dependencies())
        
        # Check system resources
        results.append(self._check_system_resources())
        
        # Check file permissions
        results.append(self._check_file_permissions())
        
        # Log overall validation result
        critical_failures = [r for r in results if r.status == 'critical']
        warnings = [r for r in results if r.status == 'warning']
        
        if critical_failures:
            self.logger.error(f"System validation failed with {len(critical_failures)} critical issues")
            for failure in critical_failures:
                self.logger.error(f"  - {failure.component}: {failure.message}")
        elif warnings:
            self.logger.warning(f"System validation completed with {len(warnings)} warnings")
            for warning in warnings:
                self.logger.warning(f"  - {warning.component}: {warning.message}")
        else:
            self.logger.info("System validation completed successfully")
        
        return results
    
    def _check_python_version(self) -> HealthCheckResult:
        """Check Python version compatibility."""
        start_time = time.time()
        
        try:
            version = sys.version_info
            required_major, required_minor = 3, 8
            
            if version.major < required_major or (version.major == required_major and version.minor < required_minor):
                status = "critical"
                message = f"Python {version.major}.{version.minor} is too old. Requires Python {required_major}.{required_minor}+"
            else:
                status = "healthy"
                message = f"Python {version.major}.{version.minor}.{version.micro} is compatible"
            
            metrics = {
                "python_version": f"{version.major}.{version.minor}.{version.micro}",
                "required_version": f"{required_major}.{required_minor}+"
            }
            
        except Exception as e:
            status = "critical"
            message = f"Failed to check Python version: {e}"
            metrics = {}
        
        duration_ms = (time.time() - start_time) * 1000
        return HealthCheckResult("Python", status, message, metrics, datetime.now(), duration_ms)
    
    def _check_directories(self) -> HealthCheckResult:
        """Check required directories exist and are writable."""
        start_time = time.time()
        
        try:
            required_dirs = [
                config.data.data_directory,
                config.model.model_save_directory,
                config.logging.log_directory,
                "models/checkpoints"
            ]
            
            missing_dirs = []
            permission_issues = []
            
            for dir_path in required_dirs:
                path = Path(dir_path)
                
                if not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                        self.logger.info(f"Created directory: {dir_path}")
                    except Exception as e:
                        missing_dirs.append(f"{dir_path}: {e}")
                        continue
                
                # Check write permissions
                try:
                    test_file = path / ".write_test"
                    test_file.write_text("test")
                    test_file.unlink()
                except Exception as e:
                    permission_issues.append(f"{dir_path}: {e}")
            
            if missing_dirs or permission_issues:
                status = "critical"
                issues = missing_dirs + permission_issues
                message = f"Directory issues: {'; '.join(issues)}"
            else:
                status = "healthy"
                message = f"All {len(required_dirs)} required directories are accessible"
            
            metrics = {
                "required_directories": len(required_dirs),
                "missing_directories": len(missing_dirs),
                "permission_issues": len(permission_issues)
            }
            
        except Exception as e:
            status = "critical"
            message = f"Failed to check directories: {e}"
            metrics = {}
        
        duration_ms = (time.time() - start_time) * 1000
        return HealthCheckResult("Directories", status, message, metrics, datetime.now(), duration_ms)
    
    def _check_configuration(self) -> HealthCheckResult:
        """Check system configuration validity."""
        start_time = time.time()
        
        try:
            is_valid = config.validate_config()
            
            if is_valid:
                status = "healthy"
                message = "Configuration validation passed"
            else:
                status = "critical"
                message = "Configuration validation failed"
            
            metrics = {
                "environment": config.environment,
                "data_directory": config.data.data_directory,
                "model_directory": config.model.model_save_directory,
                "log_level": config.logging.log_level
            }
            
        except Exception as e:
            status = "critical"
            message = f"Configuration check failed: {e}"
            metrics = {}
        
        duration_ms = (time.time() - start_time) * 1000
        return HealthCheckResult("Configuration", status, message, metrics, datetime.now(), duration_ms)
    
    def _check_dependencies(self) -> HealthCheckResult:
        """Check required Python packages are installed."""
        start_time = time.time()
        
        try:
            required_packages = [
                'pandas', 'numpy', 'yfinance', 'tensorflow', 
                'transformers', 'scikit-learn', 'psutil'
            ]
            
            missing_packages = []
            version_info = {}
            
            for package in required_packages:
                try:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                    version_info[package] = version
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                status = "critical"
                message = f"Missing required packages: {', '.join(missing_packages)}"
            else:
                status = "healthy"
                message = f"All {len(required_packages)} required packages are installed"
            
            metrics = {
                "required_packages": len(required_packages),
                "missing_packages": len(missing_packages),
                "package_versions": version_info
            }
            
        except Exception as e:
            status = "critical"
            message = f"Dependency check failed: {e}"
            metrics = {}
        
        duration_ms = (time.time() - start_time) * 1000
        return HealthCheckResult("Dependencies", status, message, metrics, datetime.now(), duration_ms)
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource availability."""
        start_time = time.time()
        
        try:
            # Memory check
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            # Disk space check
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            
            # CPU check
            cpu_count = psutil.cpu_count()
            
            warnings = []
            if available_gb < 2.0:
                warnings.append(f"Low available memory: {available_gb:.1f}GB")
            if free_gb < 5.0:
                warnings.append(f"Low disk space: {free_gb:.1f}GB")
            if cpu_count < 2:
                warnings.append(f"Limited CPU cores: {cpu_count}")
            
            if warnings:
                status = "warning"
                message = f"Resource warnings: {'; '.join(warnings)}"
            else:
                status = "healthy"
                message = "Sufficient system resources available"
            
            metrics = {
                "available_memory_gb": round(available_gb, 2),
                "free_disk_gb": round(free_gb, 2),
                "cpu_cores": cpu_count,
                "total_memory_gb": round(memory.total / (1024**3), 2)
            }
            
        except Exception as e:
            status = "critical"
            message = f"Resource check failed: {e}"
            metrics = {}
        
        duration_ms = (time.time() - start_time) * 1000
        return HealthCheckResult("SystemResources", status, message, metrics, datetime.now(), duration_ms)
    
    def _check_file_permissions(self) -> HealthCheckResult:
        """Check file system permissions for critical paths."""
        start_time = time.time()
        
        try:
            test_paths = [
                config.data.data_directory,
                config.logging.log_directory,
                config.model.model_save_directory
            ]
            
            permission_issues = []
            
            for path_str in test_paths:
                path = Path(path_str)
                if path.exists():
                    # Test read permission
                    if not os.access(path, os.R_OK):
                        permission_issues.append(f"{path_str}: No read permission")
                    
                    # Test write permission
                    if not os.access(path, os.W_OK):
                        permission_issues.append(f"{path_str}: No write permission")
            
            if permission_issues:
                status = "critical"
                message = f"Permission issues: {'; '.join(permission_issues)}"
            else:
                status = "healthy"
                message = "File permissions are correct"
            
            metrics = {
                "checked_paths": len(test_paths),
                "permission_issues": len(permission_issues)
            }
            
        except Exception as e:
            status = "critical"
            message = f"Permission check failed: {e}"
            metrics = {}
        
        duration_ms = (time.time() - start_time) * 1000
        return HealthCheckResult("FilePermissions", status, message, metrics, datetime.now(), duration_ms)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('.')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Uptime
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            return PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                uptime_seconds=uptime_seconds
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return PerformanceMetrics(0, 0, 0, {}, 0, 0)
    
    def check_alert_rules(self, metrics: PerformanceMetrics):
        """Check performance metrics against alert rules."""
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            try:
                # Get metric value
                metric_value = getattr(metrics, rule.metric_name, None)
                if metric_value is None:
                    continue
                
                # Check if alert should trigger
                should_alert = False
                if rule.comparison == 'gt' and metric_value > rule.threshold:
                    should_alert = True
                elif rule.comparison == 'lt' and metric_value < rule.threshold:
                    should_alert = True
                elif rule.comparison == 'eq' and metric_value == rule.threshold:
                    should_alert = True
                
                if should_alert:
                    # Check cooldown period
                    last_alert_key = f"{rule.metric_name}_{rule.severity}"
                    last_alert_time = self.last_alert_times.get(last_alert_key)
                    
                    if (last_alert_time is None or 
                        current_time - last_alert_time > timedelta(minutes=rule.cooldown_minutes)):
                        
                        # Trigger alert
                        message = rule.message_template.format(value=metric_value)
                        alert_metrics = {rule.metric_name: metric_value, 'threshold': rule.threshold}
                        
                        log_performance_alert(
                            alert_type=f"{rule.metric_name}_threshold",
                            message=message,
                            severity=rule.severity.upper(),
                            metrics=alert_metrics
                        )
                        
                        self.last_alert_times[last_alert_key] = current_time
                        
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule.metric_name}: {e}")
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous system monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info(f"Started system monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous system monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Stopped system monitoring")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Get performance metrics
                metrics = self.get_performance_metrics()
                
                # Store in history
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'metrics': metrics
                })
                
                # Trim history if too large
                if len(self.performance_history) > self.max_history_size:
                    self.performance_history = self.performance_history[-self.max_history_size:]
                
                # Check alert rules
                self.check_alert_rules(metrics)
                
                # Log system health periodically (every 10 minutes)
                if len(self.performance_history) % 10 == 0:
                    log_system_health(
                        component="SystemMonitor",
                        status="healthy",
                        metrics={
                            'cpu_usage': metrics.cpu_usage,
                            'memory_usage': metrics.memory_usage,
                            'disk_usage': metrics.disk_usage,
                            'uptime_hours': metrics.uptime_seconds / 3600
                        }
                    )
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                log_error_with_context(
                    self.logger, e, "monitoring_loop",
                    {'interval_seconds': interval_seconds}
                )
                time.sleep(interval_seconds)
    
    def check_system_health(self) -> bool:
        """
        Check if the system is healthy.
        Returns True if system is healthy, False otherwise.
        """
        try:
            metrics = self.get_performance_metrics()
            
            # Check for critical issues
            if metrics.cpu_usage > 95 or metrics.memory_usage > 95 or metrics.disk_usage > 98:
                self.logger.warning(f"System health check failed: CPU={metrics.cpu_usage}%, Memory={metrics.memory_usage}%, Disk={metrics.disk_usage}%")
                return False
                
            # Validate critical directories
            required_dirs = [
                config.data.data_directory,
                config.model.model_save_directory,
                config.logging.log_directory
            ]
            
            for dir_path in required_dirs:
                if not os.path.exists(dir_path) or not os.access(dir_path, os.W_OK):
                    self.logger.warning(f"System health check failed: Directory issue with {dir_path}")
                    return False
            
            self.logger.debug("System health check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"System health check failed with error: {str(e)}")
            return False
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        try:
            metrics = self.get_performance_metrics()
            
            # Determine overall health status
            status = "healthy"
            if metrics.cpu_usage > 90 or metrics.memory_usage > 90:
                status = "critical"
            elif metrics.cpu_usage > 80 or metrics.memory_usage > 80:
                status = "warning"
            
            return {
                'overall_status': status,
                'uptime_hours': round(metrics.uptime_seconds / 3600, 2),
                'performance_metrics': {
                    'cpu_usage': metrics.cpu_usage,
                    'memory_usage': metrics.memory_usage,
                    'disk_usage': metrics.disk_usage,
                    'process_count': metrics.process_count
                },
                'monitoring_active': self.monitoring_active,
                'history_size': len(self.performance_history),
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            log_error_with_context(self.logger, e, "get_health_summary")
            return {
                'overall_status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
            
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status for the orchestrator.
        This is an alias for get_health_summary to maintain compatibility.
        
        Returns:
            Dictionary with system health information
        """
        return self.get_health_summary()


# Global system health monitor instance
system_monitor = SystemHealthMonitor()


def validate_startup() -> bool:
    """Validate system at startup and return success status."""
    results = system_monitor.validate_system_startup()
    critical_failures = [r for r in results if r.status == 'critical']
    return len(critical_failures) == 0


def start_system_monitoring(interval_seconds: int = 60):
    """Start system monitoring with specified interval."""
    system_monitor.start_monitoring(interval_seconds)


def stop_system_monitoring():
    """Stop system monitoring."""
    system_monitor.stop_monitoring()


def get_system_health() -> Dict[str, Any]:
    """Get current system health summary."""
    return system_monitor.get_health_summary()


def add_custom_alert_rule(metric_name: str, threshold: float, comparison: str,
                         severity: str, message_template: str, cooldown_minutes: int = 5):
    """Add a custom alert rule to the system monitor."""
    rule = AlertRule(metric_name, threshold, comparison, severity, 
                    message_template, cooldown_minutes)
    system_monitor.alert_rules.append(rule)