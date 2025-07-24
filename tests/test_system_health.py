"""
Integration tests for system health monitoring.
Tests startup validation, performance monitoring, and alerting system.
"""

import unittest
import tempfile
import shutil
import time
import threading
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.monitoring.system_health import (
    SystemHealthMonitor,
    HealthCheckResult,
    PerformanceMetrics,
    AlertRule,
    validate_startup,
    start_system_monitoring,
    stop_system_monitoring,
    get_system_health,
    add_custom_alert_rule,
    system_monitor
)


class TestHealthCheckResult(unittest.TestCase):
    """Test HealthCheckResult dataclass."""
    
    def test_health_check_result_creation(self):
        """Test creating a health check result."""
        result = HealthCheckResult(
            component="TestComponent",
            status="healthy",
            message="Test message",
            metrics={'test_metric': 100},
            timestamp=datetime.now(),
            duration_ms=50.0
        )
        
        self.assertEqual(result.component, "TestComponent")
        self.assertEqual(result.status, "healthy")
        self.assertEqual(result.message, "Test message")
        self.assertEqual(result.metrics['test_metric'], 100)
        self.assertIsInstance(result.timestamp, datetime)
        self.assertEqual(result.duration_ms, 50.0)


class TestPerformanceMetrics(unittest.TestCase):
    """Test PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            cpu_usage=45.5,
            memory_usage=60.2,
            disk_usage=75.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            process_count=150,
            uptime_seconds=3600.0
        )
        
        self.assertEqual(metrics.cpu_usage, 45.5)
        self.assertEqual(metrics.memory_usage, 60.2)
        self.assertEqual(metrics.disk_usage, 75.0)
        self.assertEqual(metrics.network_io['bytes_sent'], 1000)
        self.assertEqual(metrics.process_count, 150)
        self.assertEqual(metrics.uptime_seconds, 3600.0)


class TestAlertRule(unittest.TestCase):
    """Test AlertRule dataclass."""
    
    def test_alert_rule_creation(self):
        """Test creating an alert rule."""
        rule = AlertRule(
            metric_name="cpu_usage",
            threshold=80.0,
            comparison="gt",
            severity="warning",
            message_template="High CPU: {value}%",
            cooldown_minutes=5
        )
        
        self.assertEqual(rule.metric_name, "cpu_usage")
        self.assertEqual(rule.threshold, 80.0)
        self.assertEqual(rule.comparison, "gt")
        self.assertEqual(rule.severity, "warning")
        self.assertEqual(rule.message_template, "High CPU: {value}%")
        self.assertEqual(rule.cooldown_minutes, 5)


class TestSystemHealthMonitor(unittest.TestCase):
    """Test SystemHealthMonitor class."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config
        self.config_patcher = patch('src.monitoring.system_health.config')
        mock_config = self.config_patcher.start()
        mock_config.data.data_directory = self.temp_dir + "/data"
        mock_config.model.model_save_directory = self.temp_dir + "/models"
        mock_config.logging.log_directory = self.temp_dir + "/logs"
        mock_config.environment = "test"
        mock_config.validate_config.return_value = True
        
        self.monitor = SystemHealthMonitor()
    
    def tearDown(self):
        self.monitor.stop_monitoring()
        self.config_patcher.stop()
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass  # Skip cleanup if files are locked
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        self.assertIsNotNone(self.monitor.logger)
        self.assertIsInstance(self.monitor.start_time, datetime)
        self.assertFalse(self.monitor.monitoring_active)
        self.assertIsNone(self.monitor.monitoring_thread)
        self.assertIsInstance(self.monitor.alert_rules, list)
        self.assertTrue(len(self.monitor.alert_rules) > 0)
    
    def test_python_version_check(self):
        """Test Python version validation."""
        result = self.monitor._check_python_version()
        
        self.assertEqual(result.component, "Python")
        self.assertIn(result.status, ['healthy', 'warning', 'critical'])
        self.assertIsInstance(result.message, str)
        self.assertIsInstance(result.metrics, dict)
        self.assertIn('python_version', result.metrics)
        self.assertGreaterEqual(result.duration_ms, 0)
    
    def test_directories_check(self):
        """Test directory validation."""
        result = self.monitor._check_directories()
        
        self.assertEqual(result.component, "Directories")
        self.assertIn(result.status, ['healthy', 'warning', 'critical'])
        self.assertIsInstance(result.message, str)
        self.assertIsInstance(result.metrics, dict)
        self.assertIn('required_directories', result.metrics)
        self.assertGreater(result.duration_ms, 0)
    
    def test_configuration_check(self):
        """Test configuration validation."""
        result = self.monitor._check_configuration()
        
        self.assertEqual(result.component, "Configuration")
        self.assertIn(result.status, ['healthy', 'warning', 'critical'])
        self.assertIsInstance(result.message, str)
        self.assertIsInstance(result.metrics, dict)
        self.assertIn('environment', result.metrics)
        self.assertGreaterEqual(result.duration_ms, 0)
    
    def test_dependencies_check(self):
        """Test dependency validation."""
        result = self.monitor._check_dependencies()
        
        self.assertEqual(result.component, "Dependencies")
        self.assertIn(result.status, ['healthy', 'warning', 'critical'])
        self.assertIsInstance(result.message, str)
        self.assertIsInstance(result.metrics, dict)
        self.assertIn('required_packages', result.metrics)
        self.assertGreater(result.duration_ms, 0)
    
    def test_system_resources_check(self):
        """Test system resources validation."""
        result = self.monitor._check_system_resources()
        
        self.assertEqual(result.component, "SystemResources")
        self.assertIn(result.status, ['healthy', 'warning', 'critical'])
        self.assertIsInstance(result.message, str)
        self.assertIsInstance(result.metrics, dict)
        self.assertIn('available_memory_gb', result.metrics)
        self.assertIn('free_disk_gb', result.metrics)
        self.assertIn('cpu_cores', result.metrics)
        self.assertGreaterEqual(result.duration_ms, 0)
    
    def test_file_permissions_check(self):
        """Test file permissions validation."""
        result = self.monitor._check_file_permissions()
        
        self.assertEqual(result.component, "FilePermissions")
        self.assertIn(result.status, ['healthy', 'warning', 'critical'])
        self.assertIsInstance(result.message, str)
        self.assertIsInstance(result.metrics, dict)
        self.assertIn('checked_paths', result.metrics)
        self.assertGreaterEqual(result.duration_ms, 0)
    
    def test_validate_system_startup(self):
        """Test complete system startup validation."""
        results = self.monitor.validate_system_startup()
        
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        
        # Check that all expected components are validated
        components = [r.component for r in results]
        expected_components = [
            "Python", "Directories", "Configuration", 
            "Dependencies", "SystemResources", "FilePermissions"
        ]
        
        for expected in expected_components:
            self.assertIn(expected, components)
        
        # All results should have required fields
        for result in results:
            self.assertIsInstance(result, HealthCheckResult)
            self.assertIn(result.status, ['healthy', 'warning', 'critical'])
            self.assertIsInstance(result.message, str)
            self.assertIsInstance(result.metrics, dict)
            self.assertIsInstance(result.timestamp, datetime)
            self.assertGreaterEqual(result.duration_ms, 0)
    
    @patch('src.monitoring.system_health.psutil')
    def test_get_performance_metrics(self, mock_psutil):
        """Test performance metrics collection."""
        # Mock psutil functions
        mock_psutil.cpu_percent.return_value = 45.5
        mock_psutil.virtual_memory.return_value = MagicMock(percent=60.2)
        mock_psutil.disk_usage.return_value = MagicMock(
            used=750*1024**3, total=1000*1024**3
        )
        mock_psutil.net_io_counters.return_value = MagicMock(
            bytes_sent=1000, bytes_recv=2000,
            packets_sent=100, packets_recv=200
        )
        mock_psutil.pids.return_value = list(range(150))
        
        metrics = self.monitor.get_performance_metrics()
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(metrics.cpu_usage, 45.5)
        self.assertEqual(metrics.memory_usage, 60.2)
        self.assertEqual(metrics.disk_usage, 75.0)
        self.assertEqual(metrics.network_io['bytes_sent'], 1000)
        self.assertEqual(metrics.process_count, 150)
        self.assertGreaterEqual(metrics.uptime_seconds, 0)
    
    @patch('src.monitoring.system_health.log_performance_alert')
    def test_check_alert_rules(self, mock_log_alert):
        """Test alert rule checking."""
        # Create test metrics that should trigger alerts
        metrics = PerformanceMetrics(
            cpu_usage=85.0,  # Should trigger warning (threshold 80)
            memory_usage=50.0,
            disk_usage=50.0,
            network_io={},
            process_count=100,
            uptime_seconds=3600
        )
        
        self.monitor.check_alert_rules(metrics)
        
        # Should have triggered at least one alert
        self.assertTrue(mock_log_alert.called)
        
        # Check alert was called with correct parameters
        call_args = mock_log_alert.call_args
        self.assertIn('cpu_usage', call_args[1]['alert_type'])
        self.assertEqual(call_args[1]['severity'], 'WARNING')
    
    def test_alert_cooldown(self):
        """Test alert cooldown functionality."""
        with patch('src.monitoring.system_health.log_performance_alert') as mock_log_alert:
            # Create metrics that trigger alert
            metrics = PerformanceMetrics(
                cpu_usage=85.0,
                memory_usage=50.0,
                disk_usage=50.0,
                network_io={},
                process_count=100,
                uptime_seconds=3600
            )
            
            # First check should trigger alert
            self.monitor.check_alert_rules(metrics)
            first_call_count = mock_log_alert.call_count
            
            # Immediate second check should not trigger (cooldown)
            self.monitor.check_alert_rules(metrics)
            second_call_count = mock_log_alert.call_count
            
            self.assertEqual(first_call_count, second_call_count)
    
    def test_monitoring_start_stop(self):
        """Test starting and stopping monitoring."""
        # Start monitoring
        self.monitor.start_monitoring(interval_seconds=1)
        
        self.assertTrue(self.monitor.monitoring_active)
        self.assertIsNotNone(self.monitor.monitoring_thread)
        self.assertTrue(self.monitor.monitoring_thread.is_alive())
        
        # Let it run briefly
        time.sleep(2)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        self.assertFalse(self.monitor.monitoring_active)
        
        # Wait for thread to finish
        time.sleep(1)
        if self.monitor.monitoring_thread:
            self.assertFalse(self.monitor.monitoring_thread.is_alive())
    
    def test_get_health_summary(self):
        """Test health summary generation."""
        with patch.object(self.monitor, 'get_performance_metrics') as mock_metrics:
            mock_metrics.return_value = PerformanceMetrics(
                cpu_usage=45.0,
                memory_usage=60.0,
                disk_usage=70.0,
                network_io={},
                process_count=100,
                uptime_seconds=3600
            )
            
            summary = self.monitor.get_health_summary()
            
            self.assertIsInstance(summary, dict)
            self.assertIn('overall_status', summary)
            self.assertIn('uptime_hours', summary)
            self.assertIn('performance_metrics', summary)
            self.assertIn('monitoring_active', summary)
            self.assertIn('last_check', summary)
            
            # Check performance metrics are included
            perf_metrics = summary['performance_metrics']
            self.assertEqual(perf_metrics['cpu_usage'], 45.0)
            self.assertEqual(perf_metrics['memory_usage'], 60.0)


class TestModuleFunctions(unittest.TestCase):
    """Test module-level functions."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config
        self.config_patcher = patch('src.monitoring.system_health.config')
        mock_config = self.config_patcher.start()
        mock_config.data.data_directory = self.temp_dir + "/data"
        mock_config.model.model_save_directory = self.temp_dir + "/models"
        mock_config.logging.log_directory = self.temp_dir + "/logs"
        mock_config.environment = "test"
        mock_config.validate_config.return_value = True
    
    def tearDown(self):
        stop_system_monitoring()
        self.config_patcher.stop()
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_validate_startup(self):
        """Test startup validation function."""
        result = validate_startup()
        self.assertIsInstance(result, bool)
    
    def test_start_stop_system_monitoring(self):
        """Test starting and stopping system monitoring."""
        # Start monitoring
        start_system_monitoring(interval_seconds=1)
        
        # Check it's running
        summary = get_system_health()
        self.assertTrue(summary.get('monitoring_active', False))
        
        # Stop monitoring
        stop_system_monitoring()
        
        # Brief wait for cleanup
        time.sleep(0.5)
    
    def test_get_system_health(self):
        """Test getting system health summary."""
        health = get_system_health()
        
        self.assertIsInstance(health, dict)
        self.assertIn('overall_status', health)
        self.assertIn('last_check', health)
    
    def test_add_custom_alert_rule(self):
        """Test adding custom alert rules."""
        initial_rule_count = len(system_monitor.alert_rules)
        
        add_custom_alert_rule(
            metric_name="custom_metric",
            threshold=100.0,
            comparison="gt",
            severity="warning",
            message_template="Custom alert: {value}",
            cooldown_minutes=10
        )
        
        self.assertEqual(len(system_monitor.alert_rules), initial_rule_count + 1)
        
        # Check the new rule
        new_rule = system_monitor.alert_rules[-1]
        self.assertEqual(new_rule.metric_name, "custom_metric")
        self.assertEqual(new_rule.threshold, 100.0)
        self.assertEqual(new_rule.comparison, "gt")
        self.assertEqual(new_rule.severity, "warning")
        self.assertEqual(new_rule.cooldown_minutes, 10)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config
        self.config_patcher = patch('src.monitoring.system_health.config')
        mock_config = self.config_patcher.start()
        mock_config.data.data_directory = self.temp_dir + "/data"
        mock_config.model.model_save_directory = self.temp_dir + "/models"
        mock_config.logging.log_directory = self.temp_dir + "/logs"
        mock_config.environment = "test"
        mock_config.validate_config.return_value = True
        
        self.monitor = SystemHealthMonitor()
    
    def tearDown(self):
        self.monitor.stop_monitoring()
        self.config_patcher.stop()
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_complete_startup_and_monitoring_cycle(self):
        """Test complete startup validation and monitoring cycle."""
        # 1. Validate startup
        startup_results = self.monitor.validate_system_startup()
        self.assertIsInstance(startup_results, list)
        self.assertTrue(len(startup_results) > 0)
        
        # 2. Start monitoring
        self.monitor.start_monitoring(interval_seconds=1)
        self.assertTrue(self.monitor.monitoring_active)
        
        # 3. Let monitoring run briefly
        time.sleep(2)
        
        # 4. Check health summary
        health = self.monitor.get_health_summary()
        self.assertIsInstance(health, dict)
        self.assertIn('overall_status', health)
        
        # 5. Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring_active)
    
    @patch('src.monitoring.system_health.psutil')
    @patch('src.monitoring.system_health.log_performance_alert')
    def test_alert_triggering_scenario(self, mock_log_alert, mock_psutil):
        """Test scenario where alerts are triggered."""
        # Mock high resource usage
        mock_psutil.cpu_percent.return_value = 95.0  # Critical level
        mock_psutil.virtual_memory.return_value = MagicMock(percent=90.0)  # Warning level
        mock_psutil.disk_usage.return_value = MagicMock(
            used=950*1024**3, total=1000*1024**3  # 95% usage
        )
        mock_psutil.net_io_counters.return_value = MagicMock(
            bytes_sent=1000, bytes_recv=2000,
            packets_sent=100, packets_recv=200
        )
        mock_psutil.pids.return_value = list(range(200))
        
        # Get metrics and check alerts
        metrics = self.monitor.get_performance_metrics()
        self.monitor.check_alert_rules(metrics)
        
        # Should have triggered multiple alerts
        self.assertTrue(mock_log_alert.called)
        self.assertGreater(mock_log_alert.call_count, 1)
        
        # Check that alerts were triggered (simplified check)
        alert_calls = mock_log_alert.call_args_list
        self.assertTrue(len(alert_calls) > 0, "No alerts were triggered")
        
        # Check that at least one alert contains CPU information
        has_cpu_alert = any(
            'cpu' in str(call).lower() 
            for call in alert_calls
        )
        self.assertTrue(has_cpu_alert, f"No CPU alert found in calls: {alert_calls}")


if __name__ == '__main__':
    unittest.main()