#!/usr/bin/env python3
"""
Data Pipeline System for IFD Signal Analysis Utility.

This module provides the infrastructure for routing data between plots through
signal processing chains. It manages connections, executes processing pipelines,
and handles data caching for optimal performance.
"""

import uuid
import copy
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal


class ConnectionStatus(Enum):
    """Status of a pipeline connection."""
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


class PipelineError(Exception):
    """Exception raised for pipeline execution errors."""
    pass


class CyclicDependencyError(PipelineError):
    """Exception raised when a pipeline would create a cycle."""
    pass


@dataclass
class PipelineConnection:
    """
    Represents a connection between two plots through a signal processor.
    
    Attributes:
        connection_id: Unique identifier for this connection
        source_plot_id: ID of the plot providing input data
        target_plot_id: ID of the plot receiving processed data
        processor_name: Name of the signal processor to apply
        processor_parameters: Parameters for the signal processor
        status: Current status of the connection
        created_timestamp: When the connection was created
        last_executed: When the connection was last executed
        execution_count: Number of times this connection has been executed
        cache_enabled: Whether to cache processed results
        auto_update: Whether to automatically update when source changes
        connection_name: Optional user-friendly name for the connection
    """
    connection_id: str
    source_plot_id: str
    target_plot_id: str
    processor_name: str
    processor_parameters: Dict[str, Any]
    status: ConnectionStatus = ConnectionStatus.ACTIVE
    created_timestamp: Optional[str] = None
    last_executed: Optional[str] = None
    execution_count: int = 0
    cache_enabled: bool = True
    auto_update: bool = False
    connection_name: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_timestamp is None:
            import datetime
            self.created_timestamp = datetime.datetime.now().isoformat()
        
        if self.connection_name is None:
            self.connection_name = f"{self.source_plot_id} → {self.target_plot_id}"


class DataPipeline(QObject):
    """
    Manages data routing and processing pipeline between multiple plots.
    
    This class coordinates the flow of data from source plots through
    signal processors to target plots, with support for:
    - Connection management and validation
    - Cycle detection and prevention
    - Data caching for performance optimization
    - Pipeline execution and error handling
    - Automatic updates when source data changes
    
    Signals:
        connection_added: Emitted when a new connection is created
        connection_removed: Emitted when a connection is deleted
        connection_executed: Emitted when a connection processes data
        pipeline_error: Emitted when pipeline execution fails
        cache_updated: Emitted when cached data is refreshed
    """
    
    # Signals for pipeline events
    connection_added = pyqtSignal(str)  # connection_id
    connection_removed = pyqtSignal(str)  # connection_id
    connection_executed = pyqtSignal(str, float)  # connection_id, execution_time
    pipeline_error = pyqtSignal(str, str)  # connection_id, error_message
    cache_updated = pyqtSignal(str, int)  # plot_id, data_size
    
    def __init__(self):
        """Initialize the data pipeline system."""
        super().__init__()
        
        # Connection storage
        self.connections: Dict[str, PipelineConnection] = {}
        
        # Data cache for processed results
        self.data_cache: Dict[str, Dict[str, Any]] = {}  # plot_id -> processed_data
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}  # plot_id -> cache_info
        
        # Pipeline execution tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        # Performance settings
        self.cache_size_limit = 100 * 1024 * 1024  # 100MB cache limit
        self.max_execution_time = 30.0  # 30 second timeout
        
    def add_connection(self, source_plot_id: str, target_plot_id: str,
                      processor_name: str, processor_parameters: Dict[str, Any],
                      connection_name: Optional[str] = None,
                      auto_update: bool = False) -> str:
        """
        Add a new pipeline connection between plots.
        
        Args:
            source_plot_id: ID of the source plot
            target_plot_id: ID of the target plot
            processor_name: Name of the signal processor
            processor_parameters: Parameters for the processor
            connection_name: Optional name for the connection
            auto_update: Whether to auto-update when source changes
            
        Returns:
            str: Connection ID of the created connection
            
        Raises:
            CyclicDependencyError: If connection would create a cycle
            PipelineError: If connection parameters are invalid
        """
        # Generate unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Validate connection parameters
        self._validate_connection_params(
            source_plot_id, target_plot_id, processor_name, processor_parameters
        )
        
        # Create temporary connection for cycle detection
        temp_connection = PipelineConnection(
            connection_id=connection_id,
            source_plot_id=source_plot_id,
            target_plot_id=target_plot_id,
            processor_name=processor_name,
            processor_parameters=processor_parameters.copy(),
            connection_name=connection_name,
            auto_update=auto_update
        )
        
        # Check for cycles before adding
        if self._would_create_cycle(temp_connection):
            raise CyclicDependencyError(
                f"Adding connection from {source_plot_id} to {target_plot_id} "
                f"would create a cycle in the pipeline"
            )
        
        # Add connection
        self.connections[connection_id] = temp_connection
        
        # Emit signal
        self.connection_added.emit(connection_id)
        
        return connection_id
    
    def remove_connection(self, connection_id: str) -> bool:
        """
        Remove a pipeline connection.
        
        Args:
            connection_id: ID of the connection to remove
            
        Returns:
            bool: True if connection was removed, False if not found
        """
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            
            # Clear any cached data for the target plot
            target_plot_id = connection.target_plot_id
            if target_plot_id in self.data_cache:
                del self.data_cache[target_plot_id]
            if target_plot_id in self.cache_metadata:
                del self.cache_metadata[target_plot_id]
            
            # Remove connection
            del self.connections[connection_id]
            
            # Emit signal
            self.connection_removed.emit(connection_id)
            
            return True
        
        return False
    
    def get_connection(self, connection_id: str) -> Optional[PipelineConnection]:
        """
        Get a pipeline connection by ID.
        
        Args:
            connection_id: ID of the connection
            
        Returns:
            PipelineConnection or None if not found
        """
        return self.connections.get(connection_id)
    
    def get_connections_for_plot(self, plot_id: str, 
                               as_source: bool = True, 
                               as_target: bool = True) -> List[PipelineConnection]:
        """
        Get all connections involving a specific plot.
        
        Args:
            plot_id: ID of the plot
            as_source: Include connections where plot is source
            as_target: Include connections where plot is target
            
        Returns:
            List of PipelineConnection objects
        """
        connections = []
        
        for connection in self.connections.values():
            include = False
            
            if as_source and connection.source_plot_id == plot_id:
                include = True
            if as_target and connection.target_plot_id == plot_id:
                include = True
                
            if include:
                connections.append(connection)
        
        return connections
    
    def get_all_connections(self) -> Dict[str, PipelineConnection]:
        """
        Get all pipeline connections.
        
        Returns:
            Dict mapping connection IDs to PipelineConnection objects
        """
        return self.connections.copy()
    
    def execute_connection(self, connection_id: str,
                          source_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a single pipeline connection.
        
        Args:
            connection_id: ID of the connection to execute
            source_data: Optional source data (if not provided, gets from cache)
            
        Returns:
            Dict containing processed data
            
        Raises:
            PipelineError: If execution fails
        """
        if connection_id not in self.connections:
            raise PipelineError(f"Connection {connection_id} not found")
        
        connection = self.connections[connection_id]
        
        if connection.status != ConnectionStatus.ACTIVE:
            raise PipelineError(f"Connection {connection_id} is not active")
        
        try:
            import time
            start_time = time.time()
            
            # Get source data if not provided
            if source_data is None:
                source_data = self._get_plot_data(connection.source_plot_id)
            
            if not source_data or 'channels' not in source_data:
                raise PipelineError(f"No valid data available from plot {connection.source_plot_id}")
            
            # Get signal processor
            from signal_processing import get_processor
            processor = get_processor(connection.processor_name)
            if processor is None:
                raise PipelineError(f"Signal processor '{connection.processor_name}' not found")
            
            # Execute processing
            processed_data = processor.process(source_data, connection.processor_parameters)
            
            # Add pipeline information to processed data
            if 'source_info' not in processed_data:
                processed_data['source_info'] = {}
            
            processed_data['source_info']['pipeline_info'] = {
                'connection_id': connection_id,
                'source_plot_id': connection.source_plot_id,
                'target_plot_id': connection.target_plot_id,
                'processor_name': connection.processor_name,
                'parameters': connection.processor_parameters.copy(),
                'execution_timestamp': self._get_current_timestamp()
            }
            
            # Cache processed data if enabled
            if connection.cache_enabled:
                self._cache_data(connection.target_plot_id, processed_data)
            
            # Update connection statistics
            execution_time = time.time() - start_time
            connection.execution_count += 1
            connection.last_executed = self._get_current_timestamp()
            
            # Add to execution history
            self._add_execution_record(connection_id, execution_time, success=True)
            
            # Emit success signal
            self.connection_executed.emit(connection_id, execution_time)
            
            return processed_data
            
        except Exception as e:
            # Mark connection as error
            connection.status = ConnectionStatus.ERROR
            
            # Add to execution history
            self._add_execution_record(connection_id, 0.0, success=False, error_msg=str(e))
            
            # Emit error signal
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.pipeline_error.emit(connection_id, error_msg)
            
            raise PipelineError(error_msg)
    
    def execute_pipeline_for_plot(self, plot_id: str, 
                                 source_data: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute all pipeline connections that have the specified plot as source.
        
        Args:
            plot_id: ID of the source plot
            source_data: Optional source data
            
        Returns:
            Dict mapping target plot IDs to lists of processed data
        """
        results = {}
        source_connections = self.get_connections_for_plot(plot_id, as_source=True, as_target=False)
        
        for connection in source_connections:
            if connection.status == ConnectionStatus.ACTIVE:
                try:
                    processed_data = self.execute_connection(connection.connection_id, source_data)
                    
                    if connection.target_plot_id not in results:
                        results[connection.target_plot_id] = []
                    
                    results[connection.target_plot_id].append(processed_data)
                    
                except PipelineError as e:
                    print(f"Warning: Pipeline execution failed for connection {connection.connection_id}: {e}")
                    continue
        
        return results
    
    def get_pipeline_chain(self, plot_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        Get the complete pipeline chain starting from a plot.
        
        Args:
            plot_id: Starting plot ID
            max_depth: Maximum depth to traverse
            
        Returns:
            Dict representing the pipeline chain structure
        """
        visited = set()
        
        def build_chain(current_plot_id: str, depth: int = 0) -> Dict[str, Any]:
            if depth >= max_depth or current_plot_id in visited:
                return {'plot_id': current_plot_id, 'connections': []}
            
            visited.add(current_plot_id)
            
            node = {
                'plot_id': current_plot_id,
                'connections': []
            }
            
            # Get outgoing connections
            connections = self.get_connections_for_plot(current_plot_id, as_source=True, as_target=False)
            
            for connection in connections:
                connection_info = {
                    'connection_id': connection.connection_id,
                    'processor_name': connection.processor_name,
                    'parameters': connection.processor_parameters,
                    'status': connection.status.value,
                    'target': build_chain(connection.target_plot_id, depth + 1)
                }
                node['connections'].append(connection_info)
            
            return node
        
        return build_chain(plot_id)
    
    def clear_cache(self, plot_id: Optional[str] = None) -> None:
        """
        Clear cached data for a specific plot or all plots.
        
        Args:
            plot_id: Plot ID to clear cache for, or None for all plots
        """
        if plot_id is None:
            # Clear all cache
            self.data_cache.clear()
            self.cache_metadata.clear()
        else:
            # Clear cache for specific plot
            if plot_id in self.data_cache:
                del self.data_cache[plot_id]
            if plot_id in self.cache_metadata:
                del self.cache_metadata[plot_id]
        
        # Emit cache update signal
        if plot_id:
            self.cache_updated.emit(plot_id, 0)
    
    def get_execution_history(self, connection_id: Optional[str] = None,
                            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get execution history for connections.
        
        Args:
            connection_id: Optional connection ID to filter by
            limit: Optional limit on number of records
            
        Returns:
            List of execution records
        """
        history = self.execution_history
        
        if connection_id:
            history = [record for record in history if record.get('connection_id') == connection_id]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def export_pipeline_config(self) -> Dict[str, Any]:
        """
        Export pipeline configuration to a dictionary.
        
        Returns:
            Dict containing pipeline configuration
        """
        config = {
            'version': '1.0',
            'export_timestamp': self._get_current_timestamp(),
            'connections': {}
        }
        
        for connection_id, connection in self.connections.items():
            config['connections'][connection_id] = asdict(connection)
        
        return config
    
    def import_pipeline_config(self, config: Dict[str, Any], 
                             clear_existing: bool = False) -> List[str]:
        """
        Import pipeline configuration from a dictionary.
        
        Args:
            config: Configuration dictionary
            clear_existing: Whether to clear existing connections first
            
        Returns:
            List of imported connection IDs
        """
        if clear_existing:
            self.connections.clear()
            self.clear_cache()
        
        imported_ids = []
        
        if 'connections' in config:
            for connection_id, connection_data in config['connections'].items():
                try:
                    # Reconstruct connection object
                    connection = PipelineConnection(**connection_data)
                    
                    # Validate before adding
                    if not self._would_create_cycle(connection):
                        self.connections[connection_id] = connection
                        imported_ids.append(connection_id)
                        self.connection_added.emit(connection_id)
                    
                except Exception as e:
                    print(f"Warning: Failed to import connection {connection_id}: {e}")
                    continue
        
        return imported_ids
    
    def _validate_connection_params(self, source_plot_id: str, target_plot_id: str,
                                   processor_name: str, processor_parameters: Dict[str, Any]) -> None:
        """Validate connection parameters."""
        if not source_plot_id or not target_plot_id:
            raise PipelineError("Source and target plot IDs must be specified")
        
        if source_plot_id == target_plot_id:
            raise PipelineError("Source and target plots cannot be the same")
        
        if not processor_name:
            raise PipelineError("Processor name must be specified")
        
        if not isinstance(processor_parameters, dict):
            raise PipelineError("Processor parameters must be a dictionary")
        
        # Validate processor exists and parameters
        from signal_processing import get_processor
        processor = get_processor(processor_name)
        if processor is None:
            raise PipelineError(f"Signal processor '{processor_name}' not found")
        
        # Validate parameters
        validation_errors = processor.validate_parameters(processor_parameters)
        if validation_errors:
            error_msg = "; ".join([f"{k}: {v}" for k, v in validation_errors.items()])
            raise PipelineError(f"Parameter validation failed: {error_msg}")
    
    def _would_create_cycle(self, new_connection: PipelineConnection) -> bool:
        """Check if adding a connection would create a cycle."""
        # Use depth-first search to detect cycles
        temp_connections = self.connections.copy()
        temp_connections[new_connection.connection_id] = new_connection
        
        def has_path(start: str, target: str, visited: Set[str]) -> bool:
            if start == target:
                return True
            
            if start in visited:
                return False
            
            visited.add(start)
            
            # Check all outgoing connections from start
            for connection in temp_connections.values():
                if connection.source_plot_id == start:
                    if has_path(connection.target_plot_id, target, visited.copy()):
                        return True
            
            return False
        
        # Check if there's a path from new target back to new source
        return has_path(new_connection.target_plot_id, new_connection.source_plot_id, set())
    
    def _get_plot_data(self, plot_id: str) -> Optional[Dict[str, Any]]:
        """Get data from a plot (placeholder - will be implemented with plot integration)."""
        # This will be implemented when integrating with the actual plot system
        # For now, return cached data if available
        return self.data_cache.get(plot_id)
    
    def _cache_data(self, plot_id: str, data: Dict[str, Any]) -> None:
        """Cache processed data for a plot."""
        # Calculate data size (approximate)
        data_size = len(str(data))  # Rough approximation
        
        # Check cache size limit
        total_cache_size = sum(len(str(cached_data)) for cached_data in self.data_cache.values())
        
        if total_cache_size + data_size > self.cache_size_limit:
            # Clear oldest cached data
            self._cleanup_cache()
        
        # Cache the data
        self.data_cache[plot_id] = copy.deepcopy(data)
        self.cache_metadata[plot_id] = {
            'cached_at': self._get_current_timestamp(),
            'size': data_size,
            'channel_count': len(data.get('channels', {}))
        }
        
        # Emit cache update signal
        self.cache_updated.emit(plot_id, data_size)
    
    def _cleanup_cache(self) -> None:
        """Remove oldest cached data to free space."""
        if not self.cache_metadata:
            return
        
        # Sort by cached timestamp and remove oldest half
        sorted_items = sorted(
            self.cache_metadata.items(),
            key=lambda x: x[1]['cached_at']
        )
        
        items_to_remove = len(sorted_items) // 2
        for i in range(items_to_remove):
            plot_id = sorted_items[i][0]
            if plot_id in self.data_cache:
                del self.data_cache[plot_id]
            if plot_id in self.cache_metadata:
                del self.cache_metadata[plot_id]
    
    def _add_execution_record(self, connection_id: str, execution_time: float,
                            success: bool, error_msg: Optional[str] = None) -> None:
        """Add an execution record to history."""
        record = {
            'connection_id': connection_id,
            'timestamp': self._get_current_timestamp(),
            'execution_time': execution_time,
            'success': success,
            'error_message': error_msg
        }
        
        self.execution_history.append(record)
        
        # Limit history size
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size//2:]
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dict containing various statistics
        """
        total_connections = len(self.connections)
        active_connections = sum(1 for conn in self.connections.values() 
                               if conn.status == ConnectionStatus.ACTIVE)
        error_connections = sum(1 for conn in self.connections.values() 
                              if conn.status == ConnectionStatus.ERROR)
        
        total_executions = sum(conn.execution_count for conn in self.connections.values())
        cached_plots = len(self.data_cache)
        total_cache_size = sum(metadata['size'] for metadata in self.cache_metadata.values())
        
        return {
            'total_connections': total_connections,
            'active_connections': active_connections,
            'error_connections': error_connections,
            'total_executions': total_executions,
            'cached_plots': cached_plots,
            'cache_size_bytes': total_cache_size,
            'execution_history_length': len(self.execution_history)
        }
