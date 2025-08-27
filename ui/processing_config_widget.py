#!/usr/bin/env python3
"""
Signal Processing Configuration UI for IFD Signal Analysis Utility.

This module provides UI components for configuring signal processing pipelines,
including processor selection, parameter configuration, and pipeline management.
"""

import sys
from typing import Dict, List, Any, Optional, Tuple
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
    QScrollArea, QWidget, QMessageBox, QFrame, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette

try:
    # Try relative import
    from ..signal_processing.processor_registry import ProcessorRegistry, get_registry
    from ..signal_processing.base_processor import ProcessingError
except ImportError:
    # Fallback for direct execution
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from signal_processing.processor_registry import ProcessorRegistry, get_registry
    from signal_processing.base_processor import ProcessingError


class ParameterWidget:
    """Factory class for creating parameter input widgets."""
    
    @staticmethod
    def create_widget(param_name: str, param_def: Dict[str, Any]) -> Tuple[QWidget, callable]:
        """
        Create appropriate widget for parameter based on its definition.
        
        Args:
            param_name: Name of the parameter
            param_def: Parameter definition dictionary
            
        Returns:
            Tuple of (widget, get_value_function)
        """
        param_type = param_def.get('type', str)
        default_value = param_def.get('default')
        minimum = param_def.get('min')
        maximum = param_def.get('max')
        choices = param_def.get('choices', [])
        
        if param_type == int:
            # Integer parameter - use QSpinBox
            widget = QSpinBox()
            if minimum is not None:
                widget.setMinimum(int(minimum))
            if maximum is not None:
                widget.setMaximum(int(maximum))
            if default_value is not None:
                widget.setValue(int(default_value))
            return widget, widget.value
            
        elif param_type == float:
            # Float parameter - use QDoubleSpinBox
            widget = QDoubleSpinBox()
            widget.setDecimals(3)
            widget.setSingleStep(0.1)
            if minimum is not None:
                widget.setMinimum(float(minimum))
            if maximum is not None:
                widget.setMaximum(float(maximum))
            if default_value is not None:
                widget.setValue(float(default_value))
            return widget, widget.value
            
        elif param_type == bool:
            # Boolean parameter - use QCheckBox
            widget = QCheckBox()
            if default_value is not None:
                widget.setChecked(bool(default_value))
            return widget, widget.isChecked
            
        elif param_type == str and choices:
            # String with choices - use QComboBox
            widget = QComboBox()
            widget.addItems(choices)
            if default_value is not None and default_value in choices:
                widget.setCurrentText(str(default_value))
            return widget, widget.currentText
            
        else:
            # Generic string parameter - use QLineEdit
            widget = QLineEdit()
            if default_value is not None:
                widget.setText(str(default_value))
            return widget, widget.text


class ProcessorParametersWidget(QWidget):
    """Widget for configuring parameters of a selected processor."""
    
    parameters_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_processor = None
        self.parameter_widgets = {}
        self.parameter_getters = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # Header
        self.header_label = QLabel("Processor Parameters")
        font = QFont()
        font.setBold(True)
        self.header_label.setFont(font)
        layout.addWidget(self.header_label)
        
        # Scrollable parameter area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.parameters_container = QWidget()
        self.parameters_layout = QGridLayout(self.parameters_container)
        
        scroll.setWidget(self.parameters_container)
        layout.addWidget(scroll)
        
        # Status label
        self.status_label = QLabel("No processor selected")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)
        
    def set_processor(self, processor_name: str):
        """
        Configure parameters for the specified processor.
        
        Args:
            processor_name: Name of the processor to configure
        """
        # Clear existing widgets
        self.clear_parameters()
        
        if not processor_name:
            self.status_label.setText("No processor selected")
            return
            
        # Get processor from registry
        registry = get_registry()
        processor = registry.get_processor(processor_name)
        
        if not processor:
            self.status_label.setText(f"Processor '{processor_name}' not found")
            return
            
        self.current_processor = processor
        self.status_label.setText(f"Configuring: {processor_name}")
        
        # Get parameter definitions
        parameters = processor.get_parameters()
        
        if not parameters:
            no_params_label = QLabel("This processor has no configurable parameters.")
            self.parameters_layout.addWidget(no_params_label, 0, 0, 1, 2)
            return
            
        # Create widgets for each parameter
        row = 0
        for param_name, param_def in parameters.items():
            # Parameter label
            label = QLabel(f"{param_name}:")
            label.setToolTip(param_def.get('description', ''))
            self.parameters_layout.addWidget(label, row, 0)
            
            # Parameter widget
            widget, getter = ParameterWidget.create_widget(param_name, param_def)
            widget.setToolTip(param_def.get('description', ''))
            
            # Connect change signals
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self.parameters_changed)
            elif hasattr(widget, 'textChanged'):
                widget.textChanged.connect(self.parameters_changed)
            elif hasattr(widget, 'toggled'):
                widget.toggled.connect(self.parameters_changed)
            elif hasattr(widget, 'currentTextChanged'):
                widget.currentTextChanged.connect(self.parameters_changed)
                
            self.parameters_layout.addWidget(widget, row, 1)
            
            # Store widget and getter
            self.parameter_widgets[param_name] = widget
            self.parameter_getters[param_name] = getter
            
            row += 1
            
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter values from the widgets.
        
        Returns:
            Dictionary of parameter name -> value
        """
        parameters = {}
        for param_name, getter in self.parameter_getters.items():
            parameters[param_name] = getter()
        return parameters
        
    def validate_parameters(self) -> Dict[str, str]:
        """
        Validate current parameter values.
        
        Returns:
            Dictionary of parameter name -> error message (empty if valid)
        """
        if not self.current_processor:
            return {}
            
        parameters = self.get_parameters()
        return self.current_processor.validate_parameters(parameters)
        
    def clear_parameters(self):
        """Clear all parameter widgets."""
        # Remove all widgets from layout
        while self.parameters_layout.count():
            child = self.parameters_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        self.parameter_widgets.clear()
        self.parameter_getters.clear()
        self.current_processor = None


class ProcessingConfigDialog(QDialog):
    """Dialog for configuring signal processing pipelines."""
    
    pipeline_configured = pyqtSignal(str, str, str, dict)  # source, target, processor, params
    
    def __init__(self, available_plots: List[str], parent=None):
        super().__init__(parent)
        self.available_plots = available_plots
        self.registry = get_registry()
        
        self.setup_ui()
        self.setup_connections()
        self.update_processor_list()
        
    def setup_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Configure Signal Processing Pipeline")
        self.setModal(True)
        self.resize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Configuration
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        
        # Plot selection group
        plot_group = QGroupBox("Plot Selection")
        plot_layout = QGridLayout(plot_group)
        
        plot_layout.addWidget(QLabel("Source Plot:"), 0, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems(self.available_plots)
        plot_layout.addWidget(self.source_combo, 0, 1)
        
        plot_layout.addWidget(QLabel("Target Plot:"), 1, 0)
        self.target_combo = QComboBox()
        self.target_combo.addItems(self.available_plots)
        plot_layout.addWidget(self.target_combo, 1, 1)
        
        config_layout.addWidget(plot_group)
        
        # Processor selection group
        processor_group = QGroupBox("Processor Selection")
        processor_layout = QGridLayout(processor_group)
        
        processor_layout.addWidget(QLabel("Processor:"), 0, 0)
        self.processor_combo = QComboBox()
        processor_layout.addWidget(self.processor_combo, 0, 1)
        
        # Processor description
        self.processor_description = QTextEdit()
        self.processor_description.setMaximumHeight(80)
        self.processor_description.setReadOnly(True)
        processor_layout.addWidget(QLabel("Description:"), 1, 0, Qt.AlignmentFlag.AlignTop)
        processor_layout.addWidget(self.processor_description, 1, 1)
        
        config_layout.addWidget(processor_group)
        
        # Parameter configuration
        self.parameters_widget = ProcessorParametersWidget()
        config_layout.addWidget(self.parameters_widget)
        
        splitter.addWidget(config_widget)
        
        # Right panel - Validation and Preview
        validation_widget = QWidget()
        validation_layout = QVBoxLayout(validation_widget)
        
        # Validation status
        validation_group = QGroupBox("Validation")
        validation_group_layout = QVBoxLayout(validation_group)
        
        self.validation_status = QTextEdit()
        self.validation_status.setMaximumHeight(100)
        self.validation_status.setReadOnly(True)
        validation_group_layout.addWidget(self.validation_status)
        
        validation_layout.addWidget(validation_group)
        
        # Preview area (placeholder for now)
        preview_group = QGroupBox("Preview")
        preview_group_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("Preview functionality will be added in a future update.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("color: gray; font-style: italic;")
        preview_group_layout.addWidget(self.preview_label)
        
        validation_layout.addWidget(preview_group)
        
        splitter.addWidget(validation_widget)
        
        # Set splitter proportions
        splitter.setSizes([400, 200])
        
        # Button bar
        button_layout = QHBoxLayout()
        
        self.preview_button = QPushButton("Preview")
        self.preview_button.setEnabled(False)  # Disabled for now
        button_layout.addWidget(self.preview_button)
        
        button_layout.addStretch()
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.setDefault(True)
        button_layout.addWidget(self.apply_button)
        
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
    def setup_connections(self):
        """Connect signals to slots."""
        self.processor_combo.currentTextChanged.connect(self.on_processor_changed)
        self.parameters_widget.parameters_changed.connect(self.validate_configuration)
        self.source_combo.currentTextChanged.connect(self.validate_configuration)
        self.target_combo.currentTextChanged.connect(self.validate_configuration)
        
        self.apply_button.clicked.connect(self.apply_configuration)
        self.cancel_button.clicked.connect(self.reject)
        
        # Validation timer to avoid too frequent updates
        self.validation_timer = QTimer()
        self.validation_timer.setSingleShot(True)
        self.validation_timer.timeout.connect(self._update_validation_status)
        
    def update_processor_list(self):
        """Update the processor combo box with available processors."""
        self.processor_combo.clear()
        processor_names = self.registry.get_processor_names()
        self.processor_combo.addItems(processor_names)
        
    def on_processor_changed(self, processor_name: str):
        """Handle processor selection change."""
        if not processor_name:
            self.processor_description.clear()
            return
            
        # Update description
        processor = self.registry.get_processor(processor_name)
        if processor:
            self.processor_description.setText(processor.get_description())
        else:
            self.processor_description.setText("Processor not found")
            
        # Update parameters widget
        self.parameters_widget.set_processor(processor_name)
        
        # Validate configuration
        self.validate_configuration()
        
    def validate_configuration(self):
        """Validate the current configuration and update UI."""
        # Use timer to avoid too frequent validation
        self.validation_timer.start(100)
        
    def _update_validation_status(self):
        """Update validation status display."""
        errors = []
        warnings = []
        
        # Check plot selection
        source_plot = self.source_combo.currentText()
        target_plot = self.target_combo.currentText()
        
        if not source_plot:
            errors.append("Source plot must be selected")
        if not target_plot:
            errors.append("Target plot must be selected")
        if source_plot and target_plot and source_plot == target_plot:
            warnings.append("Source and target plots are the same")
            
        # Check processor selection
        processor_name = self.processor_combo.currentText()
        if not processor_name:
            errors.append("Processor must be selected")
        else:
            # Validate processor parameters
            param_errors = self.parameters_widget.validate_parameters()
            if param_errors:
                for param, error in param_errors.items():
                    errors.append(f"Parameter '{param}': {error}")
                    
        # Update validation display
        status_text = ""
        
        if errors:
            status_text += "ERRORS:\n"
            for error in errors:
                status_text += f"• {error}\n"
            status_text += "\n"
            
        if warnings:
            status_text += "WARNINGS:\n"
            for warning in warnings:
                status_text += f"• {warning}\n"
            status_text += "\n"
            
        if not errors and not warnings:
            status_text = "Configuration is valid and ready to apply."
            
        self.validation_status.setText(status_text)
        
        # Enable/disable apply button
        self.apply_button.setEnabled(len(errors) == 0)
        
    def apply_configuration(self):
        """Apply the current configuration."""
        source_plot = self.source_combo.currentText()
        target_plot = self.target_combo.currentText()
        processor_name = self.processor_combo.currentText()
        parameters = self.parameters_widget.get_parameters()
        
        # Final validation
        if not source_plot or not target_plot or not processor_name:
            QMessageBox.warning(self, "Invalid Configuration", 
                              "Please select source plot, target plot, and processor.")
            return
            
        # Validate parameters one more time
        param_errors = self.parameters_widget.validate_parameters()
        if param_errors:
            error_text = "Parameter validation failed:\n"
            for param, error in param_errors.items():
                error_text += f"• {param}: {error}\n"
            QMessageBox.warning(self, "Parameter Validation Error", error_text)
            return
            
        # Emit signal with configuration
        self.pipeline_configured.emit(source_plot, target_plot, processor_name, parameters)
        self.accept()


if __name__ == "__main__":
    # Test the widget standalone
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create test plots
    available_plots = ["Plot1", "Plot2", "Plot3", "Plot4", "Plot5", "Plot6"]
    
    dialog = ProcessingConfigDialog(available_plots)
    dialog.show()
    
    sys.exit(app.exec())
