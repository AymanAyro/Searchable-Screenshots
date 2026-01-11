"""Main application window for Searchable Screenshots using PySide6."""

import sys
import os
from pathlib import Path
from typing import Optional
import threading

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QScrollArea, QFrame,
    QFileDialog, QMessageBox, QProgressBar, QSplitter,
    QDialog, QTextEdit, QListWidget, QGroupBox, QSlider,
)
from PySide6.QtCore import Qt, Signal, QObject, QSize
from PySide6.QtGui import QPixmap, QFont, QIcon
import qtawesome as qta

from ..core.config import ConfigManager
from ..core.database import Database
from ..core.search import SearchEngine, SearchResult
from ..core.processor import ScreenshotProcessor, ProcessingStats
from ..services.ocr import OCRService
from ..services.vision import VisionService
from ..services.embedding import EmbeddingService
from ..services.sparse_embedding import SparseEmbeddingService
from ..services.vector_store import VectorStore
from ..services.reranker import RerankerService


class WorkerSignals(QObject):
    """Signals for background worker threads."""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int, int, str, str)  # current, total, filename, status


def is_dark_mode() -> bool:
    """Detect if the system is using dark mode."""
    app = QApplication.instance()
    if app:
        palette = app.palette()
        bg_color = palette.color(palette.ColorRole.Window)
        # If background luminance is low, we're in dark mode
        return bg_color.lightnessF() < 0.5
    return False


class ThemeColors:
    """Theme-aware color provider."""
    
    @staticmethod
    def get_colors():
        """Get colors appropriate for current theme."""
        dark = is_dark_mode()
        
        if dark:
            return {
                'bg': '#2d2d2d',
                'bg_alt': '#3d3d3d',
                'bg_hover': '#4d4d4d',
                'border': '#555555',
                'border_hover': '#2196F3',
                'text': '#ffffff',
                'text_secondary': '#bbbbbb',
                'text_muted': '#999999',
                'accent': '#2196F3',
                'accent_hover': '#1976D2',
                'input_bg': '#3d3d3d',
            }
        else:
            return {
                'bg': '#fafafa',
                'bg_alt': '#ffffff',
                'bg_hover': '#f5f5f5',
                'border': '#e0e0e0',
                'border_hover': '#2196F3',
                'text': '#212121',
                'text_secondary': '#666666',
                'text_muted': '#888888',
                'accent': '#2196F3',
                'accent_hover': '#1976D2',
                'input_bg': '#ffffff',
            }


class InfoRow(QWidget):
    """A row in the info dialog with label, value, and copy button."""
    
    def __init__(self, label: str, value: str, multiline: bool = False, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Label
        lbl = QLabel(label)
        lbl.setFixedWidth(100)
        font = lbl.font()
        font.setBold(True)
        lbl.setFont(font)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignTop)
        layout.addWidget(lbl)
        
        # Value
        self.value = value
        if multiline:
            self.val_widget = QTextEdit(value)
            self.val_widget.setReadOnly(True)
            self.val_widget.setMaximumHeight(100)
        else:
            self.val_widget = QLineEdit(value)
            self.val_widget.setReadOnly(True)
            self.val_widget.setCursorPosition(0)
        
        layout.addWidget(self.val_widget, 1)
        
        # Copy button
        colors = ThemeColors.get_colors()
        copy_btn = QPushButton()
        copy_btn.setIcon(qta.icon('fa5s.copy', color=colors['text_secondary']))
        copy_btn.setToolTip("Copy to clipboard")
        copy_btn.clicked.connect(self.copy_to_clipboard)
        layout.addWidget(copy_btn)
    
    def copy_to_clipboard(self):
        QApplication.clipboard().setText(self.value)


class ImageInfoDialog(QDialog):
    """Dialog to show detailed image information."""
    
    def __init__(self, screenshot, parent=None):
        super().__init__(parent)
        self.screenshot = screenshot
        self.setWindowTitle("Image Information")
        self.setMinimumSize(600, 500)
        self.setup_ui()
        
    def setup_ui(self):
        colors = ThemeColors.get_colors()
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {colors['bg']};
            }}
            QLabel {{
                color: {colors['text']};
            }}
            QLineEdit, QTextEdit {{
                background-color: {colors['input_bg']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: {colors['bg_hover']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {colors['border']};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("Image Details")
        font = title.font()
        font.setPointSize(14)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        
        # Content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"background-color: {colors['bg']};")
        
        content = QWidget()
        content.setStyleSheet(f"background-color: {colors['bg']};")
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(12)
        
        # Prepare data
        file_path = Path(self.screenshot.file_path)
        
        # Fields to display
        fields = [
            ("File Name", file_path.name, False),
            ("Location", str(file_path.parent), False),
            ("File Hash", self.screenshot.file_hash, False),
            ("OCR Text", self.screenshot.ocr_text or "No text detected", True),
            ("Visual Desc", self.screenshot.visual_description or "No description available", True),
        ]
        
        for label, value, multiline in fields:
            row = InfoRow(label, value, multiline)
            content_layout.addWidget(row)
            
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        # Reindex button
        self.reindex_btn = QPushButton("Reindex")
        self.reindex_btn.setIcon(qta.icon('fa5s.redo', color=colors['text_secondary']))
        self.reindex_btn.setToolTip("Re-process this image (OCR, vision, embedding)")
        self.reindex_btn.clicked.connect(self.on_reindex)
        btn_layout.addWidget(self.reindex_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setFixedWidth(100)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def on_reindex(self):
        """Reindex this specific image in background."""
        from pathlib import Path
        import threading
        
        # Get the main window to access processor
        main_window = self.parent()
        while main_window and not hasattr(main_window, 'processor'):
            main_window = main_window.parent()
        
        if not main_window or not hasattr(main_window, 'processor'):
            QMessageBox.warning(self, "Error", "Could not access processor")
            return
        
        file_path = Path(self.screenshot.file_path)
        if not file_path.exists():
            QMessageBox.warning(self, "Error", f"File no longer exists: {file_path}")
            return
        
        # Disable the reindex button and show progress
        self.reindex_btn.setEnabled(False)
        self.reindex_btn.setText("Reindexing...")
        
        # Create signals for thread communication
        signals = WorkerSignals()
        signals.finished.connect(self._on_reindex_complete)
        signals.error.connect(self._on_reindex_error)
        
        def do_reindex():
            try:
                result = main_window.processor.process_single(file_path, force=True)
                if result is not None:
                    # Get updated screenshot data
                    updated = main_window.db.get_by_id(result)
                    signals.finished.emit(updated)
                else:
                    signals.error.emit("Failed to reindex image")
            except Exception as e:
                signals.error.emit(str(e))
        
        thread = threading.Thread(target=do_reindex, daemon=True)
        thread.start()
    
    def _on_reindex_complete(self, updated_screenshot):
        """Handle successful reindex."""
        colors = ThemeColors.get_colors()
        self.reindex_btn.setEnabled(True)
        self.reindex_btn.setText("Reindex")
        self.reindex_btn.setIcon(qta.icon('fa5s.redo', color=colors['text_secondary']))
        
        if updated_screenshot:
            self.screenshot = updated_screenshot
            QMessageBox.information(
                self,
                "Reindex Complete",
                f"Successfully reindexed: {Path(self.screenshot.file_path).name}\n\nClose and reopen to see updated data."
            )
    
    def _on_reindex_error(self, error: str):
        """Handle reindex error."""
        colors = ThemeColors.get_colors()
        self.reindex_btn.setEnabled(True)
        self.reindex_btn.setText("Reindex")
        self.reindex_btn.setIcon(qta.icon('fa5s.redo', color=colors['text_secondary']))
        QMessageBox.critical(self, "Error", f"Failed to reindex: {error}")



class SettingsDialog(QDialog):
    """Dialog for configuring application settings."""
    
    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setWindowTitle("Settings")
        self.setMinimumSize(600, 500)
        self.setup_ui()
        
    def setup_ui(self):
        colors = ThemeColors.get_colors()
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {colors['bg']};
            }}
            QLabel {{
                color: {colors['text']};
            }}
            QLineEdit {{
                background-color: {colors['input_bg']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 6px;
            }}
            QListWidget {{
                background-color: {colors['input_bg']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: {colors['bg_hover']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {colors['border']};
            }}
            QGroupBox {{
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # General Settings Group
        general_group = QWidget()
        general_layout = QVBoxLayout(general_group)
        
        # API Config
        api_group = QWidget()
        api_layout = QVBoxLayout(api_group)
        
        self.ollama_url = QLineEdit(self.config_manager.config.api.ollama_url)
        self.vision_model = QLineEdit(self.config_manager.config.api.vision_model)
        self.embed_model = QLineEdit(self.config_manager.config.api.embed_model)
        
        self.add_form_row(api_layout, "Ollama URL:", self.ollama_url)
        self.add_form_row(api_layout, "Vision Model:", self.vision_model)
        self.add_form_row(api_layout, "Embedding Model:", self.embed_model)
        
        general_layout.addWidget(api_group)
        
        # Hybrid Search Weight Group
        hybrid_group = QWidget()
        hybrid_layout = QVBoxLayout(hybrid_group)
        
        weight_row = QHBoxLayout()
        weight_row.addWidget(QLabel("Hybrid Search Weight:"))
        self.hybrid_slider = QSlider(Qt.Horizontal)
        self.hybrid_slider.setRange(0, 100)
        self.hybrid_slider.setValue(int(self.config_manager.config.hybrid_search_weight * 100))
        self.hybrid_slider.setTickPosition(QSlider.TicksBelow)
        self.hybrid_slider.setTickInterval(25)
        self.hybrid_slider.valueChanged.connect(self.on_hybrid_weight_changed)
        weight_row.addWidget(self.hybrid_slider)
        
        self.weight_label = QLabel(f"{self.config_manager.config.hybrid_search_weight:.0%}")
        self.weight_label.setFixedWidth(50)
        weight_row.addWidget(self.weight_label)
        
        hybrid_layout.addLayout(weight_row)
        
        weight_hint = QLabel("← Sparse (BM25)              Dense (Semantic) →")
        weight_hint.setStyleSheet(f"color: {colors['text_muted']}; font-size: 10px;")
        weight_hint.setAlignment(Qt.AlignCenter)
        hybrid_layout.addWidget(weight_hint)
        
        general_layout.addWidget(hybrid_group)
        
        # Parallel Processing Group
        parallel_group = QWidget()
        parallel_layout = QVBoxLayout(parallel_group)
        
        parallel_row = QHBoxLayout()
        parallel_row.addWidget(QLabel("Parallel Processing:"))
        self.parallel_slider = QSlider(Qt.Horizontal)
        self.parallel_slider.setRange(1, 20)
        self.parallel_slider.setValue(self.config_manager.config.parallel_processing)
        self.parallel_slider.setTickPosition(QSlider.TicksBelow)
        self.parallel_slider.setTickInterval(5)
        self.parallel_slider.valueChanged.connect(self.on_parallel_changed)
        parallel_row.addWidget(self.parallel_slider)
        
        self.parallel_label = QLabel(f"{self.config_manager.config.parallel_processing}")
        self.parallel_label.setFixedWidth(30)
        parallel_row.addWidget(self.parallel_label)
        
        parallel_layout.addLayout(parallel_row)
        
        parallel_hint = QLabel("Higher = faster indexing, uses more VRAM. Start low.")
        parallel_hint.setStyleSheet(f"color: {colors['text_muted']}; font-size: 10px;")
        parallel_layout.addWidget(parallel_hint)
        
        general_layout.addWidget(parallel_group)
        
        # Folders Group
        folders_group = QWidget()
        folders_layout = QVBoxLayout(folders_group)
        folders_layout.addWidget(QLabel("Scan Folders:"))
        
        self.folders_list = QListWidget()
        for folder in self.config_manager.config.scan_folders:
            self.folders_list.addItem(folder.path)
            
        folders_layout.addWidget(self.folders_list)
        
        folder_btns = QHBoxLayout()
        add_btn = QPushButton("Add Folder")
        add_btn.setIcon(qta.icon('fa5s.plus', color=colors['text_secondary']))
        add_btn.clicked.connect(self.add_folder)
        
        remove_btn = QPushButton("Remove Folder")
        remove_btn.setIcon(qta.icon('fa5s.minus', color=colors['text_secondary']))
        remove_btn.clicked.connect(self.remove_folder)
        
        folder_btns.addWidget(add_btn)
        folder_btns.addWidget(remove_btn)
        folders_layout.addLayout(folder_btns)
        
        general_layout.addWidget(folders_group)
        
        # Danger Zone - Reset Database
        danger_group = QWidget()
        danger_layout = QVBoxLayout(danger_group)
        
        danger_label = QLabel("Danger Zone")
        danger_label.setStyleSheet(f"color: #dc3545; font-weight: bold;")
        danger_layout.addWidget(danger_label)
        
        reset_btn = QPushButton("Reset Database")
        reset_btn.setIcon(qta.icon('fa5s.trash-alt', color='#dc3545'))
        reset_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: #dc3545;
                border: 1px solid #dc3545;
                border-radius: 4px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: #dc3545;
                color: white;
            }}
        """)
        reset_btn.clicked.connect(self.reset_database)
        danger_layout.addWidget(reset_btn)
        
        general_layout.addWidget(danger_group)
        layout.addWidget(general_group)
        
        # Buttons
        btns = QHBoxLayout()
        btns.addStretch()
        
        save_btn = QPushButton("Save")
        save_btn.setIcon(qta.icon('fa5s.save', color=colors['text_secondary']))
        save_btn.clicked.connect(self.save_settings)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        btns.addWidget(save_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)
        
    def add_form_row(self, layout, label_text, widget):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(120)
        row.addWidget(label)
        row.addWidget(widget)
        layout.addLayout(row)
        
    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            # Check if already exists
            items = [self.folders_list.item(i).text() for i in range(self.folders_list.count())]
            if folder not in items:
                self.folders_list.addItem(folder)
                
    def remove_folder(self):
        current_row = self.folders_list.currentRow()
        if current_row >= 0:
            self.folders_list.takeItem(current_row)
            
    def save_settings(self):
        # Update config
        try:
            # Update API config
            self.config_manager.config.api.ollama_url = self.ollama_url.text()
            self.config_manager.config.api.vision_model = self.vision_model.text()
            self.config_manager.config.api.embed_model = self.embed_model.text()
            
            # Update hybrid weight
            self.config_manager.config.hybrid_search_weight = self.hybrid_slider.value() / 100.0
            
            # Update parallel processing
            self.config_manager.config.parallel_processing = self.parallel_slider.value()
            
            # Update folders (rebuild list)
            # Note: This implementation assumes all added folders include subfolders for simplicity
            # A more complex UI would be needed to toggle per-folder settings
            new_folders = []
            for i in range(self.folders_list.count()):
                path = self.folders_list.item(i).text()
                # Preserve existing settings if possible, else default
                existing = next((f for f in self.config_manager.config.scan_folders if f.path == path), None)
                if existing:
                    new_folders.append(existing)
                else:
                    from ..core.config import ScanFolder
                    new_folders.append(ScanFolder(path=path, include_subfolders=True))
            
            self.config_manager.config.scan_folders = new_folders
            self.config_manager.save()
            
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")
    
    def on_hybrid_weight_changed(self, value):
        self.weight_label.setText(f"{value}%")
    
    def on_parallel_changed(self, value):
        self.parallel_label.setText(str(value))
    
    def reset_database(self):
        reply = QMessageBox.warning(
            self,
            "Reset Database",
            "This will delete all indexed data including:\n\n"
            "• SQLite database\n"
            "• Vector store\n"
            "• BM25 sparse index\n\n"
            "You will need to re-index all screenshots.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Close connections on parent window to release file locks
                parent = self.parent()
                if hasattr(parent, 'vector_store'):
                    parent.vector_store.close()
                
                import shutil
                data_dir = self.config_manager.data_dir
                if data_dir.exists():
                    shutil.rmtree(data_dir)
                
                # Ensure data directory is recreated
                data_dir.mkdir(parents=True, exist_ok=True)
                
                # Re-initialize parent services
                if hasattr(parent, 'db'):
                    # Re-initialize DB schema since file was deleted
                    parent.db._init_db()
                
                if hasattr(parent, 'vector_store'):
                    # Re-create vector store instance
                    from ..services.vector_store import VectorStore
                    parent.vector_store = VectorStore(self.config_manager.vector_store_path)
                    
                    # Update references in other components
                    if hasattr(parent, 'search_engine'):
                        parent.search_engine.vector_store = parent.vector_store
                    if hasattr(parent, 'processor'):
                        parent.processor.vector_store = parent.vector_store
                
                # Reset sparse embedding if it exists
                if hasattr(parent, 'sparse_embedding'):
                    from ..services.sparse_embedding import SparseEmbeddingService
                    parent.sparse_embedding = SparseEmbeddingService()
                    if hasattr(parent, 'search_engine'):
                        parent.search_engine.sparse_embedding = parent.sparse_embedding
                    if hasattr(parent, 'processor'):
                        parent.processor.sparse_embedding = parent.sparse_embedding

                QMessageBox.information(
                    self,
                    "Database Reset",
                    "Database has been reset. You can now re-index your screenshots."
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to reset database: {str(e)}")


class ResultCard(QFrame):
    """A card displaying a single search result."""
    
    clicked = Signal(object)
    
    def __init__(self, result: SearchResult, parent=None):
        super().__init__(parent)
        self.result = result
        self.setup_ui()
    
    def setup_ui(self):
        colors = ThemeColors.get_colors()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet(f"""
            ResultCard {{
                background-color: {colors['bg_alt']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
                margin: 4px;
            }}
            ResultCard:hover {{
                background-color: {colors['bg_hover']};
                border-color: {colors['border_hover']};
            }}
        """)
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Thumbnail
        thumbnail = QLabel()
        thumbnail.setFixedSize(80, 80)
        thumbnail.setStyleSheet(f"background-color: {colors['border']}; border-radius: 4px;")
        
        # Try to load the actual image as thumbnail
        file_path = Path(self.result.screenshot.file_path)
        if file_path.exists():
            pixmap = QPixmap(str(file_path))
            if not pixmap.isNull():
                scaled = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                thumbnail.setPixmap(scaled)
                thumbnail.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(thumbnail)
        
        # Info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)
        
        # Filename
        name_label = QLabel(file_path.name)
        name_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        name_label.setStyleSheet(f"color: {colors['text']}; background: transparent;")
        info_layout.addWidget(name_label)
        
        # Score and type
        score_label = QLabel(f"Score: {self.result.score:.3f} | {self.result.search_type}")
        score_label.setStyleSheet(f"color: {colors['text_secondary']}; font-size: 10px; background: transparent;")
        info_layout.addWidget(score_label)
        
        # App name
        app_label = QLabel(self.result.screenshot.app_name or "Unknown app")
        app_label.setStyleSheet(f"color: {colors['text_muted']}; font-size: 10px; background: transparent;")
        info_layout.addWidget(app_label)
        
        # Preview of OCR text
        if self.result.screenshot.ocr_text:
            preview = self.result.screenshot.ocr_text[:100].replace('\n', ' ')
            if len(self.result.screenshot.ocr_text) > 100:
                preview += "..."
            ocr_label = QLabel(preview)
            ocr_label.setStyleSheet(f"color: {colors['text_muted']}; font-size: 9px; background: transparent;")
            ocr_label.setWordWrap(True)
            info_layout.addWidget(ocr_label)
        
        info_layout.addStretch()
        layout.addLayout(info_layout, 1)
        
        # Buttons layout (Right side)
        btns_layout = QVBoxLayout()
        btns_layout.setSpacing(8)
        btns_layout.addStretch()
        
        # Open button
        open_btn = QPushButton("Open")
        open_btn.setIcon(qta.icon('fa5s.external-link-alt', color='white'))
        open_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['accent']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {colors['accent_hover']};
            }}
        """)
        open_btn.clicked.connect(self.on_open_clicked)
        btns_layout.addWidget(open_btn)
        
        # Info button
        info_btn = QPushButton()
        info_btn.setIcon(qta.icon('fa5s.info-circle', color=colors['text_secondary']))
        info_btn.setFixedSize(30, 30)
        info_btn.setToolTip("View Image Info")
        info_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['bg_hover']};
                border: 1px solid {colors['border']};
                border-radius: 15px;
            }}
            QPushButton:hover {{
                background-color: {colors['border']};
                border-color: {colors['text_muted']};
            }}
        """)
        info_btn.clicked.connect(self.on_info_clicked)

        
        # Container for info button to align it to bottom right or just add it
        # User wanted "(!) on the bottom right".
        # If we put it in a HBox with Open button, it works.
        
        # Let's restructure the right side to be a VBox with Open button, 
        # and maybe the info button is small and next to it?
        # Or maybe:
        # [Open]
        #      [!]
        
        # Let's try HBox for buttons
        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)
        action_layout.addWidget(open_btn)
        action_layout.addWidget(info_btn)
        
        # Add to main layout (which is HBox)
        # We want these at the bottom right.
        # The main layout has: Thumbnail | Info | [Buttons]
        # To align to bottom, we can wrap [Buttons] in a VBox with stretch at top.
        
        right_col = QVBoxLayout()
        right_col.addStretch()
        right_col.addLayout(action_layout)
        
        layout.addLayout(right_col)
    
    def on_info_clicked(self):
        dialog = ImageInfoDialog(self.result.screenshot, self)
        dialog.exec()
    
    def on_open_clicked(self):
        self.clicked.emit(self.result)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.result)
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Searchable Screenshots")
        self.setMinimumSize(900, 700)
        
        # Initialize services
        self.config_manager = ConfigManager()
        self.db = Database(self.config_manager.db_path)
        
        api_config = self.config_manager.config.api
        app_config = self.config_manager.config
        self.ocr = OCRService()
        self.vision = VisionService(
            api_config.ollama_url, 
            api_config.vision_model,
            prompt_style=app_config.vision_prompt_style
        )
        self.embedding = EmbeddingService(
            api_config.ollama_url, 
            api_config.embed_model,
            cache_enabled=app_config.embedding_cache_enabled
        )
        
        # Detect embedding dimension and create vector store with correct dimension
        embedding_dim = None
        if self.embedding.dimension:
            embedding_dim = self.embedding.dimension
        else:
            # Try to get dimension by embedding a test string
            test_embedding = self.embedding.embed("test")
            if test_embedding:
                embedding_dim = len(test_embedding)
        
        self.vector_store = VectorStore(
            self.config_manager.vector_store_path,
            dimension=embedding_dim
        )
        
        # Initialize sparse embedding service (BM25)
        self.sparse_embedding = SparseEmbeddingService()
        if self.config_manager.sparse_index_path.exists():
            self.sparse_embedding.load(self.config_manager.sparse_index_path)
        
        self.reranker = None
        if self.config_manager.config.use_reranker:
            self.reranker = RerankerService()
        
        self.search_engine = SearchEngine(
            self.db,
            self.vector_store,
            self.embedding,
            sparse_embedding=self.sparse_embedding,
            reranker=self.reranker,
            use_reranker=self.config_manager.config.use_reranker,
            hybrid_weight=self.config_manager.config.hybrid_search_weight,
            hybrid_normalization=app_config.hybrid_normalization,
            use_query_expansion=app_config.use_query_expansion,
        )
        
        self.processor = ScreenshotProcessor(
            self.config_manager,
            self.db,
            self.vector_store,
            self.ocr,
            self.vision,
            self.embedding,
            sparse_embedding=self.sparse_embedding,
        )
        
        self.results = []
        self.is_indexing = False
        self.cancel_indexing = False
        
        self.setup_ui()
        self.apply_styles()
        self.update_status()
    
    def setup_ui(self):
        colors = ThemeColors.get_colors()
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        
        self.add_folder_btn = QPushButton("Add Folder")
        self.add_folder_btn.setIcon(qta.icon('fa5s.folder-plus', color=colors['text_secondary']))
        self.add_folder_btn.clicked.connect(self.on_add_folder)
        toolbar.addWidget(self.add_folder_btn)
        
        self.index_btn = QPushButton("Index Now")
        self.index_btn.setIcon(qta.icon('fa5s.sync', color=colors['text_secondary']))
        self.index_btn.clicked.connect(self.on_index)
        toolbar.addWidget(self.index_btn)
        
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setIcon(qta.icon('fa5s.cog', color=colors['text_secondary']))
        self.settings_btn.clicked.connect(self.on_settings)
        toolbar.addWidget(self.settings_btn)
        
        toolbar.addStretch()
        
        self.status_label = QLabel("Ready")
        toolbar.addWidget(self.status_label)
        
        layout.addLayout(toolbar)
        
        # Search bar
        search_layout = QHBoxLayout()
        search_layout.setSpacing(8)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search screenshots... (use quotes for exact match)")
        self.search_input.returnPressed.connect(self.on_search)
        search_layout.addWidget(self.search_input, 1)
        
        self.search_btn = QPushButton("Search")
        self.search_btn.setIcon(qta.icon('fa5s.search', color=colors['text_secondary']))
        self.search_btn.clicked.connect(self.on_search)
        search_layout.addWidget(self.search_btn)
        
        layout.addLayout(search_layout)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setContentsMargins(8, 8, 8, 8)
        self.results_layout.setSpacing(8)
        self.results_layout.addStretch()
        
        self.scroll_area.setWidget(self.results_widget)
        layout.addWidget(self.scroll_area, 1)
        
        # Footer
        footer = QHBoxLayout()
        self.indexed_label = QLabel("")
        footer.addWidget(self.indexed_label)
        footer.addStretch()
        layout.addLayout(footer)
    
    def apply_styles(self):
        colors = ThemeColors.get_colors()
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {colors['bg']};
            }}
            QPushButton {{
                background-color: {colors['bg_hover']};
                border: 1px solid {colors['border']};
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                color: {colors['text']};
            }}
            QPushButton:hover {{
                background-color: {colors['border']};
                border-color: {colors['text_muted']};
            }}
            QLineEdit {{
                border: 2px solid {colors['border']};
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 14px;
                background-color: {colors['input_bg']};
                color: {colors['text']};
            }}
            QLineEdit:focus {{
                border-color: {colors['accent']};
            }}
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {colors['border']};
                height: 6px;
            }}
            QProgressBar::chunk {{
                background-color: {colors['accent']};
                border-radius: 4px;
            }}
            QScrollArea {{
                border: none;
                background-color: {colors['bg']};
            }}
            QLabel {{
                color: {colors['text']};
            }}
        """)
        self.results_widget.setStyleSheet(f"background-color: {colors['bg']};")
        self.status_label.setStyleSheet(f"color: {colors['text_secondary']};")
        self.indexed_label.setStyleSheet(f"color: {colors['text_muted']}; font-size: 11px;")
    
    def update_status(self):
        count = self.db.get_count()
        folders = len(self.config_manager.config.scan_folders)
        self.indexed_label.setText(f"{count} screenshots indexed | {folders} folders configured")
    
    def on_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select a folder to scan")
        if folder:
            self.config_manager.add_scan_folder(folder, include_subfolders=True)
            QMessageBox.information(self, "Folder Added", f"Added folder: {folder}")
            self.update_status()
    
    def on_index(self):
        if self.is_indexing:
            # If already indexing, cancel it
            self.cancel_indexing = True
            return
        
        self.is_indexing = True
        self.cancel_indexing = False
        colors = ThemeColors.get_colors()
        
        # Change button to Cancel
        self.index_btn.setText("Cancel")
        self.index_btn.setIcon(qta.icon('fa5s.stop', color='white'))
        self.index_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #dc3545;
                border: 1px solid #dc3545;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                color: white;
            }}
            QPushButton:hover {{
                background-color: #c82333;
            }}
        """)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Discovering files...")
        
        # Create worker signals
        signals = WorkerSignals()
        signals.finished.connect(self.on_index_complete)
        signals.error.connect(self.on_index_error)
        signals.progress.connect(self.on_index_progress)
        
        def do_index():
            try:
                import asyncio
                
                def progress_callback(progress):
                    signals.progress.emit(
                        progress.current_index,
                        progress.total_files,
                        progress.current_file,
                        progress.status
                    )
                
                def cancel_check():
                    return self.cancel_indexing
                
                # Run async processor in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    stats = loop.run_until_complete(
                        self.processor.process_all_async(
                            concurrency=self.config_manager.config.parallel_processing,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check
                        )
                    )
                finally:
                    loop.close()
                
                signals.finished.emit(stats)
            except Exception as e:
                signals.error.emit(str(e))
        
        thread = threading.Thread(target=do_index, daemon=True)
        thread.start()
    
    def on_index_progress(self, current: int, total: int, filename: str, status: str):
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            # Show filename (truncated) and progress
            fname = Path(filename).name if filename else ""
            if len(fname) > 40:
                fname = fname[:37] + "..."
            
            if status == "api_error":
                self.status_label.setText(f"API Error - Skipped {current}/{total}: {fname}")
            elif status == "failed":
                self.status_label.setText(f"Failed {current}/{total}: {fname}")
            else:
                self.status_label.setText(f"Processing {current}/{total}: {fname}")
    
    def on_index_complete(self, stats: ProcessingStats):
        self.is_indexing = False
        self.cancel_indexing = False
        
        # Restore button style
        colors = ThemeColors.get_colors()
        self.index_btn.setText("Index Now")
        self.index_btn.setIcon(qta.icon('fa5s.sync', color=colors['text_secondary']))
        self.index_btn.setStyleSheet("")  # Clear inline style
        self.apply_styles()  # Re-apply theme styles
        self.progress_bar.setVisible(False)
        
        # Save sparse embedding index after indexing
        if self.sparse_embedding and self.sparse_embedding.is_fitted:
            self.sparse_embedding.save(self.config_manager.sparse_index_path)
        
        # Build status message
        msg = f"Done: {stats.new_indexed} new, {stats.updated} updated, {stats.skipped} skipped"
        if stats.failed > 0:
            msg += f", {stats.failed} failed (will retry on next index)"
        self.status_label.setText(msg)
        self.update_status()
    
    def on_index_error(self, error: str):
        self.is_indexing = False
        self.cancel_indexing = False
        
        # Restore button style
        colors = ThemeColors.get_colors()
        self.index_btn.setText("Index Now")
        self.index_btn.setIcon(qta.icon('fa5s.sync', color=colors['text_secondary']))
        self.index_btn.setStyleSheet("")  # Clear inline style
        self.apply_styles()  # Re-apply theme styles
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error}")
        QMessageBox.critical(self, "Indexing Error", f"Failed to index: {error}")
    
    def on_settings(self):
        dialog = SettingsDialog(self.config_manager, self)
        if dialog.exec():
            # Re-initialize services with new model names if they changed
            api_config = self.config_manager.config.api
            old_vision_model = self.vision.model
            old_embed_model = self.embedding.model
            
            # Re-initialize vision service if model changed
            app_config = self.config_manager.config
            if old_vision_model != api_config.vision_model or self.vision.ollama_url != api_config.ollama_url:
                self.vision = VisionService(
                    api_config.ollama_url, 
                    api_config.vision_model,
                    prompt_style=app_config.vision_prompt_style
                )
                # Update processor reference
                self.processor.vision = self.vision
            
            # Re-initialize embedding service if model changed
            if old_embed_model != api_config.embed_model or self.embedding.ollama_url != api_config.ollama_url:
                self.embedding = EmbeddingService(
                    api_config.ollama_url, 
                    api_config.embed_model,
                    cache_enabled=app_config.embedding_cache_enabled
                )
                # Update processor and search engine references
                self.processor.embedding = self.embedding
                self.search_engine.embedding = self.embedding
            
            # Update search engine config
            self.search_engine.hybrid_normalization = app_config.hybrid_normalization
            self.search_engine.use_query_expansion = app_config.use_query_expansion
                
                # Note: Changing embedding model may change dimension, but we don't reinitialize
                # vector store here as it would require re-indexing all data
            
            # Update search engine with new hybrid weight
            self.search_engine.hybrid_weight = self.config_manager.config.hybrid_search_weight
            self.update_status()
    
    def on_search(self):
        query = self.search_input.text().strip()
        if not query:
            return
        
        self.status_label.setText(f"Searching: {query}...")
        QApplication.processEvents()
        
        try:
            self.results = self.search_engine.search(query, limit=20)
            self.display_results()
            self.status_label.setText(f"Found {len(self.results)} results")
        except Exception as e:
            self.status_label.setText(f"❌ Search error: {e}")
            QMessageBox.warning(self, "Search Error", str(e))
    
    def display_results(self):
        # Clear existing results
        while self.results_layout.count() > 1:  # Keep the stretch
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not self.results:
            colors = ThemeColors.get_colors()
            no_results = QLabel("No results found. Try a different search term.")
            no_results.setStyleSheet(f"color: {colors['text_muted']}; padding: 40px; font-size: 14px;")
            no_results.setAlignment(Qt.AlignCenter)
            self.results_layout.insertWidget(0, no_results)
            return

        
        for result in self.results:
            card = ResultCard(result)
            card.clicked.connect(self.on_result_clicked)
            self.results_layout.insertWidget(self.results_layout.count() - 1, card)
    
    def on_result_clicked(self, result: SearchResult):
        file_path = result.screenshot.file_path
        if sys.platform == "win32":
            os.startfile(file_path)
        elif sys.platform == "darwin":
            os.system(f'open "{file_path}"')
        else:
            os.system(f'xdg-open "{file_path}"')


def run_app():
    """Entry point for the application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Use system default palette - no forced light theme
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()

