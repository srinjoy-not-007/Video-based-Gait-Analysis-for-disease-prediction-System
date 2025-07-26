import sys
import os
import joblib
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QTextEdit, QProgressBar, QListWidget,
    QListWidgetItem, QLineEdit, QMessageBox, QComboBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

import database
import gait_processing
import prediction_model

MODEL_PATH = r"D:\GUI Gait parameters\models\random_forest_model.pkl"
REPORT_FOLDER = r"D:\GUI Gait parameters\reports"
os.makedirs(REPORT_FOLDER, exist_ok=True)


with open(MODEL_PATH, 'rb') as f:
    disease_model = joblib.load(f)

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login - Gait Analysis")
        self.setGeometry(100, 100, 400, 350)
        self.apply_styles()

        layout = QVBoxLayout()
        title = QLabel("Please Log In")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_input)

        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.handle_login)
        layout.addWidget(self.login_button)

        self.register_button = QPushButton("Register")
        self.register_button.clicked.connect(self.open_register_window)
        layout.addWidget(self.register_button)

        self.setLayout(layout)

    def handle_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        if database.validate_user(username, password):
            self.analysis_window = GaitAnalysisApp(username=username)
            self.analysis_window.show()
            self.close()
        else:
            QMessageBox.warning(self, "Login Failed", "Incorrect username or password.")

    def open_register_window(self):
        self.register_window = RegisterWindow()
        self.register_window.show()

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget { background-color: #f5fdf7; color: #333; font-family: Arial; font-size: 14px; }
            QPushButton { background-color: #4CAF50; color: white; padding: 10px; border-radius: 8px; }
            QPushButton:hover { background-color: #45a049; }
            QLineEdit { background-color: #ffffff; border: 1px solid #ccc; border-radius: 8px; padding: 8px; }
        """)

class RegisterWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Register - Gait Analysis")
        self.setGeometry(150, 150, 400, 400)
        self.apply_styles()

        layout = QVBoxLayout()
        title = QLabel("Register New User")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter Username")
        layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_input)

        self.role_selector = QComboBox()
        self.role_selector.addItems(["patient", "doctor"])
        layout.addWidget(self.role_selector)

        self.register_button = QPushButton("Register")
        self.register_button.clicked.connect(self.handle_register)
        layout.addWidget(self.register_button)

        self.setLayout(layout)

    def handle_register(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        role = self.role_selector.currentText()
        if database.add_user(username, password, role):
            QMessageBox.information(self, "Success", "User registered successfully!")
            self.close()
        else:
            QMessageBox.warning(self, "Error", "Username already exists.")

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget { background-color: #f5fdf7; color: #333; font-family: Arial; font-size: 14px; }
            QPushButton { background-color: #4CAF50; color: white; padding: 10px; border-radius: 8px; }
            QLineEdit, QComboBox { background-color: #ffffff; border: 1px solid #ccc; border-radius: 8px; padding: 8px; }
        """)

# Gait Analysis App
class GaitAnalysisApp(QMainWindow):

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #f5fdf7;
                color: #333;
                font-family: Arial;
                font-size: 14px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QTextEdit, QListWidget {
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 10px;
            }
            QProgressBar {
                height: 20px;
                border: 1px solid #ccc;
                border-radius: 10px;
                background-color: #eee;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 10px;
            }
        """)



    def __init__(self, username=None):
        super().__init__()
        self.username = username
        self.previous_data = []
        self.video_file = None
        self.gait_parameters = None
        self.predictions = None

        self.setWindowTitle(f"Gait Analysis - {self.username}")
        self.setGeometry(100, 100, 1000, 700)
        self.setup_ui()

    def setup_ui(self):
        self.apply_styles()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        self.title_label = QLabel("Gait Analysis and Diagnosis")
        self.title_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        btn_layout = QHBoxLayout()
        self.upload_button = QPushButton("Upload Gait Video")
        self.upload_button.clicked.connect(self.upload_video)
        btn_layout.addWidget(self.upload_button)

        self.analyze_button = QPushButton("Analyze Gait")
        self.analyze_button.clicked.connect(self.analyze_gait)
        btn_layout.addWidget(self.analyze_button)

        self.compare_button = QPushButton("Compare Progress")
        self.compare_button.clicked.connect(self.compare_progress)
        btn_layout.addWidget(self.compare_button)

        self.report_button = QPushButton("Export Report (PDF)")
        self.report_button.clicked.connect(self.export_report)
        btn_layout.addWidget(self.report_button)

        self.logout_button = QPushButton("Logout")
        self.logout_button.clicked.connect(self.logout)
        btn_layout.addWidget(self.logout_button)

        layout.addLayout(btn_layout)

        self.video_preview = QLabel("No video uploaded yet.")
        self.video_preview.setFixedSize(400, 300)
        self.video_preview.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        layout.addWidget(self.video_preview, alignment=Qt.AlignCenter)

        results_layout = QHBoxLayout()
        self.gait_params = QTextEdit()
        self.gait_params.setReadOnly(True)
        results_layout.addWidget(self.gait_params)

        self.prediction_list = QListWidget()
        results_layout.addWidget(self.prediction_list)
        layout.addLayout(results_layout)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

    def upload_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Gait Video", "", "Video Files (*.mp4 *.avi *.mov)", options=options)
        if file_name:
            self.video_file = file_name
            self.video_preview.setText(f"Uploaded: {os.path.basename(file_name)}")

    def analyze_gait(self):
        if not self.video_file:
            QMessageBox.warning(self, "No Video", "Please upload a video.")
            return

        self.progress_bar.setValue(10)
        gait_params = gait_processing.extract_gait_parameters(self.video_file)

        if gait_params is None:
            QMessageBox.warning(self, "Error", "Failed to extract gait parameters.")
            return

        self.progress_bar.setValue(60)
        prediction_label, risk_level = prediction_model.predict_disease(gait_params, disease_model)
        self.predictions = prediction_label
        self.gait_parameters = gait_params
        self.previous_data.append(gait_params)

        param_text = "\n".join([f"{k}: {v:.2f}" for k, v in gait_params.items()])
        self.gait_params.setText(param_text)

        self.prediction_list.clear()
        self.prediction_list.addItem(f"Prediction: {prediction_label}")
        self.prediction_list.addItem(f"Risk Level: {risk_level}")

        self.progress_bar.setValue(100)

    def compare_progress(self):
        if len(self.previous_data) < 2:
            QMessageBox.warning(self, "No Data", "Upload & analyze at least two videos.")
            return

        prev = list(self.previous_data[-2].values())
        curr = list(self.previous_data[-1].values())
        labels = list(self.previous_data[-1].keys())

        plt.plot(labels, prev, label='Previous', marker='o')
        plt.plot(labels, curr, label='Current', marker='o')
        plt.legend()
        plt.title("Progress Comparison")
        graph_path = os.path.join(REPORT_FOLDER, f"{self.username}_progress.png")
        plt.savefig(graph_path)
        plt.close()

    def export_report(self):
        if not self.gait_parameters:
            QMessageBox.warning(self, "No Analysis", "Analyze gait data first.")
            return

        report_path = os.path.join(REPORT_FOLDER, f"{self.username}_report.pdf")
        c = canvas.Canvas(report_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 750, "Gait Analysis Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, 730, f"Patient: {self.username}")
        c.drawString(50, 710, f"Date: {datetime.now().strftime('%Y-%m-%d')}")

        y = 690
        for param, value in self.gait_parameters.items():
            c.drawString(50, y, f"{param}: {value:.2f}")
            y -= 20

        c.drawString(50, y, f"Prediction: {self.predictions}")
        y -= 40

        graph_path = os.path.join(REPORT_FOLDER, f"{self.username}_progress.png")
        if os.path.exists(graph_path):
            c.drawImage(graph_path, 50, y - 200, width=500, height=200)

        c.save()
        QMessageBox.information(self, "Report", f"Report saved: {report_path}")

    def logout(self):
        self.close()
        self.login_window = LoginWindow()
        self.login_window.show()

# RUN APP
if __name__ == "__main__":
    database.initialize_db()
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec_())
