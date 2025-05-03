import sys
import os
import glob
import socket
import subprocess
import time
import argparse
import numpy as np
import threading
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QMessageBox, QScrollArea, QDialog, QInputDialog, QListWidget
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

RAW_DIR = "../assets/face_information"
MAX_PEOPLE = 5

class FaceManager(QWidget):
    update_image_signal = pyqtSignal(QImage)

    def __init__(self, server_port):
        super().__init__()
        self.setWindowTitle("Face Registration Manager")
        self.selected_file = None
        self.message_box = None
        self.server_port = server_port
        self.streaming_thread = None
        self.server_process = None

        self.image_label = QLabel("Processed image appears here")
        self.image_label.setFixedSize(480, 360)
        self.image_label.setStyleSheet("border: 1px solid gray")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.user_list = QListWidget()
        self.user_list.setFixedWidth(250)
        self.user_list.itemClicked.connect(self.handle_item_clicked)

        self.register_btn = QPushButton("Register")
        self.delete_btn = QPushButton("Delete")
        self.clear_btn = QPushButton("Delete All")

        self.register_btn.clicked.connect(self.register)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.clear_btn.clicked.connect(self.delete_all)

        self.update_image_signal.connect(self.update_image_label)

        self.init_ui()
        self.update_user_list()

    def init_ui(self):
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Registered Users:"))
        left_layout.addWidget(self.user_list)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.register_btn)
        button_layout.addWidget(self.delete_btn)
        button_layout.addWidget(self.clear_btn)
        right_layout.addLayout(button_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    def update_image_label(self, q_img):
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def update_user_list(self):
        self.user_list.clear()
        files = glob.glob(os.path.join(RAW_DIR, "*.raw"))
        for path in files:
            self.user_list.addItem(os.path.basename(path))

    def handle_item_clicked(self, item):
        self.selected_file = item.text()

    def check_file_created(self):
        if os.path.exists(self.target_path):
            self.update_user_list()
            self.show_temp_message("[INFO] Registration complete", 2000)
            self.selected_file = None
        else:
            QTimer.singleShot(500, self.check_file_created)

    def receive_frames(self):
        def recv_all(sock, length):
            data = b''
            while len(data) < length:
                packet = sock.recv(length - len(data))
                if not packet:
                    return None
                data += packet
            return data

        def run():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            for _ in range(10):
                try:
                    s.connect(("127.0.0.1", self.server_port))
                    break
                except ConnectionRefusedError:
                    time.sleep(0.5)
            else:
                print("[ERROR] Could not connect to server")
                return

            while True:
                length_bytes = recv_all(s, 4)
                if not length_bytes:
                    break
                length = int.from_bytes(length_bytes, 'big')
                if length <= 0 or length > 10_000_000:
                    break
                frame_data = recv_all(s, length)
                if not frame_data:
                    break
                img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    h, w, ch = img.shape
                    q_img = QImage(img.data, w, h, ch * w, QImage.Format_BGR888)
                    self.update_image_signal.emit(q_img)

        self.streaming_thread = threading.Thread(target=run, daemon=True)
        self.streaming_thread.start()

    def register(self):
        if len(glob.glob(os.path.join(RAW_DIR, "*.raw"))) >= MAX_PEOPLE:
            self.show_temp_message("[INFO] Max users registered.", 3000)
            return

        initial, ok = QInputDialog.getText(self, "Enter Initial", "Please enter your initials:")
        if not ok or not initial.strip():
            self.show_message("[INFO] Registration cancelled.")
            return

        initial = initial.strip()
        self.target_path = os.path.join(RAW_DIR, f"{initial}.raw")
        if os.path.exists(self.target_path):
            self.show_message("[INFO] Initial already exists.")
            return

        try:
            self.server_process = subprocess.Popen([
                "../bin/face_registeration",
                "--port", str(self.server_port),
                "--cam", str(0),
                "--name", initial
            ])
            self.receive_frames()
            QTimer.singleShot(500, self.check_file_created)
        except Exception as e:
            self.show_message(f"[ERROR] Failed to start server: {e}")

    def closeEvent(self, event):
        if self.server_process and self.server_process.poll() is None:
            self.server_process.terminate()
            time.sleep(1)
            self.server_process.kill()
        event.accept()

    def delete_selected(self):
        if self.selected_file:
            try:
                os.remove(os.path.join(RAW_DIR, self.selected_file))
                self.selected_file = None
                self.update_user_list()
            except Exception as e:
                self.show_message(f"[ERROR] Delete failed: {e}")

    def delete_all(self):
        for path in glob.glob(os.path.join(RAW_DIR, "*.raw")):
            os.remove(path)
        self.selected_file = None
        self.update_user_list()

    def show_temp_message(self, text, duration_ms):
        if self.message_box and self.message_box.isVisible():
            self.message_box.close()
            self.message_box = None

        self.message_box = QDialog(self)
        self.message_box.setWindowTitle("Notice")
        self.message_box.setModal(False)
        self.message_box.resize(400, 150)

        layout = QVBoxLayout()
        label = QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)
        self.message_box.setLayout(layout)

        QTimer.singleShot(duration_ms, self.message_box.close)
        self.message_box.show()

    def show_message(self, text):
        QMessageBox.information(self, "Notice", text)

if __name__ == "__main__":
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    parser = argparse.ArgumentParser(description="Face Manager GUI")
    parser.add_argument("--port", type=int, default=9999, help="Port number")
    args = parser.parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)

    app = QApplication(sys.argv)
    manager = FaceManager(server_port=args.port)
    manager.resize(800, 400)
    manager.show()
    sys.exit(app.exec_())
