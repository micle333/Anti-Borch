import sys
import os
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
    QLabel,
    QProgressBar,
    QMainWindow,
    QSizePolicy,
    QHBoxLayout,
    QLineEdit
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage

import cv2
import numpy as np
from skimage.transform import resize
import tensorflow as tf

CLASSES = 2
COLORS = ['black', 'red']
SAMPLE_SIZE = (512, 512)
OUTPUT_SIZE = (1080, 1920)

def input_layer():
    return tf.keras.layers.Input(shape=SAMPLE_SIZE + (3,))

def downsample_block(filters, size, batch_norm=True):
    initializer = tf.keras.initializers.GlorotNormal()
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    if batch_norm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample_block(filters, size, dropout=False):
    initializer = tf.keras.initializers.GlorotNormal()
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if dropout:
        result.add(tf.keras.layers.Dropout(0.25))
    result.add(tf.keras.layers.ReLU())
    return result

def output_layer(size):
    initializer = tf.keras.initializers.GlorotNormal()
    return tf.keras.layers.Conv2DTranspose(CLASSES, size, strides=2, padding='same', kernel_initializer=initializer, activation='sigmoid')

inp_layer = input_layer()

downsample_stack = [
    downsample_block(64, 4, batch_norm=False),
    downsample_block(128, 4),
    downsample_block(256, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
]

upsample_stack = [
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(256, 4),
    upsample_block(128, 4),
    upsample_block(64, 4)
]

out_layer = output_layer(4)

x = inp_layer
downsample_skips = []

for block in downsample_stack:
    x = block(x)
    downsample_skips.append(x)

downsample_skips = reversed(downsample_skips[:-1])

for up_block, down_block in zip(upsample_stack, downsample_skips):
    x = up_block(x)
    x = tf.keras.layers.Concatenate()([x, down_block])

out_layer = out_layer(x)

unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)

def dice_mc_metric(a, b):
    a = tf.unstack(a, axis=3)
    b = tf.unstack(b, axis=3)
    dice_summ = 0
    for i, (aa, bb) in enumerate(zip(a, b)):
        numenator = 2 * tf.math.reduce_sum(aa * bb) + 1
        denomerator = tf.math.reduce_sum(aa + bb) + 1
        dice_summ += numenator / denomerator
    avg_dice = dice_summ / CLASSES
    return avg_dice

def dice_mc_loss(a, b):
    return 1 - dice_mc_metric(a, b)

def dice_bce_mc_loss(a, b):
    return 0.3 * dice_mc_loss(a, b) + tf.keras.losses.binary_crossentropy(a, b)

rgb_colors = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 165, 0),
    (255, 192, 203),
    (0, 255, 255),
    (255, 0, 255)
]

unet_like.compile(optimizer='adam', loss=[dice_bce_mc_loss], metrics=[dice_mc_metric])
unet_like.load_weights('/Users/dronoved/AntiBorchevic/m_model.h5')

def frame_processing(frame):
    sample = resize(frame, SAMPLE_SIZE)
    predict = unet_like.predict(sample.reshape((1,) + SAMPLE_SIZE + (3,)))
    predict = predict.reshape(SAMPLE_SIZE + (CLASSES,))
    predict_layer = predict[:, :, 1]
    predict_layer_resized = resize(predict_layer, frame.shape[:2])
    predict_layer_resized = np.where(predict_layer_resized > 0.99999997, predict_layer_resized, 0)
    predict_layer_resized = (predict_layer_resized * 125).astype(np.uint8)
    heatmap = cv2.applyColorMap(predict_layer_resized, cv2.COLORMAP_HOT)
    overlayed_frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)
    return overlayed_frame

class Worker(QThread):
    progress_changed = Signal(int)
    finished = Signal(str)
    frame_ready = Signal(np.ndarray)

    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name
        self.output_path = None
        self.cap = None
        self.out = None

    def run(self):
        directory, filename = os.path.split(self.file_name)
        _, extension = os.path.splitext(filename)
        new_filename = filename.replace(extension, f"_processed{extension}")
        self.output_path = os.path.join(directory, new_filename)

        video_path = self.file_name
        fps = 15
        frame_size = (1920, 1080)
        self.cap = cv2.VideoCapture(video_path)
        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.output_path, fourcc, fps, frame_size)

        frame_num = 1

        while True:
            self.progress_changed.emit(frame_num * 100 // length)
            frame_num += 1
            ret, frame = self.cap.read()
            if not ret:
                break

            overlayed_frame = frame_processing(frame)
            self.out.write(overlayed_frame)
            self.frame_ready.emit(overlayed_frame)

        self.cap.release()
        self.out.release()
        self.finished.emit(self.output_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.file_name = ''
        self.setWindowTitle("Anti Borshevic")

        self.setStyleSheet("background-color: rgb(0,0,0);")  # Полупрозрачный серый фон

        self.btn_browse = QPushButton("Выбрать файл")
        self.btn_browse.clicked.connect(self.browse_file)

        self.file_path = QLineEdit('')
        # Горизонтальный макет для кнопки и текстового поля
        browse_layout = QHBoxLayout()
        browse_layout.addWidget(self.file_path)
        browse_layout.addWidget(self.btn_browse)

        self.btn_process = QPushButton("Обработать")
        self.btn_process.clicked.connect(self.process_video)
        self.btn_process.setEnabled(False)
        self.btn_process.setStyleSheet("background-color: blue; color: white; border-radius: 5px;  width: 10px; height: 20px;")  # Синий цвет кнопки

        self.export_label = QLabel('')
        self.progress_bar = QProgressBar()
        self.frame_display = QLabel()
        self.frame_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        fixed_width = 640
        fixed_height = 360
        self.frame_display.setFixedSize(fixed_width, fixed_height)
        self.frame_display.setAlignment(Qt.AlignCenter)

        initial_image_path = '/Users/dronoved/test_gpt/import_file.jpg'
        initial_pixmap = QPixmap(initial_image_path)
        scaled_initial_pixmap = initial_pixmap.scaled(fixed_width, fixed_height, Qt.KeepAspectRatio)
        self.frame_display.setPixmap(scaled_initial_pixmap)

        layout = QVBoxLayout()
        layout.addWidget(self.frame_display)
        layout.addWidget(self.progress_bar)
        layout.addLayout(browse_layout)  # Добавляем горизонтальный макет
        layout.addWidget(self.btn_process)
        layout.addWidget(self.export_label)


        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def browse_file(self):
        self.file_name, _ = QFileDialog.getOpenFileName(self,'Выбрать файл', "/Users/dronoved/Downloads/", "Image Files (*.mp4 *.mov *.MOV)", '', )
        if self.file_name:
            self.file_path.setText(self.file_name)
            self.btn_process.setEnabled(True)

    def process_video(self):
        self.worker = Worker(self.file_name)
        self.worker.progress_changed.connect(self.update_progress_bar)
        self.worker.finished.connect(self.on_finished)
        self.worker.frame_ready.connect(self.display_frame)
        self.worker.start()

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def display_frame(self, frame):
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        qimage = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimage)

        # Масштабируем изображение, чтобы оно соответствовало размеру QLabel
        scaled_pixmap = pixmap.scaled(self.frame_display.size(), Qt.KeepAspectRatio)
        self.frame_display.setPixmap(scaled_pixmap)

    def on_finished(self, output_path):
        self.export_label.setText(f'Export path: {output_path}')
        self


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
