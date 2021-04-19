import sys
import os
import time
import json
# import win32con, win32api
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog

from MainWindow import Ui_MainWindow
from Mode_A import Ui_ModeAWindow
from Mode_B import Ui_ModeBWindow
from Mode_B_Result import Ui_ModeBResultWindow
from Mode_A_Result import Ui_ModeAResultWindow
from Mode_C import Ui_ModeCWindow
from Mode_C_Result import Ui_ModeCResultWindow

from predict import *
from partA import *

cost_time = 0
accuracy = 0
total_classified_image_number = 0
correct = 0


class MyPyQt_MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyPyQt_MainWindow, self).__init__()
        self.setupUi(self)

    # 实现next_button_click()函数
    def next_button_click(self):
        self.close()
        if my_pyqt_main_window.mode.currentIndex() == 0:
            my_pyqt_mode_a_window.show()
        elif my_pyqt_main_window.mode.currentIndex() == 1:
            my_pyqt_mode_b_window.show()
        else:
            my_pyqt_mode_c_window.show()

    # 实现quit_button_click()函数
    def quit_button_click(self):
        self.close()


class MyPyQt_ModeAWindow(QtWidgets.QMainWindow, Ui_ModeAWindow):
    def __init__(self):
        super(MyPyQt_ModeAWindow, self).__init__()
        self.setupUi(self)

    # 实现next_button_click()函数
    def next_button_click(self):
        target_img_dir = my_pyqt_mode_a_window.target_image_path.text()
        # print(target_img_dir)
        patch_pix = int(my_pyqt_mode_a_window.patch_pixel_dir_2.text())
        patch_pix_input = (patch_pix, patch_pix)
        mode = my_pyqt_mode_a_window.mode.currentIndex()
        main("./"+target_img_dir.split("/")[-1], patch_pix_input, mode)
        self.close()
        my_pyqt_mode_a_result_window.label_2.setPixmap(QtGui.QPixmap('composite.png'))
        my_pyqt_mode_a_result_window.show()

    # 实现quit_button_click()函数
    def quit_button_click(self):
        self.close()

    # 实现browse_button_click()函数
    def browse_button_click(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "Choose Target File", os.getcwd(),
                                                                   "All Files(*);;Image Files(*.jpg)")
        my_pyqt_mode_a_window.target_image_path.setText(fileName)


class MyPyQt_ModeBWindow(QtWidgets.QMainWindow, Ui_ModeBWindow):
    def __init__(self):
        super(MyPyQt_ModeBWindow, self).__init__()
        self.setupUi(self)

    # 实现next_button_click()函数
    def next_button_click(self):
        global cost_time, accuracy, total_classified_image_number, correct
        class_nums = 2
        image_dir = 'test_image'
        labels_filename = 'dataset\\label.txt'
        # models_path = 'models/model.ckpt-10000'
        models_path = 'models\\BestModels\\best_models_26600_0.9234.ckpt'

        batch_size = 1  #
        resize_height = 299
        resize_width = 299
        depths = 3
        data_format = [batch_size, resize_height, resize_width, depths]
        cost_time, accuracy, total_classified_image_number, correct = predict(models_path, image_dir, labels_filename, class_nums, data_format)
        textSet = "Total Classified Images: " + str(total_classified_image_number) + "\nCorrect: " + str(correct) \
                  + "; Wrong: " + str(total_classified_image_number - correct) + "\nAccuracy: " + str(accuracy) + \
                  '\nTime cost: ' + '%.4f' % cost_time + 's'
        my_pyqt_mode_b_result_window.label_result.setText(textSet)
        self.close()
        my_pyqt_mode_b_result_window.show()

    # 实现quit_button_click()函数
    def quit_button_click(self):
        self.close()

    # 实现browse_button_click()函数
    def browse_button_click(self):
        path = QFileDialog.getExistingDirectory(self, "Choose Original Image Directory", "C:\\Users\Admin\Desktop")
        # path_str = unicode(path.toUtf8(), 'utf-8', 'ignore')
        my_pyqt_mode_b_window.original_image_path.setText(path)


class MyPyQt_ModeBResultWindow(QtWidgets.QWidget, Ui_ModeBResultWindow):
    def __init__(self):
        global cost_time, accuracy, total_classified_image_number, correct
        super(MyPyQt_ModeBResultWindow, self).__init__()
        self.setupUi(self)

    # 实现home_button_click()函数
    def next_button_click(self):
        self.close()
        my_pyqt_main_window.show()

    # 实现exit_button_click()函数
    def exit_button_click(self):
        self.close()


class MyPyQt_ModeAResultWindow(QtWidgets.QWidget, Ui_ModeAResultWindow):
    def __init__(self):
        super(MyPyQt_ModeAResultWindow, self).__init__()
        self.setupUi(self)

    # 实现home_button_click()函数
    def next_button_click(self):
        self.close()
        my_pyqt_main_window.show()

    # 实现exit_button_click()函数
    def exit_button_click(self):
        self.close()


class MyPyQt_ModeCWindow(QtWidgets.QMainWindow, Ui_ModeCWindow):
    def __init__(self):
        super(MyPyQt_ModeCWindow, self).__init__()
        self.setupUi(self)

    # 实现next_button_click()函数
    def next_button_click(self):
        labels_nums = 2
        image_dir = my_pyqt_mode_c_window.original_image_path.text()
        labels_filename = 'dataset\\label.txt'
        models_path = 'models\\BestModels\\best_models_26600_0.9234.ckpt'
        batch_size = 1
        resize_height = 299
        resize_width = 299
        depths = 3
        data_format = [batch_size, resize_height, resize_width, depths]
        labels = np.loadtxt(labels_filename, str, delimiter='\t')
        input_images = tf.placeholder(dtype=tf.float32,
                                      shape=[None, resize_height, resize_width, depths],
                                      name='input')
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            out, end_points = inception_v3.inception_v3(inputs=input_images,
                                                        num_classes=labels_nums,
                                                        dropout_keep_prob=1.0,
                                                        is_training=False)
        score = tf.nn.softmax(out, name='pre')
        class_id = tf.argmax(score, 1)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, models_path)
        images_list = glob.glob(os.path.join("./"+image_dir.split("/")[-1], '*.jpg'))
        for image_path in images_list:
            im = read_image(image_path, resize_height, resize_width, normalization=True)
            im = im[np.newaxis, :]
            # pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
            pre_score, pre_label = sess.run([score, class_id], feed_dict={input_images: im})
            max_score = pre_score[0, pre_label]
            print("{} is: pre labels:{},name:{} score: {}".format(image_path, pre_label, labels[pre_label], max_score))
            patch_pix_input = (16, 16)
            mode = pre_label
            main(image_path, patch_pix_input, mode)
            if mode == 0:
                type = 'Natural'
            else:
                type = 'Man-made'
            textSet = "Classification Type: " + type

        sess.close()
        self.close()
        my_pyqt_mode_c_result_window.classification_type.setText(textSet)
        my_pyqt_mode_c_result_window.label_2.setPixmap(QtGui.QPixmap('composite.png'))
        my_pyqt_mode_c_result_window.show()


    # 实现quit_button_click()函数
    def quit_button_click(self):
        self.close()

    # 实现browse_button_click()函数
    def browse_button_click(self):
        path = QFileDialog.getExistingDirectory(self, "Choose Original Image Directory", "C:\\Users\Admin\Desktop")
        # path_str = unicode(path.toUtf8(), 'utf-8', 'ignore')
        my_pyqt_mode_c_window.original_image_path.setText(path)


class MyPyQt_ModeCResultWindow(QtWidgets.QWidget, Ui_ModeCResultWindow):
    def __init__(self):
        super(MyPyQt_ModeCResultWindow, self).__init__()
        self.setupUi(self)

    # 实现home_button_click()函数
    def next_button_click(self):
        self.close()
        my_pyqt_main_window.show()

    # 实现exit_button_click()函数
    def exit_button_click(self):
        self.close()


if __name__ == '__main__':
    window = QtWidgets.QApplication(sys.argv)
    my_pyqt_main_window = MyPyQt_MainWindow()
    my_pyqt_mode_a_window = MyPyQt_ModeAWindow()
    my_pyqt_mode_b_window = MyPyQt_ModeBWindow()
    my_pyqt_mode_b_result_window = MyPyQt_ModeBResultWindow()
    my_pyqt_mode_a_result_window = MyPyQt_ModeAResultWindow()
    my_pyqt_mode_c_window = MyPyQt_ModeCWindow()
    my_pyqt_mode_c_result_window = MyPyQt_ModeCResultWindow()

    my_pyqt_main_window.show()

    sys.exit(window.exec_())
