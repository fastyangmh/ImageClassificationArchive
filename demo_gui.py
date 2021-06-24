# import
from src.project_parameters import ProjectParameters
from src.predict import Predict
import tkinter as tk
from tkinter import Tk, Button, Label, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

# class


class GUI:
    """Constructs a ProjectParameters class to store the parameters.
    """

    def __init__(self, project_parameters):
        """Initialize the class.

        Args:
            project_parameters (argparse.Namespace): the parameters for this project.
        """
        self.project_parameters = project_parameters
        self.predict_object = Predict(project_parameters=project_parameters)
        self.data_path = None

        # window
        self.window = Tk()
        self.window.geometry('{}x{}'.format(
            self.window.winfo_screenwidth(), self.window.winfo_screenheight()))
        self.window.title('Demo GUI')

        # button
        self.load_image_button = Button(
            self.window, text='load image', fg='black', bg='white', command=self._load_image)
        self.recognize_button = Button(
            self.window, text='recognize', fg='black', bg='white', command=self._recognize)

        # label
        self.data_path_label = Label(self.window, text='', fg='black')
        self.gallery_image_label = Label(self.window, text='', fg='black')
        self.probability_label = Label(self.window, text='', fg='black')
        self.result_label = Label(
            self.window, text='', fg='black', font=(None, 50))

    def _resize_image(self, image):
        """Resize the given image .

        Args:
            image (PIL.PngImagePlugin.PngImageFile): the image.

        Returns:
            PIL.PngImagePlugin.PngImageFile: the resized image.
        """
        width, height = image.size
        ratio = max(self.window.winfo_height(),
                    self.window.winfo_width())/max(width, height)
        ratio *= 0.5
        image = image.resize((int(width*ratio), int(height*ratio)))
        return image

    def _load_image(self):
        """Load image file .
        """
        self.data_path = filedialog.askopenfilename(
            initialdir='./', title='Select image file', filetypes=(('png files', '*.png'), ('jpeg files', '*.jpg')))
        image = self._resize_image(image=Image.open(
            fp=self.data_path).convert('RGB'))
        imageTk = ImageTk.PhotoImage(image)
        self.gallery_image_label.config(image=imageTk)
        self.gallery_image_label.image = imageTk
        self.data_path_label.config(
            text='image path: {}'.format(self.data_path))

    def _recognize(self):
        """Predict the image through the neural network model and display the result.
        """
        if self.data_path is not None:
            probability = self.predict_object(data_path=self.data_path)
            self.probability_label.config(text=('probability:\n'+' {}:{},'*len(probability)).format(
                *np.concatenate(list(zip(self.project_parameters.classes, probability))))[:-1])
            self.result_label.config(
                text=self.project_parameters.classes[probability.argmax()])
        else:
            messagebox.showinfo(
                title='Error!', message='please select an image!')

    def run(self):
        """Run the mainloop .
        """
        # button
        self.load_image_button.pack(anchor=tk.NW)
        self.recognize_button.pack(anchor=tk.NW)

        # label
        self.data_path_label.pack(anchor=tk.N)
        self.gallery_image_label.pack(anchor=tk.N)
        self.probability_label.pack(anchor=tk.N)
        self.result_label.pack(anchor=tk.N)

        # run
        self.window.mainloop()


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # GUI
    gui = GUI(project_parameters=project_parameters)
    gui.run()
