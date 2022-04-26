import os
import time
import json
import cv2
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

VOCABULARY = []
VOCABULARY += [chr(ord('A') + i) for i in range(26)]
VOCABULARY += [chr(ord('0') + i) for i in range(10)]


class TextImageGenerator:

    def __init__(self,
                 output_dir,
                 min_text_len=1,
                 max_text_len=3,
                 min_text_font_size=50,
                 max_text_font_size=50,
                 text_fonts=("arial.ttf", "ARLRDBD.TTF"),
                 img_mode="L",
                 img_size=(128, 128),
                 random_seed=20220209,
                 text_vocabulary=VOCABULARY,
                 gaussian_blur_radius_range=None,
                 inclusive_text=[]
                 ):

        # Init variables.
        self.output_dir = output_dir
        self.output_pattern = "{text:s}_{timestamp:d}.bmp"
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.min_text_font_size = min_text_font_size
        self.max_text_font_size = max_text_font_size
        self.text_fonts = text_fonts
        self.img_mode = img_mode
        self.img_size = img_size
        self.img_center = tuple(int(item / 2) for item in self.img_size)
        self.random_seed = random_seed
        self.text_vocabulary = VOCABULARY
        self.gaussian_blur_radius_range = gaussian_blur_radius_range
        self.inclusive_text = inclusive_text

        # Init random generator.
        self.random_state = np.random.RandomState(self.random_seed)

        # Memory for stats.
        self.all_generated_text_lens = []
        self.all_generated_font_sizes = []
        self.all_generated_font_weights = []
        self.all_generated_texts = set()
        self.all_generated_gaussian_radii = []

    '''
    Link generator behaviour to image generator. 
    '''
    def __next__(self):

        return self.__generate_text_image()

    '''
    Generating training data for superimposed texts.
    '''
    def __generate_text_image(self):

        # Get random variables.
        text_font = self.random_state.choice(self.text_fonts)
        text_font_size = self.random_state.randint(self.min_text_font_size, self.max_text_font_size+1)
        text_weight = self.random_state.choice(["normal", "bold", "extra_bold", "black"])

        # Get texts.
        if len(self.inclusive_text):
            text = self.inclusive_text.pop()
            text_len = len(text)
        else:
            text_len = self.random_state.randint(self.min_text_len, self.max_text_len + 1)
            text = "".join(self.random_state.choice(self.text_vocabulary, text_len, replace=True))
            while text in self.all_generated_texts:
                text_len = self.random_state.randint(self.min_text_len, self.max_text_len + 1)
                text = "".join(self.random_state.choice(self.text_vocabulary, text_len, replace=True))

        self.all_generated_text_lens.append(text_len)
        self.all_generated_font_sizes.append(text_font_size)
        self.all_generated_font_weights.append(text_weight)
        self.all_generated_texts.add(text)

        # Printing text on image.
        # Init image.
        pil_img = Image.new(self.img_mode, size=self.img_size, color=255)

        # Get font.
        # In Windows, Pillow looks for the default font directory if font file not found.
        pil_font = ImageFont.truetype(text_font, text_font_size, index=0)

        # Draw the text.
        pil_draw = ImageDraw.Draw(pil_img)
        pil_draw.text(self.img_center, text, anchor="mm", font=pil_font, fontweight=text_weight, fill=0)

        # Gaussian blurring.
        if self.gaussian_blur_radius_range:
            gaussian_blur_radius = self.random_state.uniform(*self.gaussian_blur_radius_range)
            self.all_generated_gaussian_radii.append(gaussian_blur_radius)
            pil_img_output = pil_img.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_radius))
        else:
            pil_img_output = pil_img

        # Save the image.
        img_output_path = os.path.join(self.output_dir, self.output_pattern.format(text=text, timestamp=int(time.time()*1000)))
        while os.path.exists(img_output_path):
            time.sleep(0.001)
            img_output_path = os.path.join(self.output_dir, self.output_pattern.format(text=text, timestamp=int(time.time()*1000)))
        pil_img_output.save(img_output_path)

        return pil_img_output

    """
    Return the history of generator. 
    """
    def get_generated_text_data(self):

        return {
            "text_len": self.all_generated_text_lens.copy(),
            "font_sizes": self.all_generated_font_sizes.copy(),
            "font_weight": self.all_generated_font_weights.copy(),
            "texts": list(self.all_generated_texts.copy()),
            "gaussian_blur_radius": self.all_generated_gaussian_radii.copy()
        }


# Module test
if __name__ == "__main__":

    # Generating.
    # output_path = "./train_data/bold_uniform_font30"
    # num_data = 5000
    # random_seed = 20220209

    output_path = "./test_data/bold_uniform_font30"
    num_data = 1000
    random_seed = 20220226

    img_gen = TextImageGenerator(
        output_path,
        min_text_len=2,
        max_text_len=5,
        min_text_font_size=30,
        max_text_font_size=30,
        gaussian_blur_radius_range=None,
        text_fonts=[
            # "arial.ttf",  # Normal
            # "arialbd.ttf",  # Bold
            # "ariblk.ttf",  # Black
            "ARLRDBD.TTF",  # Round black
        ],
        random_seed=random_seed,
    )

    for _ in range(num_data):
        next(img_gen)

    # Export generator data.
    generator_data = img_gen.get_generated_text_data()
    with open(os.path.join(output_path, "_labels.txt"), "w") as f:
        json.dump(generator_data, f)

    # Print stats.
    print(f"{len(generator_data['text_len']):d} texts generated.")
