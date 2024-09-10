from typing import List, Union, NoReturn

import matplotlib.pyplot as plt
from PIL import Image


def show_image(img: Union[Image,str], axis: str='off') -> NoReturn:
    if isinstance(img, str):
        img = Image.open(img)
    plt.imshow(img)
    plt.axis(axis)
    plt.show()

def show_images(imgs: List[Union[Image, str]], 
                figsize=(15, 5), 
                axis='off',
                wspace=0.3) -> NoReturn:
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize)

    for index, ax, img in enumerate(zip(axs, imgs)):
        path = index
        if instance(img, str):
            path = img
            img = Image.open(img)
        ax.imshow(img)
        ax.axis(axis)

    plt.subplots_adjust(wspace=wspace)
    plt.show()
