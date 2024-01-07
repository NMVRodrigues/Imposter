import pandas as pd
from PIL import Image
import argparse
from rich.progress import track
import numpy as np
import cv2
from torchvision.io import read_image

def convert_srgb_to_rgb(image_path, output_path):
    # Load the image
    image = Image.open(image_path)

    # Convert to RGB mode
    image = image.convert('RGB')

    # Convert to numpy array
    image_data = np.array(image)

    # Apply gamma correction to convert from sRGB to RGB
    # Gamma correction is a non-linear operation that helps to compensate for the way human eyes perceive light.
    # The sRGB color space uses a gamma of 2.2, while the RGB color space uses a gamma of 1.0.
    # To convert from sRGB to RGB, we can apply the following formula:
    # R' = R^2.2 / 255
    # G' = G^2.2 / 255
    # B' = B^2.2 / 255
    image_data = np.power(image_data, 2.2) / 255

    # Save the image in RGB mode
    new_image = Image.fromarray(image_data, 'RGB')
    new_image.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset', dest='dataset', type=str, required=True,
        help="path to dataset json that will be read")
    parser.add_argument(
        '--key', dest='key', type=str, required=True,
        help="Key to access the image columns of the dataset")

    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    to_remove = []
    for i, elem in track(enumerate(df[args.key]), description='Getting problematic IDs...', total=len(df[args.key])):
        if read_image(elem).shape[0] > 3:
            to_remove.append(i)

    df = df.drop(to_remove).reset_index()
    df.to_csv(args.dataset, index=False)
