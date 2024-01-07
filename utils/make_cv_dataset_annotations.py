import os
import pandas as pd
from glob import glob
from rich.progress import track
from PIL import Image

name = 'animals10'

if name == 'animals10':
    files = list(glob(os.path.join("C:\\Users\\Utilizador\\Desktop\\Impostor\\Datasets", 'CV', name, '*', '*')))
    paths, classes, dimensions = [], [], []

    for file in track(files, description='Reading files...'):
        paths.append(file)
        classes.append(file.split(os.sep)[-2])
        dimensions.append(Image.open(file).size)

    df = pd.DataFrame({'image': files, 'class': classes, 'size': dimensions})
    df.to_csv(os.path.join("C:\\Users\\Utilizador\\Desktop\\Impostor\\Datasets", 'CV', name, 'annotations.csv'), index=False)
