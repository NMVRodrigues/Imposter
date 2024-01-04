import pandas as pd
from PIL import Image


df = pd.read_csv(args.dataset)
for elem in df[args.key]:
    Image.open(elem).convert("RGB").save(elem)