import glob
from PIL import Image
ID = 506
images=glob.glob(f"results/res{ID}*")
images = [Image.open(img) for img in images]

images[0].save(f'figures/res{ID}.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=1000, loop=0)

