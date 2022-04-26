import imageio
import os
import argparse
import numpy as np
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image

#Code to convert images to gif from CMU 16-726 HW5
def save_images(image, fname, col=8):
    #image = image.cpu().detach()
    #open image here
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    image = Image.open(image)
    image = transform(image)

    image = image / 2 + 0.5

    image = vutils.make_grid(image, nrow=col)  # (C, H, W)
    image = image.numpy().transpose([1, 2, 0])
    image = np.clip(255 * image, 0, 255).astype(np.uint8)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        imageio.imwrite(fname + '.png', image)
    return image


def save_gifs(image_list, fname, col=1):
    """
    :param image_list: [(N, C, H, W), ] in scale [-1, 1]
    """
    image_list = [save_images(each, None, col) for each in image_list]
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    imageio.mimsave(fname + '.gif', image_list)


def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir', type=str, default='~/Desktop/depth', help="path to the input image")
    parser.add_argument("--envName", type=str, default="2n8kARJN3HM")
    parser.add_argument("--trajectoryName", type=str, default="110")
    parser.add_argument("--gifType", type=str, default="depth")
    return parser.parse_args()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_arg()
    #save gif of depth, rgb, and semantic information
    imagesPath = os.path.expanduser(args.inputDir)+"/"
    imagePathsList = os.listdir(imagesPath)
    imagePathsList.sort() #organize these by correct order
    image_list = [imagesPath+s for s in imagePathsList]
    print(image_list)
    outputDir = '%s/output/%s_%s_%s' % (os.path.expanduser("~/Desktop"), args.envName, args.trajectoryName, args.gifType)
    print(outputDir)
    save_gifs(image_list, outputDir)

