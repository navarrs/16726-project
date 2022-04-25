import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

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
    parser.add_argument('--inputDir', type=str, default='data/sketch/*.png', help="path to the input image")

    return parser.parse_args()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_arg()
    print(args)

    #save gif of depth, rgb, and semantic information
    image_list = args.inputDir
    for i in range(len(image_list)):
        outputDir = 'output/%d_%s' % (i, image_list[i])
        save_gifs(image_list, outputDir)

