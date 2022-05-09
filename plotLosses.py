import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plotLosses(dirs):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    dfTrainLoss = pd.DataFrame()
    dfValidLoss = pd.DataFrame()
    for i in range(len(dirs)):
        df = pd.read_csv(dirs[i])  # read the specific file for necessary stuff
        df[['train_loss', 'validation_loss']] = df["train_loss\tvalidation_loss"].str.split('\t', expand=True)
        del df["train_loss\tvalidation_loss"] #drop this monster of a column
        trainingColName = ""
        validationColName = ""
        if (i==0):
            trainingColName = "training_exp%d" % (i+1) + "_rgb"
            validationColName = "validation_exp%d" % (i+1) + "_rgb"
        elif (i==1):
            trainingColName = "training_exp%d" % (i + 1) + "_rgb_and_artifact"
            validationColName = "validation_exp%d" % (i + 1) + "_rgb_and_artifact"
        else:
            trainingColName = "training_exp%d" % (i + 1) + "_rgb_and_semantic"
            validationColName = "validation_exp%d" % (i + 1) + "_rgb_and_semantic"

        dfTrainLoss[trainingColName] = df["train_loss"]
        dfValidLoss[validationColName] = df["validation_loss"]

    dfTrainLoss = dfTrainLoss.astype('float', copy=False)
    dfValidLoss = dfValidLoss.astype('float', copy=False)
    dfTrainLoss.plot()
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    dfValidLoss.plot(linestyle="dotted")
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir', type=str, default='~/Desktop/expLogs', help="path to the input image")
    parser.add_argument("--envName", type=str, default="2n8kARJN3HM")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    inputDir = os.path.expanduser(args.inputDir)
    #load the directories
    exp1Dir = inputDir+"/exp1_progress_log_summary.csv"
    exp2Dir = inputDir+"/exp2_progress_log_summary.csv"
    exp3Dir = inputDir+"/exp3_progress_log_summary.csv"
    #print(exp1Dir, exp2Dir, exp3Dir)

    dirs = [exp1Dir, exp2Dir, exp3Dir]
    plotLosses(dirs)

