import argparse


"""
This code takes the test, validation, and training directories and calculates the total trajectories,
minimum, maximum and average trajectory lengths, and the number of environments involved.
"""

def calculateStatistics(inputDir):
    totalTrajectories = 0
    minTrajectoryLen = 0
    maxTrajectoryLen = 0
    avgTrajectoryLen = 0
    numEnvs = 0
    return totalTrajectories, minTrajectoryLen, maxTrajectoryLen, avgTrajectoryLen, numEnvs

def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir', type=str, default='val_seen/', help="test, validation or training directory")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arg()
    print(args)

    directory = args.inputDir
    totalTrajectories, minTrajectoryLen, maxTrajectoryLen, avgTrajectoryLen, numEnvs = calculateStatistics(directory)
    print(totalTrajectories, minTrajectoryLen, maxTrajectoryLen, avgTrajectoryLen, numEnvs)
