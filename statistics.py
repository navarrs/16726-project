import argparse
import os


"""
This code takes the test, validation, and training directories and calculates the total trajectories,
minimum, maximum and average trajectory lengths, and the number of environments involved.
"""

def calculateStatistics(inputDir):
    totalTrajectories = 0
    trajectoryLens = []
    numEnvs = 0

    #iteration code from: https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/
    for env in os.listdir(inputDir):
        numEnvs += 1 #first layer of directory is the number of environments
        if env.startswith('.'): continue #skip useless files
        fEnv = os.path.join(inputDir, env)
        print("fEnv is", fEnv)
        if os.path.isfile(fEnv): # checking if it is a file. should not be file yet
            raise Exception("There should be additional trajectories within env folder.")

        for traj in os.listdir(fEnv): #iterate through all trajectories in each env subfolder
            totalTrajectories += 1 #second layer of directory is the number of trajectories
            fTraj = os.path.join(fEnv, traj)
            print("fTraj is ", fTraj)
            if fTraj.startswith('.'): continue #skip useless file
            if os.path.isfile(fTraj):  # checking if it is a file. should not be file yet
                raise Exception("There should be depth, semantic, and rgb info within trajectory folder.")
            #take the first folder in this case to check len.from

            for dir in os.listdir(fTraj):
                depthDir = os.path.join(fTraj, dir)
                print("depth dir is", depthDir)
                if depthDir.startswith('.'): continue #skip useless file
                if os.path.isfile(depthDir): #should also not be a file.
                    raise Exception("There should be pictures in the depthDir.")

                tmpTrajectoryLen = 0
                for img in os.listdir(depthDir):  # collect trajectory len info.
                    tmpTrajectoryLen += 1
                print("tmp trajectory length is", tmpTrajectoryLen)
                trajectoryLens.append(tmpTrajectoryLen)

                break #everything is same len, only need info from the first folder (which is depth)


    minTrajectoryLen = min(trajectoryLens)
    maxTrajectoryLen = max(trajectoryLens)
    avgTrajectoryLen = sum(trajectoryLens)/len(trajectoryLens)
    return totalTrajectories, minTrajectoryLen, maxTrajectoryLen, avgTrajectoryLen, numEnvs

#Code for parse_arg and main from CMU 16-726 HW5
def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir', type=str, default='~/Desktop/val_seen/', help="test, validation or training directory")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arg()

    directory = os.path.expanduser(args.inputDir)
    print(directory)
    totalTrajectories, minTrajectoryLen, maxTrajectoryLen, avgTrajectoryLen, numEnvs = calculateStatistics(directory)
    print(totalTrajectories, minTrajectoryLen, maxTrajectoryLen, avgTrajectoryLen, numEnvs)
