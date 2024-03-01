import cv2
import sys
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm

path = sys.argv[1]

print(sys.argv, len(sys.argv))

dfrows = []


def getOriginalFiles(d: pathlib.Path):
    files = [f.absolute() for f in d.iterdir() if f.is_file() and f.suffix == ".png"]
    files.sort(key=lambda e: int(e.stem))
    ret = [cv2.imread(str(f)) for f in files]
    return ret

def getBinaryImgs(d: pathlib.Path):
    binPath = d.joinpath(pathlib.PurePath("axis/binaris"))
    files = [f.absolute() for f in binPath.iterdir() if f.is_file() and f.suffix == ".png"]
    files.sort(key=lambda e: int(e.stem))
    ret = [cv2.imread(str(f)) for f in files]
    return ret

def getThresholds(d: pathlib.Path):
    f = d.joinpath("axis/thr/thr.txt")
    thrs = []
    with open(str(f), "r") as thr:
        for l in thr:
            thrs.append(float(l.strip()))
    return thrs


def matchBinary(bimg, thr, imgs):
    diffs = []
    for i in range(len(imgs)):
        ret, tholded = cv2.threshold(imgs[i], thr, 255, cv2.THRESH_BINARY)
        diff = np.average((bimg-tholded)**2)
        diffs.append(diff)
    m = np.array(diffs)
    return m

def writeMins(d: pathlib.Path, start, l):
    p = d.joinpath("axis/range.txt")
    with open(str(p), "w") as f:
        f.write(str(start)+ " " + str(l) + "\n")

def calcStart(E):
    b = E.shape[0]
    n = E.shape[1]
    ss = []
    for i in range(n-b+1):
        ss.append(np.trace(E, i))
    a = np.array(ss)
    m = np.argmin(a)
    return m



def processDir(d : pathlib.Path):
    imgs = getOriginalFiles(d)
    bins = getBinaryImgs(d)
    thrs = getThresholds(d)

    rows = []

    print(f"Processing dir: {str(d)}")

    for i in tqdm(range(len(bins))):
        mi = matchBinary(bins[i], thrs[i], imgs)
        rows.append(mi)

    E = np.array(rows)
    s = calcStart(E)

    writeMins(d, s, len(bins))
    dfrows.append([d.stem, s, len(bins)])


dirs = pathlib.Path(path).iterdir()
for d in dirs:
    if d.is_dir():
        try:
            processDir(d)
        except:
            pass

DF = pd.DataFrame(data=dfrows, columns=["dir", "start", "length"])
DF.to_csv(f"{path}/intervals.csv")
