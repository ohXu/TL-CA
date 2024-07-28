from codes.tool import *
from codes.dataset import *
from codes.model import *
import random

DIR = "../data/"
RADIUS = 7

land2000 = loadGridData(os.path.join(DIR, "Y_2000.tif"))
land2010 = loadGridData(os.path.join(DIR, "Y_2010.tif"))
land2020 = loadGridData(os.path.join(DIR, "Y_2020.tif"))
row, col = land2000.shape
unique, counts = np.unique(land2000, return_counts=True)
unique2, counts2 = np.unique(land2010, return_counts=True)
unique3, counts3 = np.unique(land2020, return_counts=True)
print(dict(zip(unique, counts)), dict(zip(unique2, counts2)), dict(zip(unique3, counts3)))

if not os.path.exists(os.path.join(DIR, "D1.npy")):
    drivingFactors1 = loadDrivingFactor2(os.path.join(DIR), 1)
    drivingFactors2 = loadDrivingFactor2(os.path.join(DIR), 2)
    positionFactors = loadPositionFactor2(os.path.join(DIR))
    drivingFactors1 = np.concatenate((drivingFactors1, positionFactors), axis=-1)
    drivingFactors2 = np.concatenate((drivingFactors2, positionFactors), axis=-1)
    np.save(os.path.join(DIR, "D1.npy"), drivingFactors1)
    np.save(os.path.join(DIR, "D2.npy"), drivingFactors2)
else:
    drivingFactors1 = np.load(os.path.join(DIR, "D1.npy"))
    drivingFactors2 = np.load(os.path.join(DIR, "D2.npy"))
    print(drivingFactors1.shape, drivingFactors2.shape)

if not os.path.exists(os.path.join(DIR, "nonurbanArray.npy")):
    nonurbanArray, nonurbanNochange, nonurbanGrowth, inIndex = getValidValueF(land2000, land2010, RADIUS)
    nonurbanArray2, nonurbanNochange2, nonurbanGrowth2, inIndex2 = getValidValueF(land2010, land2020, RADIUS)
    np.save(os.path.join(DIR, "nonurbanArray.npy"), nonurbanArray)
    np.save(os.path.join(DIR, "nonurbanNochange.npy"), nonurbanNochange)
    np.save(os.path.join(DIR, "nonurbanGrowth.npy"), nonurbanGrowth)
    np.save(os.path.join(DIR, "inIndex.npy"), inIndex)
    np.save(os.path.join(DIR, "nonurbanArray2.npy"), nonurbanArray2)
    np.save(os.path.join(DIR, "nonurbanNochange2.npy"), nonurbanNochange2)
    np.save(os.path.join(DIR, "nonurbanGrowth2.npy"), nonurbanGrowth2)
    np.save(os.path.join(DIR, "inIndex2.npy"), inIndex2)
else:
    nonurbanArray = np.load(os.path.join(DIR, "nonurbanArray.npy"))
    nonurbanNochange = np.load(os.path.join(DIR, "nonurbanNochange.npy"))
    nonurbanGrowth = np.load(os.path.join(DIR, "nonurbanGrowth.npy"))
    inIndex = np.load(os.path.join(DIR, "inIndex.npy"))
    nonurbanArray2 = np.load(os.path.join(DIR, "nonurbanArray2.npy"))
    nonurbanNochange2 = np.load(os.path.join(DIR, "nonurbanNochange2.npy"))
    nonurbanGrowth2 = np.load(os.path.join(DIR, "nonurbanGrowth2.npy"))
    inIndex2 = np.load(os.path.join(DIR, "inIndex2.npy"))

allSampleNochange = np.where(nonurbanNochange == 1)
allSampleGrowth = np.where(nonurbanGrowth == 1)
allSampleNochangeNumber = allSampleNochange[0].shape[0]
allSampleGrowthNumber = allSampleGrowth[0].shape[0]
ratio = allSampleGrowthNumber / (allSampleGrowthNumber + allSampleNochangeNumber)
sampleGrowthNumber = int((allSampleGrowthNumber + allSampleNochangeNumber) * 0.05 * ratio)
sampleNochangeNumber = int((allSampleGrowthNumber + allSampleNochangeNumber) * 0.05 * (1 - ratio))

print("正样本数量:" + str(allSampleGrowthNumber) + " 负样本数量:" + str(allSampleNochangeNumber) + " 正样本采样数:" + str(
    sampleGrowthNumber) + " 负样本采样数:" + str(sampleNochangeNumber) + " ratio:" + str(ratio))
sampleGrowthList = np.linspace(0, allSampleGrowthNumber, num=sampleGrowthNumber, endpoint=False, dtype=int)
sampleNochangeList = np.linspace(0, allSampleNochangeNumber, num=sampleNochangeNumber, endpoint=False, dtype=int)
np.random.shuffle(sampleGrowthList)
np.random.shuffle(sampleNochangeList)

trainWeights = []
trainSamples = []
valSamples = []
testSamples = []
testSamples2 = []
trainNum1 = int(sampleGrowthList.shape[0] * 0.95)
trainNum2 = int(sampleNochangeList.shape[0] * 0.95)
growth_train_img = np.zeros((land2010.shape[0], land2010.shape[1], 3), dtype=np.uint8)
for i in range(row):
    for j in range(col):
        if land2000[i][j] == 2:
            growth_train_img[i][j] = [255, 0, 0]
growth_val_img = growth_train_img.copy()
nochange_train_img = growth_train_img.copy()
nochange_val_img = growth_train_img.copy()
for i in range(trainNum1):
    idm = sampleGrowthList[i]
    irow = allSampleGrowth[0][idm]
    icol = allSampleGrowth[1][idm]
    growth_train_img[irow - 2: irow + 2 + 1, icol - 2: icol + 2 + 1] = [255, 255, 255]
    factorsX = drivingFactors1[irow - RADIUS: irow + RADIUS + 1, icol - RADIUS: icol + RADIUS + 1]
    growthY = 1
    trainSamples.append([factorsX, growthY])
    if land2000[irow, icol] != 0 or land2010[irow, icol] != 2:
        print("X1")

for i in range(trainNum1, sampleGrowthList.shape[0]):
    idm = sampleGrowthList[i]
    irow = allSampleGrowth[0][idm]
    icol = allSampleGrowth[1][idm]
    growth_val_img[irow - 2: irow + 2 + 1, icol - 2: icol + 2 + 1] = [255, 255, 255]
    factorsX = drivingFactors1[irow - RADIUS: irow + RADIUS + 1, icol - RADIUS: icol + RADIUS + 1]
    growthY = 1
    valSamples.append([factorsX, growthY])
    if land2000[irow, icol] != 0 or land2010[irow, icol] != 2:
        print("X1")

for i in range(trainNum2):
    idm = sampleNochangeList[i]
    irow = allSampleNochange[0][idm]
    icol = allSampleNochange[1][idm]
    nochange_train_img[irow, icol] = [255, 255, 255]
    factorsX = drivingFactors1[irow - RADIUS: irow + RADIUS + 1, icol - RADIUS: icol + RADIUS + 1]
    growthY = 0
    trainSamples.append([factorsX, growthY])
    trainWeights.append(1)
    if land2000[irow, icol] != 0 or land2010[irow, icol] != 0:
        print("X2")

for i in range(trainNum2, sampleNochangeList.shape[0]):
    idm = sampleNochangeList[i]
    irow = allSampleNochange[0][idm]
    icol = allSampleNochange[1][idm]
    nochange_val_img[irow, icol] = [255, 255, 255]
    factorsX = drivingFactors1[irow - RADIUS: irow + RADIUS + 1, icol - RADIUS: icol + RADIUS + 1]
    growthY = 0
    valSamples.append([factorsX, growthY])
    if land2000[irow, icol] != 0 or land2010[irow, icol] != 0:
        print("X2")

for i in range(row):
    for j in range(col):
        if nonurbanNochange[i][j] == 1:
            factorsX = drivingFactors1[i - RADIUS: i + RADIUS + 1, j - RADIUS: j + RADIUS + 1]
            growthY = 0
            testSamples.append([factorsX, growthY])
        if nonurbanGrowth[i][j] == 1:
            factorsX = drivingFactors1[i - RADIUS: i + RADIUS + 1, j - RADIUS: j + RADIUS + 1]
            growthY = 1
            testSamples.append([factorsX, growthY])
        if nonurbanNochange2[i][j] == 1:
            factorsX = drivingFactors2[i - RADIUS: i + RADIUS + 1, j - RADIUS: j + RADIUS + 1]
            growthY = 0
            testSamples2.append([factorsX, growthY])
        if nonurbanGrowth2[i][j] == 1:
            factorsX = drivingFactors2[i - RADIUS: i + RADIUS + 1, j - RADIUS: j + RADIUS + 1]
            growthY = 1
            testSamples2.append([factorsX, growthY])
print("训练样本：", len(trainSamples), "验证样本：", len(valSamples), "测试样本：", len(testSamples), "测试样本2：", len(testSamples2))
batchSize = 64
train_dataset = CustomDataset(trainSamples)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batchSize, num_workers=multiprocessing.cpu_count())
val_dataset = CustomDataset(valSamples)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batchSize, num_workers=multiprocessing.cpu_count())
test_dataset = CustomDataset(testSamples)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batchSize, num_workers=multiprocessing.cpu_count())
test_dataset2 = CustomDataset(testSamples2)
test_loader2 = DataLoader(test_dataset2, shuffle=False, batch_size=batchSize, num_workers=multiprocessing.cpu_count())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")
MODEL = "ViT"
model = ViT(image_size=15, patch_size=1, num_classes=1, dim=32, depth=6, heads=1, mlp_dim=32, dim_head=16,
            channels=10, dropout=0.2, emb_dropout=0.2).to(device)

model_size = sum(p.numel() for p in model.parameters())
model_size += sum(p.numel() for p in model.buffers())
print(model_size)

loss_fn = nn.BCELoss()
loss_fn2 = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

epochs = 30
min_loss = 1000000
max_loss = 0

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    time_start = time.time()
    train(train_loader, model, loss_fn, loss_fn2, optimizer, device, 64, float(t/epochs))
    print("Time:", time.time() - time_start)
    valid_loss, valid_loss2 = val(val_loader, model, loss_fn, loss_fn2, device, 64, float(t/epochs))
    if valid_loss < min_loss:
        print(f'Validation loss decreased ({min_loss:.6f} --> {valid_loss:.6f}).  Saving model ...')
        min_loss = valid_loss
        torch.save(model, os.path.join(f"../output/{MODEL}_{batchSize}_checkpoint2.pt"))
print("Done!")

model = torch.load(os.path.join(f"../output/{MODEL}_{batchSize}_checkpoint2.pt")).to(device)
time_start = time.time()
potentialMap2, trueMap2 = test(test_loader2, model, nn.BCELoss(), device)
print("Test Time:", time.time() - time_start)

LDS = np.zeros((row, col))
NE = np.zeros((row, col))
RA = np.zeros((row, col))
n_helper = 0
validNonurbanArray = np.zeros((row, col))

data1 = np.zeros((row, col), dtype=np.float16)
data2 = np.zeros((row, col), dtype=np.float16)
data2.fill(-1)
errorArray = np.zeros((row, col), dtype=np.float16)
errorArray.fill(-1)
errorArray2 = np.zeros((row, col), dtype=np.uint8)

for i in range(row):
    for j in range(col):
        if nonurbanNochange2[i][j] == 1 or nonurbanGrowth2[i][j] == 1:
            validNonurbanArray[i][j] = 1
            LDS[i][j] = potentialMap2[n_helper]
            if nonurbanNochange2[i][j] == 1:
                errorArray[i][j] = (0 - potentialMap2[n_helper])
                data2[i][j] = 0
            if nonurbanGrowth2[i][j] == 1:
                errorArray[i][j] = (1 - potentialMap2[n_helper])
                data2[i][j] = 1
            errorArray2[i][j] = 1
            data1[i][j] = potentialMap2[n_helper]

            n_helper += 1
            DIAM = 2 * RADIUS + 1
            NE[i][j] = \
                np.where(np.array(land2010[i - RADIUS: i + RADIUS + 1, j - RADIUS: j + RADIUS + 1]) == 2)[0].shape[
                    0] / (DIAM * DIAM - 1)
            RA[i][j] = 1 + math.pow((-1 * math.log(math.e, np.random.random())), 0.05)
print("getPotential", n_helper, np.max(LDS), np.min(LDS), np.max(NE), np.min(NE), np.max(RA), np.min(RA))
validNonurbanIndex = np.where(validNonurbanArray.ravel() == 1)[0]

# totalProbability = LDS
totalProbability = np.add(LDS * 1.01, NE * 0.8)
totalProbability = np.multiply(totalProbability, RA)
totalProbability_sort = np.sort(totalProbability.ravel())

threshold = np.where(nonurbanGrowth2 == 1)[0].shape[0]
threshold = totalProbability_sort[-threshold]
print("threshold:", threshold, totalProbability_sort[-1],
      totalProbability_sort[-allSampleGrowthNumber - 5:-allSampleGrowthNumber + 5])
print(np.where(totalProbability_sort >= threshold)[0].shape)

simulateLand = np.zeros((row, col))
simulateLand2 = np.zeros((row, col, 3))
for i in range(row):
    for j in range(col):
        if land2010[i][j] == 0:
            simulateLand[i][j] = 1
            if totalProbability[i][j] >= threshold:
                simulateLand[i][j] = 3
                simulateLand2[i][j] = [255, 255, 255]
        if land2010[i][j] == 1:
            simulateLand[i][j] = 2
        if land2010[i][j] == 2:
            simulateLand[i][j] = 3

generateImage(simulateLand, f"../output/{MODEL}_{batchSize}.tif")
np.save(f"../output/{MODEL}_{batchSize}_s.npy", simulateLand)
np.save(f"../output/{MODEL}_{batchSize}_t.npy", totalProbability)

simulateLand -= 1
OAandKappa(simulateLand.ravel()[inIndex2], land2020.ravel()[inIndex2])
Fom(simulateLand.ravel()[validNonurbanIndex], land2020.ravel()[validNonurbanIndex],
    land2010.ravel()[validNonurbanIndex])