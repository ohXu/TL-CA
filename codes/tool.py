import numpy as np
from osgeo import gdal
import os
from PIL import Image
import multiprocessing
import time
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
import math
from tqdm import tqdm
import pandas as pd
from dbfread import DBF


# import libpysal
#
# w = libpysal.weights.lat2W(721125, 721125)
# print(w.full()[0])
# w1 = w.full()[0]
# print(type(w1))
# np.save("./output/weight.npy", w1)

def loadDBF(dir, number, array):
    table = DBF(dir)
    moran = np.zeros(shape=(number,))
    for record in table:
        # print(record['grid_code'], " ", record['grid_code_'], " ", record['LISA_I'])
        moran[int(record['grid_code_']) - 1] = float(record['LISA_I'])
        if array[int(record['grid_code_']) - 1] != int(record['grid_code']):
            print("XXXXX")
    return moran


def loadGridData(name):
    dataset = gdal.Open(name)
    im_height = dataset.RasterYSize
    im_width = dataset.RasterXSize
    # print(im_height, im_width)
    data = dataset.ReadAsArray(0, 0, im_width, im_height)
    return data


def extractChangeArea(newData, oldData, radius):
    row = newData.shape[0]
    col = newData.shape[1]
    new_array = np.zeros((row, col), dtype=np.uint8)
    new_array2 = np.zeros((row, col), dtype=np.uint8)
    row_array = np.zeros((row, col), dtype=np.uint32)
    array = []
    # for i in range(row):
    #     for j in range(col):
    #         if newData[i][j] == 65535:
    #             new_array2[i, j] = 0
    #         else:
    #             if newData[i][j] == 2 and oldData[i][j] == 0:
    #                 new_array[i, j] = 1
    #                 new_array2[i, j] = 2
    #             if newData[i][j] != 2 and oldData[i][j] == 0:
    #                 new_array2[i, j] = 1
    id = 1
    for i in range(radius, row - radius):
        for j in range(radius, col - radius):
            data = np.array(oldData[i - radius: i + radius + 1, j - radius: j + radius + 1])
            if np.where(data == 65535)[0].shape[0] == 0:
                if oldData[i, j] == 0:
                    if newData[i, j] == 0:
                        new_array2[i, j] = 1
                        row_array[i, j] = id
                        id += 1
                        array.append(1)
                    elif newData[i, j] == 2:
                        new_array[i, j] = 1
                        new_array2[i, j] = 2
                        row_array[i, j] = id
                        id += 1
                        array.append(2)
            else:
                new_array2[i, j] = 0
    print(np.where(new_array2 == 1)[0].shape[0] + np.where(new_array2 == 2)[1].shape[0])
    return new_array, new_array2, row_array, np.array(array)


def extractChangeArea2(newData, oldData, radius):
    row = newData.shape[0]
    col = newData.shape[1]
    new_array = np.zeros((row, col), dtype=np.uint8)
    new_array2 = np.zeros((row, col), dtype=np.uint8)
    row_array = np.zeros((row, col), dtype=np.uint32)
    array = []
    id = 1
    for i in range(radius, row - radius):
        for j in range(radius, col - radius):
            data = np.array(oldData[i - radius: i + radius + 1, j - radius: j + radius + 1])
            if np.where(data == 65535)[0].shape[0] == 0:
                if oldData[i, j] == 0:
                    if newData[i, j] == 0:
                        new_array2[i, j] = 1
                        row_array[i, j] = id
                        id += 1
                        array.append(1)
                    elif newData[i, j] == 2:
                        new_array[i, j] = 1
                        new_array2[i, j] = 2
                        row_array[i, j] = id
                        id += 1
                        array.append(2)
            else:
                new_array2[i, j] = 0
    print(np.where(new_array2 == 1)[0].shape[0] + np.where(new_array2 == 2)[1].shape[0])
    return new_array, new_array2, row_array, np.array(array)


def loadDrivingFactor(dir):
    dataProximity = []
    for root, dirs, files in os.walk(dir):
        for name in sorted(files):
            file = os.path.join(root, name)
            if "2000" not in name and "2010" not in name and "2015" not in name and "2020" not in name and "Lat" not in name and "Lon" not in name:
                data = loadGridData(file)
                datamax = np.max(data)
                datamin = np.min(data)
                print(name, np.min(data), np.max(data), data.dtype)
                if "DEM" not in name:
                    data = 1 - (data - datamin) / (datamax - datamin)
                else:
                    data = np.float32(data)
                    datasort = data.copy()
                    datasort = datasort.ravel()
                    datasort.sort()
                    datasort = np.unique(datasort)
                    datamax = datasort[-2]
                    datamin = datasort[0]
                    print(datamax, datamin)
                    start_time = time.time()
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            if data[i][j] != 32767:
                                data[i][j] = (data[i][j] - datamin) / (datamax - datamin)
                            else:
                                data[i][j] = -9999
                    print("普通的 for 循环时间：", time.time() - start_time)

                Image.fromarray(np.array(data * 255, dtype=np.uint8)).save(
                    os.path.join("./output/", name.split(".")[0] + ".png"))
                data = data.reshape((data.shape[0], data.shape[1], 1))
                dataProximity.append(data)

    factors = dataProximity[0]
    for i in range(1, len(dataProximity)):
        factors = np.concatenate((factors, dataProximity[i]), axis=-1)
    return factors


def loadDrivingFactor2(dir, index):
    dataProximity = []
    for root, dirs, files in os.walk(dir):
        for name in sorted(files):
            file = os.path.join(root, name)
            if "primary" in name or "secondary" in name or "motorway" in name or "tertiary" in name or "trunk" in name:
                data = loadGridData(file)
                datamax = np.max(data)
                datamin = np.min(data)
                print(name, np.min(data), np.max(data), data.dtype)
                data = 1 - (data - datamin) / (datamax - datamin)
                Image.fromarray(np.array(data * 255, dtype=np.uint8)).save(
                    os.path.join("./output2/", name.split(".")[0] + ".png"))
                data = data.reshape((data.shape[0], data.shape[1], 1))
                dataProximity.append(data)

            if "slope" in name or "dem" in name:
                data = loadGridData(file)
                datamax = np.max(data)
                datamin = np.min(data)
                print(name, np.min(data), np.max(data), data.dtype)
                data = np.float32(data)
                data = (data - datamin) / (datamax - datamin)
                Image.fromarray(np.array(data * 255, dtype=np.uint8)).save(
                    os.path.join("./output2/", name.split(".")[0] + ".png"))
                data = data.reshape((data.shape[0], data.shape[1], 1))
                dataProximity.append(data)

            if "pop" in name or "GDP" in name or "EC" in name:
                if index == 1 and "2010" in name:
                    data = loadGridData(file)
                    datamax = np.max(data)
                    datamin = np.min(data)
                    data = np.float32(data)
                    datasort = data.copy()
                    datasort = datasort.ravel()
                    datasort.sort()
                    datasort = np.unique(datasort)
                    datamax = datasort[-1]
                    dataminSec = datasort[1]
                    print(name, np.min(data), np.max(data), data.dtype, datamax, dataminSec)
                    start_time = time.time()
                    # data[np.where(data != datamin)[0][0], np.where(data != datamin)[0][1]] = (data - dataminSec) / (
                    #             datamax - dataminSec)
                    # data[np.where(data == datamin)[0][0], np.where(data == datamin)[0][1]] = 0
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            if data[i][j] != datamin:
                                data[i][j] = (data[i][j] - dataminSec) / (datamax - dataminSec)
                            else:
                                data[i][j] = 0
                    print("普通的 for 循环时间：", time.time() - start_time)
                    Image.fromarray(np.array(data * 255, dtype=np.uint8)).save(
                        os.path.join("./output2/", name.split(".")[0] + ".png"))
                    data = data.reshape((data.shape[0], data.shape[1], 1))
                    dataProximity.append(data)

                if index == 2 and "2010" not in name:
                    data = loadGridData(file)
                    datamax = np.max(data)
                    datamin = np.min(data)
                    data = np.float32(data)
                    datasort = data.copy()
                    datasort = datasort.ravel()
                    datasort.sort()
                    datasort = np.unique(datasort)
                    datamax = datasort[-1]
                    dataminSec = datasort[1]
                    print(name, np.min(data), np.max(data), data.dtype, datamax, dataminSec)
                    start_time = time.time()
                    # data[np.where(data != datamin)[0][0], np.where(data != datamin)[0][1]] = (data - dataminSec) / (
                    #             datamax - dataminSec)
                    # data[np.where(data == datamin)[0][0], np.where(data == datamin)[0][1]] = 0
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            if data[i][j] != datamin:
                                data[i][j] = (data[i][j] - dataminSec) / (datamax - dataminSec)
                            else:
                                data[i][j] = 0
                    print("普通的 for 循环时间：", time.time() - start_time)
                    Image.fromarray(np.array(data * 255, dtype=np.uint8)).save(
                        os.path.join("./output2/", name.split(".")[0] + ".png"))
                    data = data.reshape((data.shape[0], data.shape[1], 1))
                    dataProximity.append(data)

    factors = dataProximity[0]
    for i in range(1, len(dataProximity)):
        factors = np.concatenate((factors, dataProximity[i]), axis=-1)
    return factors


def loadPositionFactor(dir):
    dataProximity = []
    for root, dirs, files in os.walk(dir):
        for name in sorted(files):
            file = os.path.join(root, name)
            if "Lat" in name or "Lon" in name:
                data = loadGridData(file)
                data = data.reshape((data.shape[0], data.shape[1], 1))
                dataProximity.append(data)
    factors = dataProximity[0]
    for i in range(1, len(dataProximity)):
        factors = np.concatenate((factors, dataProximity[i]), axis=-1)
    return factors


def loadPositionFactor2(dir):
    dataProximity = []
    for root, dirs, files in os.walk(dir):
        for name in sorted(files):
            file = os.path.join(root, name)
            if "lat" in name or "lon" in name:
                data = loadGridData(file)
                print(np.max(data), np.min(data), name)
                data = data.reshape((data.shape[0], data.shape[1], 1))
                dataProximity.append(data)
    factors = dataProximity[0]
    for i in range(1, len(dataProximity)):
        factors = np.concatenate((factors, dataProximity[i]), axis=-1)
    return factors


def loadProximityDataEachYear(year, dir):
    dataProximity = []
    for root, dirs, files in os.walk(dir + 'Wuhan_F/DIS'):
        for name in sorted(files):
            if str(year) in name:
                file = os.path.join(root, name)
                # if "railways" in name or "secondary" in name or "tertiary" in name or "primary" in name \
                #         or "trunk" in name or "motorway" in name:
                print(name)
                data = loadGridData(file)
                datamax = np.max(data)
                datamin = np.min(data)
                data = 1 - (data - datamin) / (datamax - datamin)
                data = data.reshape((data.shape[0], data.shape[1], 1))
                dataProximity.append(data)

    factors = dataProximity[0]
    for i in range(1, len(dataProximity)):
        factors = np.concatenate((factors, dataProximity[i]), axis=-1)
    return factors


def loadNatureData(dir):
    dataNature = []
    for root, dirs, files in os.walk(dir + 'Wuhan_F/NA'):
        for name in sorted(files):
            if 'dem' in name or 'slope' in name:
                print(name)
                file = os.path.join(root, name)
                data = loadGridData(file)
                datamax = np.max(data)
                datamin = np.min(data)
                data = (data - datamin) / (datamax - datamin)
                data = data.reshape((data.shape[0], data.shape[1], 1))
                dataNature.append(data)

    factors = dataNature[0]
    for i in range(1, len(dataNature)):
        factors = np.concatenate((factors, dataNature[i]), axis=-1)
    return factors


def loadPositionData(dir):
    dataNature = []
    for root, dirs, files in os.walk(dir + 'Wuhan_F/NA'):
        for name in sorted(files):
            if 'Lon' in name or 'Lat' in name:
                print(name)
                file = os.path.join(root, name)
                data = loadGridData(file)
                data = data.reshape((data.shape[0], data.shape[1], 1))
                dataNature.append(data)

    factors = dataNature[0]
    for i in range(1, len(dataNature)):
        factors = np.concatenate((factors, dataNature[i]), axis=-1)
    return factors


def getValidValue(array, array2, radius):
    inArray = np.zeros((array.shape[0], array.shape[1]))
    nonurbanArray = np.zeros((array.shape[0], array.shape[1]))
    nonurbanNochange = np.zeros((array.shape[0], array.shape[1]))
    nonurbanGrowth = np.zeros((array.shape[0], array.shape[1]))
    for i in range(radius, array.shape[0] - radius):
        for j in range(radius, array.shape[1] - radius):
            data = np.array(array[i - radius: i + radius + 1, j - radius: j + radius + 1])
            if np.where(data == 65535)[0].shape[0] == 0:
                inArray[i][j] = 1
                if array[i, j] == 0:
                    nonurbanArray[i, j] = 1
                    if array2[i, j] == 0:
                        nonurbanNochange[i, j] = 1
                    elif array2[i, j] == 2:
                        nonurbanGrowth[i, j] = 1

    data_new_img = np.array(nonurbanGrowth * 255, dtype=np.uint8)
    img = Image.fromarray(data_new_img)
    img.save(os.path.join("./output/", "1.png"))
    # print(np.where(nonurbanArray == 1)[0].shape[0], np.where(nonurbanNochange == 1)[0].shape[0],
    #       np.where(nonurbanGrowth == 1)[0].shape[0])

    return nonurbanArray, nonurbanNochange, nonurbanGrowth, np.where(inArray.ravel() == 1)[0]


def getValidValue2(array, array2, radius, rowdata):
    inArray = np.zeros((array.shape[0], array.shape[1]))
    nonurbanArray = np.zeros((array.shape[0], array.shape[1]))
    nonurbanNochange = np.zeros((array.shape[0], array.shape[1]))
    nonurbanGrowth = np.zeros((array.shape[0], array.shape[1]))
    for i in range(radius, array.shape[0] - radius):
        for j in range(radius, array.shape[1] - radius):
            if rowdata[i][j] != 15:
                inArray[i][j] = 1
                if array[i, j] == 0:
                    nonurbanArray[i, j] = 1
                    if array2[i, j] == 0:
                        nonurbanNochange[i, j] = 1
                    elif array2[i, j] == 2:
                        nonurbanGrowth[i, j] = 1

    data_new_img = np.array(nonurbanGrowth * 255, dtype=np.uint8)
    img = Image.fromarray(data_new_img)
    img.save(os.path.join("./output/", "1.png"))

    return nonurbanArray, nonurbanNochange, nonurbanGrowth, np.where(inArray.ravel() == 1)[0]


def getValidValueF(array, array2, radius):
    inArray = np.zeros((array.shape[0], array.shape[1]))
    nonurbanArray = np.zeros((array.shape[0], array.shape[1]))
    nonurbanNochange = np.zeros((array.shape[0], array.shape[1]))
    nonurbanGrowth = np.zeros((array.shape[0], array.shape[1]))
    for i in range(radius, array.shape[0] - radius):
        for j in range(radius, array.shape[1] - radius):
            data = np.array(array[i - radius: i + radius + 1, j - radius: j + radius + 1])
            if np.where(data == -128)[0].shape[0] == 0:
                inArray[i][j] = 1
                if array[i, j] == 0:
                    nonurbanArray[i, j] = 1
                    if array2[i, j] == 0:
                        nonurbanNochange[i, j] = 1
                    elif array2[i, j] == 2:
                        nonurbanGrowth[i, j] = 1

    data_new_img = np.array(nonurbanGrowth * 255, dtype=np.uint8)
    img = Image.fromarray(data_new_img)
    img.save(os.path.join("./output2/", "1.png"))
    print(np.where(nonurbanArray == 1)[0].shape[0], np.where(nonurbanNochange == 1)[0].shape[0],
          np.where(nonurbanGrowth == 1)[0].shape[0])

    return nonurbanArray, nonurbanNochange, nonurbanGrowth, np.where(inArray.ravel() == 1)[0]


def generateImage(data, dir):
    img = np.zeros((data.shape[0], data.shape[1], 3), dtype="uint8")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] == 1:
                img[i, j, :] = [239, 228, 190]
            if data[i][j] == 2:
                img[i, j, :] = [0, 0, 255]
            if data[i][j] == 3:
                img[i, j, :] = [255, 0, 0]
    img = Image.fromarray(img)
    img.save(dir)


def OAandKappa(newLandPre, newLandTrue):
    oa = accuracy_score(newLandTrue, newLandPre)
    kappa_Value = cohen_kappa_score(newLandTrue, newLandPre)
    print("oa值为 %f" % oa, "\tkappa值为 %f" % kappa_Value)


def Fom(y_pred, y_true, x_true):
    hit = miss = false = 0
    number1 = number2 = number3 = 0
    for i in range(y_true.shape[0]):
        if x_true[i] == 0 and y_true[i] == 2 and y_pred[i] == 2:
            hit += 1
        if x_true[i] == 0 and y_true[i] == 0 and y_pred[i] == 2:
            false += 1
        if x_true[i] == 0 and y_pred[i] == 0 and y_true[i] == 2:
            miss += 1
        if x_true[i] == 0 and y_pred[i] == 2:
            number1 += 1
        if x_true[i] == 0 and y_true[i] == 2:
            number2 += 1
        if x_true[i] == 0 and y_true[i] == 1 and y_pred[i] == 2:
            number3 += 1

    fom = hit / (hit + miss + false)
    print(np.where(x_true == 0)[0].shape, x_true.shape)
    print(hit + miss, hit + false, number1, number2, number3)
    print("fom", fom, "hit", hit, "miss", miss, "false", false)
    return fom


def saveTif(array, dir):
    dataset = gdal.Open("/home/kwan2080/pan1/data/SCI_31/WUHAN/Land_2010R_WH.tif")
    print('处理图像的栅格波段数总共有：', dataset.RasterCount)

    projection = dataset.GetProjection()  # 投影
    transform = dataset.GetGeoTransform()  # 几何信息
    arr = dataset.ReadAsArray()

    row = arr.shape[0]  # 行数
    columns = arr.shape[1]  # 列数
    dim = 1  # 通道数

    driver = gdal.GetDriverByName('GTiff')
    # 创建文件
    dst_ds = driver.Create(dir, columns, row, dim, gdal.GDT_UInt32)
    # 设置几何信息
    dst_ds.SetGeoTransform(transform)
    dst_ds.SetProjection(projection)
    # 将数组写入
    dst_ds.GetRasterBand(1).WriteArray(array)
    # 写入硬盘
    dst_ds.FlushCache()
    dst_ds = None


def raster_to_polygon(rasterfile, nodata=0):
    '''
        rasterfile ： 输入要转换的栅格文件
    '''
    import rasterio as rio
    from rasterio import features
    from shapely.geometry import shape
    import geopandas as gpd
    import numpy as np

    out_shp = gpd.GeoDataFrame(columns=['category', 'geometry'])
    with rio.open(rasterfile) as f:
        image = f.read(1)
        img_crs = f.crs
        image[image == f.nodata] = nodata
        image = image.astype(np.float32)  # 上面那步把缺失值处理为0之后加上这步可以防止数据类型出错导致的报错
        i = 0
        id = 0
        for coords, value in features.shapes(image, transform=f.transform):
            id += 1
            if value != nodata:
                geom = shape(coords)
                out_shp.loc[i] = [value, geom]
                i += 1
    # print(i, id)
    out_shp.set_geometry('geometry', inplace=True)
    out_shp = out_shp.dissolve(by='category', as_index=False)
    out_shp.set_crs(img_crs, inplace=True)
    print('raster to polygon have finished!')
    out_shp.to_file("./output/output3.shp", encoding="utf-8", driver="ESRI Shapefile",
                    engine="pyogrio")
    return out_shp


def calMoran(array):
    number = np.where(array == 1)[0].shape[0] + np.where(array == 2)[0].shape[0]
    # print(number)
    row = array.shape[0]
    col = array.shape[1]
    y = []
    rows = []
    cols = []
    for i in range(row):
        for j in range(col):
            if array[i][j] == 1 or array[i][j] == 2:
                y.append(array[i][j])
                rows.append(i)
                cols.append(j)

    y = np.array(y) - 1
    rows = np.array(rows)
    cols = np.array(cols)
    y_mean = np.mean(y)
    weights_sum = 0
    y_dis_sum = 0
    yw_dis_sum = 0
    yw_dises = []
    number_sum = 0
    for i in tqdm(range(number)):
        i_row = rows[i]
        i_col = cols[i]

        dis_yi = y[i] - y_mean
        dis_yj = y - y_mean

        # for j in range(number):
        #     if i == j:
        #         w = 0
        #     else:
        #         j_row = rows[j]
        #         j_col = cols[j]
        #         dis = math.sqrt((i_row - j_row) * (i_row - j_row) + (i_col - j_col) * (i_col - j_col))
        #         if dis < 2:
        #             w = 1
        #         else:
        #             w = 0
        #     weights.append(w)
        dis_row = rows - i_row
        dis_col = cols - i_col
        dis_row = np.square(dis_row)
        dis_col = np.square(dis_col)
        dis = np.sqrt(dis_row + dis_col)
        weights = np.zeros(shape=(number,))
        weights[np.where((dis > 0) & (dis < 2))] = 1
        # for j in range(number):
        #     if dis[j] > 0 and dis[j] < 2:
        #         weights.append(1)
        #     else:
        #         weights.append(0)
        w_sum = np.sum(weights)
        if w_sum != 0:
            weights = weights / w_sum
        # else:
        #     continue
        weights_sum += np.sum(weights)
        yw_dis = weights * dis_yi
        yw_dis = np.multiply(yw_dis, dis_yj)
        yw_dises.append(np.sum(yw_dis))
        yw_dis_sum += np.sum(yw_dis)
        y_dis_sum += dis_yi * dis_yi
        number_sum += 1
        # if number_sum == 990:
        #     print((0.125 * 0.988 * 5 - 0.012 * 0.125 * 3) * 0.988)
        #     print(dis_yi, weights[np.where(weights != 0)], dis_yj[np.where(weights != 0)], np.sum(yw_dis))
        # if number_sum == 1000:
        #     break
    res = number_sum * yw_dis_sum / weights_sum / y_dis_sum
    yw_dises = np.array(yw_dises) * number_sum / y_dis_sum
    np.save("./output/moran2.npy", yw_dises)
    print(yw_dises, np.min(yw_dises), np.max(yw_dises), y_mean, number_sum / y_dis_sum, weights_sum, number_sum,
          y_dis_sum)
    print("Moran: ", res)


def calMoran1(array):
    number = np.where(array == 1)[0].shape[0] + np.where(array == 2)[0].shape[0]
    # print(number)
    row = array.shape[0]
    col = array.shape[1]
    y = []
    rows = []
    cols = []
    for i in range(row):
        for j in range(col):
            if array[i][j] == 1 or array[i][j] == 2:
                y.append(array[i][j])
                rows.append(i)
                cols.append(j)

    y = np.array(y) - 1
    rows = np.array(rows)
    cols = np.array(cols)
    y_mean = np.mean(y)
    weights_sum = 0
    y_dis_sum = 0
    yw_dis_sum = 0
    number_sum = 0
    for i in tqdm(range(number)):
        i_row = rows[i]
        i_col = cols[i]

        dis_yi = y[i] - y_mean
        dis_yj = y - y_mean

        # for j in range(number):
        #     if i == j:
        #         w = 0
        #     else:
        #         j_row = rows[j]
        #         j_col = cols[j]
        #         dis = math.sqrt((i_row - j_row) * (i_row - j_row) + (i_col - j_col) * (i_col - j_col))
        #         if dis < 2:
        #             w = 1
        #         else:
        #             w = 0
        #     weights.append(w)
        dis_row = rows - i_row
        dis_col = cols - i_col
        dis_row = np.square(dis_row)
        dis_col = np.square(dis_col)
        dis = np.sqrt(dis_row + dis_col)
        weights = np.zeros(shape=(number,))
        weights[np.where((dis > 0) & (dis < 2))] = 1
        # for j in range(number):
        #     if dis[j] > 0 and dis[j] < 2:
        #         weights.append(1)
        #     else:
        #         weights.append(0)
        w_sum = np.sum(weights)
        if w_sum == 0:
            continue
        weights_sum += np.sum(weights)
        yw_dis = weights * dis_yi
        yw_dis = np.multiply(yw_dis, dis_yj)
        yw_dis_sum += np.sum(yw_dis)
        y_dis_sum += dis_yi * dis_yi
        number_sum += 1

    res = number_sum * yw_dis_sum / weights_sum / y_dis_sum
    print("Moran: ", res)
