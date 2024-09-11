import numpy as np
from decode import decode
from math import log10, sqrt

RGB = np.array([0.299, 0.587, 0.114])

def PSNR(original, compressed): 
	mse = np.mean((original - compressed) ** 2) 
	if(mse == 0): 
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse)) 
	return psnr 


def predictV(value, grayij, X):
    beta = np.linalg.pinv(X.T * X) * X.T * value
    r_predict = np.linalg.det([1, grayij, grayij**2] * beta)
    if r_predict <= min(value[1, 0], value[0, 0]): r_predict = min(value[1, 0], value[0, 0])
    elif r_predict >= max(value[1, 0], value[0, 0]):
        r_predict = max(value[1, 0], value[0, 0])
    return np.round(r_predict)


def PEs(gray, img):
    pError = np.zeros(img.shape)
    predict = img.copy().astype(np.int32)
    rho = np.zeros(gray.shape)
    for i in range(2, img.shape[0] - 2):
        for j in range(2, img.shape[1] - 2):
            r = np.array([img[i + 1, j, 0], img[i, j + 1, 0], img[i + 1, j + 1, 0]]).reshape(3, 1)
            b = np.array([img[i + 1, j, 2], img[i, j + 1, 2], img[i + 1, j + 1, 2]]).reshape(3, 1)
            gr = np.array([gray[i + 1, j], gray[i, j + 1], gray[i + 1, j + 1]]).reshape(3, 1)
            X = np.mat(np.column_stack(([1] * 3, gr, gr**2)))
            predict[i, j, 0] = predictV(r, gray[i, j], X)
            predict[i, j, 2] = predictV(b, gray[i, j], X)
            pError[i, j] = img[i, j] - predict[i, j]
            rho[i, j] = np.var([gray[i - 1, j], gray[i, j - 1], gray[i, j], gray[i + 1, j], gray[i, j + 1]], ddof=1)
    return predict, pError, rho


def invariant(rgb):
    return np.round(rgb[:2].dot(RGB[:2]) + 2 * (rgb[2] // 2) * RGB[2]) == np.round(rgb[:2].dot(RGB[:2]) + (2 * (rgb[2] // 2) + 1) * RGB[2])


def embedMsg(img, gray, msg, mesL, selected, predict, pError, Dt):
    IMG, GRAY, pERROR = img.copy(), gray.copy(), pError.copy()
    tags = []
    La = 0
    tagsCode = '0'
    ec = 0
    location = 0
    msgIndex = 0
    for i in zip(*selected):
        if tags.count(0) < mesL:
            # 遍歷滿足 rho < rhoT 的像素點進行插入訊息
            pERROR[i][0] = 2 * pERROR[i][0] + int(msg[msgIndex])
            pERROR[i][2] = 2 * pERROR[i][2] + ec
            ec = abs(int(IMG[i][1] - np.round((GRAY[i] - IMG[i][0] * RGB[0] - IMG[i][2] * RGB[2]) / RGB[1])))
            rgb = np.array([predict[i][loc] + pERROR[i][loc] for loc in range(3)])
            rgb[1] = np.floor((GRAY[i] - rgb[0] * RGB[0] - rgb[2] * RGB[2]) / RGB[1])
            if np.round(rgb.dot(RGB)) != GRAY[i]:
                rgb[1] = np.ceil((GRAY[i] - rgb[0] * RGB[0] - rgb[2] * RGB[2]) / RGB[1])
            if np.round(rgb.dot(RGB)) != GRAY[i]: print(f'error: {i}')
            D = np.linalg.norm(rgb - IMG[i])
            if np.max(rgb) > 255 or np.min(rgb) < 0 or D > Dt:
                tags.append(1)  # 設置當前的tag為非法（tag為1）
            else:
                tags.append(0)
                msgIndex += 1
                IMG[i] = rgb
        else:
            if La == 0:
                if np.unique(tags).size > 1:
                    tagsCode, La = ''.join([str(char) for char in tags]), len(tags)
                else:
                    La = 1
            if location == La: break
            if invariant(IMG[i]):
                IMG[i][2] = 2 * (IMG[i][2] // 2) + int(tagsCode[location])
                location += 1
    if len(tags) < mesL or location < La: return False, ec, La, len(tags), tagsCode
    print(f"Message: {decode(msg)}")
    return (IMG, GRAY, pERROR), ec, La, len(tags), tagsCode


def cvtGray(img):
    gray = np.zeros(img.shape[:-1])
    for i in np.argwhere(img[:, :, -1]):
        gray[i] = np.round(img[i].dot(RGB))
    return gray
