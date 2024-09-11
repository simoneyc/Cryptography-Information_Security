from collections import defaultdict
import cv2
import numpy as np
from fractions import Fraction
from functions import *

def build_prob(input_codes):
    counts = defaultdict(int)

    for code in input_codes:
        counts[code] += 1

    counts[256] = 1

    output_prob = dict()
    length = len(input_codes)
    cumulative_count = 0

    for code in sorted(counts, key=counts.get, reverse=True):
        current_count = counts[code]
        prob_pair = Fraction(cumulative_count, length), Fraction(current_count, length)
        output_prob[code] = prob_pair
        cumulative_count += current_count

    return output_prob

def find_binary_fraction(input_start, input_end):
    output_fraction = Fraction(0, 1)
    output_denominator = 1

    while not (input_start <= output_fraction < input_end):
        output_numerator = 1 + ((input_start.numerator * output_denominator) // input_start.denominator)
        output_fraction = Fraction(output_numerator, output_denominator)
        output_denominator *= 2

    return output_fraction


def decode_fraction(input_fraction, input_prob):
    output_codes = []
    code = 257

    while code != 256:
        for code, (start, width) in input_prob.items():
            if 0 <= (input_fraction - start) < width:
                input_fraction = (input_fraction - start) / width

                if code < 256:
                    output_codes.append(code)
                break

    return ''.join([chr(code) for code in output_codes])

def decode(msg):
    return ''.join([chr(int(i, 2)) for i in (msg[i:i + 8] for i in range(0, len(msg), 8))])

if __name__ == '__main__':
    
    # 讀取嵌入訊息並計算 predication error
    encode_img = cv2.imread("encoded_image.png")
    imgRcv = cv2.cvtColor(encode_img, cv2.COLOR_BGR2RGB)
    grayRcv = cvtGray(imgRcv)
    predictRcv, pErrorRcv, rhoRcv = PEs(grayRcv, imgRcv)
    print(f'Finish reading embeded image and calculating predication error!')

    border = sorted(
        list(
            set(map(tuple, np.argwhere(grayRcv == grayRcv))) -
            set(map(tuple,
                    np.argwhere(grayRcv[1:-1, 1:-1] == grayRcv[1:-1, 1:-1]) + 1))))
    border = [str(imgRcv[loc][2] % 2) for loc in filter(lambda xy: invariant(imgRcv[xy]), border)]
    rhoT = int(''.join(border[:16]), 2)
    lastEc = int(''.join(border[16:24]), 2)
    La = int(''.join(border[24:40]), 2)
    N = int(''.join(border[40:56]), 2)
    selected = [tuple(n + 2) for n in np.argwhere(rhoRcv[2:-2, 2:-2] < rhoT)]
    tagsCode = [imgRcv[value][2] % 2
                for value in filter(lambda xy: invariant(imgRcv[xy]), selected[N:])][:La] if La != 1 else [0] * N
    # print(
    #     f'Finish extractig parameters:\n\trhoT: {rhoT}, lastEc: {lastEc}, La: {La}, N: {N}, tagsCode: {"".join([str(i) for i in tagsCode])}'
    # )
    # 根據參數提取嵌入的訊息
    candidate = reversed([selected[:N][index] for index, value in enumerate(tagsCode) if value == 0])
    predictRcv = imgRcv.copy().astype(np.int32)
    pErrorRcv = np.zeros(imgRcv.shape)
    msgRcv = ''
    for i in candidate:
        rM = np.array([imgRcv[i[0] + 1, i[1], 0], imgRcv[i[0], i[1] + 1, 0],
                       imgRcv[i[0] + 1, i[1] + 1, 0]]).reshape(3, 1)
        bM = np.array([imgRcv[i[0] + 1, i[1], 2], imgRcv[i[0], i[1] + 1, 2],
                       imgRcv[i[0] + 1, i[1] + 1, 2]]).reshape(3, 1)
        grM = np.array([grayRcv[i[0] + 1, i[1]], grayRcv[i[0], i[1] + 1], grayRcv[i[0] + 1, i[1] + 1]]).reshape(3, 1)
        X = np.mat(np.column_stack(([1] * 3, grM, grM**2)))
        predictRcv[i][0] = predictV(rM, grayRcv[i], X)
        predictRcv[i][2] = predictV(bM, grayRcv[i], X)
        pErrorRcv[i] = imgRcv[i] - predictRcv[i]

        msgRcv += str(int(pErrorRcv[i][0]) % 2)

        nextEc = pErrorRcv[i][2] % 2
        pErrorRcv[i] = pErrorRcv[i] // 2
        imgRcv[i] = predictRcv[i] + pErrorRcv[i]
        imgRcv[i][1] = np.round((grayRcv[i] - imgRcv[i][0] * RGB[0] - imgRcv[i][2] * RGB[2]) / RGB[1])
        if lastEc != 0:
            if np.round(np.array([imgRcv[i][0], imgRcv[i][1] + lastEc, imgRcv[i][2]]).dot(RGB)) == grayRcv[i]:
                imgRcv[i][1] += lastEc
            elif np.round(np.array([imgRcv[i][0], imgRcv[i][1] - lastEc, imgRcv[i][2]]).dot(RGB)) == grayRcv[i]:
                imgRcv[i][1] -= lastEc
        else:
            if np.round(np.array([imgRcv[i][0], imgRcv[i][1], imgRcv[i][2]]).dot(RGB)) != grayRcv[i]:
                print(f"index {i} has no matched ec")
        lastEc = abs(nextEc)
    print(f"Received Message: {decode(msgRcv[::-1])}")
    # cv2.imwrite("decoded_image.png", cv2.cvtColor(imgRcv, cv2.COLOR_RGB2BGR))
    
    # org_img = cv2.imread('input.jpg')
    # print("decode -> PSNR : ", PSNR(org_img, imgRcv))
