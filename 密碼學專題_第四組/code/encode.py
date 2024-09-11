from collections import defaultdict
from fractions import Fraction
import cv2
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


def encode_fraction_range(input_codes, input_prob):
    start = Fraction(0, 1)
    width = Fraction(1, 1)

    for code in input_codes:
        d_start, d_width = input_prob[code]
        start += d_start * width
        width *= d_width

    return start, start + width


def find_binary_fraction(input_start, input_end):
    output_fraction = Fraction(0, 1)
    output_denominator = 1

    while not (input_start <= output_fraction < input_end):
        output_numerator = 1 + ((input_start.numerator * output_denominator) // input_start.denominator)
        output_fraction = Fraction(output_numerator, output_denominator)
        output_denominator *= 2

    return output_fraction

def encode(msg):
    return ''.join([f"{bin(ord(i))[2:]:>08}" for i in msg])


if __name__ == '__main__':

    Dt = 20
    rhoT = 0
    msg = 'welovecryptography'
    mesL = len(encode(msg))
    print(f"Message you want to encode : {msg}")
    
    org_img = cv2.imread('input.jpg')
    img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    gray = cvtGray(img)

    print(f'{img}\nFinish reading image!')
    predict, pError, rho = PEs(gray, img)
    print(f'Finish calculating predication error!')
    
    # 根據訊息長度初選 ⍴
    while np.count_nonzero(rho < rhoT) <= mesL:
        if np.count_nonzero(rho < rhoT) == rho.size:
            print('The picture is too small! Exit!')
            exit()
        rhoT += 1
        
    # 考慮參數後再選 ⍴
    enough = 0
    while not enough:
        selected = [n + 2 for n in np.where(rho[2:-2, 2:-2] < rhoT)]
        if selected[0].size >= (img.shape[0] - 4)**2:
            print('The picture is too small! Exit!')
            exit()
        enough, lastEc, La, N, tagsCode = embedMsg(img, gray, encode(msg), mesL, selected, predict, pError, Dt)
        rhoT += 0 if enough else 1
    print(f'Finish embeding msg with the critical value of ⍴ being {rhoT}')
    img, gray, pError = enough
    
    # 在邊框中嵌入參數
    border = sorted(
        list(
            set(map(tuple, np.argwhere(gray == gray))) -
            set(map(tuple,
                    np.argwhere(gray[1:-1, 1:-1] == gray[1:-1, 1:-1]) + 1))))
    border = list(filter(lambda xy: invariant(img[xy]), border))
    if len(border) < 56:
        print('The size of image is too small to contain the necessary parameters')
        exit()
    for char, loc in zip(f'{rhoT:016b}' + f'{lastEc:08b}' + f'{La:016b}' + f'{N:016b}',
                         filter(lambda xy: invariant(img[xy]), border)):
        img[loc][2] = 2 * (img[loc][2] // 2) + int(char)
    # print(f'=> Finish embeding parameters:\n\trhoT: {rhoT}, lastEc: {lastEc}, La: {La}, N: {N}, tagsCode: {tagsCode}')
    
    cv2.imwrite("encoded_image.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print("encode -> PSNR : ", PSNR(org_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
