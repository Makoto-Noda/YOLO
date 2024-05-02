# %%
import cv2
import os
import shutil
import random
import numpy as np
import glob
from PIL import Image
import PySimpleGUI as sg

random.seed(123)
np.random.seed(123)

width = 640
height = 640

COLORS = [(0, 0, 175), (175, 0, 0), (0, 175, 0), (175, 0, 175)]

# %%
class Background:
    def __init__(self, backPath):
        self.__backPath = backPath

    def get(self):
        bgImagePath = random.choice(glob.glob(self.__backPath + '/bg*.png'))
        bgImage = cv2.imread(bgImagePath, cv2.IMREAD_UNCHANGED)
        bgImage = cv2.resize(bgImage, (int(width), int(height))) 
        return bgImage
    
class Item:
    def __init__(self, itemPath, className):
        self.__itemPath = itemPath
        self.__className = className

    def get(self, classID):

        itemImagePath = random.choice(glob.glob(f'{self.__itemPath}/item{classID}.png'))
        itemImage = cv2.imread(itemImagePath, cv2.IMREAD_UNCHANGED)
        
        dice = random.randint(1, 3)
        if dice == 1:
            itemImage = cv2.rotate(itemImage, cv2.ROTATE_90_CLOCKWISE)
        elif dice == 2:
            itemImage = cv2.rotate(itemImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif dice == 3:
            itemImage = cv2.rotate(itemImage, cv2.ROTATE_180)
        
        # 透過PNGへの変換
        # Point 1: 白色部分に対応するマスク画像を生成
        mask = np.all(itemImage[:, :, :] == [255, 255, 255], axis=-1)
        # Point 2: 元画像をBGR形式からBGRA形式に変換
        itemImage = cv2.cvtColor(itemImage, cv2.COLOR_BGR2BGRA)
        # Point3: マスク画像をもとに、白色部分を透明化
        itemImage[mask, 3] = 0

        return itemImage

# %%
class Transformer:
    def __init__(self):
        self.__width = width
        self.__height = height
        self.__min_scale = 0.4
        self.__max_scale = 1
    
    def __resize(self, itemImage):
        
        scale = random.uniform(self.__min_scale, self.__max_scale)
        h, w, _ = itemImage.shape
        
        return cv2.resize(itemImage, (int(w * scale), int(h * scale)))
    
    def __synthesize(self, itemImage, left, top):
        
        background_image = np.zeros((self.__height, self.__width, 4), np.uint8)
        back_pil = Image.fromarray(background_image)
        front_pil = Image.fromarray(itemImage)
        back_pil.paste(front_pil, (left, top), front_pil)
        
        return np.array(back_pil)    

    
    def warp(self, itemImage):

        itemImage = self.__resize(itemImage)
        
        h, w, _ = itemImage.shape
        if self.__width - w >=0:
            left = random.randint(0, self.__width - w)
        else:
            left = 0
        if self.__height - h >=0:
            top = random.randint(0, self.__height - h)
        else:
            top = 0
        rectangle = ((left, top), (left + w, top + h))

        newImage = self.__synthesize(itemImage, left, top)
        
        return (newImage, rectangle)

# %%
class Effecter:
    
    def gauss(self, img, level):
        
        return cv2.blur(img, (level * 2 + 1, level * 2 + 1))

    def noise(self, img):
        img = img.astype("float64")
        img[:, :, 0] = self.__single_channel_noise(img[:, :, 0])
        img[:, :, 1] = self.__single_channel_noise(img[:, :, 1])
        img[:, :, 2] = self.__single_channel_noise(img[:, :, 2])
        
        return img.astype("uint8")

    def __single_channel_noise(self, single):
        diff = 255 - single.max()
        noise = np.random.normal(0, random.randint(1, 100), single.shape)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = diff * noise
        noise = noise.astype(np.uint8)
        dst = single + noise
        
        return dst

# %%
def box(frame, rectangle, classID):
    
    ((x1, y1), (x2, y2)) = rectangle
    label = CLASS_NAME[classID]
    img = cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[classID], 2)
    img = cv2.rectangle(img, (x1, y1), (x1 + 150, y1 - 20), COLORS[classID], -1)
    cv2.putText(
        img,
        label,
        (x1 + 2, y1 - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,)
    
    return img

# %%
def create_label(rectangle, classID):
    
    ((x1, y1), (x2, y2)) = rectangle
    center_x = ((x1 + x2) / 2) / width
    center_y = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    
    label = f'{classID} {center_x} {center_y} {w} {h}'
    
    return label

# %%
def marge_image(background_image, front_image):
    
    back_pil = Image.fromarray(background_image)
    front_pil = Image.fromarray(front_image)
    back_pil.paste(front_pil, (0, 0), front_pil)
    
    return np.array(back_pil)

# %%
class Data:
    
    def __init__(self, rate):
        self.__rectangles = []
        self.__images = []
        self.__classIDs = []
        self.__rate = rate

    def get_classIDs(self):
        return self.__classIDs

    def max(self):
        return len(self.__rectangles)

    def get(self, i):
        return (self.__images[i], self.__rectangles[i], self.__classIDs[i])
    
    # 重複率
    def __iou(self, a, b):
        
        (ax_mn, ay_mn) = a[0]
        (ax_mx, ay_mx) = a[1]
        (bx_mn, by_mn) = b[0]
        (bx_mx, by_mx) = b[1]
        a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
        b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)
        abx_mn = max(ax_mn, bx_mn)
        aby_mn = max(ay_mn, by_mn)
        abx_mx = min(ax_mx, bx_mx)
        aby_mx = min(ay_mx, by_mx)
        w = max(0, abx_mx - abx_mn + 1)
        h = max(0, aby_mx - aby_mn + 1)
        intersect = w * h
        
        return intersect / (a_area + b_area - intersect)

    # 追加（重複率が指定値以上の場合は失敗する）
    def append(self, itemImage, rectangle, classID):
        
        conflict = False
        
        for i in range(len(self.__rectangles)):
            iou = self.__iou(self.__rectangles[i], rectangle)
            if iou > self.__rate:
                conflict = True
                break
        if conflict == False:
            self.__rectangles.append(rectangle)
            self.__images.append(itemImage)
            self.__classIDs.append(classID)
            return True
        
        return False

# %%
class Counter:
    
    def __init__(self, max):
        self.__counter = np.zeros(max)

    def get(self):
        n = np.argmin(self.__counter)
        return int(n)

    def increase(self, index):
        self.__counter[index] += 1

    def print(self):
        print(self.__counter)

# %%
def main(n_of_data, item_dir, bg_dir, CLASS_NAME, images_output_dir, labels_output_dir, images_box_output_dir):

    if os.path.exists(images_output_dir):
        shutil.rmtree(images_output_dir)
    os.mkdir(images_output_dir)
    
    if os.path.exists(images_box_output_dir):
        shutil.rmtree(images_box_output_dir)
    os.mkdir(images_box_output_dir)
    
    if os.path.exists(labels_output_dir):
        shutil.rmtree(labels_output_dir)
    os.mkdir(labels_output_dir)

    item = Item(item_dir, CLASS_NAME)
    background = Background(bg_dir)

    transformer = Transformer()
    counter = Counter(len(CLASS_NAME))
    effecter = Effecter()

    n_of_data = int(n_of_data)
    
    n = 0
    
    while True:

        backgroundImage = background.get()

        rate = 0.05
        data = Data(rate)
        labels = []
        
        for _ in range(10):
            # 現時点で作成数の少ないクラスIDを取得
            classID = counter.get()

            itemImage = item.get(classID)

            (transformImage, rectangle) = transformer.warp(itemImage)
            frame = marge_image(backgroundImage, transformImage)

            bool = data.append(transformImage, rectangle, classID)
            if bool:
                counter.increase(classID)
        
        frame = backgroundImage
        
        for index in range(data.max()):
            (itemImage, rectangle, classID) = data.get(index)
            frame = marge_image(frame, itemImage)
            label = create_label(rectangle, classID)
            labels.append(label)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frame = effecter.gauss(frame, random.randint(0, 2))
        frame = effecter.noise(frame)

        fileName = f'image{n:04}.png'
        fileName_box = f'box{n:04}.png'
        labelName = f'image{n:04}.txt'
        
        cv2.imwrite(f'{images_output_dir}/{fileName}', frame)
        
        for i in range(data.max()):
            (_, rectangle, classID) = data.get(i)
            frame_box = box(frame, rectangle, classID)
       
        cv2.imwrite(f'{images_box_output_dir}/{fileName_box}', frame_box)

        with open(f'{labels_output_dir}/{labelName}', 'w') as f:
            for line in labels:
                f.write(line + "\n")
        
        n += 1
        
        if n_of_data <= n:
            break


# %%
sg.theme('Default1')

layout = [  [sg.Text('生成する学習データの数を指定してください'), sg.Input('100', key='NOfData')],
            [sg.Text('商品画像のフォルダを選んでください(商品画像ファイル名: item{クラスID}.png)'), sg.Input(), sg.FolderBrowse('フォルダを選択', key='ItemDir')],
            [sg.Text('背景画像のフォルダを選んでください(背景画像ファイル名: bg*.png)'), sg.Input(), sg.FolderBrowse('フォルダを選択', key='BgDir')],
            [sg.Text('クラスIDに対応したクラス名の設定ファイルを指定してください(txt形式)'), sg.Input(''), sg.FileBrowse('ファイルを選択', key='ClassName')],
            [sg.Text('生成したデータの出力先を選んでください'), sg.Input(), sg.FolderBrowse('フォルダを選択', key='ImagesOutputDir')],
            [sg.Text('アノテーションラベルの出力先を選んでください'), sg.Input(), sg.FolderBrowse('フォルダを選択', key='LabelsOutputDir')],
            [sg.Text('アノテーション確認用画像ファイルの出力先を選んでください'), sg.Input(), sg.FolderBrowse('フォルダを選択', key='BoxedImagesOutputDir')],
            [sg.Button('OK'), sg.Button('Cancel')]]

window = sg.Window('学習データ生成システム', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':   
        break   
    elif event == 'OK':    
        n_of_data = values['NOfData']
        item_dir = values['ItemDir']
        bg_dir = values['BgDir']
        CLASS_NAME = []
        with open(values['ClassName'], "r") as file:
            for line in file:
                CLASS_NAME.append(line.strip())
        images_output_dir = values['ImagesOutputDir']
        labels_output_dir = values['LabelsOutputDir']
        images_box_output_dir = values['BoxedImagesOutputDir']
        
        main(n_of_data, item_dir, bg_dir, CLASS_NAME, images_output_dir, labels_output_dir, images_box_output_dir)

            
window.close()

# %%



