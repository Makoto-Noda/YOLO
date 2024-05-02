# %%
from ultralytics import YOLO
import os
import shutil
from PIL import Image
import PySimpleGUI as sg
import io

# %%
conf = 0.1

# %%
class PredictAndView:
    
    def __init__(self, model, image_path, output_dir, conf):

        self.image_path = image_path
        self.output_dir = output_dir
        self.model = model
        self.conf = conf
    
    def _predict(self):
        
        result = self.model.predict(image_path, project=self.output_dir, save=True, imgsz=640, conf=self.conf, exist_ok=True)
        
        result_object = result[0]
        pred_classes = result_object.boxes.cls.tolist()
        pred_classes = [int(c) for c in pred_classes]
        pred_names = {0:'美しい日本語選び辞典', 1:'古代中国の24時間', 2:'チェンソーマン12', 3:'中国語ナビ2022年4月号'}
        pred_boxes = result_object.boxes.xywhn.tolist()
        
        unique_classes = []
        for pred_classe in pred_classes:
            if pred_classe not in unique_classes:
                unique_classes.append(pred_classe)
            
        classes_count= {}
        for classe_value in unique_classes:
            classes_count[classe_value] = 0
        
        for pred_class in pred_classes:
            if pred_class in classes_count:
                classes_count[pred_class] += 1
                
        str_list = []
        for i, pred_class in enumerate(unique_classes):
            globals()[f'str_{i}'] = f'商品名: {pred_names[pred_class]}, 数量：{classes_count[pred_class]}\n'
            str_list.append(globals()[f'str_{i}'])
        
        display_str = ""
        for str in str_list:
            display_str += str
        
        return pred_boxes, display_str
        
    def _alert(self):
        
        pred_boxes, display_str = self._predict()
        
        coordinates = []
        for sublist in pred_boxes:
            coordinates.extend(sublist[:2])
        
        detection = False
        
        if len(coordinates) == 0:
            detection = False
        else:
            detection = True
        
        alert_trigger = False

        alert_str = ''
        for coordinate in coordinates:
            if coordinate < 0.1 or coordinate > 0.9:
                alert_str = '商品を枠の中央に寄せてください'
                alert_trigger = True
            else:
                alert_trigger = False

        return detection, alert_trigger, alert_str, display_str, coordinates
    
    def view(self):
        
        detection, alert_trigger, alert_str, display_str, coordinates = self._alert()
        
        input_image_name = self.image_path.split('/')[-1]
        output_image_path = f'{self.output_dir}/predict/{input_image_name}'
        
        return detection, alert_trigger, alert_str, display_str, coordinates, output_image_path

# %%
def main(model_path, image_path, output_dir):
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    
    model = YOLO(model_path)
    
    predict_and_view = PredictAndView(model, image_path, output_dir, conf)
    
    detection, alert_trigger, alert_str, display_str, coordinates, output_image_path = predict_and_view.view()
    
    return detection, alert_trigger, alert_str, display_str, coordinates, output_image_path

# %%
sg.theme('Default1')

layout = [  [sg.Text('使用するAIモデルを選択してください'), sg.Input(), sg.FileBrowse('ファイルを選択', key='InputModelPath')],
            [sg.Text('識別したい画像を選んでください'), sg.Input(), sg.FileBrowse('ファイルを選択', key='InputFilePath')],
            [sg.Text('識別結果の出力先フォルダを選んでください'), sg.Input(), sg.FolderBrowse('フォルダを選択', key='OutputFolderPath')],
            [sg.Button('OK'), sg.Button('Cancel')]]

window = sg.Window('AI商品識別システム', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':   
        break   
    elif event == 'OK': 
        model_path = values['InputModelPath']
        image_path = values['InputFilePath']
        output_dir = values['OutputFolderPath']
        detection, alert_trigger, alert_str, display_str, coordinates, output_image_path = main(model_path, image_path, output_dir)
        if len(alert_str) > 0:
            sg.popup(alert_str)
        elif len(display_str) > 0:            
            output_image = Image.open(output_image_path)
            w, h = output_image.size
            ratio = h / w
            new_w = 640
            new_h = int(new_w * ratio)
            output_image = output_image.resize((new_w, new_h))
            bio = io.BytesIO()
            output_image.save(bio, format='PNG')
            del output_image
            output = bio.getvalue()
            sg.popup_ok(display_str, title='識別結果', image=output)
        else:
            pass
            
window.close()


