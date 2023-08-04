import os
import re
import sys
import time
import torch
import queue
import threading
import pdfplumber
from PIL import Image
from tqdm import tqdm
from pdf2image import convert_from_path
from transformers import DetrImageProcessor,TableTransformerForObjectDetection
from multiprocessing import Process
from pypdf import PdfReader

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Image.MAX_IMAGE_PIXELS = None

#判断元素是否在表格内
def check(word, tables, rdx):
    boxes_list = tables['boxes'].tolist()
    for i in range(len(boxes_list)):
        if len(boxes_list) == 0:
            continue
        xmin, ymin, xmax, ymax = boxes_list[i]
        xmid = (word['x0'] + word['x1']) * rdx / 2
        ymid = (word['top'] + word['bottom']) * rdx / 2
        if xmid >= xmin and xmid <= xmax and ymid >= ymin and ymid <= ymax:
            return True
    return False


#先读左边部分，再读右边部分
def recombine_text(lines, mid_words, left_words, right_words):
    mid_lines = []
    left_lines = []
    right_lines = []
    for line in lines:
        words = line.split(' ')
        left_tmp = []
        right_tmp = []
        mid_tmp = []
        for word in words:
            if word in mid_words:
                mid_tmp.append(word)
                continue
            if word not in right_words:
                left_tmp.append(word)
            if word not in left_words:
                right_tmp.append(word)
        mid_lines.append(' '.join(mid_tmp))
        left_lines.append(' '.join(left_tmp))
        right_lines.append(' '.join(right_tmp))
    
    return mid_lines + left_lines + right_lines


def pdf_convert(model, file_name, output_path):
    print(file_name + ' begins to convert.')
    try:
        p = pdfplumber.open(file_name)
    except:
        print(file_name + 'is unable to open.')
        return False
    
    output_name = output_path + '/' + file_name.split('/')[-1].split('.')[0] + '.txt'
    
    #文件已存在
    if os.path.exists(output_name):
        return True
    
    data_text = []
    total_text = []

    start_time = time.time()
    #将pdf转为图片
    try:
        images = convert_from_path(file_name, dpi = 150)
    except:
        print('{} is unable to convert to images.'.format(file_name))
        return False
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('pdf2image Time cost: {:0>2}:{:0>2}:{:05.2f}'.format( int(hours), int(minutes), seconds))

    # for i, image in enumerate(images):
    #     fname = "image" + str(i) + ".png"
    #     image.save(fname, "PNG")
    table_detection_time = 0.0
    delete_time = 0.0
    recombline_time = 0.0
    write_time = 0.0

    #表格识别
    with pdfplumber.open(file_name) as p:
        try:
            minn = min(len(p.pages),len(images))
        except:
            print(file_name + ' has page problem.')
            try:
                reader = PdfReader(file_name)
                for i in range(len(reader.pages)):
                    page = reader.pages[i]
                    text = page.extract_text()
                    data_text.append(text + '\n')
            except:
                print(file_name + ' has page problem. ---double')
                return False
            else:
                data = open(output_name, 'w', encoding='utf-8')
                for text_ in data_text:
                    data.write(text_)
                data.close()
                return True
            
        for i in range(min(len(p.pages),len(images))):
            page = p.pages[i]
            try:
                text = page.extract_text()
            except:
                print('Page ' + str(i) + ' is unable to extract text.')
                continue
            total_text.append(text)
            words = page.extract_words()
            table_words_list = []

            #整份PDF都为一个表
            # for j in range(0, min(8, len(words))):
            #     if '记录表' in words[j]['text']:
            #         full_table_flag = True
            # if full_table_flag == True:
            #     tables = page.extract_tables()
            #     for table in tables:
            #         for line in table:
            #             for ele in line:
            #                 if ele is None:
            #                     continue
            #                 data.write(ele + '\n')
            #     continue

            
            #将pdf转为图片，用于后面的表格识别
            # image_path = 'image' + str(i) + '.png'
            # image = Image.open(image_path).convert("RGB")
            image = images[i]
            width, height = image.size
            rdx = width / page.width       #转为图片后长宽会发生同比变化
            
            start_time = time.time()

            feature_extractor = DetrImageProcessor()
            encoding = feature_extractor(image, return_tensors="pt")
            #使用GPU时去掉encoding和model的注释
            #encoding.to(device)
            with torch.no_grad():
                outputs = model(**encoding)
            
            #识别结果
            tables = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]   

            end_time = time.time()
            table_detection_time += end_time - start_time

            start_time = time.time()
            #如果是表中元素，加入到list中
            for word in words:
                if check(word, tables, rdx):
                    table_words_list.append(word['text'])
            # print(table_words_list)
            
            #去掉表中元素
            text_lines = text.split('\n')
            after_text_lines = []
            for text_line in text_lines:
                text_words = text_line.split(' ')
                after_text_words = []
                for text_word in text_words:
                    if text_word not in table_words_list:
                        after_text_words.append(text_word)
                after_text_line = ' '.join(after_text_words)
                after_text_lines.append(after_text_line)
            
            end_time = time.time()
            delete_time += end_time - start_time
            
            start_time = time.time()
            #分左右
            #----
            # 先计算所有words元素的x1-x0值，并存储在一个列表中
            x_diffs = [(word['x1'] - word['x0']) for word in words]

            # 用一个字典存储x0和x1的值
            x_values = [{'x0': word['x0'], 'x1': word['x1']} for word in words]
            flag = any(x > 250 and x_values[i]['x0'] < 150 for i, x in enumerate(x_diffs))

            boundary = 300
            for i in range(290):
                tmp1 = 300 + i
                tmp2 = 300 - i
                flag1 = all(not (x <= 400 and x_values[i]['x0'] <= tmp1 and x_values[i]['x1'] >= tmp1) for i, x in
                            enumerate(x_diffs))
                flag2 = all(not (x <= 400 and x_values[i]['x0'] <= tmp2 and x_values[i]['x1'] >= tmp2) for i, x in
                            enumerate(x_diffs))
                if flag1:
                    boundary = tmp1
                    break
                if flag2:
                    boundary = tmp2
                    break

            mid_words = []
            left_words = []
            right_words = []
            for i, word in enumerate(words):
                if (x_values[i]['x1'] - x_values[i]['x0']) > 450:
                    mid_words.append(word['text'])
                    continue
                if (x_values[i]['x0'] + x_values[i]['x1']) / 2 < boundary:
                    left_words.append(word['text'])
                else:
                    right_words.append(word['text'])

            if len(mid_words) <= 0.3 * len(text_lines):
                after_text_lines = recombine_text(after_text_lines, mid_words, left_words, right_words)
            
            end_time = time.time()
            recombline_time += end_time - start_time
            
            #写入文本内容
            start_time = time.time()
            for line in after_text_lines:
                if len(line) < 2:
                    continue
                data_text.append(line + '\n')   
            
            end_time = time.time()
            write_time += end_time - start_time
    
    start_time = time.time()
    data = open(output_name, 'w', encoding='utf-8')

        #整个PDF为一张表
    if len(data_text) == 0:
        p = pdfplumber.open(file_name)
        for i in range(len(p.pages)):
            page = p.pages[i]
            tables = page.extract_tables()
            for table in tables:
                for line in table:
                    data_text.append('\n'.join(ele for ele in line if ele is not None) + '\n')
    
    #仍然没有任何内容
    if len(data_text) == 0:
        for text in total_text:
            data.write(text)
    
    for text_ in data_text:
        data.write(text_)
    data.close()
    end_time = time.time()
    write_time += end_time - start_time
   
    hours, rem = divmod(table_detection_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('table detection Time cost: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
    hours, rem = divmod(delete_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('delete Time cost: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
    hours, rem = divmod(recombline_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('recombine Time cost: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
    hours, rem = divmod(write_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('write Time cost: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))

    return True


#遍历root_path路径下的所有文件
def travel_files(input_path, output_path):
    if not os.path.isdir(input_path):
        sys.exit()
    lsdir = os.listdir(input_path)
    lsdir = [os.path.join(input_path, i) for i in lsdir]
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    # 使用GPU时去掉encoding和model的注释
    # model.to(device)
    failed_list = []
    for file_name in lsdir:
        if os.path.isdir(file_name):
            travel_files(file_name, output_path)
        elif os.path.isfile(file_name):
            #每一个子进程处理一个batch的文件
            _format = file_name.split('.')[-1]
            if _format != 'pdf' and _format != 'PDF':
                continue
            start_time = time.time()

            result = pdf_convert(model, file_name, output_path)

            end_time = time.time()
            elapsed_time = end_time - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            if result == True:
                print('{}\'s convertion is finished. Time cost: {:0>2}:{:0>2}:{:05.2f}'.format(file_name, int(hours), int(minutes), seconds))
            else:
                failed_list.append(file_name)
                print('{} is unable to convert.'.format(file_name))
    f = open(output_path + '\\failed_list.txt', 'a', encoding='utf-8')
    for file in failed_list:
        f.write(file + '\n')
    f.close()


if __name__ == '__main__':
    travel_files(sys.argv[1], sys.argv[2])

