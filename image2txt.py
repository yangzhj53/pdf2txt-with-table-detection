import redis
from redis_client import RedisClient
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
import subprocess
from pypdf import PdfReader

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Image.MAX_IMAGE_PIXELS = None

output_path = '/mnt/SSD_732G/lm_corpus/txt_data'

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

#删除转换完毕的图片
def remove_images(images_list):
    for i in range(1, len(images_list)):
        subprocess.run(['rm', images_list[i]])

def pdf_convert():
    que = RedisClient('127.0.0.1', 6379)
    images_str = que.pop('images')
    
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    # 使用GPU时去掉encoding和model的注释
    model.to(device)
    
    failed_list = []

    while images_str != None:
        images_list = images_str.lstrip('[').rstrip(']').split(',')
        for i in range(len(images_list)):
            images_list[i] = images_list[i].strip(' ').strip('\'')
        file_name = images_list[0]
        start_time = time.time()
        data_text = []
        total_text = []
        
        print(file_name + ' begins to convert.')
        try:
            p = pdfplumber.open(file_name)
        except:
            print('{} is unable to open.'.format(file_name))
            images_str = que.pop('images')
            failed_list.append(file_name)
            remove_images(images_list)
            continue

        output_name = output_path + '/' + file_name.split('/')[-1].split('.')[0] + '.txt'
        
        #文件已存在
        if os.path.exists(output_name):
            print('{} already exists.'.format(file_name))
            images_str = que.pop('images')
            remove_images(images_list)
            continue
        
        #表格识别
        with pdfplumber.open(file_name) as p:
            try:
                minn = min(len(p.pages),len(images_list) - 1)
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
                    failed_list.append(file_name)
                    images_str = que.pop('images')
                    remove_images(images_list)
                    continue
                else:
                    data = open(output_name, 'w', encoding='utf-8')
                    for text_ in data_text:
                        data.write(text_)
                    data.close()
                    images_str = que.pop('images')
                    remove_images(images_list)
                    continue
                
            for i in range(len(p.pages)):

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
                image_path = images_list[i + 1]
                image = Image.open(image_path).convert("RGB")
                width, height = image.size
                rdx = width / page.width       #转为图片后长宽会发生同比变化

                feature_extractor = DetrImageProcessor()
                encoding = feature_extractor(image, return_tensors="pt")
                #使用GPU时去掉encoding和model的注释
                encoding.to(device)
                with torch.no_grad():
                    outputs = model(**encoding)
                
                #识别结果
                tables = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]   

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
                

                
                #写入文本内容
                for line in after_text_lines:
                    if len(line) < 2:
                        continue
                    data_text.append(line + '\n')   
                
        
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
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print('{}\'s convertion is finished. Time cost: {:0>2}:{:0>2}:{:05.2f}'.format(file_name, int(hours), int(minutes), seconds))
        images_str = que.pop('images')
        remove_images(images_list)
    
    f = open(output_path + '/failed_list.txt', 'a', encoding='utf-8')
    for file in failed_list:
        f.write(file + '\n')
    f.close()

if __name__ == '__main__':
    pdf_convert()
