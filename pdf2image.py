import redis
from redis_client import RedisClient
import os
import re
import sys
import time
import torch
import threading
import pdfplumber
from PIL import Image
from tqdm import tqdm
from pdf2image import convert_from_path
from transformers import DetrImageProcessor,TableTransformerForObjectDetection
from multiprocessing import Process

os.environ['KMP_DUPLICATE_LIB_OK']='True'
Image.MAX_IMAGE_PIXELS = None

batch_size = 20
output_path = '/mnt/SSD_732G/lm_corpus/images'

def Pdf2Image(batch_list):
    failed_list = []
    for file_name in batch_list:
        output_name = output_path + '/' + file_name.split('/')[-1].split('.')[0]
    
        #文件已存在
        if os.path.exists(output_name + '_0.png'):
            continue
        
        start_time = time.time()

        #将pdf转为图片
        try:
            images = convert_from_path(file_name, dpi = 150)
        except:
            print('{} is unable to convert to images.'.format(file_name))
            failed_list.append(file_name)
            continue
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print('pdf2image Time cost: {:0>2}:{:0>2}:{:05.2f}'.format( int(hours), int(minutes), seconds))
        
        #保存图片同时将所有图片的路径列表传入到redis中
        que = RedisClient('127.0.0.1', 6379)
        images_list = []
        images_list.append(file_name)
        for i, image in enumerate(images):
            fname = output_name + '_' + str(i) + ".png"
            image.save(fname, "PNG")
            images_list.append(fname)
        
        que.push('images', images_list)

    f = open(output_path.rstrip('/images') + '/txt_data/failed_list.txt', 'a', encoding='utf-8')
    for file in failed_list:
        f.write(file + '\n')
    f.close() 

#遍历root_path路径下的所有文件
def travel_files(batch, batch_list, input_path):
    if not os.path.isdir(input_path):
        sys.exit()
    lsdir = os.listdir(input_path)
    lsdir = [os.path.join(input_path, i) for i in lsdir]

    for file_name in lsdir:
        if os.path.isdir(file_name):
            travel_files(batch, batch_list, file_name)
        elif os.path.isfile(file_name):
            #每一个子进程处理一个batch的文件
            _format = file_name.split('.')[-1]
            if _format != 'pdf' and _format != 'PDF':
                continue
            batch.append(file_name)
            if len(batch) == batch_size:
                batch_list.append(batch[:])
                batch.clear()


if __name__ == '__main__':
    # model = TableTransformerForObjectDetection.from_pretrained("/cloud/data2/ethan/data/pdf2txt/microsoft_table-transformer-detection")
    batch = []
    batch_list = []
    travel_files(batch, batch_list, sys.argv[1])
    if len(batch) > 0:
        batch_list.append(batch[:])
        batch.clear()
    processes = []
    for i in range(0, len(batch_list)):
        processes.append(Process(target=Pdf2Image, args=(batch_list[i], )))
    for p in processes:
        p.start()
    for p in processes:
        p.join()