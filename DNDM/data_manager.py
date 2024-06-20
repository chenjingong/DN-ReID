from __future__ import print_function, absolute_import
import os
import numpy as np
import random
import pdb

def process_query_sysu(data_path, mode = 'all', relabel=False):
    if mode== 'all':
        ir_cameras = ['cam3','cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3','cam6']
    
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_sysu(data_path, mode = 'all', trial = 0, relabel=False):
    
    random.seed(trial)
    
    if mode== 'all':
        rgb_cameras = ['cam1','cam2','cam4','cam5']
    elif mode =='indoor':
        rgb_cameras = ['cam1','cam2']
        
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_query_llcm(data_path, mode = 1, relabel=False):
    if mode== 1:
        cameras = ['test_vis/cam1','test_vis/cam2','test_vis/cam3','test_vis/cam4','test_vis/cam5','test_vis/cam6','test_vis/cam7','test_vis/cam8','test_vis/cam9']
    elif mode ==2:
        cameras = ['test_nir/cam1','test_nir/cam2','test_nir/cam4','test_nir/cam5','test_nir/cam6','test_nir/cam7','test_nir/cam8','test_nir/cam9']
    
    file_path = os.path.join(data_path,'idx/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_llcm(data_path, mode = 1, trial = 0, relabel=False):
    
    random.seed(trial)
    
    if mode== 1:
        cameras = ['test_vis/cam1','test_vis/cam2','test_vis/cam3','test_vis/cam4','test_vis/cam5','test_vis/cam6','test_vis/cam7','test_vis/cam8','test_vis/cam9']
    elif mode ==2:
        cameras = ['test_nir/cam1','test_nir/cam2','test_nir/cam4','test_nir/cam5','test_nir/cam6','test_nir/cam7','test_nir/cam8','test_nir/cam9']
        
    file_path = os.path.join(data_path,'idx/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)

def process_test_dn348(img_dir, modal = 'day'):
    if modal=='day':
        input_data_path = img_dir + 'train_test_split/test_list_day.txt'
    elif modal=='night':
        input_data_path = img_dir + 'train_test_split/test_list_night.txt'
    
    img_dir =img_dir + modal
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split('/')[0]) for s in data_file_list]
        
        pid_container= set()

        for i in range(len(file_label)):
            pid = int(file_label[i])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for i in range(len(file_label)):
            pid = int(file_label[i])
            pid = pid2label[pid]
            file_label[i]=pid
            
    return file_image, np.array(file_label)

def process_test_dnwild(img_dir, modal='day'):
    if modal == 'day':
        input_data_path = img_dir + 'train_test_split/test_list_day.txt'
    elif modal == 'night':
        input_data_path = img_dir + 'train_test_split/query_list_night.txt'

    img_dir = img_dir + modal
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split('/')[0]) for s in data_file_list]

        pid_container = set()

        for i in range(len(file_label)):
            pid = int(file_label[i])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for i in range(len(file_label)):
            pid = int(file_label[i])
            pid = pid2label[pid]
            file_label[i] = pid

    return file_image, np.array(file_label)

def process_test_regdb(img_dir, trial = 1, modal = 'visible'):
    if modal=='visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal=='thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
    

    return file_image, np.array(file_label)