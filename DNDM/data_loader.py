import numpy as np
from PIL import Image
import torch.utils.data as data
import pdb


class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
        
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            #print(pix_array.shape)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
class LLCMData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_vis.txt'
        train_thermal_list = data_dir + 'idx/train_nir.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            #print(pix_array.shape)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
class DN348Data(data.Dataset): #new
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        train_day_list   = data_dir + 'train_test_split/train_list_day.txt'
        train_night_list = data_dir + 'train_test_split/train_list_night.txt'
        train_day_path = data_dir + 'day/'
        train_night_path = data_dir + 'night/'

        day_img_file, train_day_label = load_data1(train_day_list,train_day_path)
        night_img_file, train_night_label = load_data1(train_night_list,train_night_path)
        
        train_day_image = []
        for i in range(len(day_img_file)):
   
            img = Image.open(train_day_path+ day_img_file[i])
            img = img.resize((256, 256), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_day_image.append(pix_array)
        train_day_image = np.array(train_day_image) 
        
        train_night_image = []
        for i in range(len(night_img_file)):
            img = Image.open(train_night_path+ night_img_file[i])
            img = img.resize((256, 256), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_night_image.append(pix_array)
            #print(pix_array.shape)
        train_night_image = np.array(train_night_image)
        
        # BGR to RGB
        self.train_color_image = train_day_image  
        self.train_color_label = train_day_label
        
        # BGR to RGB
        self.train_thermal_image = train_night_image
        self.train_thermal_label = train_night_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img01,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img02,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img01)
        img2 = self.transform(img02)

        return img1, img2, img01, img02, target1, target2

    def __len__(self):
        return len(self.train_day_label)


class DNwildData(data.Dataset):  # new
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_day_list = data_dir + 'train_test_split/train_list_day.txt'
        train_night_list = data_dir + 'train_test_split/train_list_night.txt'
        train_day_path = data_dir + 'day/'
        train_night_path = data_dir + 'night/'

        day_img_file, train_day_label = load_data1(train_day_list, train_day_path)
        night_img_file, train_night_label = load_data1(train_night_list, train_night_path)

        train_day_image = []
        for i in range(len(day_img_file)):
            img = Image.open(train_day_path + day_img_file[i])
            img = img.resize((256, 256), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_day_image.append(pix_array)
        train_day_image = np.array(train_day_image)

        train_night_image = []
        for i in range(len(night_img_file)):
            img = Image.open(train_night_path + night_img_file[i])
            img = img.resize((256, 256), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_night_image.append(pix_array)
            # print(pix_array.shape)
        train_night_image = np.array(train_night_image)

        # BGR to RGB
        self.train_color_image = train_day_image
        self.train_color_label = train_day_label

        # BGR to RGB
        self.train_thermal_image = train_night_image
        self.train_thermal_label = train_night_label

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img01, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img02, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img01)
        img2 = self.transform(img02)

        return img1, img2, img01, img02, target1, target2

    def __len__(self):
        return len(self.train_day_label)
    
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform
        self.test_img_file = test_img_file

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            pdb.set_trace()
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)     

class TestDataOld1(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img01,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img01)
        return img1, img01, target1

    def __len__(self):
        return len(self.test_image)  

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label

def load_data1(input_data_path, dn_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        
        file_label = [int(s.split('/')[0]) for s in data_file_list]
        pid_container= set()

        for i in range(len(file_label)):            
            #str1 = '%s/%s.jpg'%(self.all_dir,data_mat[i][0])
           # train_paths.append(img_paths[img_paths.index(str1)]) 
            pid = int(file_label[i])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for i in range(len(file_label)):
            pid = int(file_label[i])
            pid = pid2label[pid]
            file_label[i]=pid

    return file_image, file_label