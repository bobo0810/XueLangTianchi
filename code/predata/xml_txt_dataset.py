import xml.etree.ElementTree as ET
import os
import glob


class Xml_Txt_Dataset():

    def xml_txt_dataset(self):

        # 新建一个名为train_bobo的txt文件，准备写入数据
        root = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '/'
        txt_file = open(root+'../../data/xuelangtrainpart123_dataset.txt','w')

        # 存有jpg及xml的文件夹
        file = '../../data'


        img_list = ['xuelang_round1_train_part1_20180628', 'xuelang_round1_train_part2_20180705',
                    'xuelang_round1_train_part3_20180709']

        for files_part in img_list:

            jpg_file = os.path.join(file, files_part)
            # 拿到该文件夹下所有jpg文件
            jpg_files = glob.glob(root+os.path.join(jpg_file, '*/*.jpg'))
            jpg_files.sort()

            # 遍历所有jpg文件，将文件写入txt中
            for jpg_dir in jpg_files:
                #写入 图片地址，备注为 origin
                if '正常' in jpg_dir:
                    # txt 写入图像名字（绝对路径）
                    txt_file.write(jpg_dir + ',')
                    txt_file.write('origin')
                    # 读取完一张图片后换行
                    txt_file.write('\n')
                else:
                    # txt 写入图像名字
                    txt_file.write(jpg_dir + ',')
                    txt_file.write('origin_J01')
                    # 读取完一张图片后换行
                    txt_file.write('\n')

                    txt_file.write(jpg_dir + ',')
                    txt_file.write('J01')
                    # 读取完一张图片后换行
                    txt_file.write('\n')
        # 关闭txt
        txt_file.close()
        print('xuelangtrainpart123_dataset.txt generate success')

xml_txt_dataset=Xml_Txt_Dataset().xml_txt_dataset()