#coding=utf-8
import tensorflow as tf
import cv2
import numpy as np
import os
#import classify_image
import get_features_by_CNN
import shutil
import random

video_path = '/media/wyd/h51/hmdb51/'
tfrecords_path = '/media/wyd/h51/tfrecords/'
npy_path = '/media/wyd/h51/npys/'
f=open('shuffled_filename.txt','r')
rows = [row.split()[0] for row in f]
#print rows
row_split1 = rows[0:1190]
row_split2 = rows[1190:2*1190]
row_split3 = rows[2*1190:-1]
row_split1_test = rows[0:10]
#row_list = [row_split1,row_split2,row_split3,row_split4]
#print len(row_list[1])
globle_total_size = 0
vocab = os.listdir(video_path)
vocab = sorted(vocab)
#print vocab

def encode_to_tfrecords(row_split,split_number):


    writer = tf.python_io.TFRecordWriter(tfrecords_path+'train_'+split_number+'.tfrecords')
    
    if os.path.exists(npy_path):
        shutil.rmtree(npy_path)
    os.mkdir(npy_path)
    number = 0
    for filename in row_split:

        fn=video_path+filename
        label_name = filename.split('/')[0]
        num_label = vocab.index(label_name)


        # to get frame_matrix restoring N*image
        cap = cv2.VideoCapture(fn)
        if not cap.isOpened():
            print "could not open : ",fn
            sys.exit()
        ret = True
        frame_matrix=[]
        while(ret):
            ret, frame = cap.read()
            if not ret:
                break
            else:
                frame = cv2.resize(frame,(224,224))
                frame_matrix.append(frame)
        cap.release()
        frame_matrix=np.array(frame_matrix)



        print frame_matrix.shape

        



        features_matrix=get_features_by_CNN.get_features_matrix(frame_matrix)

        len_of_frames = len(frame_matrix)
        feature_batch_matrix=[]
        for i in range(0,len_of_frames-30+1,2):
            feature_batch_matrix.append(features_matrix[i:i+30])




        f = file(npy_path+str(num_label)+'.npy','ab')
        for i in range(len(feature_batch_matrix)):
            fbm=feature_batch_matrix[i].astype(np.float32)              
            np.save(f,fbm)
        f.close()



        number += 1
        print filename+'has been writen into a npy file. '
        print 'total number = ',number





    name_dict={}
    for i in os.listdir(npy_path):
        name_dict[i]= file(npy_path+i,'rb')


    #npy_to_tfrecord(writer)
    npy_to_tfrecord_V2(writer)
    shutil.rmtree(npy_path,ignore_errors=1)
    
    writer.close()
def npy_to_tfrecord(writer):

    f={}
    u=0
    for i in os.listdir(npy_path):
        f[i]= file(npy_path+i,'rb')
    # print f


    # writer = tf.python_io.TFRecordWriter(tfrecords_path+'train_'+str(number)+'.tfrecords')

    while True:
        # print f.keys()
        for i in f.keys():
            try:
                # print i
                num_label=int(i.split('.')[0])
                fbm = np.load(f[i])
                #print fbm.shape
                fbm_raw=fbm.tostring()
                example=tf.train.Example(
                    features=tf.train.Features(
                        feature={
                                'feature_batch':tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[fbm_raw])
                                ),
                                'label_batch':tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[num_label])
                                )
                            }
                        )
                    )
                serialized=example.SerializeToString()
                writer.write(serialized)
            except IOError:
                print i+' runs out'
                del f[i]




        if not f.keys():
            break
    # writer.close()

def npy_to_tfrecord_V2(writer):
    f={}
    u=0
    for i in os.listdir(npy_path):
        f[i]= file(npy_path+i,'rb')
    # print f


    # writer = tf.python_io.TFRecordWriter(tfrecords_path+'train_'+str(number)+'.tfrecords')

    name_list,size_list = get_Npy_Size(npy_path)
    nvs = zip(name_list,size_list)
    name_size_dict = dict( (name,value) for name,value in nvs)




    while True:
        # print f.keys()

        i = random_pick(name_size_dict.keys(),name_size_dict.values())
        #print i
        #for i in f.keys():
        try:
            # print i
            num_label=int(i.split('.')[0])
            fbm = np.load(f[i])
            #print fbm.shape
            fbm_raw=fbm.tostring()
            example=tf.train.Example(
                features=tf.train.Features(
                    feature={
                            'feature_batch':tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[fbm_raw])
                            ),
                            'label_batch':tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[num_label])
                            )
                        }
                    )
                )
            serialized=example.SerializeToString()
            writer.write(serialized)
        except IOError:
            print i+' runs out'
            del f[i]
            del name_size_dict[i]




        if not f.keys():
            break

def random_pick(some_list, probabilities):  
    x = random.uniform(0,1)  
    cumulative_probability = 0.0  
    for item, item_probability in zip(some_list, probabilities):  
        cumulative_probability += item_probability  
        if x < cumulative_probability:
            break  
    return item


def get_Npy_Size(npy_path):
    name_size_dict={}
    total_size = 0.
    for root, dirs, files in os.walk(npy_path):
            for ele in files:
                ele_size = os.path.getsize(root+ele)
                name_size_dict[ele]= ele_size
                total_size += ele_size
            print name_size_dict.items()
            size_list = [name_size_dict[i]/total_size for i in name_size_dict.keys()]
            name_list = name_size_dict.keys()
            print name_list,size_list
    return name_list,size_list
#some_list = [1,2,3,4]  
#probabilities = [0.2,0.1,0.6,0.1]  
  
#print random_pick(some_list,probabilities) 





#encode_to_tfrecords(row_split1_test,'split1_test')
encode_to_tfrecords(row_split1,'split1')
encode_to_tfrecords(row_split2,'split2')
encode_to_tfrecords(row_split3,'split3')
#get_Npy_Size(tfrecords_path)