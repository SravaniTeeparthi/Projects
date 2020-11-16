from random import sample
import numpy as np
import pdb
import os,shutil

#indication num of samples rquires for testing
def create_test_data(num_samples,f_path,dest_path):
    """ This function randomly generates testdata by using num_smaples"""
    file_names  = os.listdir(f_path)
    file_names  = sample(file_names,num_samples)
    full_list=[]
    for file in file_names:
        full_path=f_path+file
        full_list.append(full_path)
    for files in full_list:
        shutil.move(files,dest_path)


create_test_data(30,'./videos/talking/','./videos/talking_100x100_test/')
create_test_data(30,'./videos/no_talking/','./videos/no_talking_100x100_test/')
