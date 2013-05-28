# -*- coding: utf-8 -*-
"""
Created on Sun May 19 18:54:36 2013

@author: rk
"""
import nltk
import os
import math

train_data = "./hw1_data/train/"
test_data = "./hw1_data/test/"
ham = "ham/"
spam = "spam/"
MAX_NUM = 10000 
K = 2

def sort_by_value(d): 
    return sorted(d.items(), lambda x, y: cmp(x[1], y[1]), reverse = True)

def word_process(word):
#     lemmatizer = nltk.WordNetLemmatizer()    
    #stop words
    #lower characters
    word_low = word.strip().lower()
    #lemmatize word    
    #word_final = lemmatizer.lemmatize(word_low)
    word_final = word_low
    return word_final

def add_to_dict(word, dict_name):
    if(word in dict_name):
        num = dict_name[word]
        num += 1
        dict_name[word] = num
    else:
        dict_name[word] = 1

def negative_dict_maker(dictionary):
    d = dict()
    for (key, value) in dictionary.items():
        if(value >= MAX_NUM or value <= 1):
            d[key] = 1
    return d

def text_reader(file_name, dict_name):
    tokenizer = nltk.RegexpTokenizer("[\w']{2,}")   #leave the word with length > 1
    f = open(file_name, 'r')
    for line in f:
        words = tokenizer.tokenize(line)
        for word in words:
            word = word_process(word)
            add_to_dict(word, dict_name)
    f.close()

def save_dict(dict_name, file_path, all_flag):
    f = open(file_path, 'w')#"dict_file.data", 'w')
    word_max = ""
    value_max = 0;
    for (key, value) in dict_name.items():
        if(not all_flag):
            if value > 1 and value < MAX_NUM:
                f.writelines(key+" "+str(value)+"\n")
            if value > value_max:
                word_max = key
                value_max = value
        else:
            f.writelines(key+" "+str(value)+"\n")
            if value > value_max:
                word_max = key
                value_max = value
    f.close()
    print("Save_dict-----> Max_key:"+word_max+", Max_value:"+str(value_max))
        
def load_dict(file_path):
    dict_loaded = dict()
    f = open(file_path, 'r')
    while 1:
        line = f.readline()
        if not line:
            break
        words = line.split()
        dict_loaded[words[0]] = int(words[1])
    f.close()
    return dict_loaded

def save_file_number(ham, spam, total):
    f = open("file_number.data", 'w')
    f.writelines(str(ham)+"\n")
    f.writelines(str(spam)+"\n")
    f.writelines(str(total)+"\n")
    f.close()

#make the master dictionary and calculate the number of ham or spam
def traverse_dictionary_maker(file_path):#the path is the ham/spam's parent
    dictionary = dict()
    ham_path = file_path+ham
    spam_path = file_path+spam
    path = {ham_path, spam_path}
    path_order = 0
    num_ham = 0
    num_spam = 0
    for i in path:
        folders = os.listdir(i)
        for file_name in folders:
            if os.path.isfile(i+file_name):
                text_reader(i+file_name, dictionary)
                if(path_order == 0):
                    num_ham += 1
                else:
                    num_spam += 1
        path_order += 1
    #initialize
    save_file_number(num_ham, num_spam, num_ham + num_spam)
    return dictionary

#create the ham/spam email dictionary
def dict_creator(file_path, negative_dict):
    dictionary = load_dict("dict_file.data")
    
    #initialize the dictionary item value
    for key in dictionary:
        dictionary[key] = 0
    if(not os.path.isfile(file_path)): #if file_path is a folder
        folders = os.listdir(file_path)
        for file_name in folders:
            single_dict = dict()
            if os.path.isfile(file_path+file_name):
                text_reader(file_path+file_name, single_dict)
            for key in single_dict:
                if key not in negative_dict:
                    num = dictionary[key]
                    num += 1
                    dictionary[key] = num
    else:
        single_dict = dict()
        text_reader(file_path, single_dict)
        for key in single_dict:
#             if key not in negative_dict:
            if(key in dictionary):
                num = dictionary[key]
                num += 1
                dictionary[key] = num
    print("Dict_creator")
    return dictionary
    
    #output the file data after process

def save_vector(dictionary, target_path):
    f = open(target_path, 'w')
    for (key, value) in dictionary.items():
        if (value != 0) :
            f.writelines(key+" "+str(value)+"\n")
    print("test_dict: "+target_path + " written!")
    f.close()

def vector_creator(file_path, negative_dict, target_path):
    dictionary = dict_creator(file_path, negative_dict)
    save_vector(dictionary, target_path)

def vector_loader(target_path):
    dictionary = load_dict(target_path)
    return dictionary


def read_w_number():
    f = open("file_number.data", 'r')
    lines = f.readlines()
    w_num = [int(lines[0]), int(lines[1]), int(lines[2])]
    f.close()
    return w_num

def print_top_twenty(list_name):
    index = 0
    while(index < 20):
        print(list_name[index])
        index += 1

#to calculate the probability of P(xi|w) with smoothing log function
def calculate_log_p_xi_w(word, dict_name, n_w, exist_flag):
    if(exist_flag):
        result = math.log(dict_name[word]+1)
    else:
        result = math.log(n_w+K-dict_name[word]-1)
    return result

#to calculate the probability of P(x|w) with log function
def calculate_log_p_x_w(vector, dict_name, n_w, n_t, denominator_all):
    result = 0.0
    for (key, value) in dict_name.items():
        exist_flag = (key in vector)
        result += calculate_log_p_xi_w(key, dict_name, n_w, exist_flag)
    result -= denominator_all
    result += math.log(n_w)
    result -= math.log(n_t)
    return result

def predict(file_path, w_num, ham_dict, spam_dict, ham_denominator_all, spam_denominator_all):
    vector = vector_loader(file_path)    
    prob_ham = calculate_log_p_x_w(vector, ham_dict, w_num[0], w_num[2], ham_denominator_all)
    prob_spam = calculate_log_p_x_w(vector, spam_dict, w_num[1], w_num[2], spam_denominator_all)
    if prob_ham > prob_spam :
        return 0
    else :
        return 1
    

file_path = train_data
dictionary = traverse_dictionary_maker(file_path)
negative_dict = negative_dict_maker(dictionary) #filter the word with number >= MAX_NUM negative[key]=1
# print ("negative: "+str(len(negative_dict))) #39624
save_dict(dictionary, "dict_file.data", False)   #include the filtering of number of words
dictionary = load_dict("dict_file.data")
w_num = read_w_number() #the number of ham emails and spam emails
#==================================================================================#
#print the number of each email set
print(w_num)

#traverse ham emails, create ham dictionary
ham_dict = dict_creator(train_data + ham, negative_dict)
save_dict(ham_dict, "ham_dict.data", True)
#traverse spam emails, create spam dictionary
spam_dict = dict_creator(train_data + spam, negative_dict)
save_dict(spam_dict, "spam_dict.data", True)

#train process
list_ham = sort_by_value(ham_dict)
print_top_twenty(list_ham)
print("-------------------------------------------")
list_spam = sort_by_value(spam_dict)
print_top_twenty(list_spam)

ham_length = len(ham_dict)
# print ham_length #46328
ham_denominator_all = math.log(w_num[0]+K) * len(dictionary)# w_num[2]
spam_length = len(spam_dict)
# print spam_length #46328
# print len(dictionary) #46328
spam_denominator_all = math.log(w_num[1]+K) * len(dictionary)# w_num[2]

#initial process for test set
def test_process(data_set):
    test_set = [ham, spam]
    for w in test_set:
        folder_path = data_set + w
        files = os.listdir(folder_path)
    #     total_num = 0
    #     correct_num = 0
        for file in files:
        #     dict_temp = dict()
            if os.path.isfile(folder_path+file):
                vector_creator(folder_path+file, negative_dict, folder_path+"dict/"+file)

def test_prob(data_set):
    test_set = [ham, spam]
    for w in test_set:
        folder_path = data_set + w + "dict/"
        print folder_path
        files = os.listdir(folder_path)
        total_num = 0
        ham_predict_num = 0
        spam_predict_num = 0
        print len(files)
        f = open("file.data", 'w')
        f.writelines(folder_path+"\n")
        for file in files:
        #     dict_temp = dict()
            if os.path.isfile(folder_path+file):
                f.writelines(folder_path+file+"\n")
                result = predict(folder_path+file, w_num, ham_dict, spam_dict, ham_denominator_all, spam_denominator_all)
#                 print result
                total_num += 1
                if(w == ham and result == 0):
                    ham_predict_num += 1
                elif (w == spam and result == 1):
                    spam_predict_num += 1
        print "Total test number: " + str(total_num)
        print "Ham predicted number: " + str(ham_predict_num)
        print "Spam predicted number: " + str(spam_predict_num)
# test_process(test_data) #This function is to pre-process the test file
# test_process(train_data) #This function is to pre-process the train file
test_prob(test_data) #This function is to calculate the probability of test_data
test_prob(train_data) ##This function is to calculate the probability of train_data

def get_top_ten_word_ratio():
    dict_temp = dict()
    for key in dictionary:
        log2 = calculate_log_p_xi_w(key, ham_dict, w_num[0], True) - math.log(w_num[0]+K)
        log1 = calculate_log_p_xi_w(key, spam_dict, w_num[1], True) - math.log(w_num[1]+K)
        result = log1-log2
        dict_temp[key] = result
    list_temp = sort_by_value(dict_temp)
    print("--------------TOP 20 spam words-----------------")
    print_top_twenty(list_temp)

get_top_ten_word_ratio()

print("-------------output ends------------------")





