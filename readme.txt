源代码：
Filter.py

请保证同目录下有以下文件夹：
hw1_data/
其中有train和test（与原data文件夹一致）
但是在train和test两个文件夹下的ham、spam文件夹内都需要手动新建dict文件夹用以存放临时变量，否则python将报错…

Filter.py内有四个测试用函数：
# test_process(test_data) #This function is to pre-process the test file
# test_process(train_data) #This function is to pre-process the train file
test_prob(test_data) #This function is to calculate the probability of test_data
test_prob(train_data) ##This function is to calculate the probability of train_data

前两个用以生成测试文件的临时变量，只需运行一遍，后两个预测函数每次预测都需要运行。