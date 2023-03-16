import os
from os.path import exists
import numpy as np
from numpy import genfromtxt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, f1_score, precision_score, recall_score, roc_curve, roc_auc_score

class Neural_Network_Dataset_Creator():
    def __init__(self):
        # file_path_all = "C:\\Users\\Dimos\\Desktop\\Dissertaion Project\\Neural_Network_All_Signals_Filter_Order_4.csv"
        file_path_selected = "C:\\Users\\Dimos\\Desktop\\Dissertaion Project\\Neural_Network_Selected_Windows_FilterOrder_8.csv"
        
        # self.DataSet = np.genfromtxt(file_path_all, delimiter=",")
        self.DataSet = np.genfromtxt(file_path_selected, delimiter=",")
        self.Classification_Label = self.DataSet[:,-1] # dataset calssification
        self.Training_Set = self.DataSet[:,:-1] # dataset attributes
        self.Check_Loading()
    
    def Check_Loading(self):
        print('\t\t\t self.DataSet - Shape: \t\t\t', self.DataSet.shape)
        print('\t\t\t self.Training_Set - Shape: \t\t', self.Training_Set.shape)
        print('\t\t\t self.Classification_Label - Shape: \t', self.Classification_Label.shape)
        print('\n\n')
        # input('\n \t\t\t Press Enter to continue.... \t\t\t \n')
    
    def Training_Data_Split(self):
        # self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(self.Training_Set, self.Classification_Label, test_size=0.2, random_state=42)
        # self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(self.Training_Set, self.Classification_Label, test_size=0.3, random_state=42)
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(self.Training_Set, self.Classification_Label, test_size=0.4, random_state=42)
        
        NN_Train_Set = np.column_stack((self.X_Train,self.Y_Train))
        NN_Test_Set = np.column_stack((self.X_Test,self.Y_Test))
        print('\t\t\t NN_Train_Set - Shape: \t\t', NN_Train_Set.shape)
        print('\t\t\t NN_Test_Set - Shape: \t\t', NN_Test_Set.shape)
        
        print('\n\n')
        
        print('\t\t\t Creating and saving CSV files... \t\t\t \n')
        # os.chdir('C:\\Users\\Dimos\\Desktop\\Dissertaion Project\\DF04')
        # os.chdir('C:\\Users\\Dimos\\Desktop\\Dissertaion Project\\DF06')
        os.chdir('C:\\Users\\Dimos\\Desktop\\Dissertaion Project\\DF08')
        
                            ## Whole Signals - File Names      
        np.savetxt('NN_Train_Set_60_40_Selected_FilterOrder_8_Sliced.csv', NN_Train_Set, delimiter=",")
        np.savetxt('NN_Test_Set_60_40_Selected_FilterOrder_8_Sliced.csv', NN_Test_Set, delimiter=",")
        
        print('\n \t\t\t CSV file was created and saved without errors! \t\t\t \n')

def main():
    
    Data_Splitter = Neural_Network_Dataset_Creator()
    Data_Splitter.Training_Data_Split()
    
if __name__ == '__main__':
    main()