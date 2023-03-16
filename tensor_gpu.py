import os
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras,metrics, math
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input, LSTM, ConvLSTM1D, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.metrics import Accuracy, Precision, PrecisionAtRecall, Recall, RecallAtPrecision, AUC
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping
import sklearn
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc, roc_auc_score, roc_curve
import numpy as np
from numpy import genfromtxt, argmax
import matplotlib.pyplot as plt
import time


class Load_Train_Test_Data(object):
    def __init__(self):
        # self.filepath_train = 'Repoducability\\All_Signals\\Data_Frame_04\\Sliced\\NN_Train_Set_60_40_FilterOrder_4_Sliced.csv'
        # self.filepath_test = 'Repoducability\\All_Signals\\Data_Frame_04\\Sliced\\NN_Test_Set_60_40_FilterOrder_4_Sliced.csv'
        # self.filepath_train = 'Repoducability\\All_Signals\\Data_Frame_04\Whole\\NN_Train_Set_60_40_FilterOrder_4_Whole.csv'
        # self.filepath_test = 'Repoducability\\All_Signals\\Data_Frame_04\Whole\\NN_Test_Set_60_40_FilterOrder_4_Whole.csv'
        
        self.filepath_train = 'Repoducability\\Selected_Signals\\DataFrame_06\\Sliced\\NN_Train_Set_60_40_FilterOrder_6_Selected_Sliced.csv'
        self.filepath_test = 'Repoducability\\Selected_Signals\\DataFrame_06\\Sliced\\NN_Test_Set_60_40_FilterOrder_6_Selected_Sliced.csv'
        # self.filepath_train = 'Repoducability\\Selected_Signals\\DataFrame_04\\Whole\\NN_Train_Set_60_40_FilterOrder_4_Selected_Whole.csv'
        # self.filepath_test = 'Repoducability\\Selected_Signals\\DataFrame_04\\Whole\\NN_Test_Set_60_40_FilterOrder_4_Selected_Whole.csv'
        
        self.Target_Names = ['Noise', 'Quake', 'Rockfall', 'Seism']
        self.Learning_Rate = [1e-3, 1e-4, 1e-5]
        self.Epochs = [10, 50, 100]
        self.Batch_Size = [100, 128, 150]
        self.color_list = ['red','blue','yellow','black']
        
        self.Early_Stop = EarlyStopping(monitor='loss',mode='min',verbose=1,patience=10)
    
    def Generate_NumPy_Array(self):
        loading_time_start = time.time()
        self.Train_Dataset = np.genfromtxt(self.filepath_train, delimiter=",", dtype=np.float64)
        self.Test_Dataset = np.genfromtxt(self.filepath_test, delimiter=",", dtype=np.float64)
        
        self.X_Train_Values = self.Train_Dataset[:,:-1]
        self.Y_Train_Labels = self.Train_Dataset[:,-1]
        self.Y_Train_Labels_Categorical = to_categorical(self.Y_Train_Labels)
        
        self.X_Test_Values = self.Test_Dataset[:,:-1]
        self.Y_Test_Labels = self.Test_Dataset[:,-1]
        self.Y_Test_Labels_Categorical = to_categorical(self.Y_Test_Labels)
        loading_time_end = time.time()
        self.loading_execution_time = loading_time_end-loading_time_start
    
    def Train_Test_Information(self):
        print('\n \t\t\t Training and Testing data information: \t\t\t \n')
        print("\t\t\t Train_Dataset - Shape: \t\t\t", self.Train_Dataset.shape)
        print("\t\t\t Test_Dataset - Shape: \t\t\t\t", self.Test_Dataset.shape)
        print("\t\t\t X_Train_Values - Shape: \t\t\t", self.X_Train_Values.shape)
        print("\t\t\t Y_Train_Labels - Shape: \t\t\t", self.Y_Train_Labels.shape)
        print("\t\t\t Y_Train_Labels_Categorical - Shape: \t\t", self.Y_Train_Labels_Categorical.shape)
        print("\t\t\t X_Test_Values - Shape: \t\t\t", self.X_Test_Values.shape)
        print("\t\t\t Y_Test_Labels - Shape: \t\t\t", self.Y_Test_Labels.shape)
        print("\t\t\t Y_Test_Labels_Categorical - Shape: \t\t", self.Y_Test_Labels_Categorical.shape)
        print("\t\t\t Loading Time in seconds: \t\t\t", self.loading_execution_time)
        
        print(self.Y_Test_Labels_Categorical[0])
    
    def Data_Normalization(self):
        self.n_traces_train = self.X_Train_Values.shape[0]
        self.n_features_train = self.X_Train_Values.shape[1]
        self.n_output_train = len(np.unique(self.Y_Train_Labels))
        self.n_output_train_categorical = len(np.unique(self.Y_Train_Labels_Categorical))
        
        print("\n \t\t\t Number of Training Traces: \t\t\t", self.n_traces_train)
        print("\t\t\t Number of Training Features: \t\t\t", self.n_features_train)
        print("\t\t\t Number of Training Output Labels: \t\t", self.n_output_train)
        print("\t\t\t Number of Training Output Categorical Labels: \t", self.n_output_train_categorical)
        
        self.n_traces_test = self.X_Test_Values.shape[0]
        self.n_features_test = self.X_Test_Values.shape[1]
        self.n_output_test = len(np.unique(self.Y_Test_Labels))
        self.n_output_train_categorical = len(np.unique(self.Y_Test_Labels_Categorical))
        
        print("\n \t\t\t Number of Testing Traces: \t\t\t", self.n_traces_test)
        print("\t\t\t Number of Testing Features: \t\t\t", self.n_features_test)
        print("\t\t\t Number of Testing Output Labels: \t\t", self.n_output_test)
        print("\t\t\t Number of Testing Output Categorical Labels: \t", self.n_output_train_categorical)
        
        
        self.X_Train = self.X_Train_Values.reshape(self.n_traces_train,self.n_features_train,1)
        self.X_Test = self.X_Test_Values.reshape(self.n_traces_test,self.n_features_test,1)
        print("\n \t\t\t X_Train - shape: \t\t\t", self.X_Train.shape)
        print("\t\t\t X_Test - shape: \t\t\t", self.X_Test.shape)
    
    def CNN_Model_Generator(self):
        CNN_Model = Sequential()
        CNN_Model.add(Conv1D(filters=32, kernel_size=3,  activation='relu', padding='same'))
        CNN_Model.add(Conv1D(filters=32, kernel_size=3,  activation='relu', padding='same'))
        CNN_Model.add(Dropout(0.2))
        CNN_Model.add(MaxPooling1D(pool_size=2, padding='same'))
        CNN_Model.add(Conv1D(filters=64, kernel_size=3,  activation='relu', padding='same'))
        CNN_Model.add(Conv1D(filters=64, kernel_size=3,  activation='relu', padding='same'))
        CNN_Model.add(Dropout(0.2))
        CNN_Model.add(MaxPooling1D(pool_size=2, padding='same'))
        CNN_Model.add(Conv1D(filters=64, kernel_size=3,  activation='relu', padding='same'))
        CNN_Model.add(MaxPooling1D(pool_size=2, padding='same'))
        CNN_Model.add(Flatten())
        CNN_Model.add(Dense(units=16, activation='relu'))
        CNN_Model.add(BatchNormalization())
        CNN_Model.add(Dense(units=8, activation='relu'))
        CNN_Model.add(BatchNormalization())
        CNN_Model.add(Dense(4, activation='softmax'))
        
        CNN_Model.build(self.X_Train.shape)
        CNN_Model.summary()
        
        return CNN_Model
    
    def LSTM_Model_Generator(self):
        LSTM_Model = Sequential()
        LSTM_Model.add(LSTM(units=256, activation='tanh', return_sequences=False, input_shape=(self.X_Train.shape[1], self.X_Train.shape[2])))
        LSTM_Model.add(Dropout(0.2))
        LSTM_Model.add((Dense(units=4, activation='softmax')))
        
        LSTM_Model.build(self.X_Train.shape)
        LSTM_Model.summary()
        
        return LSTM_Model
    
    def Model_Training(self):
        # CNN_Scores = []
        # CCN_Y_Pred = []
        # CNN_Execution_Time = []
        # self.Best_Scores = []
        # for self.rate in self.Learning_Rate:
        #     for self.epoch in self.Epochs:
        #         for self.batch in self.Batch_Size:
        #             training_time_start_cnn = time.time()
        #             CNN_Model = self.CNN_Model_Generator()
        #             CNN_Model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=self.rate),
        #                                                                   loss=tf.keras.losses.MeanSquaredError(), 
        #                                                                   metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        #             self.History_Model = CNN_Model.fit(x=self.X_Train, y=self.Y_Train_Labels_Categorical, 
        #                                                 epochs=self.epoch, batch_size=self.batch, 
        #                                                 validation_data = (self.X_Test, self.Y_Test_Labels_Categorical), 
        #                                                 callbacks=[self.Early_Stop])
        #             CNN_Scores.append(CNN_Model.evaluate((self.X_Test, self.Y_Test_Labels_Categorical, batch_size=self.batch))
        #             self.Best_Scores.append(self.History_Model.history.get('accuracy')[-1])
        #             CCN_Y_Pred.append(CNN_Model.predict(self.X_Test))
        #             training_time_end_cnn = time.time()
        #             training_execution_time_cnn = training_time_end_cnn-training_time_start_cnn
        #             CNN_Execution_Time.append(training_execution_time_cnn)
        #             self.Generate_Acc_Loss_Plot()
        # CNN_Scores_Update = self.Score_Normalization(CNN_Scores)
        
        LSTM_Scores = []
        LSTM_Y_Pred = []
        LSTM_Execution_Time = []
        self.Best_Scores = []
        for self.rate in self.Learning_Rate:
            for self.epoch in self.Epochs:
                for self.batch in self.Batch_Size:
                    training_time_start_lstm = time.time()
                    LSTM_Model = self.LSTM_Model_Generator()
                    LSTM_Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.rate),loss=[tf.keras.losses.MeanSquaredError()], 
                                        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
                    self.History_Model = LSTM_Model.fit(x=self.X_Train, y=self.Y_Train_Labels_Categorical, 
                                                        epochs=self.epoch, batch_size=self.batch,
                                                        validation_data = (self.X_Test, self.Y_Test_Labels_Categorical),
                                                        callbacks=[self.Early_Stop])
                    LSTM_Scores.append(LSTM_Model.evaluate(self.X_Test, self.Y_Test_Labels_Categorical, batch_size=self.batch))
                    self.Best_Scores.append(max(self.History_Model.history.get('accuracy')))
                    LSTM_Y_Pred.append(LSTM_Model.predict(self.X_Test))
                    training_time_end_lstm = time.time()
                    training_execution_time_lstm = training_time_end_lstm-training_time_start_lstm
                    LSTM_Execution_Time.append(training_execution_time_lstm)
                    self.Generate_Acc_Loss_Plot()
        LSTM_Scores_Update = self.Score_Normalization(LSTM_Scores)
        
        # return CNN_Scores_Update, CCN_Y_Pred, CNN_Execution_Time
        return LSTM_Scores_Update, LSTM_Y_Pred, LSTM_Execution_Time
    
    def Generate_Acc_Loss_Plot(self):
        os.chdir('C:\\Users\\Dimos\\Desktop\\Result_Images')
        
        plt.subplot(211)
        plt.plot(self.History_Model.history['accuracy'])
        plt.plot(self.History_Model.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Eopch')
        plt.legend(['Training', 'Validation'], loc='best')
        
        plt.subplot(212)
        plt.plot(self.History_Model.history['loss'])
        plt.plot(self.History_Model.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='best')
        
        plt.tight_layout()
        plt.gcf()
        plt.savefig('lr_'+str(self.rate)+'e_'+str(self.epoch)+'b_'+str(self.batch)+'.png', bbox_inches='tight')
        plt.show(block=False)
        plt.close()
        
    
    def Score_Normalization(self, Model_Scores):
        Model_Score_Update = []
        for i in range(len(Model_Scores)):
            temp = str(Model_Scores[i]).replace("[","").replace("]","").replace(" ","")
            Model_Score_Update.append(temp.split(","))
        return Model_Score_Update
    
    def Print_Model_Evaluation(self, Model, Execution):
        i=0
        for rate in self.Learning_Rate:
            for epoch in self.Epochs:
                for batch in self.Batch_Size:
                    if ((float(Model[i][3])+float(Model[i][4])) == 0):
                        F1_Score = 0
                    else:
                        F1_Score = 2*((float(Model[i][3])*float(Model[i][4]))/(float(Model[i][3])+float(Model[i][4])))
                    print("\t\t\t\t Learning_Rate=%f, Epoch=%d, Batch=%d \t\t\t" % (rate, epoch, batch))
                    print("\t\t\t Mean Square Error: \t\t\t", Model[i][0])
                    print("\t\t\t Validation Accuracy Score: \t\t\t", Model[i][1])
                    print("\t\t\t Best Training Accuracy Score: \t", self.Best_Scores[i]) 
                    print("\t\t\t F1 Score: \t\t\t\t", F1_Score)
                    print("\t\t\t Precision Score: \t\t\t", Model[i][3])
                    print("\t\t\t Recall Score: \t\t\t\t", Model[i][4])
                    print("\t\t\t AUC Score: \t\t\t\t", Model[i][2])
                    print("\t\t\t Execution Time: \t\t\t", Execution[i])
                    print("\n\n\n")
                    i = i+1
    
    def Plot_ROC_Curves(self, Model_Y_Pred):
        Y_Test_Labels_Bin = label_binarize(self.Y_Test_Labels, classes=[0.0, 1.0, 2.0, 3.0])
        Num_Classes = Y_Test_Labels_Bin.shape[1]
        
        fpr = dict()
        tpr = dict()
        threshold = dict()
        roc_auc = dict()
        
        os.chdir('C:\\Users\\Dimos\\Desktop\\Result_Images')
        for k in range(len(Model_Y_Pred)):
            for i in range(Num_Classes):
                fpr[i], tpr[i], threshold[i] = roc_curve(Y_Test_Labels_Bin[:, i], Model_Y_Pred[k][:, i])
                plt.figure("Figure %d" % k)
                plt.plot(fpr[i], tpr[i], color=self.color_list[i], lw=2)
                print('AUC for Class {}: {}'.format(self.Target_Names[i], auc(fpr[i], tpr[i])))
            print('\n\n')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic Curves')
            plt.legend(self.Target_Names)
            plt.gcf()
            plt.savefig('roc_Model_Y_Pred'+str(k)+'.png', bbox_inches='tight')
            plt.show(block=False)
            plt.close()

if __name__ == '__main__':
    tf.debugging.set_log_device_placement(True)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    Model_Trainer = Load_Train_Test_Data()
    print('\t\t\t Generating Training and Testing data...')
    Model_Trainer.Generate_NumPy_Array()
    print('\t\t\t Generation complete...')
    Model_Trainer.Train_Test_Information()
    input('\t\t\t Press Enter to continue...')
    print('\n \t\t\t Normalizing Data...')
    Model_Trainer.Data_Normalization()
    print('\t\t\t Normalization complete...')
    input('\t\t\tPress Enter to continue...')
    print('\t\t\t Initiating Model Training...')
    CNN_Score, CNN_Pred, CNN_Execution = Model_Trainer.Model_Training()
    print('\t\t\t Model Training complete...')
    input('\t\t\t Press Enter to continue...')
    print('\n \t\t\t CNN Model Evaluation: \n')
    Model_Trainer.Print_Model_Evaluation(CNN_Score, CNN_Execution)
    # print('\n \t\t\t LSTM Model Evaluation: \n')
    # Model_Trainer.Print_Model_Evaluation(LSTM_Score, LSTM_Execution)
    print('\n \t\t\t Plot ROC for CNN: \n')
    Model_Trainer.Plot_ROC_Curves(CNN_Pred)
    input('\t\t\t Press Enter to continue...')
    # print('\n \t\t\t Plot ROC for LSTM: \n')
    # Model_Trainer.Plot_ROC_Curves(LSTM_Pred)


