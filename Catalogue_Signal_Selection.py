import os
from os.path import exists
import pandas as pd
import obspy
import obspy.core
from obspy.core import UTCDateTime
import obspy.core.stream
from obspy.signal import filter
import datetime
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import csv
from csv import reader

Dataset_1 = 'C:\\Users\\Dimos\\Desktop\\Dissertation_Code\\continuous RESIF dataset\\dataset 1'
Dataset_2 = 'C:\\Users\\Dimos\\Desktop\\Dissertation_Code\\continuous RESIF dataset\\dataset 2'
Dataset_3 = 'C:\\Users\\Dimos\\Desktop\\Dissertation_Code\\continuous RESIF dataset\\dataset 3'
Datasets = [Dataset_1, Dataset_2, Dataset_3]
signal_index = np.genfromtxt('C:\\Users\\Dimos\\Desktop\\Dissertaion Project\\signal_selection_list.csv', delimiter =",", dtype = np.int8)

class Load_Catalogue_Files:    
    def Initializer(self):
        self.Event = ""
        self.Year_List = []
        self.Month_List = []
        self.Day_List = []
        self.Start_Hour_List = []
        self.Minute_List = []
        self.Seconds_List = []
        self.Duration_List = []
        self.Class_List = []
        
    def Noise_DataFrame(self):
        self.Noise = open('C:\\Users\\Dimos\\Desktop\\Dissertation_Code\\catalog\\Noise_fin.cat')
        self.Noise = self.Noise.read().split("\n")
        print
        self.Noise.pop()
        return Load_Catalogue_Files.Dataset_Allocation(self, self.Noise)
    
    def Quake_DataFrame(self):
        self.Quake = open('C:\\Users\\Dimos\\Desktop\\Dissertation_Code\\catalog\\Quake_fin.cat')
        self.Quake = self.Quake.read().split("\n")
        self.Quake.pop()
        return Load_Catalogue_Files.Dataset_Allocation(self, self.Quake)
    
    def Rockfall_DataFrame(self):
        self.Rockfall = open('C:\\Users\\Dimos\\Desktop\\Dissertation_Code\\catalog\\Rockfall_fin.cat')
        self.Rockfall = self.Rockfall.read().split("\n")
        self.Rockfall.pop()
        return Load_Catalogue_Files.Dataset_Allocation(self, self.Rockfall)
    
    def Seism_DataFrame(self):
        self.Seism = open('C:\\Users\\Dimos\\Desktop\\Dissertation_Code\\catalog\\Seism_fin.cat')
        self.Seism = self.Seism.read().split("\n")
        self.Seism.pop()
        return Load_Catalogue_Files.Dataset_Allocation(self, self.Seism)
    
    def Dataset_Allocation(self, Dataset):
        Entry = ""
        Load_Catalogue_Files.Initializer(self)
        for index in range(len(Dataset)):
            Entry = Dataset[index]
            self.Event = Entry.split()
            self.Year_List.append(self.Event[0])
            self.Month_List.append(self.Event[1])
            self.Day_List.append(self.Event[2])
            self.Start_Hour_List.append(self.Event[3])
            self.Minute_List.append(self.Event[4])
            self.Seconds_List.append(self.Event[5])
            self.Duration_List.append(self.Event[7])
            self.Class_List.append(self.Event[11])
        DataFrame = Load_Catalogue_Files.DataFrame_Creator(self)
        return DataFrame
    
    def DataFrame_Creator(self):
        DataFrame = pd.DataFrame({'Year': self.Year_List,
                                'Month': self.Month_List,
                                'Day': self.Day_List,
                                'Start Hour': self.Start_Hour_List,
                                'Minute': self.Minute_List,
                                'Seconds': self.Seconds_List,
                                'Duration' : self.Duration_List,
                                'Class': self.Class_List
                                })
        return DataFrame

def Singal_Plot(DataFrame, signal_index):
    Signal_List = []
    Class_List = []
    num_list = []
    Window_List= []
    Window_Class = []
    for index in range(len(DataFrame)):
        Year_Name = str(DataFrame.loc[index,:].at['Year'])
        Month_Name = str(DataFrame.loc[index,:].at['Month'])
        Day_Name = str(DataFrame.loc[index,:].at['Day'])
        Hour_Name = str(DataFrame.loc[index,:].at['Start Hour'])
        Minute_Name = str(DataFrame.loc[index,:].at['Minute'])
        Seconds_Name = str(DataFrame.loc[index,:].at['Seconds'])
        Duration_Value = float(DataFrame.loc[index,:].at['Duration'])
        Class_Name = str(DataFrame.loc[index,:].at['Class'])
        TimeStap = Year_Name + "-" + Month_Name + "-" + Day_Name + "T" + Hour_Name + ":" + Minute_Name + ":" + Seconds_Name
        TimeStap_Object = UTCDateTime(TimeStap)
        for folder in Datasets:
            os.chdir(folder)
            filename = str(folder) + "\start_hour_" + Year_Name + "-" + Month_Name + "-" + Day_Name + "-" + Hour_Name + '.miniseed'
            if (exists(filename)):
                file = obspy.core.stream.read(filename, starttime=TimeStap_Object-5.0, endtime=TimeStap_Object+Duration_Value+5.0)
                print(index)
                num_signals = signal_index[index]
                for i in (num_signals):
                    trace = file[int(i)]
                    Signal_List.append(trace)
                    Class_List.append(Class_Name)
                    for window in trace.slide(window_length=10.0, step=5.0):
                        Window_List.append(window)
                        Window_Class.append(Class_Name)
            else:
                break
    return Signal_List, Class_List, num_list, Window_List, Window_Class

def Class_Sampling(Class):
    Sampled_Class = []
    for index in range(len(Class)):
        if (Class[index] == 'Oo'):
            Sampled_Class.append(0)
        elif (Class[index] == 'Q'):
            Sampled_Class.append(1)
        elif (Class[index] == 'R'):
            Sampled_Class.append(2)
        elif (Class[index] == 'TS') or (Class[index] == 'S'):
            Sampled_Class.append(3)
        else:
            Sampled_Class.append(4)
    NP_Sampled_Class = np.array(Sampled_Class, dtype=np.float16)
    return NP_Sampled_Class

def Data_Sampling(Signal_Coefficients, Max_Length):
        Sampled = []
        for signal in Signal_Coefficients:
            if (len(signal) < Max_Length):
                Sample_Padding = Max_Length-len(signal)%Max_Length
                Sampled_Temp = np.pad(signal, (0,Sample_Padding), 'constant')
            Sampled.append(Sampled_Temp)
        NP_Sampled = np.array(Sampled, dtype=np.float32)
        return NP_Sampled
    
Creator = Load_Catalogue_Files()

Noise_DataFrame = Creator.Noise_DataFrame()
Quake_DataFrame = Creator.Quake_DataFrame()
Rockfall_DataFrame = Creator.Rockfall_DataFrame()
Seism_DataFrame = Creator.Seism_DataFrame()

Final_DataFrame = pd.concat([Noise_DataFrame, Quake_DataFrame, Rockfall_DataFrame, Seism_DataFrame], ignore_index=True)
print(Final_DataFrame)
input('Press Enter to Continue...')

Signals, Classes, index_list, Windows, Win_Class = Singal_Plot(Final_DataFrame, signal_index)
input('Press Enter to Continue...')

Processed_Signals_4 = [Signals[i].filter('bandpass', freqmin=1.0, freqmax=70.0, corners=4, zerophase=False) for i in range(0, len(Signals))]
Processed_Windows_4 = [Windows[i].filter('bandpass', freqmin=1.0, freqmax=70.0, corners=4, zerophase=False) for i in range(0, len(Windows))]


print("\t\t\t Processed_Signals_4 - Length: \t\t\t\t", len(Processed_Signals_4))
print("\t\t\t Processed_Windows_4 - Length: \t\t\t\t", len(Processed_Windows_4))

Max_Length_Training_Signals = len(max(Processed_Signals_4, key=len))
Max_Sampling_Length_Training_Signals = Max_Length_Training_Signals + (10-(Max_Length_Training_Signals%10))
Max_Length_Training_Windows = len(max(Processed_Windows_4, key=len))
Max_Sampling_Length_Training_Windows = Max_Length_Training_Windows + (10-(Max_Length_Training_Windows%10))

Sampled_Classes = Class_Sampling(Classes)
Sampled_Win_Classes = Class_Sampling(Win_Class)
Sampled_Singals_4 = Data_Sampling(Processed_Signals_4, Max_Sampling_Length_Training_Signals)
Sampled_Windows_4 = Data_Sampling(Processed_Windows_4, Max_Sampling_Length_Training_Windows)

print("\t\t\t Sampled_Singals_4 - Shape: \t\t\t\t", Sampled_Singals_4.shape)
print("\t\t\t Sampled_Windows_4 - Shape: \t\t\t\t", Sampled_Windows_4.shape)

NP_Processed_Signals_and_Classes_4 = np.column_stack((Sampled_Singals_4,Sampled_Classes))
NP_Processed_Windows_and_Classes_4 = np.column_stack((Sampled_Windows_4,Sampled_Win_Classes))

print("\t\t\t NP_Processed_Signals_and_Classes_4 - Shape: \t\t", NP_Processed_Signals_and_Classes_4.shape)
print("\t\t\t NP_Processed_Windows_and_Classes_4 - Shape: \t\t", NP_Processed_Windows_and_Classes_4.shape)

Counter_Noise_W = 0
Counter_Quake_W = 0
Counter_Rockfall_W = 0
Counter_Seism_W = 0
Counter_Other_W = 0

for index in range(len(Sampled_Classes)):
    if (Sampled_Classes[index] == 0):
        Counter_Noise_W += 1
    elif (Sampled_Classes[index] == 1):
        Counter_Quake_W += 1
    elif (Sampled_Classes[index] == 2):
        Counter_Rockfall_W += 1
    elif (Sampled_Classes[index] == 3):
        Counter_Seism_W += 1
    else:
        Counter_Other_W += 1
print("\t\t\t\t\t\t Number of Whole Samples:")
print("\t\t\t Number of Noise Samples: \t\t\t\t %d" % Counter_Noise_W)
print("\t\t\t Number of Quake Samples: \t\t\t\t %d" % Counter_Quake_W)
print("\t\t\t Number of Rockfall Samples: \t\t\t\t %d" % Counter_Rockfall_W)
print("\t\t\t Number of Seism Samples: \t\t\t\t %d" % Counter_Seism_W)
print("\t\t\t Number of Unidentified Samples: \t\t\t\t %d" % Counter_Other_W)

Counter_Noise_S = 0
Counter_Quake_S = 0
Counter_Rockfall_S = 0
Counter_Seism_S = 0
Counter_Other_S = 0

for index_2 in range(len(Sampled_Win_Classes)):
    if (Sampled_Win_Classes[index_2] == 0):
        Counter_Noise_S += 1
    elif (Sampled_Win_Classes[index_2] == 1):
        Counter_Quake_S += 1
    elif (Sampled_Win_Classes[index_2] == 2):
        Counter_Rockfall_S += 1
    elif (Sampled_Win_Classes[index_2] == 3):
        Counter_Seism_S += 1
    else:
        Counter_Other_S += 1
print("\t\t\t\t\t\t Number of Sliced Samples:")
print("\t\t\t Number of Noise Samples: \t\t\t\t %d" % Counter_Noise_S)
print("\t\t\t Number of Quake Samples: \t\t\t\t %d" % Counter_Quake_S)
print("\t\t\t Number of Rockfall Samples: \t\t\t\t %d" % Counter_Rockfall_S)
print("\t\t\t Number of Seism Samples: \t\t\t\t %d" % Counter_Seism_S)
print("\t\t\t Number of Unidentified Samples: \t\t\t\t %d" % Counter_Other_S)

# os.chdir('C:\\Users\\Dimos\\Desktop\\Dissertaion Project')
# np.savetxt('Neural_Network_Selected_Signals_FilterOrder_4.csv', NP_Processed_Signals_and_Classes_4, delimiter=",")
# np.savetxt('Neural_Network_Selected_Windows_FilterOrder_4.csv', NP_Processed_Windows_and_Classes_4, delimiter=",")

# os.chdir('C:\\Users\\Dimos\\Desktop\\Dissertaion Project')
# np.savetxt("signal_selection_list.csv", index_list, delimiter =" ",  fmt ='% s') 
# print("Done!")

# np.savetxt('NP_Selected_Signals.csv', NP_Signals, delimiter=",", fmt='%s')
# np.savetxt('NP_Selected_Windows.csv', NP_Windows, delimiter=",", fmt='%s')
# NP_Signals = np.array(Signals, dtype=np.float32)
# NP_Windows = np.array(Windows, dtype=np.float32)
# NP_Signals_and_Classes = np.column_stack((NP_Signals,Sampled_Classes))
# NP_Windows_and_Classes = np.column_stack((NP_Windows,Sampled_Win_Classes))