# encoding:utf-8
import os
import time

import scipy 
import librosa
import numpy as np
import soundfile as sf

from logger import LogHandler,LogLevel

# 辅助函数
logger_news_decoder = LogHandler( name='segmentation_audio_{}'.format(os.getpid()), 
                                log_dir=os.path.join(os.getcwd(),"log"), 
                                level=LogLevel.DEBUG, 
                                stream=False, 
                                file=True)

def get_current_time():
    return time.time()

def is_exist(path:str):
    # 判断某目录是否存在
    if not os.path.exists(path):
        os.makedirs(path)

def zero_crossing_rate(y:list, frame_length:int):
    """
    计算语音短时过零率：单位时间(每帧)穿过横轴（过零）的次数
    y: a series of audio
    frame_length: length of a frame
    """
    y = np.array(y, dtype=float)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=frame_length, center=False) 
    return np.array(zcr[0], dtype=np.uint32)
    """
    zcr = []  # 语音短时过零率列表
    counting_sum_per_frame = 0  # 每一帧过零次数累加和，即过零率
    for i in range(len(y)):  # 遍历每一个采样点数据
        if i % frame_length == 0:  # 开头采样点无过零，因此每一帧的第一个采样点跳过
            continue

        if (y[i] < 0 and y[i-1] > 0) or (y[i] > 0 and y[i-1] < 0):
        #if y[i] * y[i - 1] < 0:  # 相邻两个采样点乘积小于0，则说明穿过横轴
            counting_sum_per_frame += 1  # 过零次数加一
        if (i + 1) % frame_length == 0:  # 一帧所有采样点遍历结束
            zcr.append(counting_sum_per_frame)  # 加入短时过零率列表
            counting_sum_per_frame = 0  # 清空和
        elif i == len(y) - 1:  # 不满一帧，最后一个采样点
            zcr.append(counting_sum_per_frame)  # 将最后一帧短时过零率加入列表
    return np.array(zcr, dtype=np.uint32)
    """

def short_time_energy(y:list, frame_length:int):
    """ 
    计算语音短时能量：每一帧中所有语音信号的平方和
    y: a series of audio
    frame_length: length of a frame    
    """ 
    energy = []  # 语音短时能量列表
    energy_sum_per_frame = 0  # 每一帧短时能量累加和
    for i in range(0, len(y), frame_length):  # 遍历每一个采样点数据
        if i + frame_length > len(y):
            energy.append(np.sum(np.abs(y[i:])))
        else:
            energy.append(np.sum(np.abs(y[i: i+frame_length])))
    energy = np.array(energy)
    energy = np.where(energy == 0, np.finfo(np.float64).eps, energy)  # 避免能量值为0，防止后续取log出错(eps是取非负的最小值)
    #print("energy: ", energy)
    return energy

def save_wav(file:str, data:list, samplerate:int):
    # 写入音频文件
    path = os.path.dirname(file)
    if not os.path.exists(path):
        os.makedirs(path)
    sf.write(file, data, samplerate)

def read_audio(audio_path:str, start_time:int=0, duration:int=None):
    # 读取音频：单/双声道-->单声道，采样率不变
    if not isinstance(audio_path, str):
        raise TypeError("audio_path type error:{},{}".format(type(audio_path),audio_path))
    y, sr = sf.read(audio_path, dtype="int16")
    if len(list(y.shape)) == 2:
        y = np.mean(y.T, axis=0, dtype="int16")
    if duration is None:
        y = y[start_time * sr :]
    else:
        y = y[start_time * sr: (start_time + duration) * sr]
    return y, sr