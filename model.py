# encoding:utf-8

import os
import glob
import shutil
import argparse
import logging
import time
import json
from typing import List

import scipy 
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from utils import logger_news_decoder, zero_crossing_rate, short_time_energy,save_wav
from logger import log_info, log_debug

# 模型
class AudioSplitByTimeAndSpec(object):
    # 按照时长和频谱切割音频
    def __init__(self, 
                item:dict, 
                split_time_min:int=5, #最小切割时长
                split_time_max:int=10, #最大切割时长
                frame_length:int=400, #帧长
                e_low_multifactor:float=1.0,
                min_interval:int=10, #两段语音间的最小间隔
                right_shift_sec:float=2.5, #当无法切割时，起点右移2.5sec
                audio_length:int=30,
                top_five_energy:int=2000,
                output_path:str=".", #保存切割好的句子
                save_wav_flag:bool=True): 
        self.item = item
        self.y = self.get_waveform(item)
        self.sr = int(self.get_item_value(item, "sample_rate", None))
        self.task_name = self.get_item_value(item, "task_name", None)
        self.start_time = float(self.get_item_value(item, "start_time", None))
        self.end_time = self.get_item_value(item, "end_time", 0)
        if self.end_time is not None:
            self.end_time = float(self.end_time)
        
        self.output_path = output_path

        self.split_time_min = split_time_min
        self.split_time_max = split_time_max
        self.frame_length = frame_length
        self.min_interval = min_interval
        self.right_shift_sec = right_shift_sec
        self.e_low_multifactor = e_low_multifactor
        self.top_five_energy = top_five_energy # # 前5帧能量均值：定值

        self.zcr = zero_crossing_rate(self.y, self.frame_length)
        self.ste = short_time_energy(self.y, self.frame_length)
        self.save_wav_flag = save_wav_flag
        #判断是否是一个task_name的最后一个音频        
        self.end_audio = self.get_item_value(item, "task_end", None)

    def get_item_value(self, item:dict, key, fill_value):
        value = item.get(key, fill_value)
        if value == fill_value:
            raise TypeError("{} of item is not exist.".format(key))
        return value

    def get_waveform(self, item:dict):
        # 得到采样点数据
        if not isinstance(item, dict):
            raise TypeError("item type error:{}".format(type(item)))
        sample_bits = self.get_item_value(item, "sample_bits", None)
        if sample_bits != "16":
            log_info(logger_news_decoder, "AudioSplitByTimeAndSpec", "get_waveform", ",", sample_bits=sample_bits)
            raise TypeError("sample_bit != 16:{}".format(sample_bits))

        # left, right = get_pcm_by_file(item["task_file_path"])
        left = np.array(self.get_item_value(item, "left_pcm", None), dtype="int16")
        right = np.array(self.get_item_value(item, "right_pcm", None), dtype="int16")
        # print(np.array(left).shape) 
        return np.mean([left, right], axis=0, dtype="int16")

    def get_ste_mean(self, ste:list):
        """得到短时能量的均值"""
        if len(ste) <= 0:
            log_info(logger_news_decoder, "AudioSplitByTimeAndSpec", "get_ste_mean", ",", ste_length=len(ste))
            raise ZeroDivisionError("the length of ste is:{}".format(len(ste)))
        return sum(ste) / len(ste)

    def get_spoint_list_by_ste(self, ste:list, 
                        n_frames_max:int, 
                        n_frames_min:int, 
                        energy_low:float,
                        right_shift_frame:int,
                        min_interval:int,
                        frame_length:int):
        """时域切割"""
        total_start = [0] # 保存切割点所在的帧
        start = 0
        while start + n_frames_max < len(ste):
            count = 0
            voiced_ste = {}  # 语音段的静音部分 
            # 首先利用能量低阈值energy_high进行初步检测
            # 检测从起点start开始的第split_time_min到第split_time_max秒之间的分割点(如5-10s)
            i = start + n_frames_min
            while i < start + n_frames_max:
                if ste[i] < energy_low:
                    count += 1
                elif ste[i] > energy_low and count > 0:
                    if count > min_interval:
                        voiced_ste[i-round(count/2)] = count
                    count = 0
                i += 1
            if len(voiced_ste) == 0:
                # 若第split_time_min到第split_time_max秒之间无分割点，将起点右移right_shift_sec秒，继续检测
                start = start + right_shift_frame 
                continue
            elif len(voiced_ste) == 1:
                # 只存在一个切分点
                split_point = list(voiced_ste.keys())[0]
            else: 
                # 存在多个切分点，寻找间隔最大的切分点
                voiced_ste_copy = voiced_ste.copy()
                voiced_ste_copy = sorted(voiced_ste_copy.items(), key=lambda x:x[1], reverse=True)
                split_point = voiced_ste_copy[0][0]
            start = split_point
            total_start.append(start)
        # print(start, len(self.ste), n_frames_min)
        # print(total_start)
        total_start = [i * frame_length for i in total_start]
        return total_start

    def get_spoint_list_by_spec(self, y:list,
                        n_frames_max:int, 
                        n_frames_min:int,
                        energy_low:float, 
                        right_shift_frame:int,
                        min_interval:int,
                        frame_length:int,
                        plot:bool):
        """频域切割"""
        y = np.array(y, dtype=float)
        # Sxx = np.fft.fft(y) #对y整个信号做傅立叶变换，使用np.fft.fft需要提前分帧
        Sxx = librosa.stft(y=y, n_fft=frame_length, hop_length=frame_length, win_length=None, window='hann', center=False)
        # f, t, Sxx = scipy.signal.spectrogram(x=y, fs=self.sr, scaling='spectrum',
        #                                     mode='magnitude',
        #                                     window='hann',
        #                                     nperseg=frame_length,
        #                                     noverlap=frame_length/2)
        # print("Sxx.shape:", Sxx.shape)
        Sxx = Sxx[1:51, :]
        # spec_angle = np.sum(np.angle(Sxx)**2, axis=0).tolist()
        spec_energy = np.sum(np.abs(Sxx)**2, axis=0).tolist()
        """
        print(spec_energy)
        plt.figure(figsize=(50, 10))
        plt.subplot(3, 1, 1)
        plt.plot(spec_angle, label="angle")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(spec_mang, label="mang")
        plt.legend()
        plt.subplot(3, 1, 3)
        
        plt.plot(ste, label="energy")
        plt.legend()
        plt.grid()
        plt.savefig("apj_min(spec_energy).png", dpi=300)
        """
        k = 1 / max(min(spec_energy),1)
        self.spec = k * np.array(spec_energy)
        energy_average = self.get_ste_mean(self.spec)  # 求全部帧的短时能量均值
        energy_high = energy_average * 0.6   # 能量均值的0.6倍作为能量高阈值
        
        # exit(0)
        energy_low = (energy_average + energy_high * 0.5 ) * 0.8  # 前5帧能量均值+能量高阈值的5分之一作为能量低阈值
        # print(energy_average, energy_high, energy_low)

        total_start = [0] # 保存切割点所在的帧
        start = 0
        while start + n_frames_max < len(self.spec):
            count = 0
            voiced_ste = {}  # 语音段的静音部分 
            # 首先利用能量低阈值energy_high进行初步检测
            # 检测从起点start开始的第split_time_min到第split_time_max秒之间的分割点(如5-10s)
            i = start + n_frames_min
            while i < start + n_frames_max:
                if self.spec[i] < energy_low:
                    count += 1
                elif self.spec[i] > energy_low and count > 0:
                    if count > min_interval:
                        voiced_ste[i-round(count/2)] = count
                    count = 0
                i += 1
            if len(voiced_ste) == 0:
                # 若第split_time_min到第split_time_max秒之间无分割点，将起点右移right_shift_sec秒，继续检测
                start = start + right_shift_frame 
                continue
            elif len(voiced_ste) == 1:
                # 只存在一个切分点
                split_point = list(voiced_ste.keys())[0]
            else:
                # 存在多个切分点，寻找间隔最大的切分点
                voiced_ste_copy = voiced_ste.copy()
                voiced_ste_copy = sorted(voiced_ste_copy.items(), key=lambda x:x[1], reverse=True)
                split_point = voiced_ste_copy[0][0]

            start = split_point
            total_start.append(start)
        # print(start, len(self.ste), n_frames_min)
        if plot:
            self.plot(self.spec, total_start, energy_low, self.task_name)

        total_start = [i * frame_length  for i in total_start]    
        return total_start

    def split(self, plot:bool=False):
        """切割"""
        if len(self.y) <= 0:
            log_info(logger_news_decoder, "AudioSplitByTimeAndSpec", "split", ",", 
                        msg="wavefrom is None", 
                        task_name=self.task_name,
                        start_time=self.start_time,
                        end_time=self.end_time,
                        task_file_path=self.item["task_file_path"])
            return self.end_time, None
        n_frames_min = round(self.split_time_min * self.sr / self.frame_length) # 最小帧数
        n_frames_max = round(self.split_time_max * self.sr / self.frame_length) # 最大帧数
        right_shift_frame = round(self.right_shift_sec * self.sr / self.frame_length) # 右移帧数
        # print("n_frames_min: ", n_frames_min)
        # print("n_frames_max: ", n_frames_max)
        energy_average = self.get_ste_mean(self.ste)  # 求全部帧的短时能量均值
        energy_high = energy_average * 0.6   # 能量均值的0.6倍作为能量高阈值
        energy_low = (self.top_five_energy + energy_high / 2 ) * self.e_low_multifactor  # 前5帧能量均值+能量高阈值的5分之一作为能量低阈值      
        # 通过时域切分
        total_start = self.get_spoint_list_by_ste(ste=self.ste, n_frames_max=n_frames_max, n_frames_min=n_frames_min, 
                                      energy_low=energy_low,right_shift_frame=right_shift_frame,min_interval=self.min_interval,
                                      frame_length=self.frame_length)
        # print("total_start", total_start)
        audio_duration = int(len(self.y) / self.sr)
        if len(total_start) * 30 / audio_duration <= 2.5:
            """时域切分切出来的句子小于3个，不满足条件，则通过频域切分
            """
            log_info(logger_news_decoder, "AudioSplitByTimeAndSpec", "split", ",", 
                        msg="total_start is short by ste",
                        task_name=self.task_name,
                        start_time=self.start_time,
                        end_time=self.end_time,
                        total_start=total_start)
            min_interval = self.min_interval * 4
            while len(total_start) <= 2 and min_interval >= 10:
                min_interval = min_interval / 2
                total_start = self.get_spoint_list_by_spec(y=self.y, n_frames_max=n_frames_max, n_frames_min=n_frames_min,
                                        energy_low=50, right_shift_frame=right_shift_frame,
                                        min_interval=min_interval, frame_length=self.frame_length, plot=plot) 
        """经过频域切分却依旧无法满足条件的情况
        """
        if len(total_start) <= 1:
            log_info(logger_news_decoder, "AudioSplitByTimeAndSpec", "split", ",", 
                        msg="split failed",
                        task_name=self.task_name,
                        start_time=self.start_time,
                        end_time=self.end_time,
                        total_start=total_start)                
            # 发送整个音频给ASR            
            # file1 = os.path.join(self.output_path, os.path.join(self.task_name, f"{str(self.start_time)}.wav")) # savename
            file1 = os.path.join(self.output_path, f"{self.task_name}_0.wav") # savename
            if self.save_wav_flag:
                save_wav(file=file1, data=self.y, samplerate=self.sr)
            item = self.item.copy()
            item["left_pcm"] = self.y.tolist()
            item["right_pcm"] = self.y.tolist()             
            # 删除ASR不需要的字段
            for key in ["task_end", "data_type", "flag", "end_split_time", "remain"]:
                del item[key]
            return self.end_time, None
        elif len(total_start) <= 2:
            log_info(logger_news_decoder, "AudioSplitByTimeAndSpec", "split", ",", 
                        msg="split two audio",
                        task_name=self.task_name,
                        start_time=self.start_time,
                        end_time=self.end_time,
                        total_start=total_start)             
            end_remain_start_time, end_remain_data = self.send_split_audio(total_start)
            return end_remain_start_time, end_remain_data

        # 若不是最后一个音频，且最后一个切割点之后的时长小于最小时长，则去掉最后一个切割点
        if len(self.y) - total_start[-1] < n_frames_min * self.frame_length and not self.end_audio:
            total_start = total_start[:-1]          
        end_remain_start_time, end_remain_data = self.send_split_audio(total_start)
        return end_remain_start_time, end_remain_data

    def send_split_audio(self, total_start:list):
        """发送切割好的句子给ASR"""
        for i in range(len(total_start)):
            small_item = self.item.copy()
            file_name = os.path.join(self.output_path, f"{self.task_name}_{i}.wav") # savename
            if i == len(total_start) - 1:
                if self.end_audio:
                    split_start = total_start[i]
                    data = self.y[split_start:]
                    start_time = self.frame_to_second(split_start)
                    small_item["start_time"] = start_time
                    small_item["left_pcm"] = data.tolist()
                    small_item["right_pcm"] = data.tolist()
                    last_item = dict()
                    for key, value in small_item.items():
                        if key != "task_name":
                            last_item[key] = None
                        else:
                            last_item[key] = value
                    for key in ["task_end", "flag", "data_type", "end_split_time", "remain"]:
                        del last_item[key]                  
                else:
                    continue                
            else:
                split_start, split_end = total_start[i], total_start[i+1]
                start_time = self.frame_to_second(split_start)
                end_time = self.frame_to_second(split_end)
                data = self.y[split_start: split_end]
                # print("data", data)
                small_item["start_time"] = start_time
                small_item["end_time"] = end_time
                small_item["left_pcm"] = data.tolist()
                small_item["right_pcm"] = data.tolist()             

                # 删除ASR不需要的字段
                for key in ["task_end", "flag", "data_type", "end_split_time", "remain"]:
                    del small_item[key] 
            if self.save_wav_flag:
                save_wav(file=file_name, data=data, samplerate=self.sr)
        if not self.end_audio:
            end_split = total_start[-1]
            end_remain = self.y[end_split:]
            end_remain_start_time = self.frame_to_second(end_split)
            return end_remain_start_time, end_remain
        else:
            return None, None
    
    def frame_to_second(self, point:int):
        """帧 -> 秒"""
        time = point / self.sr
        return str(round(self.start_time + time, 3))
