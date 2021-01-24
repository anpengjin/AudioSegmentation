#!/usr/bin/env python3

import os
import time
import pytz
import logging
import logging.handlers
import multiprocessing
from datetime import datetime
from enum import Enum

class SafeTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False):
        logging.handlers.TimedRotatingFileHandler.__init__(self, filename, when, interval, backupCount, encoding, delay, utc)
        self._lock = multiprocessing.Lock()
    
    def doRollover(self):
        # 关闭原stream
        if self.stream:
            self.stream.close()
            self.stream = None
        # 当前时间
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        # 文件名称 dfn
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)
        
        # 源代码可见: /usr/python3.8/lib/python3.8/logging
        # 原来代码是直接判断文件是否存在，存在则删除文件，多进程下可能另外一个进程在操作处理
        # 这里改为判断文件是否存在，如果存在则直接用a的模式打开文件
        lock_time = time.time()
        print("[{}] [SafeTimedRotatingFileHandler.doRollover()] lock begin, src:{}, dst:{}".format(
            os.getpid(),self.baseFilename, dfn))
        self._lock.acquire()
        print("[{}] [SafeTimedRotatingFileHandler.doRollover()] lock end, src:{}, dst:{}".format(
            os.getpid(),self.baseFilename, dfn))
        # 如果文件不存在，并且basefilename 存在，则rename
        if not os.path.exists(dfn) and os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, dfn)
            print("[{}] [SafeTimedRotatingFileHandler.doRollover()] rename src:{}, dst:{}".format(
                os.getpid(),self.baseFilename, dfn))
        
        print("[{}] [SafeTimedRotatingFileHandler.doRollover()] unlock begin, src:{}, dst:{}".format(
            os.getpid(),self.baseFilename, dfn))
        self._lock.release()
        print("[{}] [SafeTimedRotatingFileHandler.doRollover()] unlock end, src:{}, dst:{},using_time:{}".format(
            os.getpid(),self.baseFilename, dfn,time.time() - lock_time))

        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
                print("[{}] [SafeTimedRotatingFileHandler.doRollover()] remove file success:{}".format(
                    os.getpid(), s))
        if not self.delay:
            self.mode = "a"
            self.stream = self._open()
            print("[{}] [SafeTimedRotatingFileHandler.doRollover()] open file success:{}".format(
                os.getpid(),self.baseFilename))
        
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        #If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:           # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt

class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False):
        logging.handlers.RotatingFileHandler.__init__(self, filename, mode, maxBytes, backupCount, encoding, delay)
        self._lock = multiprocessing.Lock()
        self._queue = multiprocessing.Queue()

    def doRollover(self):
        lock_time = time.time()
        print("SafeRotatingFileHandler.doRollover lock:{}, baseFilename:{}, qsize:{}".format(
            os.getpid(),self.baseFilename,self._queue.qsize()))
        self._lock.acquire()
        print("SafeRotatingFileHandler.doRollover lock:{}, baseFilename:{}, qsize:{}".format(
            os.getpid(),self.baseFilename,self._queue.qsize()))

        if self._queue.empty():
            super().doRollover()
            self._queue.put(( os.getpid(), int(time.time()) ))
        else:
            pid,end_time = self._queue.get(1)
            # 如果是直接第二次进入，则roll
            if pid == os.getpid():
                super().doRollover()
                self._queue.put(( os.getpid(), int(time.time()) ))
                print("SafeRotatingFileHandler.doRollover roll success:{}, baseFilename:{}, qsize:{}".format(
                    os.getpid(),self.baseFilename,self._queue.qsize()))
            else:
                # 如果是其他进程进入，判断时间间隔，设置60s，大于60s才允许roll
                cur_time = int(time.time())
                if cur_time > end_time + 60:
                    super().doRollover()
                    self._queue.put(( os.getpid(), int(time.time()) ))
                    print("SafeRotatingFileHandler.doRollover roll success:{}, baseFilename:{}, qsize:{}".format(
                        os.getpid(),self.baseFilename,self._queue.qsize()))
                else:
                    # 数据放回
                    self._queue.put((pid, end_time))
                    print("SafeRotatingFileHandler.doRollover roll failed:{}, baseFilename:{}, qsize:{}".format(
                        os.getpid(),self.baseFilename,self._queue.qsize()))
        
        print("SafeRotatingFileHandler.doRollover unlock:{}, baseFilename:{}, qsize:{}".format(
            os.getpid(), self.baseFilename,self._queue.qsize()))
        self._lock.release()
        print("SafeRotatingFileHandler.doRollover unlock:{}, baseFilename:{}, using_time:{}, qsize:{}".format(
                os.getpid(), self.baseFilename, time.time() - lock_time, self._queue.qsize()))

class LogLevel(Enum):
    # 日志级别
    FATAL:int = 50
    ERROR:int = 40
    WARN:int = 30
    INFO:int = 20
    DEBUG:int = 10
    NOTSET:int = 0

class LogHandler(logging.Logger):
    def __init__(
        self, 
        name, 
        log_dir:str, 
        level:LogLevel = LogLevel.DEBUG, 
        stream:bool = False, 
        file:bool = True, 
        rotate_type:str = "time", 
        fmt:str = None,
        max_file_count:int = 10,
        max_bytes:int = 1024*1024*1000,
        when:str='D'):

        self.root_dir = log_dir
        self.name = name
        self.current_level = level
        self.is_stream = stream
        self.is_file = file
        self.rotate_type = rotate_type
        self.fmt = fmt
        self.max_file_count = max_file_count
        self.max_bytes = max_bytes
        self.when = when

        print("log_dir:",log_dir)
        if os.path.exists(self.root_dir)==False:
            os.makedirs(self.root_dir)

        logging.Logger.__init__(self, self.name, level=level.value)
        if stream:
            self.__setStreamHandler__()
        if file:
            self.__setFileHandler__()
        print("LogHandler level:{},stream:{},file:{}".format(level,stream,file))

    @property
    def stream(self):
        return self.is_stream

    @property
    def file(self):
        return self.is_file

    def _get_formater(self,time:bool,level:bool,func:bool,thread:bool):
        '''
        %(levelno)s: 打印日志级别的数值
        %(levelname)s: 打印日志级别名称
        %(pathname)s: 打印当前执行程序的路径，其实就是sys.argv[0]
        %(filename)s: 打印当前执行文件名
        %(module)s: 打印当前执行模块名
        %(funcName)s: 打印日志的当前函数
        %(lineno)d: 打印日志的当前行号
        %(asctime)s: 打印日志的时间
        %(thread)d: 打印线程ID
        %(threadName)s: 打印线程名称
        %(process)d: 打印进程ID
        %(message)s: 打印日志信息
        '''
        if self.fmt is None:
            str_form = ""
            if time:
                str_form = "{}[%(asctime)s] ".format(str_form)
            if level:
                str_form = "{}[%(levelname)s] ".format(str_form)
            if func:
                str_form = "{}[%(filename)s:%(funcName)s:%(lineno)s] ".format(str_form)
            if thread:
                str_form = "{}[id:%(process)s] ".format(str_form)
            return logging.Formatter('{} %(message)s'.format(str_form))
        return logging.Formatter(fmt=self.fmt, datefmt='%Y-%m-%d %H:%M:%S')

    def __setFileHandler__(self,):
        file_name = os.path.join(self.root_dir, '{name}.log'.format(name=self.name))
        if self.rotate_type=="time":
            # 设置日志回滚, 保存在log目录, 一天保存一个文件, 保留15天
            file_handler = SafeTimedRotatingFileHandler(filename=file_name, 
                                                        when=self.when, 
                                                        interval=1, 
                                                        backupCount=self.max_file_count,)
            #file_handler.suffix = '%Y%m%d.log'
        elif self.rotate_type=="size":
            raise TypeError("clog is not stabel please change rotate_type==time or contact with mochou")
            # 每1000M一个日志文件, 最多10个文件
            file_handler = SafeRotatingFileHandler( filename=file_name, 
                                                    maxBytes=self.max_bytes, 
                                                    backupCount=self.max_file_count, 
                                                    encoding='utf-8')
        elif self.rotate_type=="clog":
            raise TypeError("clog is not stabel please change rotate_type==time or contact with mochou")
            # cloghandler.ConcurrentRotatingFileHandler
            #file_handler = cloghandler.ConcurrentRotatingFileHandler(filename=file_name, 
            #                                                        mode="a", 
            #                                                        maxBytes=self.max_bytes, 
            #                                                        backupCount=self.max_file_count, 
            #                                                        encoding='utf-8')
            #file_handler.suffix = '%Y%m%d.log'

        file_handler.setLevel(self.current_level.value)
        formatter = self._get_formater(time=True,level=True,func=False,thread=True)
        file_handler.setFormatter(formatter)
        self.file_handler = file_handler
        self.addHandler(file_handler)

    def __setStreamHandler__(self,):
        stream_handler = logging.StreamHandler()

        formatter = self._get_formater(time=True,level=True,func=True,thread=True)
        #formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(self.current_level.value)
        self.addHandler(stream_handler)

    def resetName(self, name):
        if name == self.name:
            return
        self.name = name
        self.removeHandler(self.file_handler)
        self.__setFileHandler__()

    def resetLevel(self, level=LogLevel.DEBUG):
        if level == self.current_level:
            return
        self.current_level = level
        for handler in self.handlers:
            handler.setLevel(level.value)

def log_debug(logger_nlp:LogHandler,class_name:str,fuc_name:str,split:str,**kwargs):
    if logger_nlp is None or logger_nlp.file==False:
        return
    first = True
    space_num = len("[2020-09-23 02:20:14,521] [DEBUG] [id:36846] ") + len(class_name) + len(fuc_name) + 6
    space = " "*space_num 
    msg = ""
    for key, value in kwargs.items():
        if first:
            msg += "  {}:{}".format(key,value)
            first = False
        else:
            if split == "\n":
                msg += "{}{}{}:{}".format(split,space,key,value)
            else:
                msg += "{}{}:{}".format(split,key,value)
    logger_nlp.debug("[{}.{}] {}".format(class_name,fuc_name,msg))

def log_info(logger_nlp:LogHandler,class_name:str,fuc_name:str,split:str,**kwargs):
    if logger_nlp is None or logger_nlp.file==False:
        return
    first = True
    space_num = len("[2020-09-23 02:20:14,521] [DEBUG] [id:36846] ") + len(class_name) + len(fuc_name) + 6
    space = " "*space_num 
    msg = ""
    for key, value in kwargs.items():
        if first:
            msg += " {}:{}".format(key,value)
            first = False
        else:
            if split == "\n":
                msg += "{}{}{}:{}".format(split,space,key,value)
            else:
                msg += "{}{}:{}".format(split,key,value)
    logger_nlp.info("[{}.{}] {}".format(class_name,fuc_name,msg))

def log_warn(logger_nlp:LogHandler,class_name:str,fuc_name:str,split:str,**kwargs):
    if logger_nlp is None or logger_nlp.file==False:
        return
    first = True
    space_num = len("[2020-09-23 02:20:14,521] [DEBUG] [id:36846] ") + len(class_name) + len(fuc_name) + 6
    space = " "*space_num 
    msg = ""
    for key, value in kwargs.items():
        if first:
            msg += " {}:{}".format(key,value)
            first = False
        else:
            if split == "\n":
                msg += "{}{}{}:{}".format(split,space,key,value)
            else:
                msg += "{}{}:{}".format(split,key,value)
    logger_nlp.warning("[{}.{}] {}".format(class_name,fuc_name,msg))

def log_error(logger_nlp:LogHandler,class_name:str,fuc_name:str,split:str,**kwargs):
    if logger_nlp is None or logger_nlp.file==False:
        return
    first = True
    space_num = len("[2020-09-23 02:20:14,521] [DEBUG] [id:36846] ") + len(class_name) + len(fuc_name) + 6
    space = " "*space_num 
    msg = ""
    for key, value in kwargs.items():
        if first:
            msg += " {}:{}".format(key,value)
            first = False
        else:
            if split == "\n":
                msg += "{}{}{}:{}".format(split,space,key,value)
            else:
                msg += "{}{}:{}".format(split,key,value)
    logger_nlp.error("[{}.{}] {}".format(class_name,fuc_name,msg))

def log_stream(class_name:str,fuc_name:str,split:str,**kwargs):
    first = True
    space_num = len("[id:36846] [2020-09-23 02:20:14,521]") + len(class_name) + len(fuc_name) + 6
    space = " "*space_num 
    msg = ""
    for key, value in kwargs.items():
        if first:
            msg += "  {}:{}".format(key,value)
            first = False
        else:
            msg += "{}{}{}:{}".format(split,space,key,value)
    print("[id:{}] [{}] [{}.{}] {}".format(os.getpid(),
                                            datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),
                                            class_name,
                                            fuc_name,
                                            msg))

if __name__ == '__main__':
    log = LogHandler('test',".")
    log1 = LogHandler('mochou',".")
    log.info('this is a test msg')
    log1.info('this is a test msg mochou')