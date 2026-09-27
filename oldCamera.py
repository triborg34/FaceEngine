import multiprocessing
import os
import sys
import time
import threading
import numpy as np
import cv2 as cv

# also acts (partly) like a cv.VideoCapture
class FreshestFrame(threading.Thread):
    def __init__(self, rtsp_url, name='FreshestFrame'):
        # self.capture = capture
        # assert self.capture.isOpened()
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]="rtsp_transport;tcp"
        os.environ['OPENCV_FFMPEG_FFMPEG_DEBUG']="1"
        os.environ['OPENCV_FFMPEG_FFMPEG_LOGLEVEL']="48"
        cv.setNumThreads(multiprocessing.cpu_count())
        self.rtsp_url = rtsp_url
        self._create_capture()

        # this lets the read() method block until there's a new frame
        self.cond = threading.Condition()

        # this allows us to stop the thread gracefully
        self.running = False

        # keeping the newest frame around
        self.frame = None

        # passing a sequence number allows read() to NOT block
        # if the currently available one is exactly the one you ask for
        self.latestnum = 0

        # this is just for demo purposes        
        self.callback = None
        
        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()
    
    def get(self,proberty):
        self.cap.get(proberty)
    def _create_capture(self):
        self.cap = cv.VideoCapture(self.rtsp_url, cv.CAP_FFMPEG)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.cap.release()

    def run(self):
        counter = 0
        while self.running:
            rv, img = self.cap.read()
            if not rv:
                print("Lost frame, reconnecting...")
                self.cap.release()
                time.sleep(2)
                self._create_capture()
                continue

            with self.cond:
                self.frame = img
                self.latestnum = counter
                self.cond.notify_all()
            counter += 1
    def read(self, wait=True, seqnumber=None, timeout=None):
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with timeout argument, may return an earlier frame;
        #   may even be (0,None) if nothing received yet

        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum+1
                if seqnumber < 1:
                    seqnumber = 1
                
                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return (self.latestnum, self.frame)

            return (self.latestnum, self.frame)