from __future__ import print_function
import time
from multiprocessing import Process, Queue, Value
import cPickle as pickle
import numpy
from numpy import int16, uint16, uint8, float16, log2

import cv2
from cv2 import cvtColor as convertColor, COLOR_BGR2GRAY, COLOR_GRAY2RGB,\
                resize

try:                  #nearest neighboor interpolation
  from cv2.cv import CV_INTER_NN, \
                     CV_CAP_PROP_FRAME_WIDTH, \
                     CV_CAP_PROP_FRAME_HEIGHT, \
                     CV_CAP_PROP_FPS, \
                     CV_CAP_PROP_AUTO_EXPOSURE, \
                     CV_CAP_PROP_GAIN, \
                     CV_CAP_PROP_EXPOSURE
                     
except:
  from cv2 import INTER_NEAREST as CV_INTER_NN, \
                  CAP_PROP_FRAME_WIDTH as CV_CAP_PROP_FRAME_WIDTH, \
                  CAP_PROP_FRAME_HEIGHT as CV_CAP_PROP_FRAME_HEIGHT, \
                  CAP_PROP_FPS as CV_CAP_PROP_FPS, \
                  CAP_PROP_AUTO_EXPOSURE as CV_CAP_PROP_AUTO_EXPOSURE, \
                  CAP_PROP_GAIN as CV_CAP_PROP_GAIN, \
                  CAP_PROP_EXPOSURE as CV_CAP_PROP_EXPOSURE

import pyximport; pyximport.install()
from pydvs.generate_spikes import *

MODE_128 = "128"
MODE_64  = "64"
MODE_32  = "32"
MODE_16  = "16"

UP_POLARITY     = "UP"
DOWN_POLARITY   = "DOWN"
MERGED_POLARITY = "MERGED"
POLARITY_DICT   = {UP_POLARITY: uint8(0), 
                 DOWN_POLARITY: uint8(1), 
                 MERGED_POLARITY: uint8(2),
                 0: UP_POLARITY,
                 1: DOWN_POLARITY,
                 2: MERGED_POLARITY}

OUTPUT_RATE         = "RATE"
OUTPUT_TIME         = "TIME"
OUTPUT_TIME_BIN     = "TIME_BIN"
OUTPUT_TIME_BIN_THR = "TIME_BIN_THR"

BEHAVE_MICROSACCADE = "SACCADE"
BEHAVE_ATTENTION    = "ATTENTION"
BEHAVE_TRAVERSE     = "TRAVERSE"
BEHAVE_FADE         = "FADE"

IMAGE_TYPES = ["png", 'jpeg', 'jpg']


# -------------------------------------------------------------------- #
# grab / rescale frame                                                 #

def grab_first(dev, res):
  valid, raw = dev.read()
  height, width, depth = raw.shape
  new_height = res
  new_width = int( float(new_height*width)/float(height) )
  col_from = (new_width - res)//2
  col_to   = col_from + res
  img = resize(convertColor(raw, COLOR_BGR2GRAY).astype(int16),
               (new_width, new_height), interpolation=CV_INTER_NN)[:, col_from:col_to]

  return img, new_width, new_height, col_from, col_to

def grab_frame(dev, width, height, col_from, col_to):
  valid, raw = dev.read()
  img = resize(convertColor(raw, COLOR_BGR2GRAY).astype(int16),
               (width, height), interpolation=CV_INTER_NN)[:, col_from:col_to]

  return img


# -------------------------------------------------------------------- #
# process image thread function                                        #

def processing_thread(img_queue, spike_queue, running):
  frame_count = 0
  #~ start_time = time.time()
  while True:
    img = img_queue.get()
  
    if img is None or running.value == 0:
      running.value = 0
      break
    
    # do the difference
    diff[:], abs_diff[:], spikes[:] = thresholded_difference_adpt(img, ref, 
                                                                  threshold,
                                                                  min_threshold,
                                                                  max_threshold)
                                
    # inhibition ( optional ) 
    if is_inh_on:
      spikes[:] = local_inhibition(spikes, abs_diff, inh_coords, 
                                   width, height, inh_width)

    # update the reference
    ref[:], threshold[:] = update_reference_rate_adpt(abs_diff, spikes,
                                                      ref, threshold,
                                                      min_threshold,
                                                      max_threshold,
                                                      down_threshold_change,
                                                      up_threshold_change,
                                                      max_time_ms,
                                                      history_weight)
    # convert into a set of packages to send out
    neg_spks, pos_spks, max_diff = split_spikes(spikes, abs_diff, polarity)
    
    # this takes too long, could be parallelized at expense of memory
    spike_lists = make_spike_lists_rate(pos_spks, neg_spks,
                                        max_diff, min_threshold,
                                        up_down_shift, data_shift, data_mask,
                                        max_time_ms)

    spike_queue.put(spike_lists)
    
    spk_img[:] = render_frame(spikes, img, cam_res, cam_res, polarity)
    cv2.imshow ("spikes", spk_img.astype(uint8))  
    if cv2.waitKey(1) & 0xFF == ord('q'):
      running.value = 0
      break


    #~ end_time = time.time()
#~ 
    #~ if end_time - start_time >= 1.0:
      #~ print("%d frames per second"%(frame_count))
      #~ frame_count = 0
      #~ start_time = time.time()
    #~ else:
      #~ frame_count += 1
  
  cv2.destroyAllWindows()  
  running.value = 0


# -------------------------------------------------------------------- #
# send  image thread function                                          #

def emitting_thread(spike_queue, running):
  outfile = "recorded_spikes/spikes_%08d.pickle"
  frame_counts = 0
  fname = None
  while True:
    spikes = spike_queue.get()
    
    if spikes is None or running.value == 0:
      running.value = 0
      break
    fname = outfile%(frame_counts)
    frame_counts += 1
    pickle.dump( spikes, open( fname, "wb" ) )
    # Add favourite mechanisms to get spikes out of the pc
#    print("sending!")
    
  running.value = 0


  
#----------------------------------------------------------------------#
# global variables                                                     #

mode = MODE_128
cam_res = int(mode)
#cam_res = 256 <- can be done, but spynnaker doesn't suppor such resolution
width = cam_res # square output
height = cam_res
shape = (height, width)

data_shift = uint8( log2(cam_res) )
up_down_shift = uint8(2*data_shift)
data_mask = uint8(cam_res - 1)

polarity = POLARITY_DICT[ MERGED_POLARITY ]
output_type = OUTPUT_RATE
history_weight = 1.0
min_threshold = 12 # -> 48
max_threshold = 240 # 12*15 ~ 0.7*255
down_threshold_change = 0#4
up_threshold_change = 0#10
threshold = numpy.ones(shape, dtype=int16)*min_threshold

scale_width = 0
scale_height = 0
col_from = 0
col_to = 0

curr     = numpy.zeros(shape,     dtype=int16) 
ref      = 128*numpy.ones(shape,  dtype=int16) 
spikes   = numpy.zeros(shape,     dtype=int16) 
diff     = numpy.zeros(shape,     dtype=int16) 
abs_diff = numpy.zeros(shape,     dtype=int16) 

# just to see things in a window
spk_img  = numpy.zeros((height, width, 3), uint8)

num_bits = 6   # how many bits are used to represent exceeded thresholds
num_active_bits = 2 # how many of bits are active
log2_table = generate_log2_table(num_active_bits, num_bits)[num_active_bits - 1]
spike_lists = None
pos_spks = None
neg_spks = None
max_diff = 0


# -------------------------------------------------------------------- #
# inhibition related                                                   #

inh_width = 2
is_inh_on = False
inh_coords = generate_inh_coords(width, height, inh_width)


# -------------------------------------------------------------------- #
# camera/frequency related                                             #

video_dev = cv2.VideoCapture(0) # webcam
#~ video_dev = cv2.VideoCapture('./120fps HFR Sample.mp4') # webcam

#ps3 eyetoy can do 125fps
try:
  video_dev.set(CV_CAP_PROP_FRAME_WIDTH, 320)
  video_dev.set(CV_CAP_PROP_FRAME_HEIGHT, 240)
  video_dev.set(CV_CAP_PROP_FPS, 125)
except:
  pass

try:
  video_dev.set(CV_CAP_PROP_AUTO_EXPOSURE, 0)
except:
  pass
try:
  video_dev.set(CV_CAP_PROP_GAIN, 16)
except:
  pass
try:
  video_dev.set(CV_CAP_PROP_EXPOSURE, 64)
except:
  pass
  
fps = video_dev.get(CV_CAP_PROP_FPS)
max_time_ms = int(1000./fps)


# -------------------------------------------------------------------- #
# threading related                                                    #

running = Value('i', 1)

spike_queue = Queue()
spike_emitting_proc = Process(target=emitting_thread, 
                              args=(spike_queue, running))
spike_emitting_proc.start()

img_queue = Queue()
#~ spike_gen_proc = Process(target=self.process_frame, args=(img_queue,))
spike_gen_proc = Process(target=processing_thread, 
                         args=(img_queue, spike_queue, running))
spike_gen_proc.start()


# -------------------------------------------------------------------- #
# main loop                                                            #

is_first_pass = True
start_time = time.time()
end_time = 0
frame_count = 0

while(running.value == 1):
  # get an image from video source
  if is_first_pass:
    curr[:], scale_width, scale_height, col_from, col_to = grab_first(video_dev, cam_res)
    first_pass = False
  else:
    curr[:] = grab_frame(video_dev, scale_width,  scale_height, col_from, col_to)
  
  img_queue.put(curr)
  
  
running.value == 0

img_queue.put(None)
spike_gen_proc.join()
print("generation thread stopped")

spike_queue.put(None)
spike_emitting_proc.join()
print("emission thread stopped")

if video_dev is not None:
  video_dev.release()
