import pyaudio
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

time = 1
samplerate = 44100
fs = 1024
index = 0

def record(index, samplerate, fs, time):
  pa = pyaudio.PyAudio()
  data = []
  dt = 1 / samplerate

  stream = pa.open(format = pyaudio.paInt16,
                   channels =1,
                   rate = samplerate,
                   input = True,
                   input_device_index = index,
                   frames_per_buffer = fs)
  for i in range( int ( ((time / dt) / fs) ) ):
    frame = stream.read(fs)
    data.append(frame)

  stream.stop_stream()
  stream.close()
  pa.terminate()

  data = b"".join(data)

  data = np.frombuffer(data, dtype="int16") / float((np.power(2, 16) / 2) - 1)

  return data, i

data, i = record(index, samplerate, fs, time)
data = data / 32768
fft_data = np.abs(np.fft.fft(data))
fft_freq = np.fft.fftfreq(fft_data.shape[0],1.0 / samplerate)

fft_max_data = signal.argrelmax(fft_data)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Frequency[s]')
ax.set_ylabel('gain')
ax.loglog(fft_freq,fft_data,".-",label="signal")
ax.plot(fft_freq[fft_max_data],fft_data[fft_max_data],"ro",label="peak")
ax.legend()
plt.show()
plt.close()