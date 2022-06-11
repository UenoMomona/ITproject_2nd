# from re import S
import pyaudio
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack
import requests
import math


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
  t = np.arange(0, fs * (i + 1) * (1 / samplerate), 1 / samplerate)

  return data, t

def calc_fft(data, samplerate, dbref, A):
  spectrum = fftpack.fft(data)
  amp = np.sqrt((spectrum.real ** 2) + (spectrum.imag ** 2))
  amp = amp / (len(data) / 2)
  phase = np.arctan2(spectrum.imag, spectrum.real)
  phase = np.degrees(phase)
  freq = np.linspace(0, samplerate, len(data))

  if dbref > 0:
    amp = 20 * np.log10(amp / dbref)

    if A == True:
      amp += aweightings(freq)

  return spectrum, amp, phase, freq

def aweightings(f):
  if f[0] == 0:
    f[0] = 1e-6
  else:
    pass
  ra = (np.power(12194, 2) * np.power(f, 4)) / \
    ((np.power(f, 2) + np.power(20.6, 2)) * \
      np.sqrt((np.power(f, 2) + np.power(107.2, 2)) *\
        (np.power(f, 2) + np.power(737.9, 2))) * \
          (np.power(f, 2) + np.power(12194, 2)))
  a = 20 * np.log10(ra) + 2.00
  return a

def send_line_notify(notification_message):
  line_notify_token = 'M3SP374QjzAUr3yl0TdizUdYg5M5Vgw8kAITLE1t59f'
  line_notify_api = 'https://notify-api.line.me/api/notify'
  headers = {'Authorization': f'Bearer {line_notify_token}'}
  data = {'message': f'message: {notification_message}'}
  requests.post(line_notify_api, headers = headers, data = data)

# for i in range(5):






while True:
  data, t = record(index, samplerate, fs, time)

  dbref = 2e-5
  spectrum, amp,phase, freq = calc_fft(data, samplerate, dbref, True)

  # 極大値を求める
  max_amp = np.max(amp)
  max_index = np.argmax(amp)

  print(f'音量:{math.floor(max_amp)}, 周波数:{math.floor(freq[max_index])}')

  if freq[max_index] >= 1500 :
    send_line_notify("アラーム音がなっているかもしれません")
    break


# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)

# ax1.set_xlabel('Times[s]')
# ax1.set_ylabel('amplitude')
# ax2.set_xlabel('Frequency[Hz]')
# ax2.set_ylabel('Amplitude[dBA]')

# ax2.set_xticks(np.arange(0, 25600, 1000))
# ax2.set_xlim(0,5000)
# ax2.set_ylim(np.max(amp) - 100, np.max(amp) + 10)

# ax1.plot(t, data, label='Time waveform', lw=1, color='red')
# ax2.plot(freq, amp, label="Amplitude", lw=1, color='blue')
# ax2.plot(freq[max_index],np.max(amp),'ro',label='peak_maximal')

# fig.tight_layout()

# plt.show()
# plt.close()