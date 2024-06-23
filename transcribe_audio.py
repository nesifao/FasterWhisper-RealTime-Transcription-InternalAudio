from faster_whisper import WhisperModel
import soundcard as sc
import time 
import os
import threading
import warnings 
import tkinter as tk 
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog
import wave
from datetime import datetime

# TO DO
# change record sec, work on audio click, select translation

# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 

model_size = "large-v3"
# model_size = "distil-large-v3"
# model_size = "medium.en"
# Run on GPU with FP16
start_time = time.time()
model = WhisperModel(model_size, device="cuda", compute_type="float16")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Model loaded in: {elapsed_time:.6f}")
OUTPUT_FILE_NAME = "out.wav"    # file name.
SAMPLE_RATE = 44100              # [Hz]. sampling rate.
RECORD_SEC = 2                  # [sec]. duration recording audio.
elapsed_time = 0.0
counter = 0
list_audio_files = []
lock = threading.Lock()
is_recording = False
folder_path = "./record_audio/"

def record_audio():
    global counter
    global is_recording
    folder_tmp_path = "./tmp_audio/"
    if not os.path.exists(folder_tmp_path):
        os.makedirs(folder_tmp_path)    

    while is_recording: 
        try:
            with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
                    data = mic.record(numframes=SAMPLE_RATE*RECORD_SEC) 
            tmp_output_file = folder_tmp_path + str(counter) + ".wav"
            
            # Format data to 16-bit PCM
            data_16bit = (data[:, 0] * 32767).astype('<h')
            with wave.open(tmp_output_file, mode="wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.setnframes(SAMPLE_RATE*RECORD_SEC)
                wav_file.writeframes(data_16bit.tobytes())                
            with lock:
                list_audio_files.append(tmp_output_file)
            counter += 1 
        except Exception as e:
            print(f"Error recording audio: {e}")   

def transcribe_sample_audio():
    global is_recording 
    while is_recording:
        try:
            with lock:
                if list_audio_files:
                    segments, _ = model.transcribe(list_audio_files[0], beam_size=5, without_timestamps=True, word_timestamps=False, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))     
                    for segment in segments:
                        update_text(segment.text)              
                    list_audio_files.pop(0)           
        except Exception as e:
            print (f"Error transcribing audio: {e}")
        time.sleep(0.2)

def update_text(text):
    text_box.insert(tk.END, text)
    text_box.see(tk.END)

def start_recording():
    global thread_record_audio
    global thread_transcribe_audio
    global is_recording    
    if not is_recording:           
        is_recording = True
        thread_record_audio = threading.Thread(target=record_audio)     
        thread_record_audio.start()  
        print ("Start record")
        thread_transcribe_audio = threading.Thread(target=transcribe_sample_audio)     
        thread_transcribe_audio.start() 
        print ("Start transcribe")

def stop_recording():    
    global thread_record_audio 
    global thread_transcribe_audio   
    global is_recording    
    if is_recording:
        is_recording = False
        # thread_record_audio.join()
        print ("Stop record")
        # thread_transcribe_audio.join()
        print ("Stop transcribe")

def clear_text():
    text_box.delete('1.0', tk.END)

def transcribe_audio_file():
    audio_file_name = filedialog.askopenfilename(filetypes=(("Audio Files", ".wav .mp3"),   ("All Files", "*.*")), initialdir=folder_path)
    if (audio_file_name):
        clear_text()
        start_time = time.time()
        segments, _ = model.transcribe(audio_file_name, beam_size=5, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))                
        for segment in segments:
            update_text("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Transcribe all audio sample in: {elapsed_time}")    

def save_audio_file():
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) 

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    name_file = dt_string + ".wav"
    final_path = filedialog.asksaveasfilename(initialfile=name_file , initialdir=folder_path)
    list_tmp_audio = []
    # sort the files by digit increasing
    for file in sorted(os.listdir("./tmp_audio"), key=lambda x: int(x.split('.')[0])):
        if file.endswith(".wav"):
            list_tmp_audio.append("./tmp_audio/" + file)
    if list_tmp_audio:
        with wave.open(list_tmp_audio[0], 'rb') as wave_file:
            params = wave_file.getparams()
    
        with wave.open(final_path, 'wb') as output:
            output.setparams(params)
            for file in list_tmp_audio:
                with wave.open(file, 'rb') as wave_file:
                    data = wave_file.readframes(wave_file.getnframes())
                    output.writeframes(data)
        print (f"Record audio created at {final_path}")


# GUI
root = tk.Tk()
root.title("Audio Transcription")
root.resizable(True, True)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="Start recording and transcription", command=start_recording)
start_button.pack(side="left", padx=10)

stop_button = tk.Button(button_frame, text="Stop recording and transcription", command=stop_recording)
stop_button.pack(side="left", padx=10)

clear_button = tk.Button(button_frame, text="Clear text", command=clear_text)
clear_button.pack(side="left", padx=10)

clear_button = tk.Button(button_frame, text="Save audio file", command=save_audio_file)
clear_button.pack(side="left", padx=10)

clear_button = tk.Button(button_frame, text="Transcribe audio file", command=transcribe_audio_file)
clear_button.pack(side="left", padx=10)

text_box = ScrolledText(root, wrap=tk.WORD, width=100, height=40)
text_box.pack(pady=1)

root.mainloop()

stop_recording()
# Clear
with lock:
    is_recording = False     
    if os.path.exists("./tmp_audio"):
        for file in os.listdir("./tmp_audio"):
            os.remove("./tmp_audio/"+file)  
    print ("Stop record")
print ("End transcribe")