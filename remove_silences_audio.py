import librosa
import soundfile as sf
import os
from scipy.io import savemat

audio_dir = 'Audio_data/All_audio_new_subs'
write_dir = 'Audio_data_trimmed/Trimmed_audio_new_subs'
index_dir = 'Index_dir_new_subs'

if not os.path.exists(write_dir):
    os.makedirs(write_dir)

if not os.path.exists(index_dir):
    os.makedirs(index_dir)

#read wav data
for audio_file in os.listdir(audio_dir):
    if audio_file.endswith('.wav'):
        audio_path = audio_dir + '/' + audio_file
        audio, sr = librosa.load(audio_path, sr=51200, mono=True)
        clip, index = librosa.effects.trim(audio, top_db=20)
        sf.write(write_dir +'/' + audio_file, clip[:], sr)

        mdic_i = {"timestamps": index}
        savemat(index_dir + '/' + audio_file[:-4] + '_indices.mat', mdic_i)





