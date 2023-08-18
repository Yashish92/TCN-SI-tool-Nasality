import librosa
import soundfile as sf
import noisereduce as nr

# load audio and downsample to 16KHz
audio_low, sr = librosa.load('Read_speech_data/BPRS_low_audio/5_4_2022_BPRS49.wav', sr=16000)
audio_high, sr = librosa.load('Read_speech_data/BPRS_high_audio/09_24_2021_BPRS67.wav', sr=16000)

# do speech enhancement
audio_high_enh = nr.reduce_noise(y=audio_high, sr=sr)

# # write downsampled audio
# # s_rate = 16000
# sf.write('Read_speech_data/BPRS_low_audio/5_4_2022_BPRS49_16khz.wav', audio_low, sr)
# sf.write('Read_speech_data/BPRS_high_audio/09_24_2021_BPRS67_16khz.wav', audio_high, sr)

# write enhanced audio
# s_rate = 16000
# sf.write('Read_speech_data/BPRS_low_audio/5_4_2022_BPRS49_16khz.wav', audio_low, sr)
sf.write('Read_speech_data/BPRS_high_audio/09_24_2021_BPRS67_16khz_enhanced.wav', audio_high_enh, sr)