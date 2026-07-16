from asr_data import AudioDB

db = AudioDB("房产客服电话.vasr")
audios = db.query(10)
audio = audios[0]
print(audio.source)
waveform = audio.load()
print(audios[0].timeline("mono"))
print(audios[0])
