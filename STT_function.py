from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch
from pydub import AudioSegment
import os
import librosa 

# model_id = "facebook/mms-1b-fl102"
model_id = "facebook/mms-1b-all"


processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

def split_audio_to_chunks(file_path, output_dir, chunk_length_ms=60000):
    # Load the audio file
    audioFull = AudioSegment.from_file(file_path)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the length of the audio file in milliseconds
    audio_length_ms = len(audioFull)
    
    final_transcription = ""
    
    # Split the audio file into chunks
    for i in range(0, audio_length_ms, chunk_length_ms):
        # Determine the start and end of the chunk
        start_ms = i
        end_ms = min(i + chunk_length_ms, audio_length_ms)
        
        # Extract the chunk
        chunk = audioFull[start_ms:end_ms]
        
        
        # Create the output file name
        chunk_filename = os.path.join(output_dir, f"chunk_{i // chunk_length_ms + 1}.wav")
        
        # Export the chunk to a file
        chunk.export(chunk_filename, format="wav")
        
        print(f"Exported {chunk_filename}")
        
        audio_path= chunk_filename
        sampling_rate=16000
        audio, _ = librosa.load(audio_path, sr=sampling_rate)
        
        processor.tokenizer.set_target_lang("gle")
        model.load_adapter("gle")

        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor.decode(ids)
        print (transcription)
        final_transcription = final_transcription + transcription

    return final_transcription