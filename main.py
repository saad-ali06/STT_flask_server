from flask import Flask, request, jsonify
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch
from pydub import AudioSegment
import os
import librosa
import tempfile

app = Flask(__name__)

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
        
        audio_path = chunk_filename
        sampling_rate = 16000
        audio, _ = librosa.load(audio_path, sr=sampling_rate)
        
        # Set target language and load adapter
        processor.tokenizer.set_target_lang("gle")
        model.load_adapter("gle")

        # Process and run model for transcription
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor.decode(ids)
        final_transcription += transcription + " "
    
    return final_transcription.strip()

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save file to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        # Output directory for audio chunks
        output_dir = os.path.join(temp_dir, "chunks")
        
        try:
            transcription = split_audio_to_chunks(file_path, output_dir)
            return jsonify({"transcription": transcription}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# @app.route('/transcribe', methods=['POST'])
# def transcribe():
#     return jsonify({"transcription":'DONE'}), 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
