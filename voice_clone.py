from flask import Flask, request, jsonify, send_file
from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark
import scipy.io.wavfile
import os
import tempfile
import shutil
import requests

# Initialize Flask app
app = Flask(__name__)

# Load TTS model (this part stays the same as in your original script)
config = BarkConfig()
model = Bark.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="bark/", eval=True)

TEMP_DIR = 'temp_audio_files/'
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    try:
        # 1. Get transcription text and audio file from request
        transcription = request.form.get('transcription')
        audio_file = request.files.get('audio_file')
        
        if not transcription or not audio_file:
            return jsonify({"error": "Both 'transcription' and 'audio_file' are required"}), 400
        
        # 2. Save the uploaded audio file temporarily
        audio_path = os.path.join(TEMP_DIR, "input_audio.wav")
        audio_file.save(audio_path)

        speaker_id = "speaker"  # This would be based on the uploaded audio (or predefined)
        voice_dirs = "bark_voices/"

        # 4. Process the text into speech (synthesize)
        output_dict = model.synthesize(
            transcription, 
            config, 
            speaker_id=speaker_id, 
            voice_dirs=voice_dirs
        )

        # 5. Save the synthesized audio to a temporary file
        output_audio_path = os.path.join(TEMP_DIR, "generated_audio.wav")
        sample_rate = 24000
        scipy.io.wavfile.write(output_audio_path, rate=sample_rate, data=output_dict["wav"])

        # 6. Send the generated audio file back to the client (Node.js app)
        return send_file(output_audio_path, mimetype='audio/wav', as_attachment=True, download_name="generated_audio.wav")

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # 7. Cleanup: Remove the temporary files after the process
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)