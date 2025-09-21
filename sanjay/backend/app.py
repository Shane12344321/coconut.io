from flask import Flask, render_template, request, jsonify
from processing import transcribe
import os, subprocess

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, "input.mp4")
    file.save(filepath)

    audio_path = os.path.join(UPLOAD_FOLDER, "audio.wav")
    subprocess.run([r"C:\ffmpeg\ffmpeg.exe", "-y", "-i", filepath, audio_path])
    transcript, segments = transcribe(audio_path)
    return jsonify({"message": "Video uploaded and transcribed","transcript": transcript,"segments": segments, "audio_path": audio_path})

if __name__ == "__main__":
    app.run(debug=True)
