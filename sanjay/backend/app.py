from flask import Flask, render_template, request, jsonify, send_from_directory
import os, subprocess
from processing import transcribe_audio, make_clip

app = Flask(__name__, static_folder="clips", template_folder="templates")

UPLOAD_FOLDER = "uploads"
CLIPS_FOLDER = "clips"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLIPS_FOLDER, exist_ok=True)

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

    # Extract audio
    audio_path = os.path.join(UPLOAD_FOLDER, "audio.wav")
    subprocess.run(["ffmpeg", "-y", "-i", filepath, audio_path],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Transcribe
    transcription_result = transcribe_audio(audio_path)

    # Generate clips for each segment
    clip_paths = []
    for seg in transcription_result["segments"]:
        clip_path = make_clip(filepath, seg["start_time"], seg["end_time"], CLIPS_FOLDER)
        clip_paths.append(clip_path)

    # Make paths relative for frontend
    clip_urls = [os.path.basename(c) for c in clip_paths]

    return jsonify({
        "message": "Video uploaded, transcribed, and clips generated",
        "transcription": transcription_result,
        "clips": clip_urls
    })

# Optional: serve clips
@app.route("/clips/<path:filename>")
def serve_clip(filename):
    return send_from_directory(CLIPS_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
