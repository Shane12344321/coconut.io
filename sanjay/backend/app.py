from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
import os, subprocess, uuid, shutil
from threading import Thread
from processing import transcribe_audio, make_clip
import logging

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Flask + SocketIO setup
# -----------------------------
app = Flask(__name__, template_folder="templates")
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------------
# Folders
# -----------------------------
UPLOAD_FOLDER = "uploads"
CLIPS_FOLDER = "clips"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLIPS_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'wmv'}

# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_ffmpeg():
    return shutil.which("ffmpeg") is not None

def cleanup_old_files():
    import time
    current_time = time.time()
    for folder in [UPLOAD_FOLDER, CLIPS_FOLDER]:
        if not os.path.exists(folder):
            continue
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath) and current_time - os.path.getctime(filepath) > 3600:
                try:
                    os.remove(filepath)
                    logger.info(f"Removed old file: {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to remove {filepath}: {e}")

# -----------------------------
# Background video processing
# -----------------------------
def process_video(video_path, audio_path, job_id):
    try:
        socketio.emit("processing_start", {"job_id": job_id, "message": "Starting transcription..."}, namespace="/")
        transcription_result = transcribe_audio(audio_path)
        segments = transcription_result.get("segments", [])

        clip_filenames = []
        total_segments = len(segments)

        for idx, seg in enumerate(segments, start=1):
            try:
                clip_path = make_clip(video_path, seg["start_time"], seg["end_time"], CLIPS_FOLDER)
                clip_filenames.append(os.path.basename(clip_path))
                socketio.emit("clip_progress", {
                    "job_id": job_id,
                    "current": idx,
                    "total": total_segments,
                    "text": seg.get("text", ""),
                    "clip_created": True
                }, namespace="/")
            except Exception as e:
                logger.error(f"Failed to create clip {idx}: {e}")
                socketio.emit("clip_progress", {
                    "job_id": job_id,
                    "current": idx,
                    "total": total_segments,
                    "text": seg.get("text", ""),
                    "clip_created": False
                }, namespace="/")

        socketio.emit("clip_complete", {
            "job_id": job_id,
            "clips": clip_filenames,
            "transcription": transcription_result
        }, namespace="/")

    finally:
        for f in [video_path, audio_path]:
            try: os.remove(f)
            except: pass
        cleanup_old_files()

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    try:
        if not check_ffmpeg():
            return jsonify({"error": "FFmpeg not found. Please install FFmpeg and add it to PATH."}), 500

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": f"File type not allowed. Allowed: {ALLOWED_EXTENSIONS}"}), 400

        unique_id = str(uuid.uuid4())
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'mp4'
        video_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.{ext}")
        audio_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.wav")
        file.save(video_path)

        # Extract audio for Whisper
        cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", audio_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg audio extraction failed: {result.stderr}")
            return jsonify({"error": "Audio extraction failed"}), 500

        # Start background processing safely
        socketio.start_background_task(process_video, video_path, audio_path, unique_id)

        return jsonify({"message": "Upload successful, processing started", "job_id": unique_id})

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/clips/<path:filename>")
def serve_clip(filename):
    if '..' in filename or filename.startswith('/'):
        return jsonify({"error": "Invalid filename"}), 400
    filepath = os.path.join(CLIPS_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(CLIPS_FOLDER, filename)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    logger.info("Starting Flask + SocketIO app...")
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)
