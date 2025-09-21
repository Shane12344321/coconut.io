from faster_whisper import WhisperModel
import subprocess
import os
import uuid

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
# Preprocess audio to mono 16kHz
def preprocess_audio(input_path, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        output_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Load Whisper model once
model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio(audio_path):
    processed_path = os.path.splitext(audio_path)[0] + "_16k.wav"
    preprocess_audio(audio_path, processed_path)

    segments, info = model.transcribe(processed_path, beam_size=1)

    full_transcript = " ".join([seg.text.strip() for seg in segments])
    formatted_segments = []
    for seg in segments:
        formatted_segments.append({
            "start_time": round(seg.start, 2),
            "end_time": round(seg.end, 2),
            "text": seg.text.strip()
        })

    return {
        "language": info.language,
        "duration": round(info.duration, 2),
        "full_transcript": full_transcript,
        "segments": formatted_segments
    }

# Generate clip from start to end
def make_clip(video_path, start, end, output_folder="clips"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    clip_name = f"clip_{uuid.uuid4().hex}.mp4"
    clip_path = os.path.join(output_folder, clip_name)
    duration = end - start
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ss", str(start), "-t", str(duration), clip_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return clip_path
