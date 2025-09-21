import subprocess, os, uuid
from faster_whisper import WhisperModel

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

# Initialize Whisper model once
model = WhisperModel("base", device="cpu", compute_type="int8")

def preprocess_audio(input_path, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        output_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def transcribe_audio(audio_path):
    processed_path = os.path.splitext(audio_path)[0] + "_16k.wav"
    preprocess_audio(audio_path, processed_path)

    segments, info = model.transcribe(processed_path, beam_size=1)

    # Take top 5 longest segments
    segments = sorted(segments, key=lambda s: s.end - s.start, reverse=True)[:5]

    full_transcript = " ".join([s.text.strip() for s in segments])
    formatted_segments = []
    for s in segments:
        formatted_segments.append({
            "start_time": round(s.start, 2),
            "end_time": round(s.end, 2),
            "text": s.text.strip()
        })

    return {
        "language": info.language,
        "duration": round(info.duration, 2),
        "full_transcript": full_transcript,
        "segments": formatted_segments
    }

def make_clip(video_path, start, end, output_folder="clips"):
    os.makedirs(output_folder, exist_ok=True)
    clip_name = f"clip_{uuid.uuid4().hex}.mp4"
    clip_path = os.path.join(output_folder, clip_name)
    duration = end - start
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ss", str(start), "-t", str(duration), clip_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return clip_path
