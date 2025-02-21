import os
import sys
from pydub import AudioSegment

# List of supported audio file extensions
AUDIO_EXTENSIONS = ('.wav')

def get_audio_duration(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        return audio.duration_seconds
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return 0

def scan_folder(folder_path):
    total_duration = 0.0
    file_count = 0

    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(AUDIO_EXTENSIONS):
                file_path = os.path.join(root, f)
                duration = get_audio_duration(file_path)
                if duration > 0:
                    total_duration += duration
                    file_count += 1
    return total_duration, file_count

if __name__ == "__main__":
    # folder names as command line arguments
    folders = []
    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        folders = ['High', 'Low']
    base_dir = os.path.dirname(os.path.abspath(__file__))

    overall_duration = 0.0
    overall_count = 0

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        folder_duration, count = scan_folder(folder_path)
        overall_duration += folder_duration
        overall_count += count

        if count > 0:
            avg = folder_duration / count
            print(f"Folder '{folder}': {count} audio files, average duration {avg:.2f} seconds.")
        else:
            print(f"Folder '{folder}': No valid audio files found.")

    if overall_count > 0:
        overall_avg = overall_duration / overall_count
        print(f"\nOverall: {overall_count} audio files, total duration {overall_duration:.2f} seconds, average duration {overall_avg:.2f} seconds.")
    else:
        print("No valid audio files were processed.")
