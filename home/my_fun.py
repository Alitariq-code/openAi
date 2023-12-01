import whisper
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from difflib import SequenceMatcher
import pandas as pd
from difflib import SequenceMatcher
import requests
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from difflib import SequenceMatcher

def transcribe_audio(file_path):
    # Load the Whisper model
    print('doing')
    model = whisper.load_model("base")
    print('model', model)  # You can choose different model sizes like tiny, small, medium, large, etc.
    # Transcribe the audio file
    result = model.transcribe(file_path)
    print('result', result)
    # Return the transcription
    return result["text"]
def get_wav_duration(file_path):
    """
    Get the duration of a WAV file.

    Parameters:
    - file_path (str): Path to the WAV file.

    Returns:
    - float: Duration in seconds.
    """
    audio = AudioSegment.from_file(file_path)
    duration_in_seconds = len(audio) / 1000.0
    return duration_in_seconds

def analyze_audio(file_path):
    """
    Analyze audio for word repetitions, short pauses, and long pauses.

    Parameters:
    - file_path (str): Path to the audio file.

    Returns:
    - dict: Analysis results.
    """
    audio = AudioSegment.from_file(file_path, format="wav")
    
    # Set thresholds for pause durations
    short_pause_threshold = 3000  # 3 seconds
    long_pause_threshold = 3000   # 3 seconds
    
    # Initialize variables for counting repetitions and pauses
    word_repetitions = 0
    short_pauses = 0
    long_pauses = 0
    
    # Split audio on silence
    segments = split_on_silence(audio, silence_thresh=-40)  # Adjust the threshold based on your audio
    
    # Iterate through the segments
    for i in range(len(segments)):
        segment_duration = len(segments[i])
        
        # Check if the segment duration falls within the pause thresholds
        if segment_duration >= short_pause_threshold and segment_duration <= long_pause_threshold:
            short_pauses += 1
        elif segment_duration > long_pause_threshold:
            long_pauses += 1
        
        # Check for repeated words within a 3-second window
        window_start = max(0, i - 1)
        window_end = min(len(segments), i + 1)
        window = segments[window_start:window_end]
        
        if len(window) > 1 and compare_segments(segments[i], sum(window)):
            word_repetitions += 1
    
    return {
        "word_repetitions": word_repetitions,
        "short_pauses": short_pauses,
        "long_pauses": long_pauses
    }

def compare_lines(original_lines, spoken_lines, similarity_threshold=0.1):
    """
    Compare original and spoken lines for deleted, inserted, substituted, and repeated words.

    Parameters:
    - original_lines (str): Original text.
    - spoken_lines (str): Spoken text.
    - similarity_threshold (float): Similarity threshold for considering lines similar (default is 0.1).

    Returns:
    - tuple: Lists of deleted, inserted, substituted, and repeated words.
    """
    original_lines = original_lines.lower()
    spoken_lines = spoken_lines.lower()

    original_lines = remove_newlines(original_lines)
    original_lines = remove_punctuation(original_lines)
    spoken_lines = remove_newlines(spoken_lines)
    original_lines = original_lines.strip().split('\n')
    spoken_lines = spoken_lines.strip().split('\n')

    deleted_words = []
    inserted_words = []
    substituted_words = []
    repeated_words = []

    for spoken_line in spoken_lines:
        max_similarity = 0
        most_similar_original_line = None

        # Calculate similarity for each original line
        for original_line in original_lines:
            similarity = SequenceMatcher(None, original_line, spoken_line).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_original_line = original_line

        if max_similarity >= similarity_threshold:
            original_words = most_similar_original_line.split()
            spoken_words = spoken_line.split()

            # Identify deleted words
            deleted_words.extend([word for word in original_words if word not in spoken_words])

            # Identify inserted words
            inserted_words.extend([word for word in spoken_words if word not in original_words])

            # Identify substituted words (original word not in deleted or inserted)
            substituted_words.extend([(original_word, spoken_word) for original_word, spoken_word in zip(original_words, spoken_words) if original_word not in spoken_words])

            # Identify consecutive repeated words
            current_repeated_words = []
            for word in spoken_words:
                if spoken_words.count(word) > 1 and word not in current_repeated_words:
                    current_repeated_words.append(word)
                else:
                    if len(current_repeated_words) > 1:
                        repeated_words.extend(current_repeated_words)
                    current_repeated_words = []

    subt = list(dict(substituted_words).keys())
    substituted_words = [word for word in subt if word not in inserted_words]
    repeated_words = remove_duplicates(repeated_words)

    return deleted_words, inserted_words, substituted_words, repeated_words

def count_duplicate_lines(text_data):
    """
    Count the number of duplicate lines in the given text data.

    Parameters:
    - text_data (str): Text data.

    Returns:
    - int: Number of duplicate lines.
    """
    seen_lines = set()
    duplicate_count = 0

    lines = text_data.split('\n')  # Split the input text into lines

    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespaces
        if not line:
            continue  # Skip empty lines

        if line in seen_lines:
            duplicate_count += 1
        else:
            seen_lines.add(line)

    return duplicate_count

def count_skipped_lines(text_data):
    """
    Count the number of skipped lines in the given text data.

    Parameters:
    - text_data (str): Text data.

    Returns:
    - int: Number of skipped lines.
    """
    lines = text_data.split('\n')  # Split the input text into lines
    skipped_count = 0

    for i in range(1, len(lines) - 1):
        current_line = lines[i].strip()
        next_line = lines[i + 1].strip()

        if not next_line:
            skipped_count += 1

    return skipped_count

def count_words(text):
    """
    Count the number of words in the given text.

    Parameters:
    - text (str): Text.

    Returns:
    - int: Number of words.
    """
    words = text.split()
    return len(words)

def remove_duplicates(word_list):
    """
    Remove duplicates from a list while maintaining the order.

    Parameters:
    - word_list (list): List of words.

    Returns:
    - list: List with duplicates removed.
    """
    unique_words = []
    seen_words = set()

    for word in word_list:
        if word not in seen_words:
            unique_words.append(word)
            seen_words.add(word)

    return unique_words

def remove_newlines(text):
    """
    Remove newline characters from the given text.

    Parameters:
    - text (str): Text.

    Returns:
    - str: Text with newlines removed.
    """
    return text.replace('\n', '')

def remove_punctuation(text):
    """
    Remove commas and periods from the given text.

    Parameters:
    - text (str): Text.

    Returns:
    - str: Text with commas and periods removed.
    """
    cleaned_text = text.replace(',', '').replace('.', '')
    return cleaned_text

def compare_segments(segment1, segment2):
    """
    Compare two audio segments based on their root mean square (RMS) values.

    Parameters:
    - segment1 (AudioSegment): First audio segment.
    - segment2 (AudioSegment): Second audio segment.

    Returns:
    - bool: True if the RMS of segment1 is greater than 80% of the RMS of segment2, else False.
    """
    return segment1.set_frame_rate(44100).set_channels(1).rms > segment2.set_frame_rate(44100).set_channels(1).rms * 0.8


def find_repeated_words(text):
    words = text.split()
    repeated_words = []
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            repeated_words.append(words[i])
    return repeated_words



def calculate_word_count_ratio(transcribed_text, original_text, max_ratio=100):
    """
    Calculate the ratio of word count in the transcribed text compared to the original text.

    Parameters:
    - transcribed_text (str): Transcribed text.
    - original_text (str): Original text.
    - max_ratio (float): Maximum value for the ratio (default is 100).

    Returns:
    - float: Calculated ratio.
    """
    word_count_org = count_words(original_text)
    word_count_transcribed = count_words(transcribed_text)

    # Calculate the ratio and limit it to the maximum value
    ratio = min(word_count_org / word_count_transcribed * 100, max_ratio)
    return ratio



def calculate_error_metrics(original_text, transcribed_text):
    """
    Calculate error metrics between original and transcribed text.

    Parameters:
    - original_text (str): Original text.
    - transcribed_text (str): Transcribed text.

    Returns:
    - dict: Error metrics.
    """
    words_original = original_text.split()
    words_transcribed = transcribed_text.split()

    wc = len(set(words_original) & set(words_transcribed))  # Words Correct
    wr = len(words_transcribed)  # Words Reads

    # Implement your logic to calculate other error metrics
    # ...

    error_metrics = {
        'WR': wr,
        'WC': wc,
        'Words Correct per Minute': calculate_words_per_minute(wc, transcribed_text),
        # Add other error metrics as needed
    }

    return error_metrics

def calculate_words_per_minute(words_correct, transcribed_text):
    """
    Calculate words correct per minute.

    Parameters:
    - words_correct (int): Number of words correct.
    - transcribed_text (str): Transcribed text.

    Returns:
    - float: Words correct per minute.
    """
    # Implement your logic to calculate words correct per minute
    # ...

    # Placeholder value, replace with actual calculation
    audio_duration = 120  # seconds
    words_per_minute = (words_correct / audio_duration) * 60

    return words_per_minute

def calculate_pause_metrics(transcribed_text):
    """
    Calculate pause metrics in the transcribed text.

    Parameters:
    - transcribed_text (str): Transcribed text.

    Returns:
    - dict: Pause metrics.
    """
    # Implement your logic to calculate pause metrics
    # ...

    # Placeholder values, replace with actual calculations
    pauses_1_3_seconds = 5
    hesitations_3_seconds = 2

    pause_metrics = {
        'Pauses (1-3 seconds)': pauses_1_3_seconds,
        'Hesitations (3+ seconds)': hesitations_3_seconds,
        # Add other pause metrics as needed
    }

    return pause_metrics
