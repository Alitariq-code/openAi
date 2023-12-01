from django.shortcuts import render
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from difflib import SequenceMatcher
from home.my_fun import transcribe_audio
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json
import requests
from .my_fun import analyze_audio, compare_lines, count_duplicate_lines, count_skipped_lines, count_words,calculate_word_count_ratio,get_wav_duration,transcribe_audio,calculate_error_metrics,calculate_pause_metrics


class ProcessApiView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            data = request.data
            audio_url = data.get('audio_url')
            original_text = data.get('original_text')

            # Download the audio file from the provided URL
            response = requests.get(audio_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download audio from the provided URL. Status code: {response.status_code}")

            # Save audio data to a temporary file
            audio_file_path = "temp_original.wav"
            with open(audio_file_path, "wb") as temp_file:
                temp_file.write(response.content)

            # Convert audio to mono and set sample width to 2 bytes
            audio = AudioSegment.from_file(audio_file_path)
            audio = audio.set_channels(1).set_sample_width(2)
            audio.export("temp.wav", format="wav")

            # Transcribe audio using Vosk
            transcribed_text = transcribe_audio("temp.wav")
            print('transcribed_text', transcribed_text)
            result = compare_lines(original_text, transcribed_text)
            result = compare_lines(original_text, transcribed_text)
            df_delete = result[0]
            df_subtititue = result[1]
            df_insert = result[2]

            duplicate_lines = count_duplicate_lines(transcribed_text)
            skipped_lines = count_skipped_lines(transcribed_text)
            word_count = count_words(transcribed_text)

            # Calculate error metrics
            error_metrics = calculate_error_metrics(original_text, transcribed_text)

            # Calculate pause metrics
            pause_metrics = calculate_pause_metrics(transcribed_text)

            # Calculate accuracy, audio duration, and transcription confidence score
            accuracy = error_metrics.get('Words Correct per Minute', 0) / word_count * 100 if word_count != 0 else 0
            audio_duration = get_wav_duration('temp.wav')

            original_vs_audio = calculate_word_count_ratio(transcribed_text, original_text)
            analysis_result = analyze_audio('temp.wav')
            Correct = word_count - (len(df_delete) + len(df_insert))
            Acuracy = Correct / word_count

            # Convert lists to JSON format
            df_delete_json = json.dumps(df_delete)
            df_insert_json = json.dumps(df_insert)
            df_subtititue_json = json.dumps(df_subtititue)
            # Prepare JSON response with additional outcomes
            response_data = {
                'transcribed_text': transcribed_text,
                'analysis_result': analysis_result,
                'deleted_words': df_delete_json,
                'inserted_words': df_insert_json,
                'substituted_words': df_subtititue_json,
                'duplicate_lines': duplicate_lines,
                'skipped_lines': skipped_lines,
                'word_count': word_count,
                'error_metrics': error_metrics,
                'pause_metrics': pause_metrics,
                'correct_words': Correct,
                'accuracy': f"{Acuracy:.2f}%",
                'original_vs_audio': original_vs_audio,
                # 'accuracy': accuracy,
                'audio_duration': audio_duration,
                'transcription_confidence': 76,
                # Add other metrics as needed
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)