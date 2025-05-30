from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ImageClip
import argparse
import yt_dlp
import os
import re
import tempfile
import time
from google import genai
from google.genai import types


# Function to convert HH:MM:SS.mmm, MM:SS.mmm, SS.mmm, MM:SS, HH:MM:SS to seconds
def time_to_seconds(time_str):
    parts = time_str.split(":")
    h, m, s_float = 0, 0, 0.0

    if len(parts) == 3:  # HH:MM:SS or HH:MM:SS.mmm
        h = int(parts[0])
        m = int(parts[1])
        s_float = float(parts[2])
    elif len(parts) == 2:  # MM:SS or MM:SS.mmm
        m = int(parts[0])
        s_float = float(parts[1])
    elif len(parts) == 1:  # SS or SS.mmm (just seconds)
        s_float = float(parts[0])
    else:
        raise ValueError(f"Invalid time format structure: {time_str}")

    return float(h * 3600 + m * 60) + s_float


def parse_vtt_captions(vtt_file_path):
    """Parses a VTT caption file and returns a list of caption segments."""
    captions = []
    if not os.path.exists(vtt_file_path):
        print(f"Error: VTT file not found at {vtt_file_path}")
        return captions

    try:
        with open(vtt_file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading VTT file {vtt_file_path}: {e}")
        return captions

    # Regex to match VTT time cues and text blocks
    # It handles optional hours in timestamps and multiline caption text
    # Example: 00:00:04.260 --> 00:00:05.710 or 00:04.260 --> 00:05.710
    # The text can span multiple lines and may have styling tags, which we'll strip roughly.
    # Simpler regex for cues, assuming WEBVTT and KIND/LANGUAGE lines are ignored for now.
    # We will process line by line for more robust parsing.

    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if "-->" in line:
            try:
                start_time_str, end_time_str = line.split(" --> ")
                # Further processing for time strings if they have extra info beyond timestamp
                start_time_str = start_time_str.split(" ")[0]
                end_time_str = end_time_str.split(" ")[0]

                start_seconds = time_to_seconds(start_time_str)
                end_seconds = time_to_seconds(end_time_str)

                caption_text_lines = []
                # Read subsequent lines as caption text until a blank line or another timestamp
                while i < len(lines) and lines[i].strip():
                    caption_text_lines.append(lines[i].strip())
                    i += 1

                if caption_text_lines:
                    full_caption_text = " ".join(caption_text_lines)
                    # Basic cleaning of VTT tags like <v Roger Goodell>
                    full_caption_text = re.sub(r"<[^>]+>", "", full_caption_text)
                    captions.append(
                        {
                            "start": start_seconds,
                            "end": end_seconds,
                            "text": full_caption_text.strip(),
                        }
                    )
            except ValueError as ve:
                # print(f"Skipping invalid time format in VTT: {line} - {ve}")
                pass  # Skip malformed time lines
            except Exception as e:
                # print(f"Error parsing VTT block near '{line}': {e}")
                pass  # Skip block on other errors
        # Skip other lines (like WEBVTT, empty lines, sequence numbers, etc.)

    return captions


def download_video(url, output_dir, cookies_from_browser_str=None):
    """Downloads a video and its English VTT captions from a YouTube URL."""
    video_path = None
    caption_path = None
    try:
        ydl_opts = {
            # Request 480p max, trying for mp4 directly, then best audio, fallback to best 480p overall
            "format": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio/best[height<=480][ext=mp4]/best[height<=480]",
            "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
            "quiet": False,
            "noplaylist": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "vtt",
            # Ensure the final output is mp4 by recoding if necessary after download
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
            # 'keepvideo': True, # Often needed if explicit recoding happens from a non-mp4 source,
            # but yt-dlp's default merge to mp4 might make this redundant.
            # Let's test without it first, as preferedformat should handle it.
        }

        if cookies_from_browser_str:
            parts = cookies_from_browser_str.split(":")
            # yt-dlp expects a tuple: (BROWSER, PROFILE, KEYRING, CONTAINER)
            # We'll support BROWSER and BROWSER:PROFILE for simplicity first.
            # User can extend this if keyring/container needed by providing more parts.
            # Ensure all parts of the tuple are present, even if None for later ones.
            parsed_cookie_args = [None] * 4
            for i, part in enumerate(parts):
                if i < len(parsed_cookie_args):
                    parsed_cookie_args[i] = part if part else None

            ydl_opts["cookiesfrombrowser"] = tuple(parsed_cookie_args)
            print(f"DEBUG: Using cookiesfrombrowser: {ydl_opts['cookiesfrombrowser']}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info_dict)

            # Ensure the video file has an mp4 extension if it was merged
            if (
                video_path
                and not video_path.endswith(".mp4")
                and info_dict.get("ext") == "mp4"
            ):
                base, _ = os.path.splitext(video_path)
                renamed_video_path = base + ".mp4"
                if os.path.exists(video_path):
                    os.rename(video_path, renamed_video_path)
                    video_path = renamed_video_path
                elif os.path.exists(renamed_video_path):
                    video_path = renamed_video_path
                else:
                    print(
                        f"Warning: Downloaded video {video_path} was not an mp4 and could not be renamed."
                    )

            # Attempt to find the caption file path
            if video_path and os.path.exists(video_path):
                # Option 1: Check info_dict (more reliable if available)
                if info_dict.get("requested_subtitles"):
                    sub_info = info_dict["requested_subtitles"].get("en")
                    if (
                        sub_info
                        and sub_info.get("filepath")
                        and os.path.exists(sub_info["filepath"])
                    ):
                        caption_path = sub_info["filepath"]
                    elif (
                        sub_info and sub_info.get("ext") == "vtt"
                    ):  # if filepath not absolute
                        # Construct path based on video path if relative path in info_dict
                        expected_caption_name = (
                            os.path.splitext(os.path.basename(video_path))[0]
                            + ".en.vtt"
                        )
                        potential_caption_path = os.path.join(
                            output_dir, expected_caption_name
                        )
                        if os.path.exists(potential_caption_path):
                            caption_path = potential_caption_path

                # Option 2: If not found via info_dict, construct expected path (less reliable but good fallback)
                if not caption_path:
                    base_name, _ = os.path.splitext(video_path)
                    expected_caption_path = base_name + ".en.vtt"
                    if os.path.exists(expected_caption_path):
                        caption_path = expected_caption_path

                if caption_path:
                    print(f"Found caption file: {caption_path}")
                else:
                    print(
                        "English VTT captions not found or path could not be determined."
                    )
            else:
                print(
                    "Video download failed or video path not found, skipping caption search."
                )

    except yt_dlp.utils.DownloadError as e:
        print(f"Error downloading video/captions: {e}")
        return None, None  # Return None for both if download error
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return None, None  # Return None for both for other errors

    return video_path, caption_path


def create_clip(
    input_path, output_path, clip_abs_start, clip_abs_end, all_parsed_captions=None
):
    """
    Creates a clip from the input video file, optionally adding timed text captions.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output clip.
        clip_abs_start (float): Absolute start time of the clip in the original video (seconds).
        clip_abs_end (float): Absolute end time of the clip in the original video (seconds).
        all_parsed_captions (list, optional): List of parsed caption dicts
                                            (e.g., [{'start': float, 'end': float, 'text': str}]).
    """
    video = None
    main_subclip = None
    final_composite_clip = None
    # Keep track of all created text clips for closing
    created_text_clips = []

    try:
        video = VideoFileClip(input_path)
        clip_duration = clip_abs_end - clip_abs_start

        if clip_duration <= 0:
            print(f"Error: Clip duration is zero or negative for {output_path}.")
            return

        main_subclip = video.subclip(clip_abs_start, clip_abs_end)

        text_overlays = []

        if all_parsed_captions:
            for caption in all_parsed_captions:
                cap_start_abs = caption["start"]
                cap_end_abs = caption["end"]
                cap_text = caption["text"]

                # Determine overlap between caption and current clip
                overlap_start = max(clip_abs_start, cap_start_abs)
                overlap_end = min(clip_abs_end, cap_end_abs)

                if overlap_start < overlap_end:  # There is an overlap
                    # Calculate start and duration relative to the subclip
                    text_rel_start = overlap_start - clip_abs_start
                    text_duration = overlap_end - overlap_start

                    if text_duration <= 0:
                        continue  # Skip zero or negative duration overlays

                    # Create TextClip
                    # Define text properties - consider making these configurable later
                    fontsize = 30
                    font = "Liberation-Sans"
                    color = "yellow"
                    bg_color = "rgba(0, 0, 0, 0.8)"
                    stroke_color = "black"
                    stroke_width = 2
                    interline = -10

                    text_clip_width = int(main_subclip.w * 0.9)

                    txt_clip = TextClip(
                        cap_text,
                        fontsize=fontsize,
                        color=color,
                        font=font,
                        bg_color=bg_color,
                        stroke_color=stroke_color,
                        stroke_width=stroke_width,
                        method="caption",
                        size=(text_clip_width, None),
                        align="center",
                        interline=interline,
                    )

                    # Position slightly up from the absolute bottom, ensuring space for multi-line text
                    # Estimate text height based on fontsize and number of lines (rough estimate)
                    # A more accurate way would be to get txt_clip.h after creation, but this is a circular dependency for positioning.
                    # For now, let's use a fixed bottom margin or adjust based on subclip height.
                    bottom_margin = int(main_subclip.h * 0.05)  # 5% margin from bottom
                    position_y = main_subclip.h - txt_clip.h - bottom_margin

                    # Ensure text is not positioned off-screen if it's very tall
                    if position_y < 0:
                        position_y = 0

                    txt_clip = (
                        txt_clip.set_position(("center", position_y))
                        .set_duration(text_duration)
                        .set_start(text_rel_start)
                    )

                    text_overlays.append(txt_clip)
                    created_text_clips.append(txt_clip)  # Add to list for later closing

        if text_overlays:
            # Add main subclip first, then all text overlays
            final_composite_clip = CompositeVideoClip(
                [main_subclip] + text_overlays, size=main_subclip.size
            )
            final_composite_clip.write_videofile(
                output_path, codec="libx264", audio_codec="aac"
            )
        else:
            # No text overlays, just write the subclip
            main_subclip.write_videofile(
                output_path, codec="libx264", audio_codec="aac"
            )

        print(f"Clip saved to {output_path}")

    except Exception as e:
        print(f"An error occurred while creating clip {output_path}: {e}")
        import traceback

        print(traceback.format_exc())  # Print full traceback for debugging
    finally:
        if main_subclip:
            main_subclip.close()
        for tc in created_text_clips:  # Close all created text clips
            if hasattr(tc, "close") and callable(tc.close):
                try:
                    tc.close()
                except Exception as e_tc_close:
                    print(f"Error closing a text clip: {e_tc_close}")
        if final_composite_clip:
            final_composite_clip.close()
        if video:
            video.close()


def format_captions_for_llm(parsed_captions):
    """Formats parsed captions into a single transcript string for the LLM."""
    transcript_parts = []
    for caption in parsed_captions:
        # We can include timestamps in the transcript for better context for the LLM
        # but for now, let's keep it simple with just the text.
        # transcript_parts.append(f"[{caption[start]:.2f}s - {caption[end]:.2f}s] {caption[text]}")
        transcript_parts.append(caption["text"])
    return "\n".join(transcript_parts)


def get_llm_clip_suggestions(downloaded_video_path, api_key, prompt_text):
    """Gets clip suggestions from Gemini LLM by uploading the downloaded video file."""
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return []

    uploaded_file_resource = None  # To store the File object from upload
    try:
        client = genai.Client(api_key=api_key)
        model_name = "gemini-1.5-pro"

        print(
            f"\nUploading video file ({downloaded_video_path}) to Gemini for analysis..."
        )
        if not os.path.exists(downloaded_video_path):
            print(f"Error: Video file not found at {downloaded_video_path} for upload.")
            return []

        uploaded_file_resource = client.files.upload(
            file=downloaded_video_path,
            config=types.UploadFileConfig(mime_type="video/mp4"),
        )
        print(
            f"Video uploaded successfully: {uploaded_file_resource.name} ({uploaded_file_resource.uri}). Waiting for it to become active..."
        )

        # Polling loop to check for ACTIVE state
        processing_timeout_seconds = 180  # 3 minutes timeout for processing
        polling_interval_seconds = 10  # Check every 10 seconds
        start_time = time.time()

        while True:
            current_file_state = client.files.get(name=uploaded_file_resource.name)
            print(
                f"Current file state: {current_file_state.state} (checking for ACTIVE)"
            )

            if current_file_state.state == types.FileState.ACTIVE:
                print("File is now ACTIVE and ready for use.")
                break
            elif current_file_state.state == types.FileState.FAILED:
                print(
                    f"Error: File processing FAILED for {uploaded_file_resource.name}. Error: {current_file_state.error}"
                )
                # Attempt to delete the failed upload before returning
                try:
                    client.files.delete(name=uploaded_file_resource.name)
                    print(
                        f"Successfully deleted failed upload {uploaded_file_resource.name}."
                    )
                except Exception as e_del_failed:
                    print(
                        f"Error deleting failed upload {uploaded_file_resource.name}: {e_del_failed}"
                    )
                uploaded_file_resource = (
                    None  # Ensure it's not processed in finally if it failed here
                )
                return []  # Stop processing for this LLM attempt

            if time.time() - start_time > processing_timeout_seconds:
                print(
                    f"Timeout: File {uploaded_file_resource.name} did not become ACTIVE within {processing_timeout_seconds} seconds."
                )
                # Attempt to delete the timed-out upload before returning
                try:
                    client.files.delete(name=uploaded_file_resource.name)
                    print(
                        f"Successfully deleted timed-out upload {uploaded_file_resource.name}."
                    )
                except Exception as e_del_timeout:
                    print(
                        f"Error deleting timed-out upload {uploaded_file_resource.name}: {e_del_timeout}"
                    )
                uploaded_file_resource = None  # Ensure it's not processed in finally
                return []

            time.sleep(polling_interval_seconds)

        print(
            f"Sending uploaded video URI ({uploaded_file_resource.uri}) and prompt to LLM ({model_name}) for clip suggestions..."
        )

        llm_contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        file_data=types.FileData(
                            file_uri=uploaded_file_resource.uri, mime_type="video/mp4"
                        )
                    ),
                    types.Part.from_text(text=prompt_text),
                ],
            )
        ]

        # Configuration for the generation - ensuring text response
        gen_config_obj = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )

        response = client.models.generate_content(
            model=model_name, contents=llm_contents, config=gen_config_obj
        )

        suggestions = []
        if response.text:
            print(f"LLM Response Text:\n{response.text}")
            for line in response.text.splitlines():
                line = line.strip()
                # Updated regex to capture times with colons, e.g., M:SS or HH:MM:SS or S.ms
                match = re.search(
                    r"START:\s*([\d:\.]+)\s*END:\s*([\d:\.]+)", line, re.IGNORECASE
                )
                if match:
                    try:
                        start_time_str = match.group(1)
                        end_time_str = match.group(2)

                        start_time = time_to_seconds(start_time_str)
                        end_time = time_to_seconds(end_time_str)

                        if start_time < end_time:
                            suggestions.append({"start": start_time, "end": end_time})
                        else:
                            print(
                                f"Warning: LLM suggested start time {start_time} is not before end time {end_time}. Skipping."
                            )
                    except ValueError:
                        print(
                            f"Warning: Could not parse time from LLM suggestion: {line}"
                        )
        else:
            print("LLM returned no text.")
            # Add more detailed feedback if available, e.g., response.prompt_feedback for older genai client
            # For google.generativeai it was response.prompt_feedback
            # For the new client, need to check how to get safety ratings or block reasons if any.
            # This might be part of the response object directly or within response.candidates.
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if (
                    hasattr(candidate, "finish_reason")
                    and candidate.finish_reason != "STOP"
                ):
                    print(f"Generation stopped. Reason: {candidate.finish_reason}")
                if hasattr(candidate, "safety_ratings"):
                    print(f"Safety Ratings: {candidate.safety_ratings}")

        if not suggestions:
            print(
                "LLM did not provide any valid clip suggestions in the expected format."
            )
        return suggestions

    except Exception as e:
        print(f"Error communicating with LLM or processing response: {e}")
        if hasattr(e, "errors"):
            print(f"API Errors: {e.errors}")
        if hasattr(e, "message"):
            print(f"API Message: {e.message}")
        return []
    finally:
        if uploaded_file_resource:
            print(
                f"Deleting uploaded file {uploaded_file_resource.name} from Google servers..."
            )
            try:
                client.files.delete(name=uploaded_file_resource.name)
                print(f"Successfully deleted {uploaded_file_resource.name}.")
            except Exception as e_del:
                print(
                    f"Error deleting uploaded file {uploaded_file_resource.name}: {e_del}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Create short clips from YouTube videos, optionally using LLM for time suggestions."
    )
    parser.add_argument("youtube_url", help="YouTube video URL.")
    parser.add_argument(
        "output_path",
        help="Path to save the output clip (base name if multiple clips). Example: my_clip.mp4",
    )
    parser.add_argument(
        "--start_time",
        type=float,
        help="(Optional) Start time of the clip in seconds. LLM suggestions override this.",
    )
    parser.add_argument(
        "--end_time",
        type=float,
        help="(Optional) End time of the clip in seconds. LLM suggestions override this.",
    )
    parser.add_argument(
        "--use_llm",
        action="store_true",
        help="Use LLM to suggest clip times (requires GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="(Optional) Specify browser to load cookies from (e.g., 'chrome', 'firefox:default'). See yt-dlp docs for more.",
    )

    args = parser.parse_args()
    print(f"DEBUG: Parsed youtube_url: '{args.youtube_url}'")

    youtube_url_to_process = args.youtube_url.strip()
    downloaded_temp_file = None
    temp_dir = None
    caption_file_to_process = None  # Added for caption file
    parsed_captions = []  # To store parsed captions
    llm_suggested_clips = []  # To store LLM suggestions
    api_key = os.environ.get("GEMINI_API_KEY")

    # Define the refined prompt here to pass to the LLM function
    llm_prompt_text = """
Analyze the following video. Your goal is to identify 1 to 3 distinct, highly engaging, and concise segments that would make excellent short video clips (e.g., for TikTok, YouTube Shorts, Instagram Reels). 
These clips should be catchy and serve as potential hooks or highlight the most impactful moments.

**Key criteria for suggested clips:**
- **Duration:** Ideally between 10 and 30 seconds. Avoid suggesting clips longer than 45 seconds unless absolutely necessary for a crucial point.
- **Content:** Focus on peak moments, strong statements, key takeaways, intriguing questions, or emotionally resonant parts.
- **Conciseness:** Each clip should be able to stand alone and deliver a punch.

Please format EACH suggestion STRICTLY as follows, with each suggestion on a new line:
START: [start_time_seconds] END: [end_time_seconds]

Example:
START: 30.5 END: 45.2
START: 120.0 END: 135.7

Video analysis and suggestions:
"""

    print(
        f"Input is a YouTube URL: {youtube_url_to_process}. Attempting to download video and captions..."
    )
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory

    # Pass the cookies_from_browser argument to download_video
    downloaded_video_path, downloaded_caption_path = download_video(
        youtube_url_to_process, temp_dir, args.cookies_from_browser
    )
    print(
        f"DEBUG: download_video('{youtube_url_to_process}') returned video: '{downloaded_video_path}', caption: '{downloaded_caption_path}'"
    )

    if not downloaded_video_path or not os.path.exists(downloaded_video_path):
        print("Failed to download video or downloaded video file not found. Exiting.")
        if temp_dir and os.path.exists(temp_dir):
            # Clean up temp directory if download failed or file not found
            if downloaded_video_path and os.path.exists(
                downloaded_video_path
            ):  # Though unlikely if first check failed
                try:
                    os.remove(downloaded_video_path)
                except OSError as e:
                    print(
                        f"Error removing temporary video file {downloaded_video_path}: {e}"
                    )
            if downloaded_caption_path and os.path.exists(
                downloaded_caption_path
            ):  # Also clean up caption if it exists
                try:
                    os.remove(downloaded_caption_path)
                except OSError as e:
                    print(
                        f"Error removing temporary caption file {downloaded_caption_path}: {e}"
                    )
            try:
                # Ensure directory is empty or handle specific files before rmdir
                files_in_temp = os.listdir(temp_dir)
                if not files_in_temp:  # Only remove if empty
                    os.rmdir(temp_dir)
                else:
                    # Attempt to remove known files if they exist to help make dir empty
                    # This part might be redundant if files are already handled or removed
                    # For safety, just print a warning if not empty
                    print(
                        f"Warning: Temporary directory {temp_dir} not empty after failed download. Contains: {files_in_temp}. Manual cleanup may be needed."
                    )

            except OSError as e:
                print(f"Error removing temporary directory {temp_dir}: {e}")
        return

    input_to_clip = downloaded_video_path
    downloaded_temp_file = downloaded_video_path  # Keep track to delete later
    if downloaded_caption_path and os.path.exists(downloaded_caption_path):
        caption_file_to_process = downloaded_caption_path
        print(f"Video downloaded to: {input_to_clip}")
        print(f"Captions downloaded to: {caption_file_to_process}")

        # Parse the downloaded VTT file
        if caption_file_to_process:
            print(f"Parsing VTT captions from {caption_file_to_process}...")
            parsed_captions = parse_vtt_captions(caption_file_to_process)
            if parsed_captions:
                print(f"Successfully parsed {len(parsed_captions)} caption segments.")
                # For debugging, print the first few parsed captions
                # for idx, cap in enumerate(parsed_captions[:3]):
                #     print(f"  {idx + 1}: Start={{cap[start]:.2f}}s, End={{cap[end]:.2f}}s, Text=\"{cap[text][:50]}...\"")

                if args.use_llm:
                    if not api_key:
                        print(
                            "\n--use_llm flag is set, but GEMINI_API_KEY environment variable is not found. Skipping LLM suggestions."
                        )
                    else:
                        # Removed transcript_for_llm = format_captions_for_llm(parsed_captions)
                        # The new LLM function takes the URL directly
                        # if transcript_for_llm: # This check is no longer needed for the primary LLM path
                        llm_suggested_clips = get_llm_clip_suggestions(
                            downloaded_video_path, api_key, llm_prompt_text
                        )
                        if llm_suggested_clips:
                            print(
                                f"\nLLM suggested {len(llm_suggested_clips)} clip(s):"
                            )
                            for i, clip_times in enumerate(llm_suggested_clips):
                                print(
                                    f"  Suggestion {i+1}: Start={clip_times['start']:.2f}s, End={clip_times['end']:.2f}s"
                                )
                        else:
                            print("\nLLM did not provide usable clip suggestions.")
                else:
                    print("No captions were parsed or VTT file was empty/invalid.")
    else:
        print(f"Video downloaded to: {input_to_clip}")
        print("Captions were not successfully downloaded or found.")

    # Determine clip times
    clips_to_create = []
    if args.use_llm and llm_suggested_clips:
        clips_to_create = llm_suggested_clips
        print(f"\nUsing {len(clips_to_create)} clip suggestion(s) from LLM.")
    elif args.start_time is not None and args.end_time is not None:
        clips_to_create.append({"start": args.start_time, "end": args.end_time})
        print(
            f"\nUsing manually specified start and end times: {args.start_time}s to {args.end_time}s."
        )
    else:
        print(
            "\nNo clip times specified (manual or via LLM) and --use_llm not successful. No clips will be created."
        )
        # Cleanup and exit if no clips to create
        if downloaded_temp_file:
            try:
                os.remove(downloaded_temp_file)
            except OSError as e:
                print(f"Error deleting temporary video file: {e}")
        if caption_file_to_process:
            try:
                os.remove(caption_file_to_process)
            except OSError as e:
                print(f"Error deleting temporary caption file: {e}")
        if temp_dir and os.path.exists(temp_dir):
            try:
                if not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                    print(f"Successfully removed temporary directory: {temp_dir}")
                else:
                    print(
                        f"Warning: Temporary directory {temp_dir} is not empty. Contains: {os.listdir(temp_dir)}. Manual cleanup needed."
                    )
            except OSError as e:
                print(f"Error deleting temporary directory {temp_dir}: {e}")
        return

    # Proceed to create clip(s)
    output_base, output_ext = os.path.splitext(args.output_path)
    if not output_ext:
        output_ext = ".mp4"  # Default to .mp4 if no extension

    for i, clip_times in enumerate(clips_to_create):
        clip_start_time = clip_times["start"]
        clip_end_time = clip_times["end"]

        final_output_path = args.output_path
        if len(clips_to_create) > 1:
            final_output_path = f"{output_base}_clip{i+1}{output_ext}"
        elif i > 0:
            final_output_path = f"{output_base}_clip{i+1}{output_ext}"

        print(
            f"\nCreating clip {i+1}/{len(clips_to_create)}: {final_output_path} (start: {clip_start_time:.2f}s, end: {clip_end_time:.2f}s)"
        )
        create_clip(
            input_to_clip,
            final_output_path,
            clip_start_time,
            clip_end_time,
            parsed_captions,
        )  # Pass parsed_captions

    # Cleanup (already partially handled if no clips, ensure it runs after clips too)
    if downloaded_temp_file:
        print(f"Cleaning up temporary video file: {downloaded_temp_file}")
        try:
            os.remove(downloaded_temp_file)
        except OSError as e:
            print(f"Error deleting temporary video file: {e}")

    if caption_file_to_process:  # Also clean up caption file
        print(f"Cleaning up temporary caption file: {caption_file_to_process}")
        try:
            os.remove(caption_file_to_process)
        except OSError as e:
            print(f"Error deleting temporary caption file: {e}")

    if temp_dir and os.path.exists(temp_dir):  # also remove the directory
        try:
            if not os.listdir(temp_dir):  # Only remove if empty
                os.rmdir(temp_dir)
                print(f"Successfully removed temporary directory: {temp_dir}")
            else:
                print(
                    f"Warning: Temporary directory {temp_dir} is not empty after processing. Contains: {os.listdir(temp_dir)}. Manual cleanup may be needed."
                )
        except OSError as e:
            print(f"Error deleting temporary directory {temp_dir}: {e}")


if __name__ == "__main__":
    main()
