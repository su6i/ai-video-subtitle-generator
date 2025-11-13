# video_multilang_translate.py
# Multi-language subtitle generator with DeepSeek API + automatic font configuration

from faster_whisper import WhisperModel
from openai import OpenAI
import argparse
import os
import re
import subprocess
from datetime import timedelta
import configparser
import shutil
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# Language configuration with native fonts
LANGUAGE_CONFIG = {
    'en': {'name': 'English', 'font': 'Arial', 'font_size': 32, 'rtl': False},
    'fa': {'name': 'Persian', 'font': 'B Nazanin', 'font_size': 48, 'rtl': True},
    'ar': {'name': 'Arabic', 'font': 'Arial', 'font_size': 36, 'rtl': True},
    'es': {'name': 'Spanish', 'font': 'Arial', 'font_size': 36, 'rtl': False},
    'fr': {'name': 'French', 'font': 'Arial', 'font_size': 36, 'rtl': False},
    'de': {'name': 'German', 'font': 'Arial', 'font_size': 36, 'rtl': False},
    'it': {'name': 'Italian', 'font': 'Arial', 'font_size': 36, 'rtl': False},
    'pt': {'name': 'Portuguese', 'font': 'Arial', 'font_size': 36, 'rtl': False},
    'ru': {'name': 'Russian', 'font': 'Arial', 'font_size': 36, 'rtl': False},
    'ja': {'name': 'Japanese', 'font': 'MS Gothic', 'font_size': 36, 'rtl': False},
    'ko': {'name': 'Korean', 'font': 'Malgun Gothic', 'font_size': 36, 'rtl': False},
    'zh': {'name': 'Chinese', 'font': 'SimHei', 'font_size': 36, 'rtl': False},
    'hi': {'name': 'Hindi', 'font': 'Mangal', 'font_size': 36, 'rtl': False},
    'tr': {'name': 'Turkish', 'font': 'Arial', 'font_size': 36, 'rtl': False},
    'nl': {'name': 'Dutch', 'font': 'Arial', 'font_size': 36, 'rtl': False},
}


def format_time(seconds: float) -> str:
    """Convert seconds to SRT time format (00:00:00,000)"""
    td = timedelta(seconds=float(seconds))
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def load_api_key(config_file: str = '.config') -> str:
    """Load API key from config file"""
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' not found!")
    
    config.read(config_file)
    
    if 'DEFAULT' in config and 'DEEPSEEK_API' in config['DEFAULT']:
        api_key = config['DEFAULT']['DEEPSEEK_API'].strip()
        if not api_key:
            raise ValueError("DEEPSEEK_API key is empty in config file!")
        return api_key
    
    raise ValueError("DEEPSEEK_API not found in config file!")


def fix_persian_text(text: str) -> str:
    """Fix common Persian typography issues"""
    if not text:
        return text
    
    # Fix ZWNJ (Zero Width Non-Joiner) - نیم‌فاصله
    # Add ZWNJ between repeated letters that should be separate
    patterns = [
        (r'(\w)(ها)(\s|$)', r'\1‌\2\3'),  # کتابها → کتاب‌ها
        (r'(\w)(های)(\s|$)', r'\1‌\2\3'), # کتابهای → کتاب‌های
        (r'می(\s)', r'می‌\1'),             # می کنم → می‌کنم
        (r'نمی(\s)', r'نمی‌\1'),           # نمی کنم → نمی‌کنم
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    
    # Fix common word breaks
    text = text.replace('صحبتهای', 'صحبت‌های')
    text = text.replace('صحبتها', 'صحبت‌ها')
    text = text.replace('ویدیوهای', 'ویدیو‌های')
    text = text.replace('ویدیوها', 'ویدیو‌ها')
    text = text.replace('فیلمهای', 'فیلم‌های')
    text = text.replace('فیلمها', 'فیلم‌ها')
    
    return text


def translate_with_deepseek(texts: List[str], target_lang: str, api_key: str, source_lang: str = 'en') -> List[str]:
    """Batch translation with DeepSeek API"""
    if not texts:
        return []
    
    lang_name = LANGUAGE_CONFIG.get(target_lang, {}).get('name', target_lang)
    source_lang_name = LANGUAGE_CONFIG.get(source_lang, {}).get('name', source_lang)
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    # Create numbered batch
    batch_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    
    # Enhanced prompt for better Persian quality
    if target_lang == 'fa':
        system_prompt = """You are a professional translator specializing in Persian (Farsi). 
Translate English texts to natural, conversational Persian following these rules:
- Use proper Persian grammar and vocabulary
- Use ZWNJ (‌) correctly: می‌کنم, نمی‌کنم, کتاب‌ها, فیلم‌های
- Keep translations concise and natural
- Only write the translation, no explanations"""
    else:
        system_prompt = f"You are a professional translator. Translate {source_lang_name} texts to natural, conversational {lang_name}. Only write the translation, no explanations."
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate these {source_lang_name} sentences to natural {lang_name}:\n\n{batch_text}"}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        output = response.choices[0].message.content.strip()
        translations = []
        
        # Parse numbered responses
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            if '. ' in line and line.split('.')[0].isdigit():
                trans = line.split('. ', 1)[1]
                # Apply Persian fixes if translating to Persian
                if target_lang == 'fa':
                    trans = fix_persian_text(trans)
                translations.append(trans)
            else:
                if target_lang == 'fa':
                    line = fix_persian_text(line)
                translations.append(line)
        
        if len(translations) != len(texts):
            print(f"Warning: Expected {len(texts)} translations, got {len(translations)}. Using original texts for missing.")
            translations.extend(texts[len(translations):])
        
        return translations[:len(texts)]
    
    except Exception as e:
        print(f"Translation error: {e}. Using original texts.")
        return texts


def clean_subtitle_line(text: str) -> str:
    """Remove trailing punctuation from subtitle line"""
    text = text.strip()
    if not text:
        return text
    
    # Remove trailing punctuation
    while text and text[-1] in ".،,؟!?…":
        text = text[:-1].strip()
    
    return text


def create_ass_with_font(srt_path: str, ass_path: str, lang_code: str = 'en') -> None:
    """Convert SRT to ASS with language-specific font"""
    lang_config = LANGUAGE_CONFIG.get(lang_code, LANGUAGE_CONFIG['en'])
    font_name = lang_config['font']
    font_size = lang_config['font_size']
    
    # Create temporary ASS with subtitles filter to force font size
    result = subprocess.run([
        'ffmpeg', '-y', '-i', srt_path,
        '-f', 'ass',
        ass_path
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
    
    # Read ASS file
    with open(ass_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace ALL font settings (both in Style line and any Fontsize parameters)
    # Replace the main Style line
    content = re.sub(
        r'Style: Default,[^,]*,\d+',
        f'Style: Default,{font_name},{font_size}',
        content
    )
    
    # Also replace any inline font size overrides
    content = re.sub(
        r'\\fs\d+',
        f'\\\\fs{font_size}',
        content
    )
    
    # Make sure PlayResY is set for proper scaling
    if 'PlayResX' not in content:
        content = content.replace('[Script Info]', 
                                '[Script Info]\nPlayResX: 1920\nPlayResY: 1080')
    
    # Save modified content
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(content)


def create_combined_ass(subtitle_files: List[Tuple[str, str]], output_ass: str) -> None:
    """
    Create ASS file with one or multiple subtitle tracks
    subtitle_files: List of (srt_path, lang_code) tuples
    """
    if not subtitle_files:
        raise ValueError("No subtitle files provided")
    
    # If only one subtitle, create it directly
    if len(subtitle_files) == 1:
        srt_path, lang_code = subtitle_files[0]
        print(f"Creating single-language ASS: {output_ass}")
        create_ass_with_font(srt_path, output_ass, lang_code)
        
        # Verify the font size in created file
        with open(output_ass, 'r', encoding='utf-8') as f:
            content = f.read()
            style_match = re.search(r'Style: Default,([^,]+),(\d+)', content)
            if style_match:
                print(f"  ✓ Font: {style_match.group(1)}, Size: {style_match.group(2)}")
        return
    
    # Multiple subtitles: create combined ASS
    print(f"Creating combined ASS with {len(subtitle_files)} languages")
    
    # First, ensure individual ASS files exist
    ass_files = []
    for srt_path, lang_code in subtitle_files:
        ass_path = srt_path.replace('.srt', '.ass')
        if not os.path.exists(ass_path):
            create_ass_with_font(srt_path, ass_path, lang_code)
        ass_files.append((ass_path, lang_code))
    
    # Write combined ASS
    with open(output_ass, 'w', encoding='utf-8') as f:
        # Header
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n")
        f.write("WrapStyle: 0\n")
        f.write("ScaledBorderAndShadow: yes\n\n")
        
        # Styles
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        
        for i, (_, lang_code) in enumerate(ass_files):
            lang_config = LANGUAGE_CONFIG.get(lang_code, LANGUAGE_CONFIG['en'])
            font_name = lang_config['font']
            font_size = lang_config['font_size']
            
            # Position: first at bottom, others stacked above
            margin_v = 20 if i == 0 else 80 + (i - 1) * 60
            
            style_line = f"Style: {lang_code.upper()},{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2.5,1,2,10,10,{margin_v},1\n"
            f.write(style_line)
            print(f"  ✓ {lang_code.upper()}: {font_name} size {font_size} at margin {margin_v}")
        
        # Events
        f.write("\n[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for ass_path, lang_code in ass_files:
            with open(ass_path, 'r', encoding='utf-8') as af:
                content = af.read()
            
            for line in re.findall(r"Dialogue: .*", content, re.MULTILINE):
                parts = line.split(",", 9)
                if len(parts) >= 10:
                    parts[3] = lang_code.upper()
                    f.write(",".join(parts) + "\n")
    
    print(f"  Combined ASS created: {output_ass}")


def render_video_with_subtitles(video_path: str, ass_path: str, output_path: str) -> None:
    """Render video with embedded subtitles"""
    print(f"Rendering video with subtitles → {os.path.basename(output_path)}")
    
    # First, get video resolution and duration
    probe_result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'default=noprint_wrappers=1',
        video_path
    ], capture_output=True, text=True)
    
    width, height, duration = 1920, 1080, 0
    if probe_result.returncode == 0:
        try:
            for line in probe_result.stdout.strip().split('\n'):
                if 'width=' in line:
                    width = line.split('=')[1]
                elif 'height=' in line:
                    height = line.split('=')[1]
                elif 'duration=' in line:
                    duration = float(line.split('=')[1])
            print(f"  Video: {width}x{height}, Duration: {duration:.1f}s")
        except:
            pass
    
    # Fix ASS file to match video resolution
    with open(ass_path, 'r', encoding='utf-8') as f:
        ass_content = f.read()
    
    ass_content = re.sub(r'PlayResX: \d+', f'PlayResX: {width}', ass_content)
    ass_content = re.sub(r'PlayResY: \d+', f'PlayResY: {height}', ass_content)
    
    fixed_ass = ass_path.replace('.ass', '_fixed.ass')
    with open(fixed_ass, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    
    print(f"  Encoding: 0%", end='', flush=True)
    
    # Start FFmpeg
    process = subprocess.Popen([
        'ffmpeg', '-y', '-i', video_path,
        '-vf', f'ass={fixed_ass}',
        '-c:a', 'copy',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-progress', 'pipe:2',
        '-nostats',
        output_path
    ], stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, bufsize=1)
    
    # Track progress
    last_percent = 0
    for line in process.stderr:
        if 'out_time_ms=' in line:
            try:
                time_ms = int(line.split('=')[1].strip())
                time_sec = time_ms / 1000000.0
                
                if duration > 0:
                    percent = min(int((time_sec / duration) * 100), 100)
                    if percent > last_percent:
                        print(f"\r  Encoding: {percent}%", end='', flush=True)
                        last_percent = percent
            except:
                pass
    
    process.wait()
    print(f"\r  Encoding: 100% ✓")
    
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg rendering failed")
    
    # Cleanup temporary fixed ASS file
    try:
        os.remove(fixed_ass)
    except:
        pass


def find_existing_subtitle(base_path: str, lang_code: str) -> Optional[str]:
    """Find existing subtitle file for a language"""
    lang_name = LANGUAGE_CONFIG.get(lang_code, {}).get('name', '').lower()
    
    possible_names = [
        f"{base_path}_{lang_code}.srt",
        f"{base_path}_{lang_name}.srt",
    ]
    
    # Special cases for common naming patterns
    if lang_code == 'en':
        possible_names.insert(0, f"{base_path}.srt")  # Plain .srt usually means English
        possible_names.append(f"{base_path}_english.srt")
    elif lang_code == 'fa':
        possible_names.extend([
            f"{base_path}_farsi.srt",
            f"{base_path}_persian.srt",
            f"{base_path}_fa.srt"
        ])
    elif lang_code == 'ar':
        possible_names.append(f"{base_path}_arabic.srt")
    elif lang_code == 'es':
        possible_names.append(f"{base_path}_spanish.srt")
    elif lang_code == 'fr':
        possible_names.append(f"{base_path}_french.srt")
    elif lang_code == 'de':
        possible_names.append(f"{base_path}_german.srt")
    elif lang_code == 'ja':
        possible_names.append(f"{base_path}_japanese.srt")
    elif lang_code == 'zh':
        possible_names.extend([
            f"{base_path}_chinese.srt",
            f"{base_path}_cn.srt"
        ])
    
    return next((f for f in possible_names if os.path.exists(f)), None)


def extract_subtitles_from_srt(srt_path: str) -> List[Dict]:
    """Extract subtitle entries from SRT file"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    return [
        {
            'index': int(m[0]),
            'start': m[1],
            'end': m[2],
            'text': m[3].strip()
        }
        for m in matches
    ]


def write_srt_file(srt_path: str, entries: List[Dict]) -> None:
    """Write subtitle entries to SRT file"""
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, entry in enumerate(entries, 1):
            text = clean_subtitle_line(entry['text'])
            f.write(f"{i}\n{entry['start']} --> {entry['end']}\n{text}\n\n")


def transcribe_video(video_path: str, model_size: str = 'base', language: str = 'en') -> str:
    """Transcribe video using Whisper and return SRT path"""
    print(f"Transcribing video with Whisper ({model_size})...")
    
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments = list(model.transcribe(
        video_path,
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )[0])
    
    base_name = os.path.splitext(video_path)[0]
    srt_path = f"{base_name}_{language}.srt" if language != 'en' else f"{base_name}.srt"
    
    print(f"Saving transcription: {srt_path}")
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            if seg.text.strip():
                f.write(f"{i}\n{format_time(seg.start)} --> {format_time(seg.end)}\n{seg.text.strip()}\n\n")
    
    return srt_path


def process_video_subtitles(
    video_path: str,
    source_lang: str,
    target_langs: List[str],
    model_size: str = 'base',
    render: bool = False,
    force_transcribe: bool = False
) -> Dict[str, str]:
    """
    Main processing function
    Returns dict of {lang_code: srt_path}
    """
    print(f"Processing video: {video_path}")
    print(f"Source language: {source_lang}")
    print(f"Target languages: {', '.join(target_langs)}\n")
    
    base_path = os.path.splitext(video_path)[0]
    api_key = load_api_key()
    
    result_files = {}
    
    # Step 1: Get source language subtitle
    source_srt = find_existing_subtitle(base_path, source_lang)
    
    if source_srt and not force_transcribe:
        print(f"Found existing {source_lang.upper()} subtitle: {source_srt}")
    else:
        if force_transcribe:
            print(f"Force transcribing (--force-transcribe enabled)...")
        else:
            print(f"No existing {source_lang.upper()} subtitle found.")
        source_srt = transcribe_video(video_path, model_size, source_lang)
    
    # Only add source to result if it's in target_langs
    if source_lang in target_langs:
        result_files[source_lang] = source_srt
    
    # Step 2: Extract source text
    source_entries = extract_subtitles_from_srt(source_srt)
    source_texts = [e['text'] for e in source_entries]
    
    # Step 3: Translate to each target language
    for target_lang in target_langs:
        if target_lang == source_lang:
            continue
        
        target_srt = f"{base_path}_{target_lang}.srt"
        
        # Check if translation already exists
        existing = find_existing_subtitle(base_path, target_lang)
        if existing:
            print(f"Found existing {target_lang.upper()} subtitle: {existing}")
            result_files[target_lang] = existing
            continue
        
        # Translate
        print(f"Translating to {target_lang.upper()}...")
        batch_size = 20
        all_translations = []
        
        for i in range(0, len(source_texts), batch_size):
            batch = source_texts[i:i+batch_size]
            print(f"  Translating lines {i+1}–{min(i+batch_size, len(source_texts))}")
            
            translations = translate_with_deepseek(batch, target_lang, api_key, source_lang)
            all_translations.extend(translations)
        
        # Write translated SRT
        translated_entries = [
            {**entry, 'text': trans}
            for entry, trans in zip(source_entries, all_translations)
        ]
        write_srt_file(target_srt, translated_entries)
        print(f"Created: {target_srt}")
        result_files[target_lang] = target_srt
    
    # Step 4: Create ASS files
    print("\nCreating ASS subtitle files...")
    for lang_code, srt_path in result_files.items():
        ass_path = srt_path.replace('.srt', '.ass')
        if not os.path.exists(ass_path) or force_transcribe:
            create_ass_with_font(srt_path, ass_path, lang_code)
            print(f"  Created: {ass_path}")
        else:
            print(f"  Using existing: {ass_path}")
    
    # Step 5: Render video if requested
    if render:
        # Determine output filename
        if len(result_files) == 1:
            lang_suffix = list(result_files.keys())[0]
        else:
            lang_suffix = "_".join(sorted(result_files.keys()))
        
        output_video = f"{base_path}_{lang_suffix}_subtitled.mp4"
        
        if os.path.exists(output_video):
            print(f"\nSubtitled video already exists: {output_video}")
        else:
            print(f"\nRendering video with subtitles...")
            
            # Create combined ASS (works for both single and multiple languages)
            combined_ass = f"{base_path}_combined.ass"
            subtitle_files = [(srt, lang) for lang, srt in result_files.items()]
            create_combined_ass(subtitle_files, combined_ass)
            
            render_video_with_subtitles(video_path, combined_ass, output_video)
            print(f"Created: {output_video}")
    
    # Summary
    print("\n" + "="*50)
    print("Processing complete!")
    print("="*50)
    for lang, path in result_files.items():
        print(f"  {lang.upper()}: {path}")
    if render:
        print(f"  Video: {output_video}")
    
    return result_files


# ====================== MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-language video subtitle generator with DeepSeek API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe English and translate to Persian
  python script.py video.mp4 -s en -t fa
  
  # Translate to multiple languages
  python script.py video.mp4 -s en -t fa ar es
  
  # Render video with subtitles
  python script.py video.mp4 -s en -t fa -r
  
  # List supported languages
  python script.py --list-languages
        """
    )
    
    parser.add_argument("video", nargs='?', help="Video file path")
    parser.add_argument(
        "-s", "--source",
        default="en",
        help=f"Source language code (default: en). Available: {', '.join(LANGUAGE_CONFIG.keys())}"
    )
    parser.add_argument(
        "-t", "--target",
        nargs='+',
        help=f"Target language code(s). Available: {', '.join(LANGUAGE_CONFIG.keys())}"
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "-r", "--render",
        action="store_true",
        help="Render video with embedded subtitles"
    )
    parser.add_argument(
        "-f", "--force-transcribe",
        action="store_true",
        help="Force re-transcription even if subtitle exists"
    )
    parser.add_argument(
        "-l", "--list-languages",
        action="store_true",
        help="List all supported languages"
    )
    
    args = parser.parse_args()
    
    # List languages
    if args.list_languages:
        print("Supported languages:")
        print("-" * 40)
        for code, info in sorted(LANGUAGE_CONFIG.items()):
            print(f"  {code:4s} - {info['name']}")
        exit(0)
    
    # Validate arguments
    if not args.video:
        parser.error("Video file path required (or use --list-languages)")
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        exit(1)
    
    if not args.target:
        parser.error("At least one target language required (-t)")
    
    # Validate language codes
    invalid_langs = [lang for lang in [args.source] + args.target if lang not in LANGUAGE_CONFIG]
    if invalid_langs:
        print(f"Error: Invalid language code(s): {', '.join(invalid_langs)}")
        print(f"Use --list-languages to see available options")
        exit(1)
    
    # Process video
    try:
        process_video_subtitles(
            args.video,
            args.source,
            args.target,
            model_size=args.model,
            render=args.render,
            force_transcribe=args.force_transcribe
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)