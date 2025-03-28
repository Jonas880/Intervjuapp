import streamlit as st
import re
from io import BytesIO
import openai
import os
from pydub import AudioSegment
import logging
import mimetypes
import time
import math

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- !!! SECURITY WARNINGS !!! ---
OPENAI_API_KEY = "" # <-- Replace or use Secrets

# --- Check for API Key ---
# if not OPENAI_API_KEY:
#     st.error("🚨 OpenAI API key is missing.")
#     st.stop()

# --- Configuration ---
DEFAULT_TARGET_WORD_COUNT = 60
HIGHLIGHT_MARKER = "[HIGHLIGHTED SEGMENT] "

# --- Constants for Dynamic Text Area Height (Revised) ---
CHARS_PER_LINE_ESTIMATE = 90  # Rough estimate (Adjust if needed)
PIXELS_PER_LINE = 20          # Estimated height of a line (Adjust if needed)
BASE_TEXT_AREA_HEIGHT = 30    # Reduced base height (Overhead for padding/border)
MIN_TEXT_AREA_HEIGHT = 60     # Reduced minimum height
MAX_TEXT_AREA_HEIGHT = 400    # Maximum height

# --- GPT Function ---
# [Function remains the same]
def fix_segment_with_gpt(text_segment):
    if not OPENAI_API_KEY:
        logging.error("fix_segment_with_gpt called without API Key."); return text_segment
    prompt = ("Professional transcription editor: Fix typos/grammar in this text. "
              "Do NOT rephrase or change meaning. Literal fixes only. Output ONLY corrected text.\n\n" + text_segment)
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.2)
        reply = response.choices[0].message.content.strip()
        if not reply: logging.warning(f"GPT empty reply for: '{text_segment[:50]}...'."); return text_segment
        logging.debug(f"GPT corrected: '{text_segment[:50]}...' -> '{reply[:50]}...'"); return reply
    except openai.AuthenticationError as e:
        logging.error(f"OpenAI Auth Error: {e}."); st.warning("OpenAI Auth Error. Segment uncorrected.", icon="🔑"); return text_segment
    except Exception as e:
        logging.error(f"OpenAI API Error: {e}."); st.warning(f"OpenAI API Error. Segment uncorrected.", icon="⚠️"); return text_segment

# --- Cleaning Function (Final Download) ---
# [Function remains the same]
def clean_final_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# --- Timestamp Parser (with number removal) ---
# [Function remains the same]
def parse_timestamped_transcript(text):
    pattern = r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})\n(.*?)(?=\n\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->|\Z)"
    matches = re.findall(pattern, text, re.DOTALL); blocks = []
    for start, end, content in matches:
        content = re.sub(r"^\s*\d+\s*\n", "", content); start_norm = start.replace(",", "."); end_norm = end.replace(",", ".")
        text_clean = re.sub(r'\s+', ' ', content.strip())
        text_clean = re.sub(r"(?<!\S)\b\d+\b(?!\S)", "", text_clean).strip()
        text_clean = re.sub(r'\s{2,}', ' ', text_clean)
        if text_clean: blocks.append({"start": start_norm, "end": end_norm, "text": text_clean.strip()})
    logging.info(f"Parsed/cleaned {len(blocks)} non-empty blocks.")
    if not blocks: st.warning("⚠️ No transcript blocks found after cleaning.")
    return blocks

# --- Audio Utils ---
# [Functions remain the same: get_audio_segment, time_to_millis, get_audio_format, get_audio_format_from_mime]
def get_audio_segment(full_audio: AudioSegment | None, start_str: str, end_str: str) -> AudioSegment | None:
    if full_audio is None: return None
    try:
        start_ms=time_to_millis(start_str); end_ms=time_to_millis(end_str)
        start_ms=max(0, start_ms); end_ms=max(start_ms, end_ms); end_ms=min(end_ms, len(full_audio))
        return full_audio[start_ms:end_ms] if start_ms < end_ms else None
    except ValueError as e: logging.error(f"Timestamp conversion error: {e}"); return None
    except Exception as e: logging.error(f"Audio slicing error: {e}"); return None

def time_to_millis(ts_str: str) -> int:
    try:
        h, m, sec_ms = ts_str.split(':'); s, ms_str = sec_ms.split('.')
        ms = int(ms_str.ljust(3, '0')[:3]) if len(sec_ms)>1 and ms_str else 0 # Handle missing ms
        return int(h)*3600000 + int(m)*60000 + int(s)*1000 + ms
    except Exception as e: raise ValueError(f"Invalid timestamp: '{ts_str}'") from e

def get_audio_format(file_obj) -> str | None:
    fmt = (get_audio_format_from_mime(file_obj.type) or
           (os.path.splitext(file_obj.name)[1].lstrip('.') if file_obj.name else None))
    if not fmt: logging.warning(f"Could not determine audio format for {file_obj.name}")
    return fmt

def get_audio_format_from_mime(mime: str | None) -> str | None:
    if not mime: return None
    mapping={"audio/mpeg":"mp3","audio/mp4":"m4a","audio/m4a":"m4a","audio/x-m4a":"m4a",
             "audio/wav":"wav","audio/x-wav":"wav","audio/ogg":"ogg","audio/aac":"aac"}
    fmt=mapping.get(mime) or (mimetypes.guess_extension(mime) or "").lstrip('.')
    if not fmt: fmt='m4a' if 'mp4' in mime else 'mp3' if 'mpeg' in mime else 'wav' if 'wav' in mime else None
    return fmt

# --- Grouping Function ---
# [Function remains the same]
def group_blocks_by_word_count(blocks, target_count=60):
    grouped=[]; current_text=""; current_start=None; current_end=None; current_count=0
    if not blocks: return grouped
    for i, block in enumerate(blocks):
        txt=block['text']; wc=len(txt.split()); is_new = (current_count == 0)
        finalize = not is_new and (current_count >= target_count or (current_count + wc > target_count * 1.5 and current_count > 0))
        if finalize:
            grouped.append({"text": current_text.strip(), "original_text": current_text.strip(), "start": current_start, "end": current_end})
            current_text=""; current_count=0; current_start=None; is_new=True
        if is_new: current_start=block['start']; current_text=txt
        else: current_text += " " + txt
        current_end=block['end']; current_count+=wc
        if i == len(blocks) - 1 and current_count > 0:
             grouped.append({"text": current_text.strip(), "original_text": current_text.strip(), "start": current_start, "end": current_end})
    logging.info(f"Grouped into {len(grouped)} segments (~{target_count} words).")
    return grouped

# --- Initialize Session State ---
# [Function remains the same]
def initialize_session_state(grouped_segments, fix_typos_flag):
    count = len(grouped_segments)
    st.session_state.segment_count = count
    processed = []; status_editor = "None"
    highlight_keys = [f"highlight_{i}" for i in range(count)]
    if fix_typos_flag and count > 0:
        status_editor = "GPT-4o (Per Segment)"; gpt_start = time.time()
        status = st.status(f"🤖 Correcting {count} segments...", expanded=False); bar = status.progress(0)
        for i, seg in enumerate(grouped_segments):
            status.write(f"Segment {i+1}/{count}..."); original = seg['text']; corrected = fix_segment_with_gpt(original)
            seg['text'] = corrected; processed.append(seg)
            bar.progress((i + 1) / count)
        gpt_end = time.time(); status.update(label=f"✅ GPT complete ({gpt_end - gpt_start:.2f}s)", state="complete", expanded=False)
        logging.info("GPT correction applied per segment.")
    else:
        processed = grouped_segments; status_editor = "None"
    st.session_state.processed_segments = {i: seg for i, seg in enumerate(processed)}
    st.session_state.edited_by = status_editor; st.session_state.processing_done = True
    st.session_state.highlight_keys = highlight_keys
    for key in highlight_keys: st.session_state.setdefault(key, False)

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("🎙️ Interview Transcript Editor")

# [Initialize Session State Keys - remains the same]
for key, default in [('processing_done', False), ('processed_segments', {}), ('segment_count', 0),
                     ('edited_by', "None"), ('full_audio_bytes', None), ('audio_format', None),
                     ('highlight_keys', [])]:
    st.session_state.setdefault(key, default)

# [UI Setup - remains the same]
col1, col2 = st.columns([2, 1])
with col1: up_file=st.file_uploader("1. Transcript (.txt)", type=["txt"], help="VTT-like")
with col1: au_file=st.file_uploader("2. Audio (.mp3/.m4a/.wav)", type=["mp3","m4a","wav"])
with col2: st.markdown("**Options:**"); tc_input=st.number_input(f"Words/Segment", 20, 300, DEFAULT_TARGET_WORD_COUNT, 10, help="~Segment Length", key="target_words")
with col2: fix_gpt=st.checkbox(f"Fix Typos (GPT)", help="Uses OpenAI per segment.", key="fix_typos")
with col2: process_btn = st.button("Process Files", type="primary", disabled=(not up_file or not au_file))

# [How to Use - remains the same]
st.markdown("""**How to Use:**
1. Upload files. Adjust options. Click **"Process Files"**.
2. **Edit:** Correct text below.
3. **Highlight:** Check **⭐** for important segments.
4. **Download:** Get the full text.
""")
st.divider()

full_audio = None

# [Processing Logic - remains the same]
if process_btn:
    st.session_state.processing_done = False; st.session_state.processed_segments = {}
    st.session_state.segment_count = 0; st.session_state.highlight_keys = []
    for k in list(st.session_state.keys()):
        if k.startswith("highlight_") or k.startswith("segment_edit_"): del st.session_state[k]

    status_proc = st.status("Starting processing...", expanded=True); t_start = time.time()
    try:
        status_proc.write("Parsing transcript..."); txt=up_file.read().decode("utf-8"); blocks=parse_timestamped_transcript(txt)
        if not blocks: raise ValueError("No valid blocks parsed.")

        status_proc.write("Loading audio..."); st.session_state.full_audio_bytes=au_file.getvalue(); data=BytesIO(st.session_state.full_audio_bytes)
        st.session_state.audio_format=get_audio_format(au_file)
        try: full_audio=AudioSegment.from_file(data, format=st.session_state.audio_format) if st.session_state.audio_format else AudioSegment.from_file(data)
        except Exception as e: raise ValueError(f"Pydub load failed: {e}")
        if not full_audio: raise ValueError("Audio load resulted in None.")
        status_proc.write(f"🎧 Audio loaded: {au_file.name}")

        status_proc.write("Grouping text..."); grouped=group_blocks_by_word_count(blocks, target_count=st.session_state.target_words)
        if not grouped: raise ValueError("Grouping failed.")

        initialize_session_state(grouped, st.session_state.fix_typos)

        t_total=time.time() - t_start; status_proc.update(label=f"✅ Processing complete! ({t_total:.2f}s)", state="complete", expanded=False)
    except Exception as e:
        st.session_state.processing_done=False; st.session_state.processed_segments={}
        status_proc.update(label=f"Processing Failed: {e}", state="error", expanded=True); logging.error(f"Processing failed: {e}", exc_info=True); st.stop()


if st.session_state.processing_done:
    st.subheader(f"Transcript Segments (~{st.session_state.target_words} words each)")
    st.caption(f"Editor: {st.session_state.edited_by}.")

    if full_audio is None and st.session_state.full_audio_bytes: # Reload if needed
        try: data=BytesIO(st.session_state.full_audio_bytes); full_audio = AudioSegment.from_file(data, format=st.session_state.audio_format) if st.session_state.audio_format else AudioSegment.from_file(data)
        except Exception as e: st.error(f"Audio reload error: {e}"); full_audio = None

    for i in range(st.session_state.segment_count):
        seg_data=st.session_state.processed_segments.get(i);
        if not seg_data: st.warning(f"Seg {i+1} data missing."); continue
        edit_key=f"segment_edit_{i}"; hl_key=f"highlight_{i}"; is_hl = st.session_state.get(hl_key, False)

        # --- Dynamic Height Calculation (Revised) ---
        current_text = seg_data['text']
        # Estimate number of lines based on characters and estimated chars per line (no +1)
        num_lines_estimate = math.ceil(len(current_text) / CHARS_PER_LINE_ESTIMATE)
        # Add count of actual newlines
        num_lines_estimate += current_text.count('\n')
        # Prevent zero lines if text is empty but base height is desired
        num_lines_estimate = max(1, num_lines_estimate) # Ensure at least 1 line counted if text exists or base needed

        # Calculate height based on lines and base height
        dynamic_height = BASE_TEXT_AREA_HEIGHT + num_lines_estimate * PIXELS_PER_LINE
        # Clamp the height within min/max bounds
        final_height = int(max(MIN_TEXT_AREA_HEIGHT, min(dynamic_height, MAX_TEXT_AREA_HEIGHT)))
        # --- End Calculation ---

        with st.container(border=True):
             col1, col2 = st.columns([3, 1])
             with col1:
                  st.markdown(f"**Segment {i+1}**{' ⭐' if is_hl else ''}")
                  st.text_area(
                      "Edit",
                      label_visibility="collapsed",
                      value=current_text, # Use the original/GPT text for initial render
                      key=edit_key,
                      height=final_height # Use calculated height
                  )
             with col2:
                  st.markdown(" ") # Spacer
                  start, end = seg_data['start'], seg_data['end']
                  try:
                      # [Audio Display Logic - remains the same]
                      if full_audio:
                            audio_clip = get_audio_segment(full_audio, start, end)
                            if audio_clip and len(audio_clip) > 0:
                                audio_bytes_io = BytesIO()
                                export_format = "mp3"
                                try:
                                    audio_clip.export(audio_bytes_io, format=export_format)
                                except Exception as export_mp3_err:
                                    logging.warning(f"MP3 export failed for seg {i+1} ({export_mp3_err}), trying WAV...")
                                    export_format = "wav"
                                    audio_bytes_io = BytesIO() # Reset buffer
                                    try:
                                        audio_clip.export(audio_bytes_io, format=export_format)
                                    except Exception as export_wav_err:
                                        logging.error(f"WAV export failed for seg {i+1}: {export_wav_err}")
                                        audio_bytes_io = None # Signal failure

                                if audio_bytes_io and audio_bytes_io.getbuffer().nbytes > 0:
                                    audio_bytes_io.seek(0)
                                    st.audio(audio_bytes_io, format=f"audio/{export_format}")
                                elif audio_clip:
                                    st.caption(f"({export_format.upper()} Export Fail)")
                            elif audio_clip is not None:
                                st.caption("(Empty Audio)")
                      else:
                            st.caption("Audio Load Err")
                      st.caption(f"{start} → {end}")
                  except Exception as e: st.caption(f"Audio Error: {e}")
                  st.checkbox("Highlight ⭐", key=hl_key, help="Mark segment")
    st.divider()

    # [Download and Comparison Section - remains the same]
    st.subheader("Full Transcript Text")
    final_texts = []
    for i in range(st.session_state.segment_count):
        txt = st.session_state.get(f"segment_edit_{i}", ""); # Read current widget value
        prefix = HIGHLIGHT_MARKER if st.session_state.get(f"highlight_{i}", False) else ""
        final_texts.append(prefix + txt)
    final_dl_txt = clean_final_text(" ".join(final_texts))

    st.text_area("Full Text Preview", final_dl_txt, height=200, key="final_preview")
    st.download_button(
        label=f"Download Text ({st.session_state.edited_by}).txt", data=final_dl_txt,
        file_name="processed_transcript.txt", mime="text/plain",
        help=f"Highlights marked with '{HIGHLIGHT_MARKER}'.")

    with st.expander("Show Original vs Edited"):
        colA, colB = st.columns(2)
        orig_txt = " ".join([st.session_state.processed_segments.get(i,{}).get('original_text','') for i in range(st.session_state.segment_count)])
        colA.text_area("Original (Joined)", clean_final_text(orig_txt), height=300)
        curr_edit = " ".join([st.session_state.get(f"segment_edit_{i}", "") for i in range(st.session_state.segment_count)])
        colB.text_area("Current Edited (Joined)", clean_final_text(curr_edit), height=300)

elif not process_btn:
    # [Initial Info messages - remains the same]
    if not up_file or not au_file: st.info("⬆️ Upload files.")
    else: st.info("➡️ Adjust options, then click 'Process Files'.")

st.markdown("---")
st.markdown("Powered by Streamlit, Pydub, OpenAI")
