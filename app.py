import os
import time
import logging
import psutil
from datetime import datetime
from functools import wraps

# ---------------------------------------------------------
# 1. LOGGING CONFIGURATION
# ---------------------------------------------------------
def setup_logging():
    """
    Configures the logging for the application.
    Creates a 'logs' directory if it doesn't exist.
    Sets up a file handler (DEBUG) with timestamped naming and a console handler (INFO).
    Returns the configured logger.
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"app_{timestamp}.log")

    logger = logging.getLogger("speech_app")
    
    # Set the root logger level to DEBUG so it passes all messages to handlers
    logger.setLevel(logging.DEBUG)

    # Prevent adding handlers multiple times if setup_logging is called more than once
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Configure File Handler (DEBUG level)
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Configure Console Handler (INFO level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = setup_logging()

# ---------------------------------------------------------
# 2. PERFORMANCE TRACKING (Decorators & Utilities)
# ---------------------------------------------------------
def track_execution_time(func):
    """
    Decorator to track and log execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        # Execution time logged at DEBUG to avoid cluttering INFO
        logger.debug(f"Execution time of {func.__name__}: {end_time - start_time:.4f}s")
        return result
    return wrapper

def log_memory_usage(context=""):
    """
    Utility to log current memory usage.
    Tracks resident set size (RSS) via psutil to debug memory bloat.
    """
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Memory usage {context}: {memory_mb:.2f} MB")
    except Exception as e:
        logger.warning(f"Failed to retrieve memory usage: {e}")

# ---------------------------------------------------------
# 3. INSTRUMENTED FUNCTIONS
# ---------------------------------------------------------

@track_execution_time
def segment_to_float32(segment_data, input_dtype="int16"):
    """
    Convert audio segment memory to float32.
    """
    # Assuming segment_data is a simulated array list or similar with a length
    shape = getattr(segment_data, "shape", (len(segment_data),))
    
    logger.info("Entering segment_to_float32()")
    logger.info(f"Input dtype={input_dtype} shape={shape}")
    
    # Mock processing
    start_time = time.perf_counter()
    output_dtype = "float32"
    # Delay to simulate computation
    time.sleep(0.002)
    execution_time = time.perf_counter() - start_time
    
    logger.info(f"Output dtype={output_dtype} shape={shape}")
    logger.info(f"Execution time={execution_time:.3f}s")
    
    return {"data": segment_data, "dtype": output_dtype, "shape": shape}

@track_execution_time
def merge_same_speaker(segments):
    """
    Merge contiguous audio segments matching the same speaker_id.
    Time complexity: O(n) where n is number of segments.
    """
    before_count = len(segments)
    
    if before_count == 0:
        logger.warning("Empty segment list provided to merge_same_speaker()")
        return []

    merged = []
    current_segment = segments[0].copy()
    
    for i in range(1, before_count):
        seg = segments[i]
        if seg['speaker_id'] == current_segment['speaker_id']:
            logger.debug(f"Merging segment {seg['id']} into current segment for {seg['speaker_id']}")
            current_segment['end'] = max(current_segment['end'], seg['end'])
        else:
            merged.append(current_segment)
            current_segment = seg.copy()
            
    merged.append(current_segment)
    after_count = len(merged)
    
    logger.info(f"Merging segments: before={before_count}, after={after_count}")
    return merged

@track_execution_time
def split_overlaps(segments):
    """
    Detect and split overlapping text segments appropriately.
    Time complexity: O(n) where n is number of segments.
    """
    before_count = len(segments)
    overlaps_detected = 0
    segments_split = 0
    
    if before_count == 0:
        logger.warning("Empty segment list provided to split_overlaps()")
        return []

    resolved = []
    
    # Process pairs and handle overlaps conservatively
    for i in range(before_count - 1):
        seg1 = segments[i].copy()
        seg2 = segments[i+1]
        
        if seg1['end'] > seg2['start']:
            overlaps_detected += 1
            logger.warning(f"Overlapping segments detected between speaker {seg1['speaker_id']} and {seg2['speaker_id']} at {seg2['start']}s")
            
            # Mock split operation by fixing the overlap
            seg1['end'] = seg2['start']
            segments_split += 1
            logger.debug(f"Split completed for overlap, adjusted '{seg1['id']}' end time.")
            
        resolved.append(seg1)
        
    resolved.append(segments[-1].copy())
    after_count = len(resolved)
    
    logger.info(f"Splitting overlaps: overlaps_detected={overlaps_detected}, segments_split={segments_split}, before={before_count}, after={after_count}")
    return resolved

@track_execution_time
def build_clean_segments(raw_segments):
    """
    Pipeline orchestrating the cleanup of raw Diarization segments.
    """
    raw_count = len(raw_segments)
    logger.info(f"Building clean segments: input={raw_count}")
    
    # Step 1: Filter empty or extremely short segments
    filtered = []
    for s in raw_segments:
        duration = s['end'] - s['start']
        if duration <= 0:
            logger.warning(f"Empty segment detected: {s['id']} at {s['start']}s")
            continue
        elif duration < 0.1:
            logger.warning(f"Extremely short segment detected: {s['id']} (duration={duration:.3f}s)")
            continue
        filtered.append(s)
        
    after_filtering = len(filtered)
    logger.info(f"Building clean segments: input={raw_count}, after_filtering={after_filtering}")
    logger.info("Processing steps applied: Filtered segments < 0.1s")
    
    # Step 2: Merge same speaker contiguous intervals
    merged = merge_same_speaker(filtered)
    
    # Step 3: Ensure overlapping times are removed
    resolved = split_overlaps(merged)
    
    return resolved

@track_execution_time
def transcribe_segment(segment):
    """
    Execute whisper transcription process per segment.
    """
    processing_start = time.perf_counter()
    success = True
    transcription = ""
    
    try:
        # Simulate authentication or resource loading exception logging 
        if segment.get('invalid_token'):
            raise PermissionError("Invalid authentication token for transcription API")
        
        # Simulate Whisper processing time
        time.sleep(0.05)
        
        duration = segment['end'] - segment['start']
        if duration == 0:
            logger.warning("Whisper returning empty transcription!")
        else:
            transcription = f"Mock text for segment {segment['id']}"
            
    except Exception:
        success = False
        logger.exception(f"Failed transcription for segment {segment['id']}")
        raise
        
    processing_time = time.perf_counter() - processing_start
    
    # Log: Segment start/end, speaker id, transcription success/failure, processing time
    logger.info(f"Segment [{segment['start']:.2f}s -- {segment['end']:.2f}s] {segment['speaker_id']}: transcribed={success}, duration={processing_time:.2f}s")
    
    return transcription

# ---------------------------------------------------------
# 4. MAIN EXECUTION FLOW
# ---------------------------------------------------------
def main():
    audio_file = "audio.mp3"
    logger.info(f"Starting audio processing: {audio_file}")
    
    try:
        # Simulate mock environment exception for demo
        if not os.path.exists("logs"):
             raise FileNotFoundError(f"Missing logs dir")
             
        # Mock metadata loading
        duration_s = 120.5
        sample_rate = 16000
        channels = 1
        logger.info(f"Audio loaded: duration={duration_s}s, channels={channels}, sample_rate={sample_rate}")
        
        log_memory_usage("before diarization")
        
        logger.info("Speaker diarization started")
        diarization_start = time.perf_counter()
        
        # Mock diarization generation
        time.sleep(0.3)
        mock_raw_segments = [
            {"id": "seg001", "start": 0.0, "end": 5.23, "speaker_id": "SPEAKER_00"},
            {"id": "seg002", "start": 5.23, "end": 8.45, "speaker_id": "SPEAKER_00"}, # Will merge with seg001
            {"id": "seg003", "start": 8.0, "end": 10.0, "speaker_id": "SPEAKER_01"},  # Overlap with seg002!
            {"id": "seg004", "start": 10.0, "end": 10.05, "speaker_id": "SPEAKER_01"}, # Extremely short
            {"id": "seg005", "start": 10.5, "end": 10.5, "speaker_id": "SPEAKER_02"}  # Empty segment
        ]
        
        diarization_time = time.perf_counter() - diarization_start
        logger.info(f"Diarization completed: {len(mock_raw_segments)} segments found, execution_time={diarization_time:.2f}s")
        
        log_memory_usage("after diarization")
        
        logger.info("Segment cleaning pipeline started")
        clean_segments = build_clean_segments(mock_raw_segments)
        
        logger.info(f"Starting transcription of {len(clean_segments)} segments")
        transcription_start = time.perf_counter()
        
        transcripts = []
        for seg in clean_segments:
            # Instrument audio conversion array loading
            segment_to_float32([0]*32000)
            
            try:
                text = transcribe_segment(seg)
                transcripts.append(text)
            except Exception:
                logger.error(f"Abandoning processing for segment {seg['id']} due to transcription error")
                
        total_time = time.perf_counter() - transcription_start
        logger.info(f"Transcription completed: {len(clean_segments)} segments processed, total_time={total_time:.2f}s")
        
        output_file = "transcript.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(transcripts))
        logger.info(f"Output written to: {output_file}")
        
    except Exception as e:
        logger.exception(f"ERROR - Critical failure in main processing pipeline: {e}")

if __name__ == "__main__":
    main()
