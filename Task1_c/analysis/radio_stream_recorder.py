import os
import time
import json
import random
import requests
import datetime
import argparse
import pandas as pd
import subprocess
import logging
import io
from pydub import AudioSegment

class RadioSonicArchiveGenerator:
    """
    RadioSonicArchive Dataset Generator
    
    Records audio streams from online radio stations to create a dataset
    for music genre classification, mood detection, or audio fingerprinting.
    """
    
    def __init__(self, output_dir="radio_sonic_archive", metadata_file="metadata.csv"):
        """Initialize the dataset generator"""
        self.output_dir = output_dir
        self.metadata_file = metadata_file
        self.metadata = []
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{output_dir}/recording.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Default radio stations (can be extended)
        self.radio_stations = {
            "SomaFM_Groove": "https://ice2.somafm.com/groovesalad-128-mp3",
            "SomaFM_Drone": "https://ice2.somafm.com/dronezone-128-mp3",
            "SomaFM_Jazz": "https://ice2.somafm.com/sonicuniverse-128-mp3",
            "SomaFM_Indie": "https://ice2.somafm.com/indiepop-128-mp3",
            "SomaFM_Folk": "https://ice2.somafm.com/folkfwd-128-mp3",
            "SomaFM_Ambient": "https://ice2.somafm.com/spacestation-128-mp3",
            "SomaFM_Lounge": "https://ice2.somafm.com/lush-128-mp3"
            #"Radio net": "https://www.radio.net/"
            
            #"Classical music":"https://www.classicalradio.com/"
           # "National Public Radio": "https://www.npr.org/streams/"
        }
    
    def add_radio_stations(self, stations_dict):
        """Add additional radio stations to the list"""
        self.radio_stations.update(stations_dict)
        self.logger.info(f"Added {len(stations_dict)} new radio stations")
    
    def test_stream(self, station_name, url, timeout=5):
        """Test if a stream is accessible"""
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            if response.status_code == 200:
                self.logger.info(f"Stream test successful for {station_name}")
                return True
            else:
                self.logger.warning(f"Stream test failed for {station_name}: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.logger.warning(f"Stream test failed for {station_name}: {str(e)}")
            return False
    
    def record_stream_ffmpeg(self, station_name, url, duration, output_file):
        """Record a stream using ffmpeg for a specified duration in seconds"""
        try:
            # First check if ffmpeg is installed
            try:
                subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            except (subprocess.SubprocessError, FileNotFoundError):
                self.logger.warning("ffmpeg not found or not working. Falling back to Python-based recording method.")
                return self.record_stream_python(station_name, url, duration, output_file)
                
            # Construct the ffmpeg command
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', url,
                '-t', str(duration),
                '-c', 'copy',  # Copy without re-encoding to preserve quality
                output_file
            ]
            
            self.logger.info(f"Starting recording: {station_name} for {duration} seconds")
            self.logger.debug(f"Command: {' '.join(cmd)}")
            
            # Execute the command
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for the process to complete
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"Recording failed for {station_name}: {stderr.decode()}")
                self.logger.info(f"Trying Python-based recording method instead...")
                return self.record_stream_python(station_name, url, duration, output_file)
            
            self.logger.info(f"Recording complete: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording {station_name} with ffmpeg: {str(e)}")
            self.logger.info(f"Trying Python-based recording method instead...")
            return self.record_stream_python(station_name, url, duration, output_file)
    
    def record_stream_python(self, station_name, url, duration, output_file):
        """Record a stream using Python requests library (no ffmpeg required)"""
        try:
            self.logger.info(f"Starting Python-based recording: {station_name} for {duration} seconds")
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            
            # Start recording
            with requests.get(url, stream=True, timeout=10) as response:
                response.raise_for_status()  # Raise exception for HTTP errors
                
                # Open the output file
                with open(output_file, 'wb') as f:
                    start_time = time.time()
                    bytes_written = 0
                    
                    # Record for the specified duration
                    for chunk in response.iter_content(chunk_size=8192):
                        if time.time() - start_time > duration:
                            break
                        
                        if chunk:
                            f.write(chunk)
                            bytes_written += len(chunk)
            
            # Check if we got enough data
            if bytes_written < 1024:  # Less than 1KB
                self.logger.warning(f"Recorded file is too small ({bytes_written} bytes)")
                return False
                
            self.logger.info(f"Python recording complete: {output_file} ({bytes_written/1024:.2f} KB)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in Python-based recording for {station_name}: {str(e)}")
            return False
    
    def get_audio_duration(self, file_path):
        """Get the actual duration of the recorded audio file"""
        try:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        except Exception as e:
            self.logger.error(f"Error getting duration for {file_path}: {str(e)}")
            
            # If we can't get the duration, try to estimate it from file size
            # This is a rough estimate assuming 128kbps MP3
            try:
                size_bytes = os.path.getsize(file_path)
                estimated_duration = (size_bytes * 8) / (128 * 1024)  # size in bits / bitrate
                self.logger.info(f"Estimated duration for {file_path}: {estimated_duration:.2f} seconds")
                return estimated_duration
            except Exception as e2:
                self.logger.error(f"Error estimating duration from file size: {str(e2)}")
                return None
    
    def record_multiple_segments(self, num_files=30, min_duration=30, max_duration=90, 
                                file_format="mp3", random_stations=True):
        """
        Record multiple audio segments from different radio stations
        
        Args:
            num_files: Number of audio files to record
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            file_format: Audio file format (mp3 or wav)
            random_stations: If True, randomly select stations for each recording
        """
        # Test all streams first
        valid_stations = {}
        for name, url in self.radio_stations.items():
            if self.test_stream(name, url):
                valid_stations[name] = url
        
        if not valid_stations:
            self.logger.error("No valid radio stations found. Aborting.")
            return False
        
        self.logger.info(f"Found {len(valid_stations)} valid radio stations")
        
        # Create the metadata dataframe structure
        self.metadata = []
        
        # Record the specified number of files
        stations_to_use = list(valid_stations.items())
        
        for i in range(num_files):
            # Select a station (either randomly or sequentially)
            if random_stations:
                station_name, url = random.choice(stations_to_use)
            else:
                station_name, url = stations_to_use[i % len(stations_to_use)]
            
            # Generate a random duration if min and max are different
            if min_duration == max_duration:
                duration = min_duration
            else:
                duration = random.randint(min_duration, max_duration)
            
            # Create a filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{i+1:03d}_{station_name}_{timestamp}.{file_format}"
            output_path = os.path.join(self.output_dir, filename)
            
            # Record the audio (try ffmpeg first, fallback to Python method)
            success = self.record_stream_ffmpeg(station_name, url, duration, output_path)
            
            if success:
                # Get actual duration
                actual_duration = self.get_audio_duration(output_path)
                
                # Store metadata
                record = {
                    "file_id": i+1,
                    "filename": filename,
                    "station_name": station_name,
                    "url": url,
                    "timestamp": timestamp,
                    "requested_duration": duration,
                    "actual_duration": actual_duration if actual_duration else "unknown",
                    "format": file_format
                }
                self.metadata.append(record)
                
                self.logger.info(f"Recorded file {i+1}/{num_files}: {filename}")
                
                # Save metadata after each successful recording
                self._save_metadata()
            else:
                self.logger.warning(f"Failed to record file {i+1}/{num_files} from {station_name}")
            
            # Add a small delay between recordings
            time.sleep(2)
        
        return True
    
    def _save_metadata(self):
        """Save metadata to CSV file"""
        if not self.metadata:
            self.logger.warning("No metadata to save")
            return
            
        df = pd.DataFrame(self.metadata)
        df.to_csv(os.path.join(self.output_dir, self.metadata_file), index=False)
        
        # Also save as JSON for easier parsing
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
        self.logger.info(f"Metadata saved to {self.metadata_file}")
    
    def get_dataset_summary(self):
        """Generate a summary of the dataset"""
        if not self.metadata:
            self.logger.warning("No metadata available for summary")
            return {}
        
        df = pd.DataFrame(self.metadata)
        
        summary = {
            "total_files": len(df),
            "total_stations": df['station_name'].nunique(),
            "stations": df['station_name'].value_counts().to_dict(),
            "total_duration": df['actual_duration'].sum() if 'actual_duration' in df and df['actual_duration'].dtype != object else "unknown",
            "formats": df['format'].value_counts().to_dict() if 'format' in df else {"unknown": len(df)},
            "earliest_recording": df['timestamp'].min() if 'timestamp' in df else "unknown",
            "latest_recording": df['timestamp'].max() if 'timestamp' in df else "unknown"
        }
        
        return summary

def main():
    """Main function to run the dataset generator"""
    parser = argparse.ArgumentParser(description="RadioSonicArchive: Online Radio Stream Dataset Generator")
    parser.add_argument('--output', type=str, default="radio_sonic_archive", 
                       help="Output directory for the dataset")
    parser.add_argument('--num-files', type=int, default=30,
                       help="Number of audio files to record")
    parser.add_argument('--min-duration', type=int, default=30,
                       help="Minimum duration in seconds")
    parser.add_argument('--max-duration', type=int, default=90,
                       help="Maximum duration in seconds")
    parser.add_argument('--format', type=str, choices=['mp3', 'wav'], default='mp3',
                       help="Audio file format")
    parser.add_argument('--random', action='store_true',
                       help="Randomly select stations for each recording")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("RadioSonicArchive Dataset Generator")
    print("=" * 50)
    print(f"Output Directory: {args.output}")
    print(f"Number of Files: {args.num_files}")
    print(f"Duration Range: {args.min_duration}-{args.max_duration} seconds")
    print(f"File Format: {args.format}")
    print(f"Random Station Selection: {args.random}")
    print("=" * 50)
    
    # Create the dataset generator
    generator = RadioSonicArchiveGenerator(args.output)
    
    # Record the audio files
    success = generator.record_multiple_segments(
        num_files=args.num_files,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        file_format=args.format,
        random_stations=args.random
    )
    
    if success:
        # Get summary
        summary = generator.get_dataset_summary()
        
        print("\nDataset Generation Complete!")
        print(f"Total Files: {summary.get('total_files', 0)}")
        print(f"Total Stations: {summary.get('total_stations', 0)}")
        print(f"Total Duration: {summary.get('total_duration', 'unknown')} seconds")
        print("\nStation Distribution:")
        for station, count in summary.get('stations', {}).items():
            print(f"  - {station}: {count} files")
            
        print(f"\nMetadata saved to {os.path.join(args.output, 'metadata.csv')}")
    else:
        print("\nDataset generation encountered errors. Check the log file for details.")

if __name__ == "__main__":
    main()