#!/usr/bin/env python3
"""
WeatherCache - A comprehensive weather data collection and caching tool
Author: Jainish Patel
Date: August 5, 2025
Project: WeatherCache CLI Tool with daily data collection, caching, and automatic visualization

Features:
- Smart caching with city/year/day structure
- Automatic visualization generation
- CSV batch processing with date/time separation
- Environment configuration management
- Concurrent API requests with rate limiting
- Comprehensive error handling and logging
"""

import os
import sys
import pickle
import time
import argparse
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from colorama import Fore, Style, init
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore') 
# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Fix Windows encoding issue
if os.name == 'nt':  # Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())


class WeatherCacheException(Exception):
    """Custom exception for WeatherCache operations"""
    pass


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


class EnvironmentManager:
    """Advanced environment variable management with interactive setup"""
    
    @staticmethod
    def load_environment() -> bool:
        """Load environment variables from .env file if it exists"""
        env_file = Path('.env')
        if env_file.exists():
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                return True
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Warning: Could not load .env file: {e}")
                return False
        return False
    
    @staticmethod
    def create_env_file(api_key: str, **kwargs) -> bool:
        """Create .env file with user configuration"""
        env_content = f"""# WeatherCache Environment Configuration
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# =============================================================================
# REQUIRED: OpenWeatherMap API Configuration
# =============================================================================
OPENWEATHER_API_KEY={api_key}

# =============================================================================
# OPTIONAL: Cache Configuration
# =============================================================================
WEATHER_CACHE_DIR={kwargs.get('cache_dir', 'cache')}
WEATHER_DATA_DIR={kwargs.get('data_dir', 'weather_data')}
WEATHER_CACHE_TTL={kwargs.get('cache_ttl', 86400)}

# =============================================================================
# OPTIONAL: Performance Configuration  
# =============================================================================
MAX_WORKERS={kwargs.get('max_workers', 5)}
LOG_LEVEL={kwargs.get('log_level', 'INFO')}

# =============================================================================
# OPTIONAL: File Paths
# =============================================================================
DEFAULT_INPUT_CSV={kwargs.get('input_csv', 'cities.csv')}
"""
        
        try:
            env_path = Path('.env')
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(env_content)
            print(f"{Fore.GREEN}✅ Environment file created at: {env_path.absolute()}")
            return True
        except IOError as e:
            print(f"{Fore.RED}❌ Failed to create .env file: {e}")
            return False
    
    @staticmethod
    def interactive_setup() -> bool:
        """Interactive environment setup"""
        print(f"\n{Fore.CYAN}🌤️ WeatherCache Environment Setup")
        print(f"{Fore.CYAN}{'='*50}")
        
        # API Key
        print(f"\n{Fore.YELLOW}1. OpenWeatherMap API Key")
        print(f"{Fore.WHITE}   Get your free API key from: https://openweathermap.org/api")
        api_key = input(f"{Fore.GREEN}Enter your API key: ").strip()
        
        if not api_key:
            print(f"{Fore.RED}❌ API key is required!")
            return False
        
        # Validate API key
        if not EnvironmentManager.validate_api_key(api_key):
            print(f"{Fore.RED}❌ API key validation failed!")
            return False
        
        # Directory settings
        print(f"\n{Fore.YELLOW}2. Directory Configuration[Press Enter for defaults]")
        cache_dir = input(f"{Fore.GREEN}Cache directory [cache]: ").strip() or "cache"
        data_dir = input(f"{Fore.GREEN}Weather data directory [weather_data]: ").strip() or "weather_data"
        
        cache_ttl_input = input(f"{Fore.GREEN}Cache TTL in seconds [86400 (1 day)]: ").strip()
        try:
            cache_ttl = int(cache_ttl_input) if cache_ttl_input else 86400
        except ValueError:
            cache_ttl = 86400
        
        # Performance settings
        print(f"\n{Fore.YELLOW}3. Performance Configuration")
        workers_input = input(f"{Fore.GREEN}Max concurrent workers [5]: ").strip()
        try:
            max_workers = int(workers_input) if workers_input else 5
            max_workers = max(1, min(max_workers, 10))  # Limit to reasonable range
        except ValueError:
            max_workers = 5
        
        log_level = input(f"{Fore.GREEN}Log level [INFO] (DEBUG/INFO/WARNING/ERROR): ").strip().upper() or "INFO"
        if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            log_level = "INFO"
        
        # File paths
        print(f"\n{Fore.YELLOW}4. Default File Paths")
        input_csv = input(f"{Fore.GREEN}Default input CSV [cities.csv]: ").strip() or "cities.csv"
        
        # Create .env file
        config = {
            'cache_dir': cache_dir,
            'data_dir': data_dir,
            'cache_ttl': cache_ttl,
            'max_workers': max_workers,
            'log_level': log_level,
            'input_csv': input_csv
        }
        
        if EnvironmentManager.create_env_file(api_key, **config):
            print(f"\n{Fore.GREEN}✅ Environment file created successfully!")
            print(f"{Fore.GREEN}📁 Working directory: {Path.cwd()}")
            return True
        else:
            print(f"\n{Fore.RED}❌ Failed to create .env file")
            return False
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get configuration from environment variables with smart defaults"""
        EnvironmentManager.load_environment()
        
        return {
            'api_key': os.getenv('OPENWEATHER_API_KEY'),
            'cache_dir': os.getenv('WEATHER_CACHE_DIR', 'cache'),
            'data_dir': os.getenv('WEATHER_DATA_DIR', 'weather_data'),
            'cache_ttl': int(os.getenv('WEATHER_CACHE_TTL', '86400')),
            'max_workers': int(os.getenv('MAX_WORKERS', '5')),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'default_input_csv': os.getenv('DEFAULT_INPUT_CSV', 'cities.csv')
        }
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key by making a test request"""
        if not api_key:
            return False
        
        try:
            test_url = "https://api.openweathermap.org/data/2.5/weather"
            params = {'q': 'London', 'appid': api_key, 'units': 'metric'}
            
            print(f"{Fore.YELLOW}🔍 Validating API key...")
            response = requests.get(test_url, params=params, timeout=10)
            
            if response.status_code == 200:
                print(f"{Fore.GREEN}✅ API key is valid!")
                return True
            elif response.status_code == 401:
                print(f"{Fore.RED}❌ Invalid API key!")
                return False
            else:
                print(f"{Fore.YELLOW}⚠️ API validation inconclusive (status: {response.status_code})")
                return True
        except requests.exceptions.RequestException:
            print(f"{Fore.YELLOW}⚠️ Could not validate API key due to network error")
            return True


class WeatherCache:
    """Advanced weather data caching system with daily data collection"""
    
    def __init__(self, api_key: str, cache_dir: str = "cache", data_dir: str = "weather_data",
                 cache_ttl: int = 86400, max_workers: int = 5):
        """
        Initialize WeatherCache with city/year/day structure
        
        Args:
            api_key: OpenWeatherMap API key
            cache_dir: Directory for cache files (cache/city/year/day)
            data_dir: Directory for processed data (weather_data/reports/visualization/data)
            cache_ttl: Cache time-to-live in seconds (default: 1 day)
            max_workers: Maximum number of concurrent threads
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.data_dir = Path(data_dir)
        self.cache_ttl = cache_ttl
        self.max_workers = max_workers
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        
        # Create directory structure
        self._create_directory_structure()
        
        # Setup logging
        self.setup_logging()
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls = 0
        
        self.logger.info(f"WeatherCache initialized - Daily data collection mode")
    
    def _create_directory_structure(self) -> None:
        """Create required directory structure"""
        try:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            self.data_dir.mkdir(exist_ok=True, parents=True)
            
            # Weather data subdirectories
            self.reports_dir = self.data_dir / "reports"
            self.visualization_dir = self.data_dir / "visualization" 
            self.processed_data_dir = self.data_dir / "data"
            
            self.reports_dir.mkdir(exist_ok=True, parents=True)
            self.visualization_dir.mkdir(exist_ok=True, parents=True)
            self.processed_data_dir.mkdir(exist_ok=True, parents=True)
            
            print(f"{Fore.CYAN}📁 Directory structure created:")
            print(f"  {Fore.WHITE}├── {self.cache_dir.absolute()}/")
            print(f"  {Fore.WHITE}│   └── [city]/[year]/[day]/")
            print(f"  {Fore.WHITE}└── {self.data_dir.absolute()}/")
            print(f"  {Fore.WHITE}    ├── reports/")
            print(f"  {Fore.WHITE}    ├── visualization/")
            print(f"  {Fore.WHITE}    └── data/")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to create directory structure: {e}")
            raise WeatherCacheException(f"Cannot create directories: {e}")
    
    def setup_logging(self) -> None:
        """Setup colored logging"""
        self.logger = logging.getLogger('WeatherCache')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = ColoredFormatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def set_log_level(self, level: str) -> None:
        """Set logging level"""
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        self.logger.setLevel(levels.get(level.upper(), logging.INFO))
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize city name for use as filename/folder name"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        
        name = '_'.join(name.split()).lower()
        
        if len(name) > 50:
            name = name[:50]
        
        return name
    
    def _get_cache_path(self, city: str, target_date: date) -> Path:
        """Get cache path following city/year/day structure"""
        sanitized_city = self._sanitize_filename(city)
        year = str(target_date.year)
        day = target_date.strftime("%Y-%m-%d")
        
        cache_path = self.cache_dir / sanitized_city / year / day
        cache_path.mkdir(exist_ok=True, parents=True)
        return cache_path
    
    def _get_city_csv_path(self, city: str) -> Path:
        """Get the main CSV file path for a city"""
        sanitized_city = self._sanitize_filename(city)
        city_csv = self.processed_data_dir / f"{sanitized_city}_weather_data.csv"
        return city_csv
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is still valid based on TTL"""
        if not cache_file.exists():
            return False
        
        file_age = time.time() - cache_file.stat().st_mtime
        return file_age < self.cache_ttl
    
    def _load_from_cache(self, cache_file: Path) -> Optional[Dict]:
        """Load weather data from pickle cache if valid"""
        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    data['cache_status'] = 'from_cache'
                    self.cache_hits += 1
                    self.logger.debug(f"💾 Cache HIT: {cache_file.name}")
                    return data
            except (pickle.PickleError, IOError) as e:
                self.logger.warning(f"🗑️ Cache file corrupted, removing: {e}")
                cache_file.unlink(missing_ok=True)
        
        self.cache_misses += 1
        self.logger.debug(f"❌ Cache MISS: {cache_file.name}")
        return None
    
    def _save_to_cache(self, cache_file: Path, data: Dict) -> None:
        """Save weather data to pickle cache"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.debug(f"💾 Cached: {cache_file}")
        except (pickle.PickleError, IOError) as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    def _append_to_city_csv(self, city: str, weather_data: Dict, request_date: date, 
                           request_time: str = None) -> None:
        """Append weather data to city's main CSV file"""
        city_csv = self._get_city_csv_path(city)
        
        # Create new row with separate date and time columns
        new_row = {
            'date': request_date.strftime('%Y-%m-%d'),
            'time': request_time or datetime.now().strftime('%H:%M:%S'),
            'city': weather_data.get('name', city),
            'country': weather_data.get('sys', {}).get('country'),
            'latitude': weather_data.get('coord', {}).get('lat'),
            'longitude': weather_data.get('coord', {}).get('lon'),
            'temperature': weather_data.get('main', {}).get('temp'),
            'feels_like': weather_data.get('main', {}).get('feels_like'),
            'temp_min': weather_data.get('main', {}).get('temp_min'),
            'temp_max': weather_data.get('main', {}).get('temp_max'),
            'humidity': weather_data.get('main', {}).get('humidity'),
            'pressure': weather_data.get('main', {}).get('pressure'),
            'sea_level': weather_data.get('main', {}).get('sea_level'),
            'ground_level': weather_data.get('main', {}).get('grnd_level'),
            'weather_main': weather_data.get('weather', [{}])[0].get('main'),
            'description': weather_data.get('weather', [{}])[0].get('description'),
            'weather_icon': weather_data.get('weather', [{}])[0].get('icon'),
            'wind_speed': weather_data.get('wind', {}).get('speed'),
            'wind_direction': weather_data.get('wind', {}).get('deg'),
            'wind_gust': weather_data.get('wind', {}).get('gust'),
            'visibility': weather_data.get('visibility', 0) / 1000 if weather_data.get('visibility') else None,
            'cloudiness': weather_data.get('clouds', {}).get('all'),
            'sunrise': datetime.fromtimestamp(weather_data.get('sys', {}).get('sunrise', 0)).strftime('%H:%M:%S') if weather_data.get('sys', {}).get('sunrise') else None,
            'sunset': datetime.fromtimestamp(weather_data.get('sys', {}).get('sunset', 0)).strftime('%H:%M:%S') if weather_data.get('sys', {}).get('sunset') else None,
            'timezone': weather_data.get('timezone'),
            'cache_status': weather_data.get('cache_status', 'fresh_api_call'),
            'fetch_timestamp': weather_data.get('fetch_timestamp', datetime.now().isoformat())
        }
        
        # Load existing data or create new DataFrame
        if city_csv.exists():
            try:
                existing_df = pd.read_csv(city_csv)
                # Check if this exact date/time already exists to avoid duplicates
                existing_mask = (existing_df['date'] == new_row['date']) & (existing_df['time'] == new_row['time'])
                if not existing_mask.any():
                    new_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    self.logger.debug(f"Skipping duplicate entry for {city} on {new_row['date']} at {new_row['time']}")
                    return
            except Exception as e:
                self.logger.warning(f"Error reading existing CSV, creating new: {e}")
                new_df = pd.DataFrame([new_row])
        else:
            new_df = pd.DataFrame([new_row])
        
        # Sort by date and time
        new_df['datetime_combined'] = pd.to_datetime(new_df['date'] + ' ' + new_df['time'])
        new_df = new_df.sort_values('datetime_combined').drop('datetime_combined', axis=1)
        
        # Save updated CSV
        try:
            new_df.to_csv(city_csv, index=False)
            self.logger.debug(f"📊 Updated city CSV: {city_csv}")
        except Exception as e:
            self.logger.error(f"Failed to save city CSV: {e}")
    
    def _fetch_weather_api(self, params: Dict) -> Dict:
        """Fetch weather data from OpenWeatherMap API"""
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            self.api_calls += 1
            
            data = response.json()
            data['fetch_timestamp'] = datetime.now().isoformat()
            data['cache_status'] = 'fresh_api_call'
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise WeatherCacheException(f"API request failed: {e}")
    
    def get_weather_for_date(self, city: str, target_date: date, target_time: str = None, 
                           use_cache: bool = True) -> Dict:
        """Get weather data for a specific city and date"""
        cache_path = self._get_cache_path(city, target_date)
        
        # Create cache filename based on date and time
        if target_time:
            cache_filename = f"weather_{target_date.strftime('%Y-%m-%d')}_{target_time.replace(':', '-')}.pkl"
        else:
            cache_filename = f"weather_{target_date.strftime('%Y-%m-%d')}_current.pkl"
        
        cache_file = cache_path / cache_filename
        
        # Try cache first if enabled
        if use_cache:
            cached_data = self._load_from_cache(cache_file)
            if cached_data:
                self.logger.info(f"🎯 Cache HIT: {city} on {target_date}")
                return cached_data
        
        # Check if requesting historical data
        today = date.today()
        if target_date < today:
            self.logger.warning(f"⚠️ Requested date {target_date} is historical. OpenWeatherMap free API only provides current weather.")
            self.logger.warning("💡 For historical data, consider OpenWeatherMap's One Call API 3.0 (requires subscription)")
        
        # Make API call for current weather
        self.logger.info(f"📡 Fetching weather for {city} (target date: {target_date})")
        try:
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            data = self._fetch_weather_api(params)
            
            # Add metadata
            data['requested_date'] = target_date.isoformat()
            data['requested_time'] = target_time
            if target_date < today:
                data['note'] = 'Current weather data returned for historical date request'
            
            # Save to cache
            if use_cache:
                self._save_to_cache(cache_file, data)
                self.logger.info(f"💾 Cached weather data for {city}")
            
            # Append to city's main CSV
            self._append_to_city_csv(city, data, target_date, target_time)
            
            return data
            
        except WeatherCacheException as e:
            self.logger.error(str(e))
            raise
    
    def process_csv(self, csv_file: str, use_cache: bool = True) -> pd.DataFrame:
        """Process CSV with separate date and time columns"""
        try:
            csv_path = Path(csv_file)
            if not csv_path.exists():
                raise WeatherCacheException(f"CSV file not found: {csv_path.absolute()}")
            
            print(f"{Fore.CYAN}📄 Reading CSV file: {csv_path.absolute()}")
            df = pd.read_csv(csv_path)
            
            # Check for required columns
            available_columns = list(df.columns)
            print(f"{Fore.CYAN}📋 Available columns: {available_columns}")
            
            required_cols = ['city', 'date']
            missing_cols = [col for col in required_cols if col not in available_columns]
            if missing_cols:
                raise WeatherCacheException(f"Missing required columns: {missing_cols}. Available: {available_columns}")
            
            print(f"{Fore.GREEN}✅ Processing {len(df)} rows")
            
            # Process each row
            processed_data = []
            
            for idx, row in df.iterrows():
                try:
                    city = str(row['city']).strip()
                    target_date = pd.to_datetime(row['date']).date()
                    target_time = str(row['time']).strip() if 'time' in row and not pd.isna(row['time']) else None
                    
                    if not city:
                        print(f"{Fore.YELLOW}⚠️ Row {idx+1}: Empty city name, skipping")
                        continue
                    
                    # Get weather data
                    weather_data = self.get_weather_for_date(city, target_date, target_time, use_cache)
                    
                    # Create processed row
                    processed_row = row.to_dict()
                    processed_row.update({
                        'api_city_name': weather_data['name'],
                        'country': weather_data['sys']['country'],
                        'temperature': weather_data['main']['temp'],
                        'feels_like': weather_data['main']['feels_like'],
                        'humidity': weather_data['main']['humidity'],
                        'pressure': weather_data['main']['pressure'],
                        'weather_main': weather_data['weather'][0]['main'],
                        'description': weather_data['weather'][0]['description'].title(),
                        'wind_speed': weather_data.get('wind', {}).get('speed', 0),
                        'cloudiness': weather_data['clouds']['all'],
                        'cache_status': weather_data.get('cache_status', 'unknown'),
                        'status': 'success'
                    })
                    
                    print(f"{Fore.GREEN}✓ Row {idx+1}: {city} on {target_date}")
                    processed_data.append(processed_row)
                    
                except Exception as e:
                    print(f"{Fore.RED}❌ Row {idx+1}: Error - {e}")
                    processed_row = row.to_dict()
                    processed_row.update({'status': 'error', 'error_message': str(e)})
                    processed_data.append(processed_row)
            
            result_df = pd.DataFrame(processed_data)
            
            # Save processing report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.reports_dir / f"processing_report_{timestamp}.csv"
            result_df.to_csv(report_file, index=False)
            print(f"{Fore.CYAN}📊 Processing report saved: {report_file}")
            
            # Automatically generate visualizations
            self._auto_generate_visualizations()
            
            return result_df
            
        except Exception as e:
            raise WeatherCacheException(f"CSV processing failed: {e}")
    
    def _auto_generate_visualizations(self) -> None:
        """Generate research-grade visualizations for all city data"""
        print(f"\n{Fore.CYAN}🎨 Generating research-grade visualizations...")
        
        city_files = list(self.processed_data_dir.glob("*_weather_data.csv"))
        
        if not city_files:
            print(f"{Fore.YELLOW}⚠️ No city data files found for visualization")
            return
        
        visualizer = WeatherVisualizer()
        
        # Individual city statistical analysis
        for city_file in city_files:
            try:
                city_name = city_file.stem.replace('_weather_data', '')
                df = pd.read_csv(city_file)
                
                if len(df) < 3:  # Need minimum data for analysis
                    print(f"{Fore.YELLOW}⚠️ Insufficient data for {city_name} (need ≥3 data points)")
                    continue
                
                # Generate statistical analysis
                chart_file = self.visualization_dir / f"{city_name}_statistical_analysis.png"
                visualizer.create_statistical_analysis(df, city_name, str(chart_file))
                
                print(f"{Fore.GREEN}✅ Generated statistical analysis for {city_name}")
                
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Failed to generate analysis for {city_file.stem}: {e}")
        
        # Multi-city comparison heatmap
        if len(city_files) > 1:
            try:
                combined_df = pd.concat([pd.read_csv(f) for f in city_files], ignore_index=True)
                heatmap_file = self.visualization_dir / "climate_comparison_heatmap.png"
                visualizer.create_climate_comparison_heatmap(combined_df, str(heatmap_file))
                print(f"{Fore.GREEN}✅ Generated climate comparison heatmap")
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Failed to generate comparison heatmap: {e}")
        
        print(f"{Fore.CYAN}📊 Research visualizations saved to: {self.visualization_dir}")
    
    def clear_cache(self, older_than_hours: int = None) -> int:
        """Clear cache files with city/year/day structure"""
        cleared_count = 0
        
        for city_dir in self.cache_dir.iterdir():
            if city_dir.is_dir():
                for year_dir in city_dir.iterdir():
                    if year_dir.is_dir():
                        for day_dir in year_dir.iterdir():
                            if day_dir.is_dir():
                                for cache_file in day_dir.glob("*.pkl"):
                                    if older_than_hours:
                                        file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                                        if file_age_hours < older_than_hours:
                                            continue
                                    
                                    try:
                                        cache_file.unlink()
                                        cleared_count += 1
                                    except OSError as e:
                                        self.logger.error(f"Failed to delete {cache_file}: {e}")
        
        self.logger.info(f"Cleared {cleared_count} cache files")
        return cleared_count
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics with new structure"""
        pickle_files = []
        csv_files = list(self.processed_data_dir.glob("*.csv"))
        
        # Count cache files in city/year/day structure
        cities_cached = 0
        for city_dir in self.cache_dir.iterdir():
            if city_dir.is_dir():
                cities_cached += 1
                for year_dir in city_dir.iterdir():
                    if year_dir.is_dir():
                        for day_dir in year_dir.iterdir():
                            if day_dir.is_dir():
                                pickle_files.extend(list(day_dir.glob("*.pkl")))
        
        total_size = sum(f.stat().st_size for f in pickle_files + csv_files)
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses) * 100,
            'api_calls': self.api_calls,
            'pickle_files': len(pickle_files),
            'csv_files': len(csv_files),
            'cities_cached': cities_cached,
            'total_cache_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_dir),
            'data_directory': str(self.data_dir)
        }
    
    def print_stats(self) -> None:
        """Print comprehensive cache statistics"""
        stats = self.get_cache_stats()
        
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}🌤️  WEATHERCACHE DAILY DATA STATISTICS")
        print(f"{Fore.CYAN}{'='*60}")
        
        # Cache Performance
        print(f"{Fore.WHITE}📊 CACHE PERFORMANCE:")
        print(f"  {Fore.GREEN}✅ Cache Hits: {stats['cache_hits']} (data from cache)")
        print(f"  {Fore.YELLOW}❌ Cache Misses: {stats['cache_misses']} (required API calls)")
        print(f"  {Fore.BLUE}🎯 Hit Rate: {stats['hit_rate']:.1f}% (higher is better)")
        
        # API Usage
        print(f"\n{Fore.WHITE}🌐 API USAGE:")
        print(f"  {Fore.MAGENTA}📡 Total API Calls: {stats['api_calls']}")
        print(f"  {Fore.GREEN}💰 API Calls Saved: {stats['cache_hits']} (thanks to caching)")
        
        # Storage Information
        print(f"\n{Fore.WHITE}💾 DATA STORAGE:")
        print(f"  {Fore.CYAN}🏙️  Cities with Data: {stats['cities_cached']}")
        print(f"  {Fore.CYAN}📦 Cache Files (pickle): {stats['pickle_files']}")
        print(f"  {Fore.CYAN}📊 City CSV Files: {stats['csv_files']}")
        print(f"  {Fore.CYAN}📁 Total Size: {stats['total_cache_size_mb']:.2f} MB")
        
        # Directory Structure
        print(f"\n{Fore.WHITE}🗂️  DIRECTORY STRUCTURE:")
        print(f"  {Fore.CYAN}📂 Cache: {stats['cache_directory']}")
        print(f"  {Fore.CYAN}    └── [city]/[year]/[day]/")
        print(f"  {Fore.CYAN}📂 Data: {stats['data_directory']}")
        print(f"  {Fore.CYAN}    ├── reports/ (processing reports)")
        print(f"  {Fore.CYAN}    ├── visualization/ (auto-generated charts)")
        print(f"  {Fore.CYAN}    └── data/ (city CSV files)")
        
        print(f"{Fore.CYAN}{'='*60}\n")


class WeatherVisualizer:
    """Research-grade weather data visualization with statistical analysis"""
    
    def __init__(self):
        # Enhanced theme for research publications
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.titlesize': 16,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
    
    def create_statistical_analysis(self, df: pd.DataFrame, city_name: str, output_file: str) -> None:
        """Create comprehensive statistical analysis for researchers"""
        if len(df) < 7:  # Need minimum data for meaningful analysis
            print(f"{Fore.YELLOW}⚠️ Insufficient data for statistical analysis (need ≥7 points)")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Statistical Analysis: {city_name.title()}', fontsize=16, fontweight='bold')
        
        # 1. Temperature distribution with statistical tests
        ax1 = axes[0, 0]
        temps = df['temperature'].dropna()
        ax1.hist(temps, bins=min(15, len(temps)//2), alpha=0.7, color='#2E86AB', edgecolor='black')
        ax1.axvline(temps.mean(), color='red', linestyle='--', label=f'Mean: {temps.mean():.1f}°C')
        ax1.axvline(temps.median(), color='orange', linestyle='--', label=f'Median: {temps.median():.1f}°C')
        ax1.set_title('Temperature Distribution\n(with normality indicators)')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Add statistical info
        from scipy import stats
        skewness = stats.skew(temps)
        kurtosis = stats.kurtosis(temps)
        ax1.text(0.02, 0.98, f'Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}', 
                transform=ax1.transAxes, va='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Box plot comparison by time periods
        ax2 = axes[0, 1]
        df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
        df['time_period'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], 
                                  labels=['Night\n(0-6h)', 'Morning\n(6-12h)', 'Afternoon\n(12-18h)', 'Evening\n(18-24h)'])
        
        if not df['time_period'].isna().all():
            df.boxplot(column='temperature', by='time_period', ax=ax2)
            ax2.set_title('Temperature by Time Period\n(with quartiles & outliers)')
            ax2.set_xlabel('Time Period')
            ax2.set_ylabel('Temperature (°C)')
        
        # 3. Correlation heatmap
        ax3 = axes[0, 2]
        corr_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
        available_cols = [col for col in corr_cols if col in df.columns and not df[col].isna().all()]
        
        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr()
            im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax3.set_xticks(range(len(available_cols)))
            ax3.set_yticks(range(len(available_cols)))
            ax3.set_xticklabels([col.replace('_', '\n') for col in available_cols], rotation=45)
            ax3.set_yticklabels([col.replace('_', '\n') for col in available_cols])
            
            # Add correlation values
            for i in range(len(available_cols)):
                for j in range(len(available_cols)):
                    text = ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            ax3.set_title('Weather Parameter Correlations\n(Pearson coefficients)')
            plt.colorbar(im, ax=ax3, shrink=0.6)
        
        # 4. Moving averages and trends
        ax4 = axes[1, 0]
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df_sorted = df.sort_values('datetime')
        
        if len(df_sorted) >= 3:
            ax4.plot(df_sorted['datetime'], df_sorted['temperature'], 'o-', alpha=0.7, 
                    label='Actual', color='#2E86AB', markersize=4)
            
            # Add moving average if enough data
            if len(df_sorted) >= 5:
                window = min(5, len(df_sorted)//2)
                rolling_mean = df_sorted['temperature'].rolling(window=window, center=True).mean()
                ax4.plot(df_sorted['datetime'], rolling_mean, '--', 
                        label=f'{window}-point Moving Avg', color='red', linewidth=2)
            
            # Add trend line
            if len(df_sorted) >= 3:
                x_numeric = np.arange(len(df_sorted))
                z = np.polyfit(x_numeric, df_sorted['temperature'], 1)
                p = np.poly1d(z)
                ax4.plot(df_sorted['datetime'], p(x_numeric), '-', 
                        label=f'Trend (slope: {z[0]:.3f}°C/period)', color='green', linewidth=2)
        
        ax4.set_title('Temperature Trends & Moving Averages')
        ax4.set_xlabel('Date/Time')
        ax4.set_ylabel('Temperature (°C)')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Weather condition frequency
        ax5 = axes[1, 1]
        if 'weather_main' in df.columns and not df['weather_main'].isna().all():
            weather_counts = df['weather_main'].value_counts()
            wedges, texts, autotexts = ax5.pie(weather_counts.values, 
                                              labels=weather_counts.index,
                                              autopct='%1.1f%%',
                                              startangle=90)
            ax5.set_title('Weather Condition Distribution\n(percentage of observations)')
        
        # 6. Statistical summary table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create statistical summary
        stats_data = []
        for col in ['temperature', 'humidity', 'pressure', 'wind_speed']:
            if col in df.columns and not df[col].isna().all():
                series = df[col].dropna()
                stats_data.append([
                    col.replace('_', ' ').title(),
                    f'{series.mean():.1f}',
                    f'{series.std():.1f}',
                    f'{series.min():.1f}',
                    f'{series.max():.1f}',
                    f'{len(series)}'
                ])
        
        if stats_data:
            table = ax6.table(cellText=stats_data,
                             colLabels=['Parameter', 'Mean', 'Std Dev', 'Min', 'Max', 'Count'],
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax6.set_title('Statistical Summary')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_climate_comparison_heatmap(self, df: pd.DataFrame, output_file: str) -> None:
        """Create heatmap comparing cities and dates for climate research"""
        if 'city' not in df.columns or len(df) < 10:
            print(f"{Fore.YELLOW}⚠️ Insufficient data for climate comparison heatmap")
            return
        
        # Create pivot table for heatmap
        df['date_short'] = pd.to_datetime(df['date']).dt.strftime('%m-%d')
        pivot_temp = df.pivot_table(values='temperature', index='city', columns='date_short', aggfunc='mean')
        
        if pivot_temp.empty:
            print(f"{Fore.YELLOW}⚠️ No temperature data available for heatmap")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Temperature heatmap
        im1 = ax1.imshow(pivot_temp.values, cmap='RdYlBu_r', aspect='auto')
        ax1.set_xticks(range(len(pivot_temp.columns)))
        ax1.set_yticks(range(len(pivot_temp.index)))
        ax1.set_xticklabels(pivot_temp.columns, rotation=45)
        ax1.set_yticklabels([city.title() for city in pivot_temp.index])
        ax1.set_title('Temperature Comparison Across Cities and Dates (°C)')
        ax1.set_xlabel('Date (MM-DD)')
        ax1.set_ylabel('City')
        
        # Add temperature values to cells
        for i in range(len(pivot_temp.index)):
            for j in range(len(pivot_temp.columns)):
                if not np.isnan(pivot_temp.iloc[i, j]):
                    text = ax1.text(j, i, f'{pivot_temp.iloc[i, j]:.1f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im1, ax=ax1, shrink=0.6)
        
        # Humidity comparison
        if 'humidity' in df.columns:
            pivot_humid = df.pivot_table(values='humidity', index='city', columns='date_short', aggfunc='mean')
            if not pivot_humid.empty:
                im2 = ax2.imshow(pivot_humid.values, cmap='Blues', aspect='auto')
                ax2.set_xticks(range(len(pivot_humid.columns)))
                ax2.set_yticks(range(len(pivot_humid.index)))
                ax2.set_xticklabels(pivot_humid.columns, rotation=45)
                ax2.set_yticklabels([city.title() for city in pivot_humid.index])
                ax2.set_title('Humidity Comparison Across Cities and Dates (%)')
                ax2.set_xlabel('Date (MM-DD)')
                ax2.set_ylabel('City')
                plt.colorbar(im2, ax=ax2, shrink=0.6)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def create_sample_csv() -> Path:
    """Create a sample CSV file with separate date and time columns"""
    sample_data = {
        'date': [
            # Recent dates for London (enough for statistical analysis - need 7+ points)
            '2025-08-01', '2025-08-02', '2025-08-03', '2025-08-04', '2025-08-05', 
            '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-09', '2025-08-10',
            
            # Recent dates for Paris (7+ points for statistical analysis)
            '2025-08-01', '2025-08-02', '2025-08-03', '2025-08-04', '2025-08-05',
            '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-09', '2025-08-10',
            
            # Recent dates for Tokyo (7+ points)
            '2025-08-01', '2025-08-02', '2025-08-03', '2025-08-04', '2025-08-05',
            '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-09', '2025-08-10',
            
            # Recent dates for New York (7+ points)
            '2025-08-01', '2025-08-02', '2025-08-03', '2025-08-04', '2025-08-05',
            '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-09', '2025-08-10',
            
            # Recent dates for Sydney (7+ points)
            '2025-08-01', '2025-08-02', '2025-08-03', '2025-08-04', '2025-08-05',
            '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-09', '2025-08-10',
            
            # Additional cities for multi-city heatmap (need 10+ total records)
            '2025-08-05', '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-09',
            '2025-08-05', '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-09',
            '2025-08-05', '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-09',
            '2025-08-05', '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-09',
            '2025-08-05', '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-09',
        ],
        'time': [
            # London - varied times for time period analysis (Night/Morning/Afternoon/Evening)
            '02:00:00', '08:00:00', '14:00:00', '20:00:00', '05:00:00',
            '11:00:00', '17:00:00', '23:00:00', '07:00:00', '13:00:00',
            
            # Paris - varied times
            '01:00:00', '09:00:00', '15:00:00', '21:00:00', '04:00:00',
            '10:00:00', '16:00:00', '22:00:00', '06:00:00', '12:00:00',
            
            # Tokyo - varied times  
            '03:00:00', '07:00:00', '13:00:00', '19:00:00', '01:00:00',
            '09:00:00', '15:00:00', '21:00:00', '05:00:00', '11:00:00',
            
            # New York - varied times
            '00:00:00', '06:00:00', '12:00:00', '18:00:00', '03:00:00',
            '09:00:00', '15:00:00', '21:00:00', '04:00:00', '10:00:00',
            
            # Sydney - varied times
            '02:00:00', '08:00:00', '14:00:00', '20:00:00', '01:00:00',
            '07:00:00', '13:00:00', '19:00:00', '05:00:00', '11:00:00',
            
            # Additional cities - varied times
            '06:00:00', '09:00:00', '12:00:00', '15:00:00', '18:00:00',  # Berlin
            '07:00:00', '10:00:00', '13:00:00', '16:00:00', '19:00:00',  # Mumbai
            '08:00:00', '11:00:00', '14:00:00', '17:00:00', '20:00:00',  # São Paulo
            '05:00:00', '08:00:00', '11:00:00', '14:00:00', '17:00:00',  # Cairo
            '04:00:00', '07:00:00', '10:00:00', '13:00:00', '16:00:00',  # Vancouver
        ],
        'city': [
            # London (10 entries for good statistical analysis)
            'London,GB', 'London,GB', 'London,GB', 'London,GB', 'London,GB',
            'London,GB', 'London,GB', 'London,GB', 'London,GB', 'London,GB',
            
            # Paris (10 entries)
            'Paris,FR', 'Paris,FR', 'Paris,FR', 'Paris,FR', 'Paris,FR',
            'Paris,FR', 'Paris,FR', 'Paris,FR', 'Paris,FR', 'Paris,FR',
            
            # Tokyo (10 entries)
            'Tokyo,JP', 'Tokyo,JP', 'Tokyo,JP', 'Tokyo,JP', 'Tokyo,JP',
            'Tokyo,JP', 'Tokyo,JP', 'Tokyo,JP', 'Tokyo,JP', 'Tokyo,JP',
            
            # New York (10 entries)
            'New York,US', 'New York,US', 'New York,US', 'New York,US', 'New York,US',
            'New York,US', 'New York,US', 'New York,US', 'New York,US', 'New York,US',
            
            # Sydney (10 entries)
            'Sydney,AU', 'Sydney,AU', 'Sydney,AU', 'Sydney,AU', 'Sydney,AU',
            'Sydney,AU', 'Sydney,AU', 'Sydney,AU', 'Sydney,AU', 'Sydney,AU',
            
            # Additional cities for multi-city heatmap (5 entries each)
            'Berlin,DE', 'Berlin,DE', 'Berlin,DE', 'Berlin,DE', 'Berlin,DE',
            'Mumbai,IN', 'Mumbai,IN', 'Mumbai,IN', 'Mumbai,IN', 'Mumbai,IN',
            'São Paulo,BR', 'São Paulo,BR', 'São Paulo,BR', 'São Paulo,BR', 'São Paulo,BR',
            'Cairo,EG', 'Cairo,EG', 'Cairo,EG', 'Cairo,EG', 'Cairo,EG',
            'Vancouver,CA', 'Vancouver,CA', 'Vancouver,CA', 'Vancouver,CA', 'Vancouver,CA',
        ],
        'region': [
            # London
            'Europe', 'Europe', 'Europe', 'Europe', 'Europe',
            'Europe', 'Europe', 'Europe', 'Europe', 'Europe',
            
            # Paris  
            'Europe', 'Europe', 'Europe', 'Europe', 'Europe',
            'Europe', 'Europe', 'Europe', 'Europe', 'Europe',
            
            # Tokyo
            'Asia', 'Asia', 'Asia', 'Asia', 'Asia',
            'Asia', 'Asia', 'Asia', 'Asia', 'Asia',
            
            # New York
            'North America', 'North America', 'North America', 'North America', 'North America',
            'North America', 'North America', 'North America', 'North America', 'North America',
            
            # Sydney
            'Oceania', 'Oceania', 'Oceania', 'Oceania', 'Oceania',
            'Oceania', 'Oceania', 'Oceania', 'Oceania', 'Oceania',
            
            # Additional cities
            'Europe', 'Europe', 'Europe', 'Europe', 'Europe',        # Berlin
            'Asia', 'Asia', 'Asia', 'Asia', 'Asia',                  # Mumbai
            'South America', 'South America', 'South America', 'South America', 'South America',  # São Paulo
            'Africa', 'Africa', 'Africa', 'Africa', 'Africa',        # Cairo
            'North America', 'North America', 'North America', 'North America', 'North America',  # Vancouver
        ]
    }
    
    df = pd.DataFrame(sample_data)
    csv_path = Path('sample_weather_schedule.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"{Fore.GREEN}✅ Sample CSV created: {csv_path.absolute()}")
    print(f"{Fore.CYAN}📋 CSV structure for daily weather collection:")
    print(f"  {Fore.WHITE}• date: Date in YYYY-MM-DD format")
    print(f"  {Fore.WHITE}• time: Time in HH:MM:SS format (optional)")
    print(f"  {Fore.WHITE}• city: City name with country code")
    print(f"  {Fore.WHITE}• region: Geographic region (optional)")
    print(f"\n{Fore.YELLOW}💡 Features:")
    print(f"  {Fore.CYAN}• Separate date and time columns")
    print(f"  {Fore.CYAN}• Automatic city CSV accumulation")
    print(f"  {Fore.CYAN}• Daily data collection ready")
    print(f"  {Fore.CYAN}• Auto-visualization generation")
    
    return csv_path


def main():
    """Main CLI interface for daily weather data collection"""
    parser = argparse.ArgumentParser(
        description="🌤️ WeatherCache - Daily Weather Data Collection & Auto-Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time setup
  python weathercache.py --set-env
  
  # Create sample CSV with date/time structure
  python weathercache.py --create-sample
  
  # Process daily weather data (auto-generates visualizations)
  python weathercache.py --csv sample_weather_schedule.csv
  
  # Single city for specific date
  python weathercache.py --city "London" --date "2025-08-05"
  
  # View cache statistics
  python weathercache.py --stats
  
  # Clear old cache
  python weathercache.py --clear-cache --older-than 48
        """
    )
    
    # Environment Management
    env_group = parser.add_argument_group('Environment Management')
    env_group.add_argument('--set-env', action='store_true', 
                          help='Interactive environment setup')
    env_group.add_argument('--validate-key', action='store_true',
                          help='Validate API key')
    env_group.add_argument('--show-config', action='store_true',
                          help='Show current configuration')
    
    # Main operations
    ops_group = parser.add_argument_group('Main Operations')
    ops_group.add_argument('--city', type=str, help='Single city to fetch weather for')
    ops_group.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)')
    ops_group.add_argument('--time', type=str, help='Target time (HH:MM:SS)')
    ops_group.add_argument('--csv', type=str, nargs='?', const='DEFAULT',
                          help='CSV file with date/time/city data')
    
    # Cache management
    cache_group = parser.add_argument_group('Cache Management')
    cache_group.add_argument('--no-cache', action='store_true', help='Disable caching')
    cache_group.add_argument('--clear-cache', action='store_true', help='Clear cache files')
    cache_group.add_argument('--older-than', type=int, help='Clear cache older than N hours')
    cache_group.add_argument('--cache-dir', type=str, help='Cache directory (overrides env)')
    cache_group.add_argument('--data-dir', type=str, help='Data directory (overrides env)')
    cache_group.add_argument('--cache-ttl', type=int, help='Cache TTL in seconds (overrides env)')
    
    # Configuration overrides
    config_group = parser.add_argument_group('Configuration Overrides')
    config_group.add_argument('--api-key', type=str, help='OpenWeatherMap API key')
    config_group.add_argument('--workers', type=int, help='Number of concurrent workers')
    config_group.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                             help='Logging level')
    
    # Utilities
    util_group = parser.add_argument_group('Utilities')
    util_group.add_argument('--stats', action='store_true', help='Show cache statistics')
    util_group.add_argument('--create-sample', action='store_true', help='Create sample CSV')
    
    args = parser.parse_args()
    
    # Load configuration
    env_config = EnvironmentManager.get_config()
    
    # Handle environment setup
    if args.set_env:
        if EnvironmentManager.interactive_setup():
            print(f"\n{Fore.GREEN}🎉 Setup complete! Daily weather collection ready.")
            print(f"{Fore.CYAN}Try: python weathercache.py --create-sample")
        else:
            print(f"{Fore.RED}❌ Setup failed!")
            sys.exit(1)
        return
    
    # Show configuration
    if args.show_config:
        print(f"\n{Fore.CYAN}🔧 WeatherCache Daily Collection Configuration")
        print(f"{Fore.CYAN}{'='*50}")
        print(f"{Fore.WHITE}API Key: {'✅ Set' if env_config['api_key'] else '❌ Not set'}")
        print(f"{Fore.WHITE}Cache Directory: {env_config['cache_dir']}")
        print(f"{Fore.WHITE}Data Directory: {env_config['data_dir']}")
        print(f"{Fore.WHITE}Cache TTL: {env_config['cache_ttl']} seconds ({env_config['cache_ttl']//3600} hours)")
        print(f"{Fore.WHITE}Max Workers: {env_config['max_workers']}")
        print(f"{Fore.WHITE}Log Level: {env_config['log_level']}")
        print(f"{Fore.WHITE}Default Input CSV: {env_config['default_input_csv']}")
        print(f"{Fore.CYAN}{'='*50}\n")
        return
    
    # Create sample CSV
    if args.create_sample:
        create_sample_csv()
        return
    
    # Resolve configuration
    api_key = args.api_key or env_config['api_key']
    cache_dir = args.cache_dir or env_config['cache_dir']
    data_dir = args.data_dir or env_config['data_dir']
    cache_ttl = args.cache_ttl or env_config['cache_ttl']
    max_workers = args.workers or env_config['max_workers']
    log_level = args.log_level or env_config['log_level']
    
    csv_file = None
    if args.csv == 'DEFAULT':
        csv_file = env_config['default_input_csv']
    elif args.csv:
        csv_file = args.csv
    
    # Validate API key
    if args.validate_key:
        if not api_key:
            print(f"{Fore.RED}❌ No API key found!")
            sys.exit(1)
        
        if EnvironmentManager.validate_api_key(api_key):
            print(f"{Fore.GREEN}🎉 API key is working correctly!")
        else:
            print(f"{Fore.RED}❌ API key validation failed!")
            sys.exit(1)
        return
    
    # Check API key requirement
    if not api_key and not args.clear_cache and not args.stats:
        print(f"{Fore.RED}❌ OpenWeatherMap API key required!")
        print(f"\n{Fore.YELLOW}Setup options:")
        print(f"{Fore.CYAN}  1. Interactive setup: python weathercache.py --set-env")
        print(f"{Fore.CYAN}  2. Use argument: python weathercache.py --api-key YOUR_KEY")
        print(f"\n{Fore.WHITE}Get your free API key: https://openweathermap.org/api")
        sys.exit(1)
    
    # Initialize WeatherCache
    weather_cache = None
    if api_key:
        weather_cache = WeatherCache(
            api_key=api_key,
            cache_dir=cache_dir,
            data_dir=data_dir,
            cache_ttl=cache_ttl,
            max_workers=max_workers
        )
        weather_cache.set_log_level(log_level)
    
    try:
        # Clear cache
        if args.clear_cache:
            if not weather_cache:
                weather_cache = WeatherCache("dummy", cache_dir=cache_dir, data_dir=data_dir)
            cleared = weather_cache.clear_cache(args.older_than)
            print(f"{Fore.GREEN}✅ Cleared {cleared} cache files")
            return
        
        # Show stats
        if args.stats:
            if not weather_cache:
                weather_cache = WeatherCache("dummy", cache_dir=cache_dir, data_dir=data_dir)
            weather_cache.print_stats()
            return
        
        # Single city request
        if args.city:
            target_date = pd.to_datetime(args.date).date() if args.date else date.today()
            target_time = args.time
            
            print(f"{Fore.CYAN}🌤️ Fetching weather for {args.city} on {target_date}")
            if target_time:
                print(f"{Fore.CYAN}⏰ Target time: {target_time}")
            
            try:
                data = weather_cache.get_weather_for_date(args.city, target_date, target_time, 
                                                        use_cache=not args.no_cache)
                
                # Display results
                print(f"\n{Fore.GREEN}{'='*50}")
                print(f"{Fore.GREEN}WEATHER DATA FOR {data['name'].upper()}")
                print(f"{Fore.GREEN}{'='*50}")
                print(f"{Fore.YELLOW}📅 Date: {target_date}")
                if target_time:
                    print(f"{Fore.YELLOW}⏰ Time: {target_time}")
                print(f"{Fore.YELLOW}🌡️  Temperature: {data['main']['temp']:.1f}°C")
                print(f"{Fore.BLUE}💧 Humidity: {data['main']['humidity']}%")
                print(f"{Fore.MAGENTA}🌬️  Pressure: {data['main']['pressure']} hPa")
                print(f"{Fore.WHITE}☁️  Conditions: {data['weather'][0]['description'].title()}")
                print(f"{Fore.CYAN}💨 Wind: {data.get('wind', {}).get('speed', 0)} m/s")
                print(f"{Fore.RED}🌍 Country: {data['sys']['country']}")
                
                cache_status = data.get('cache_status', 'unknown')
                if cache_status == 'from_cache':
                    print(f"{Fore.GREEN}💾 Data source: Cache")
                else:
                    print(f"{Fore.YELLOW}📡 Data source: Fresh API call")
                
            except WeatherCacheException as e:
                print(f"{Fore.RED}❌ Error: {e}")
                sys.exit(1)
        
        # Process CSV file
        elif csv_file:
            if not os.path.exists(csv_file):
                print(f"{Fore.RED}❌ Error: CSV file '{csv_file}' not found")
                if csv_file == env_config['default_input_csv']:
                    print(f"{Fore.YELLOW}💡 Default file: {csv_file}")
                    print(f"{Fore.CYAN}Create sample: python weathercache.py --create-sample")
                sys.exit(1)
            
            print(f"{Fore.CYAN}📄 Processing daily weather data from: {csv_file}")
            print(f"{Fore.YELLOW}🎨 Auto-visualization will be generated after processing")
            
            try:
                df = weather_cache.process_csv(csv_file, use_cache=not args.no_cache)
                
                # Show summary
                successful = len(df[df['status'] == 'success'])
                failed = len(df[df['status'] == 'failed'])
                errors = len(df[df['status'] == 'error'])
                
                print(f"\n{Fore.GREEN}✅ Daily weather collection complete!")
                print(f"{Fore.GREEN}📊 Successful: {successful} entries")
                if failed > 0:
                    print(f"{Fore.RED}❌ Failed: {failed} entries")
                if errors > 0:
                    print(f"{Fore.YELLOW}⚠️ Errors: {errors} entries")
                
                # Show cache performance
                weather_cache.print_stats()
                
                # Show generated files
                print(f"\n{Fore.CYAN}📁 Generated files:")
                print(f"  {Fore.WHITE}📊 City CSV files: {weather_cache.processed_data_dir}")
                print(f"  {Fore.WHITE}📈 Visualizations: {weather_cache.visualization_dir}")
                print(f"  {Fore.WHITE}📋 Reports: {weather_cache.reports_dir}")
                
            except WeatherCacheException as e:
                print(f"{Fore.RED}❌ Error: {e}")
                sys.exit(1)
        
        else:
            parser.print_help()
            print(f"\n{Fore.YELLOW}💡 Quick start for daily weather collection:")
            if not env_config['api_key']:
                print(f"{Fore.RED}  🔑 No API key configured!")
                print(f"{Fore.CYAN}  1. Setup: python weathercache.py --set-env")
            else:
                print(f"{Fore.GREEN}  ✅ API key configured!")
                print(f"{Fore.CYAN}  1. Create sample: python weathercache.py --create-sample")
                print(f"{Fore.CYAN}  2. Process data: python weathercache.py --csv sample_weather_schedule.csv")
                print(f"{Fore.CYAN}  3. Single city: python weathercache.py --city London --date 2025-08-05")
                print(f"{Fore.CYAN}  4. View stats: python weathercache.py --stats")
                
                print(f"\n{Fore.YELLOW}🎯 Daily collection features:")
                print(f"  {Fore.WHITE}• Separate date/time columns")
                print(f"  {Fore.WHITE}• City-based CSV accumulation")
                print(f"  {Fore.WHITE}• Automatic visualization generation")
                print(f"  {Fore.WHITE}• Cache structure: city/year/day")
                print(f"  {Fore.WHITE}• Organized data directories")
    
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️ Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"{Fore.RED}❌ Unexpected error: {e}")
        if log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


def display_banner():
    """Display application banner with version info"""
    banner = f"""
{Fore.CYAN}
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🌤️  WEATHERCACHE v1.0                                ║
║                   Enhanced Weather Data Collection Tool                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Features:                                                                   ║
║  • Smart caching with city/year/day structure                                ║
║  • Automatic visualization generation                                        ║
║  • CSV batch processing with date/time separation                            ║
║  • Environment configuration management                                      ║
║  • Concurrent API requests with rate limiting                                ║
║  • Comprehensive error handling and logging                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


class WeatherDataAnalyzer:
    """Advanced weather data analysis and reporting"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.processed_data_dir = data_dir / "data"
        self.reports_dir = data_dir / "reports"
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report of all weather data"""
        city_files = list(self.processed_data_dir.glob("*_weather_data.csv"))
        
        if not city_files:
            return {"error": "No weather data files found"}
        
        all_data = []
        city_summaries = {}
        
        for city_file in city_files:
            try:
                df = pd.read_csv(city_file)
                if len(df) == 0:
                    continue
                
                city_name = city_file.stem.replace('_weather_data', '')
                all_data.append(df)
                
                # City-specific summary
                city_summaries[city_name] = {
                    'total_records': len(df),
                    'date_range': f"{df['date'].min()} to {df['date'].max()}",
                    'avg_temperature': df['temperature'].mean(),
                    'min_temperature': df['temperature'].min(),
                    'max_temperature': df['temperature'].max(),
                    'avg_humidity': df['humidity'].mean(),
                    'avg_pressure': df['pressure'].mean(),
                    'most_common_weather': df['weather_main'].mode().iloc[0] if len(df['weather_main'].mode()) > 0 else 'N/A',
                    'countries': df['country'].unique().tolist() if 'country' in df.columns else []
                }
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Error processing {city_file}: {e}")
                continue
        
        if not all_data:
            return {"error": "No valid weather data found"}
        
        # Combined analysis
        combined_df = pd.concat(all_data, ignore_index=True)
        
        summary = {
            'generation_time': datetime.now().isoformat(),
            'total_cities': len(city_summaries),
            'total_records': len(combined_df),
            'date_range': f"{combined_df['date'].min()} to {combined_df['date'].max()}",
            'global_stats': {
                'avg_temperature': combined_df['temperature'].mean(),
                'temperature_range': {
                    'min': combined_df['temperature'].min(),
                    'max': combined_df['temperature'].max()
                },
                'avg_humidity': combined_df['humidity'].mean(),
                'avg_pressure': combined_df['pressure'].mean(),
                'weather_conditions': combined_df['weather_main'].value_counts().to_dict(),
                'countries_covered': combined_df['country'].nunique() if 'country' in combined_df.columns else 0
            },
            'city_summaries': city_summaries,
            'data_quality': {
                'cache_hit_rate': len(combined_df[combined_df['cache_status'] == 'from_cache']) / len(combined_df) * 100,
                'successful_requests': len(combined_df[combined_df['temperature'].notna()]) / len(combined_df) * 100
            }
        }
        
        return summary
    
    def export_summary_report(self) -> Path:
        """Export summary report to JSON file"""
        summary = self.generate_summary_report()
        
        if 'error' in summary:
            raise WeatherCacheException(summary['error'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"weather_summary_report_{timestamp}.json"
        
        try:
            import json
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"{Fore.GREEN}✅ Summary report exported: {report_file}")
            return report_file
        except Exception as e:
            raise WeatherCacheException(f"Failed to export summary report: {e}")
    
    def print_summary_report(self) -> None:
        """Print formatted summary report to console"""
        summary = self.generate_summary_report()
        
        if 'error' in summary:
            print(f"{Fore.RED}❌ {summary['error']}")
            return
        
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}📊 WEATHER DATA SUMMARY REPORT")
        print(f"{Fore.CYAN}{'='*80}")
        
        # Global statistics
        print(f"\n{Fore.WHITE}🌍 GLOBAL STATISTICS:")
        print(f"  {Fore.GREEN}📊 Total Cities: {summary['total_cities']}")
        print(f"  {Fore.GREEN}📈 Total Records: {summary['total_records']}")
        print(f"  {Fore.GREEN}📅 Date Range: {summary['date_range']}")
        print(f"  {Fore.GREEN}🌡️  Average Temperature: {summary['global_stats']['avg_temperature']:.1f}°C")
        print(f"  {Fore.GREEN}🔥 Temperature Range: {summary['global_stats']['temperature_range']['min']:.1f}°C to {summary['global_stats']['temperature_range']['max']:.1f}°C")
        print(f"  {Fore.GREEN}💧 Average Humidity: {summary['global_stats']['avg_humidity']:.1f}%")
        print(f"  {Fore.GREEN}🌬️  Average Pressure: {summary['global_stats']['avg_pressure']:.1f} hPa")
        print(f"  {Fore.GREEN}🌍 Countries Covered: {summary['global_stats']['countries_covered']}")
        
        # Data quality
        print(f"\n{Fore.WHITE}📊 DATA QUALITY:")
        print(f"  {Fore.BLUE}💾 Cache Hit Rate: {summary['data_quality']['cache_hit_rate']:.1f}%")
        print(f"  {Fore.BLUE}✅ Success Rate: {summary['data_quality']['successful_requests']:.1f}%")
        
        # Weather conditions
        print(f"\n{Fore.WHITE}☁️  WEATHER CONDITIONS:")
        for condition, count in list(summary['global_stats']['weather_conditions'].items())[:5]:
            print(f"  {Fore.YELLOW}{condition}: {count} records")
        
        # Top cities by data points
        print(f"\n{Fore.WHITE}🏙️  TOP CITIES BY DATA POINTS:")
        sorted_cities = sorted(summary['city_summaries'].items(), 
                             key=lambda x: x[1]['total_records'], reverse=True)
        for city, data in sorted_cities[:10]:
            print(f"  {Fore.CYAN}{city.title()}: {data['total_records']} records "
                  f"(Avg: {data['avg_temperature']:.1f}°C)")
        
        print(f"{Fore.CYAN}{'='*80}\n")


class WeatherCacheManager:
    """Enhanced cache management with advanced features"""
    
    def __init__(self, weather_cache: WeatherCache):
        self.cache = weather_cache
    
    def optimize_cache(self) -> Dict[str, int]:
        """Optimize cache by removing outdated and duplicate files"""
        optimized = {
            'removed_outdated': 0,
            'removed_duplicates': 0,
            'total_space_saved_mb': 0
        }
        
        current_time = time.time()
        seen_files = set()
        
        for city_dir in self.cache.cache_dir.iterdir():
            if not city_dir.is_dir():
                continue
                
            for year_dir in city_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                    
                for day_dir in year_dir.iterdir():
                    if not day_dir.is_dir():
                        continue
                    
                    for cache_file in day_dir.glob("*.pkl"):
                        file_size = cache_file.stat().st_size
                        file_age = current_time - cache_file.stat().st_mtime
                        
                        # Remove outdated files
                        if file_age > self.cache.cache_ttl:
                            try:
                                cache_file.unlink()
                                optimized['removed_outdated'] += 1
                                optimized['total_space_saved_mb'] += file_size / (1024 * 1024)
                            except OSError:
                                continue
                        
                        # Check for duplicates based on file content hash
                        try:
                            with open(cache_file, 'rb') as f:
                                file_hash = hash(f.read())
                            
                            if file_hash in seen_files:
                                cache_file.unlink()
                                optimized['removed_duplicates'] += 1
                                optimized['total_space_saved_mb'] += file_size / (1024 * 1024)
                            else:
                                seen_files.add(file_hash)
                        except (OSError, IOError):
                            continue
        
        return optimized
    
    def backup_cache(self, backup_dir: str) -> Path:
        """Create backup of cache directory"""
        import shutil
        
        backup_path = Path(backup_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"weathercache_backup_{timestamp}"
        full_backup_path = backup_path / backup_name
        
        try:
            backup_path.mkdir(exist_ok=True, parents=True)
            shutil.copytree(self.cache.cache_dir, full_backup_path)
            
            print(f"{Fore.GREEN}✅ Cache backup created: {full_backup_path}")
            return full_backup_path
        except Exception as e:
            raise WeatherCacheException(f"Backup failed: {e}")
    
    def restore_cache(self, backup_path: str) -> None:
        """Restore cache from backup"""
        import shutil
        
        backup_dir = Path(backup_path)
        if not backup_dir.exists():
            raise WeatherCacheException(f"Backup directory not found: {backup_path}")
        
        try:
            # Remove current cache
            if self.cache.cache_dir.exists():
                shutil.rmtree(self.cache.cache_dir)
            
            # Restore from backup
            shutil.copytree(backup_dir, self.cache.cache_dir)
            
            print(f"{Fore.GREEN}✅ Cache restored from: {backup_path}")
        except Exception as e:
            raise WeatherCacheException(f"Restore failed: {e}")


def setup_requirements():
    """Check and install required packages"""
    required_packages = [
        'requests', 'pandas', 'matplotlib', 'seaborn', 
        'colorama', 'pathlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"{Fore.YELLOW}⚠️ Missing required packages: {', '.join(missing_packages)}")
        print(f"{Fore.CYAN}Install with: pip install {' '.join(missing_packages)}")
        return False
    
    return True


def handle_advanced_operations(args, weather_cache: WeatherCache) -> None:
    """Handle advanced operations like analysis and optimization"""
    
    # Generate analysis report
    if hasattr(args, 'analyze') and args.analyze:
        analyzer = WeatherDataAnalyzer(weather_cache.data_dir)
        analyzer.print_summary_report()
        
        if hasattr(args, 'export_report') and args.export_report:
            analyzer.export_summary_report()
        return
    
    # Cache optimization
    if hasattr(args, 'optimize_cache') and args.optimize_cache:
        manager = WeatherCacheManager(weather_cache)
        results = manager.optimize_cache()
        
        print(f"{Fore.GREEN}✅ Cache optimization complete:")
        print(f"  {Fore.CYAN}Removed outdated files: {results['removed_outdated']}")
        print(f"  {Fore.CYAN}Removed duplicates: {results['removed_duplicates']}")
        print(f"  {Fore.CYAN}Space saved: {results['total_space_saved_mb']:.2f} MB")
        return
    
    # Cache backup
    if hasattr(args, 'backup_cache') and args.backup_cache:
        manager = WeatherCacheManager(weather_cache)
        backup_dir = getattr(args, 'backup_dir', 'backups')
        manager.backup_cache(backup_dir)
        return


def add_advanced_arguments(parser: argparse.ArgumentParser) -> None:
    """Add advanced command line arguments"""
    
    # Analysis group
    analysis_group = parser.add_argument_group('Data Analysis')
    analysis_group.add_argument('--analyze', action='store_true',
                               help='Generate comprehensive data analysis report')
    analysis_group.add_argument('--export-report', action='store_true',
                               help='Export analysis report to JSON file')
    
    # Advanced cache management
    advanced_cache_group = parser.add_argument_group('Advanced Cache Management')
    advanced_cache_group.add_argument('--optimize-cache', action='store_true',
                                     help='Optimize cache by removing outdated/duplicate files')
    advanced_cache_group.add_argument('--backup-cache', action='store_true',
                                     help='Create backup of cache directory')
    advanced_cache_group.add_argument('--backup-dir', type=str, default='backups',
                                     help='Directory for cache backups')
    advanced_cache_group.add_argument('--restore-cache', type=str,
                                     help='Restore cache from backup directory')


def enhanced_main():
    """Enhanced main function with all features"""
    
    # Check requirements first
    if not setup_requirements():
        sys.exit(1)
    
    # Display banner
    display_banner()
    
    # Create enhanced argument parser
    parser = argparse.ArgumentParser(
        description="🌤️ WeatherCache v1.0 - Enhanced Weather Data Collection & Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Examples:
  # Complete setup and analysis workflow
  python weathercache.py --set-env
  python weathercache.py --create-sample
  python weathercache.py --csv sample_weather_schedule.csv
  python weathercache.py --analyze --export-report
  
  # Cache management
  python weathercache.py --optimize-cache
  python weathercache.py --backup-cache --backup-dir ./my_backups
  
  # Advanced single city with time
  python weathercache.py --city "London,GB" --date "2025-08-05" --time "14:30:00"
        """
    )
    
    # Environment Management
    env_group = parser.add_argument_group('Environment Management')
    env_group.add_argument('--set-env', action='store_true', 
                          help='Interactive environment setup')
    env_group.add_argument('--validate-key', action='store_true',
                          help='Validate API key')
    env_group.add_argument('--show-config', action='store_true',
                          help='Show current configuration')
    
    # Main operations
    ops_group = parser.add_argument_group('Main Operations')
    ops_group.add_argument('--city', type=str, help='Single city to fetch weather for')
    ops_group.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)')
    ops_group.add_argument('--time', type=str, help='Target time (HH:MM:SS)')
    ops_group.add_argument('--csv', type=str, nargs='?', const='DEFAULT',
                          help='CSV file with date/time/city data')
    
    # Cache management
    cache_group = parser.add_argument_group('Cache Management')
    cache_group.add_argument('--no-cache', action='store_true', help='Disable caching')
    cache_group.add_argument('--clear-cache', action='store_true', help='Clear cache files')
    cache_group.add_argument('--older-than', type=int, help='Clear cache older than N hours')
    cache_group.add_argument('--cache-dir', type=str, help='Cache directory (overrides env)')
    cache_group.add_argument('--data-dir', type=str, help='Data directory (overrides env)')
    cache_group.add_argument('--cache-ttl', type=int, help='Cache TTL in seconds (overrides env)')
    
    # Configuration overrides
    config_group = parser.add_argument_group('Configuration Overrides')
    config_group.add_argument('--api-key', type=str, help='OpenWeatherMap API key')
    config_group.add_argument('--workers', type=int, help='Number of concurrent workers')
    config_group.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                             help='Logging level')
    
    # Utilities
    util_group = parser.add_argument_group('Utilities')
    util_group.add_argument('--stats', action='store_true', help='Show cache statistics')
    util_group.add_argument('--create-sample', action='store_true', help='Create sample CSV')
    util_group.add_argument('--version', action='store_true', help='Show version information')
    
    # Add advanced arguments
    add_advanced_arguments(parser)
    
    args = parser.parse_args()
    
    # Show version
    if args.version:
        print(f"{Fore.CYAN}WeatherCache v1.0 - Enhanced Weather Data Collection Tool")
        print(f"{Fore.WHITE}Author: Jainish Patel")
        print(f"{Fore.WHITE}Date: August 5, 2025")
        return
    
    # Run the main application logic
    main()


if __name__ == "__main__":
    enhanced_main()