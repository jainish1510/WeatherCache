# 🌤️ WeatherCache v1.0 
### Author: Jainish Patel

A comprehensive weather data collection and analysis tool with smart caching, automatic visualization generation, and research-grade statistical analysis.

## ✨ Features

- **Smart Caching System**: City/year/day structure with configurable TTL
- **Automatic Visualizations**: Research-grade statistical charts and climate comparisons
- **CSV Batch Processing**: Handle multiple cities with separate date/time columns
- **Environment Configuration**: Interactive setup with `.env` file management
- **Concurrent Processing**: Rate-limited API requests with thread pooling
- **Data Analysis**: Comprehensive statistical reports and climate comparisons
- **Cache Optimization**: Advanced cache management with backup/restore

## 📋 Requirements

- Python 3.7+
- OpenWeatherMap API key (free at [openweathermap.org/api](https://openweathermap.org/api))

### Python Dependencies

```bash
pip install requests pandas matplotlib seaborn colorama numpy scipy
```

Or install all at once:
```bash
pip install -r requirements.txt
or
pip install requests pandas matplotlib seaborn colorama numpy scipy pathlib

```

## 🚀 Quick Start

### 1. First-Time Setup

Run the interactive setup to configure your API key and preferences:

```bash
python weathercache.py --set-env
```

This will:
- Prompt for your OpenWeatherMap API key
- Set up directory structure
- Configure cache settings
- Create a `.env` file with your preferences

### 2. Create Sample Data

Generate a sample CSV file to understand the data format:

```bash
python weathercache.py --create-sample
```

### 3. Process Weather Data

Process the sample CSV (automatically generates visualizations):

```bash
python weathercache.py --csv sample_weather_schedule.csv
```

### 4. View Results

Check cache statistics and generated files:

```bash
python weathercache.py --stats
```

## 📁 Directory Structure

WeatherCache creates an organized directory structure:

```
project_root/
├── .env                           # Environment configuration
├── cache/                         # Smart cache storage
│   └── [city]/[year]/[day]/      # Hierarchical cache structure
├── weather_data/                  # Processed data and outputs
│   ├── data/                     # City CSV files
│   ├── reports/                  # Processing reports
│   └── visualization/            # Auto-generated charts
└── sample_weather_schedule.csv   # Sample data file
```

## 🛠️ Command Reference

### Environment Management

| Command | Description |
|---------|-------------|
| `--set-env` | Interactive environment setup |
| `--validate-key` | Test API key validity |
| `--show-config` | Display current configuration |

### Main Operations

| Command | Description | Example |
|---------|-------------|---------|
| `--city CITY` | Fetch weather for single city | `--city "London,GB"` |
| `--date DATE` | Specify target date (YYYY-MM-DD) | `--date "2025-08-05"` |
| `--time TIME` | Specify target time (HH:MM:SS) | `--time "14:30:00"` |
| `--csv FILE` | Process CSV file | `--csv cities.csv` |

### Cache Management

| Command | Description | Example |
|---------|-------------|---------|
| `--no-cache` | Disable caching for this run | |
| `--clear-cache` | Remove cache files | |
| `--older-than HOURS` | Clear cache older than N hours | `--older-than 48` |
| `--optimize-cache` | Remove outdated/duplicate files | |

### Data Analysis

| Command | Description |
|---------|-------------|
| `--analyze` | Generate comprehensive analysis report |
| `--export-report` | Export analysis to JSON file |
| `--stats` | Show cache statistics |

### Configuration Overrides

| Command | Description | Example |
|---------|-------------|---------|
| `--api-key KEY` | Override API key | `--api-key "your_key_here"` |
| `--cache-dir DIR` | Override cache directory | `--cache-dir "my_cache"` |
| `--data-dir DIR` | Override data directory | `--data-dir "my_data"` |
| `--workers N` | Set concurrent workers (1-5) | `--workers 3` |
| `--log-level LEVEL` | Set logging level | `--log-level DEBUG` |

### Utilities

| Command | Description |
|---------|-------------|
| `--create-sample` | Create sample CSV file |
| `--version` | Show version information |

## 📊 CSV Data Format

WeatherCache processes CSV files with the following structure:

### Required Columns
- `city`: City name (with optional country code)
- `date`: Date in YYYY-MM-DD format

### Optional Columns
- `time`: Time in HH:MM:SS format
- `region`: Geographic region (for grouping)

### Example CSV
```csv
date,time,city,region
2025-08-05,09:00:00,"London,GB",Europe
2025-08-05,12:00:00,"Paris,FR",Europe
2025-08-05,15:00:00,"Tokyo,JP",Asia
2025-08-05,18:00:00,"New York,US",North America
2025-08-06,09:00:00,"Sydney,AU",Oceania
```

## 🎨 Generated Visualizations

WeatherCache automatically generates research-grade visualizations:

### Individual City Analysis
- **Statistical Distribution**: Temperature histograms with normality indicators
- **Time Period Comparison**: Box plots by time of day
- **Correlation Heatmap**: Weather parameter relationships
- **Trend Analysis**: Moving averages and trend lines
- **Weather Conditions**: Pie charts of condition frequency
- **Statistical Summary**: Comprehensive data table

### Multi-City Comparisons
- **Climate Heatmaps**: Temperature and humidity comparisons
- **Regional Analysis**: Geographic weather patterns
- **Comparative Statistics**: Cross-city performance metrics

## 🔧 Configuration Options

### Environment Variables (.env file)

```bash
# Required
OPENWEATHER_API_KEY=your_api_key_here

# Optional - Cache Configuration
WEATHER_CACHE_DIR=cache
WEATHER_DATA_DIR=weather_data
WEATHER_CACHE_TTL=86400

# Optional - Performance
MAX_WORKERS=5
LOG_LEVEL=INFO

# Optional - Default Files
DEFAULT_INPUT_CSV=cities.csv
```

### Cache TTL (Time-To-Live)
- `3600` = 1 hour
- `21600` = 6 hours  
- `86400` = 1 day (default)
- `604800` = 1 week

## 📚 Usage Examples

### Basic Examples

```bash
# Single city, current weather
python weathercache.py --city "London"

# Single city with specific date
python weathercache.py --city "Tokyo,JP" --date "2025-08-05"

# Single city with date and time
python weathercache.py --city "Paris,FR" --date "2025-08-05" --time "14:30:00"

# Process CSV file
python weathercache.py --csv my_cities.csv

# Process without caching
python weathercache.py --csv my_cities.csv --no-cache
```

### Advanced Examples

```bash
# High-performance processing
python weathercache.py --csv large_dataset.csv --workers 8 --log-level DEBUG

# Custom directories
python weathercache.py --csv data.csv --cache-dir ./my_cache --data-dir ./my_results

# Analysis workflow
python weathercache.py --csv cities.csv
python weathercache.py --analyze --export-report
python weathercache.py --stats
```

### Cache Management

```bash
# View cache statistics
python weathercache.py --stats

# Clear old cache (older than 2 days)
python weathercache.py --clear-cache --older-than 48

# Optimize cache (remove duplicates/outdated)
python weathercache.py --optimize-cache

# Backup cache
python weathercache.py --backup-cache --backup-dir ./backups
```

## 🔍 Data Analysis Features

### Statistical Analysis
- **Descriptive Statistics**: Mean, median, standard deviation, skewness, kurtosis
- **Distribution Analysis**: Histograms with normality tests
- **Correlation Analysis**: Pearson correlations between weather parameters
- **Trend Detection**: Linear regression and moving averages
- **Time Series Analysis**: Patterns by time periods

### Research-Grade Outputs
- **Publication-Ready Charts**: High-DPI (300 DPI) visualizations
- **Statistical Summaries**: Comprehensive data tables
- **JSON Reports**: Machine-readable analysis results
- **CSV Exports**: Processed data for further analysis

## 🚨 Troubleshooting

### Common Issues

**API Key Problems**
```bash
# Validate your API key
python weathercache.py --validate-key

# Reset environment
python weathercache.py --set-env
```

**Permission Errors**
```bash
# Use different directories
python weathercache.py --cache-dir ./temp_cache --data-dir ./temp_data
```

**Large Dataset Processing**
```bash
# Reduce workers if experiencing timeouts
python weathercache.py --csv large_file.csv --workers 2

# Enable debug logging
python weathercache.py --csv large_file.csv --log-level DEBUG
```

**Cache Issues**
```bash
# Clear and rebuild cache
python weathercache.py --clear-cache
python weathercache.py --csv your_data.csv
```

### Error Messages

| Error | Solution |
|-------|----------|
| "API key required" | Run `--set-env` or use `--api-key` |
| "CSV file not found" | Check file path and permissions |
| "Invalid API key" | Verify key at openweathermap.org |
| "Network timeout" | Check internet connection, reduce `--workers` |

## 📈 Performance Tips

1. **Use Caching**: Leave caching enabled for repeat requests
2. **Optimize Workers**: Start with 3-5 workers, adjust based on network
3. **Batch Processing**: Process multiple cities in single CSV file
4. **Cache Management**: Run `--optimize-cache` periodically
5. **Large Datasets**: Process in smaller chunks if experiencing issues

## 🌍 API Limitations

**OpenWeatherMap Free Tier:**
- 60 calls/minute
- 1,000 calls/day
- Current weather only (no historical data)

**WeatherCache Optimizations:**
- Smart caching reduces API calls
- Rate limiting prevents API throttling
- Concurrent requests with safe limits
- Cache hit rates typically >80% for repeat runs

## 📄 Output Files

### Generated Files
- `{city}_weather_data.csv`: Individual city data accumulation
- `processing_report_{timestamp}.csv`: Batch processing results
- `weather_summary_report_{timestamp}.json`: Analysis report
- `{city}_statistical_analysis.png`: Individual city charts
- `climate_comparison_heatmap.png`: Multi-city comparison

### CSV Output Columns
- Date/time information: `date`, `time`
- Location data: `city`, `country`, `latitude`, `longitude`
- Temperature: `temperature`, `feels_like`, `temp_min`, `temp_max`
- Atmospheric: `humidity`, `pressure`, `sea_level`, `ground_level`
- Weather: `weather_main`, `description`, `weather_icon`
- Wind: `wind_speed`, `wind_direction`, `wind_gust`
- Other: `visibility`, `cloudiness`, `sunrise`, `sunset`
- Metadata: `cache_status`, `fetch_timestamp`

## 🤝 Contributing

WeatherCache is designed for extensibility:

1. **Custom Visualizations**: Add new chart types in `WeatherVisualizer`
2. **Data Sources**: Extend API support in `WeatherCache`
3. **Analysis Methods**: Enhance statistical analysis in `WeatherDataAnalyzer`
4. **Export Formats**: Add new output formats

## 📜 License

This tool is provided as-is for educational and research purposes. Please respect OpenWeatherMap's API terms of service.

## 🔗 Links

- [OpenWeatherMap API](https://openweathermap.org/api)
- [API Documentation](https://openweathermap.org/current)
- [Get Free API Key](https://home.openweathermap.org/users/sign_up)
