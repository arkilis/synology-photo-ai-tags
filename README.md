# synology-photo-ai-tags

A script that automatically tags photos and writes the results back into photo metadata.

The current implementation uses the Gemini Vision API to generate, for each photo:

- Chinese tags
- English tags
- OCR text detected in the image
- Chinese and English search terms derived from the OCR text

This metadata is then written back to the photo so it can be searched directly in Synology Photos using Chinese, English, and text that appears inside the image.

## Features

- Recursively scans a photo directory
- Supports common formats such as JPG, JPEG, PNG, HEIC, TIFF, WEBP, and RAW
- Writes metadata directly into regular image files
- Writes `.xmp` sidecar files for RAW images
- Extracts visible text from images and stores it in searchable fields
- Generates both Chinese and English keywords
- Supports resumable runs via `progress.json`
- Uses the file `path` to determine whether a file has already been processed
- Creates a backup copy as `original_filename_BAK` before modifying a regular image
- Restores the original `modified date/time` after writing metadata to a regular image
- Can wait for a NAS mount to become available before starting
- Supports limiting how many new files are processed per run to help control API usage

## Current Write Strategy

### Non-RAW Files

Non-RAW files are updated directly in the image metadata.

Before writing, the script creates a backup:

```text
IMG_5032.JPG -> IMG_5032.JPG_BAK
```

The backup is created only once. If it already exists, it will not be overwritten.

### RAW Files

RAW files are never modified directly. The script writes a neighboring `.xmp` sidecar file instead.

## Dependencies

- Python 3.10+ recommended
- `exiftool`
- A Gemini API key

At runtime, the current code only uses the Python standard library. It does **not** require pip packages such as `google-genai`, `httpx`, or `anyio`.

In practice, this means that on a Synology NAS you usually **do not need** to run:

```bash
pip install -r requirements.txt
```

As long as the system has:

- a working Python interpreter
- `exiftool`

the script can run directly.

## Installation

### 1. Python Environment

If you already have the virtual environment included in this repository, you can use it directly.

On Synology NAS, it is usually enough to have `python3` available.

### 2. Install exiftool

On macOS:

```bash
brew install exiftool
```

If the script runs in another environment, make sure `exiftool` is available in `PATH`.

### 3. Configure the API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key
```

You can also add optional settings such as:

```env
GEMINI_MODEL=gemini-2.5-flash
REQUESTS_PER_MINUTE=8
MAX_FILES_PER_RUN=220
WAIT_FOR_ROOT_SECONDS=300
```

## Usage

### Basic Run

```bash
python3 -m src \
  --progress /Volumes/homes/ben/Photos/.ai-tags-progress.json \
  --root /Volumes/homes/ben/Photos/
```

## Running on Synology NAS

This project currently does not require Python third-party packages at runtime.

The minimum setup is usually just:

```bash
python3 --version
exiftool -ver
```

If both work, you can run the project directly from the repository:

```bash
python3 -m src \
  --progress /volume1/photo/.ai-tags-progress.json \
  --root /volume1/photo
```

If you previously ran `pip install -r requirements.txt` and saw an error like:

```text
Could not find a version that satisfies the requirement anyio==4.12.1
```

you can safely ignore it for the current version of this project, because the runtime code does not depend on those pip packages.

### Limiting Free-Tier Usage

```bash
python3 -m src \
  --model gemini-2.5-flash \
  --requests-per-minute 8 \
  --max-files-per-run 220 \
  --wait-for-root-seconds 300 \
  --progress /Volumes/homes/ben/Photos/.ai-tags-progress.json \
  --root /Volumes/homes/ben/Photos/
```

### Test Against a Small Folder

```bash
python3 -m src \
  --progress /Volumes/homes/ben/Photos/.ai-tags-progress.json \
  --root /Volumes/homes/ben/Photos/MobileBackup/iPhone/2026/test
```

### Analyze Only, Without Writing Metadata

```bash
python3 -m src \
  --dry-run \
  --progress /Volumes/homes/ben/Photos/.ai-tags-progress.json \
  --root /Volumes/homes/ben/Photos/
```

## CLI Parameters

- `--root`: photo root directory to scan recursively
- `--progress`: path to the progress file
- `--model`: Gemini model name
- `--requests-per-minute`: client-side rate limit
- `--request-timeout`: request timeout in seconds
- `--max-inline-bytes`: send small images inline; larger ones use the Files API
- `--max-files-per-run`: maximum number of new files to process in a single run
- `--wait-for-root-seconds`: how long to wait for the NAS path to appear
- `--force`: ignore the progress file and reprocess everything
- `--dry-run`: analyze only, without writing metadata or updating progress

## Progress File

`progress.json` or `.ai-tags-progress.json` stores:

- absolute file path
- input image path
- write mode: `embedded` or `xmp`
- generated keywords
- generated description
- update timestamp

The script now uses the file `path` to decide whether a file has already been processed. As long as the same file stays at the same path, it will not be processed again.

## NAS Mount Handling

If the photo directory lives under macOS `/Volumes/...` and the NAS share is not mounted yet, the script can wait before starting.

Default behavior:

- If `--root` is under `/Volumes/...`, the script waits 120 seconds by default
- If the path appears during that time, processing starts automatically
- If the timeout is reached and the path is still unavailable, the script exits with an error

You can override the wait time:

```bash
--wait-for-root-seconds 300
```

## Time and Backup Guarantees

For non-RAW files:

- a `*_BAK` backup is created before writing
- the original `modified date/time` is restored after writing
- if restoration fails, the script raises an error instead of silently treating the write as successful

This ensures that newly processed images do not end up with “today” as their modified time just because metadata was added.

## Known Limitations

- The project currently depends on the Gemini API, so API usage may cost money
- Writing metadata into non-RAW files depends on `exiftool`
- If a file’s timestamp was already changed by an older version of the script, this project does not automatically restore that historical value
- Synology Photos does not index metadata exactly the same way for every file format, so you should validate search behavior on a small sample set first

## Project Files

- `src/config.py`: configuration and CLI argument parsing
- `src/gemini_client.py`: Gemini API integration
- `src/photo_tagger.py`: scanning, skip logic, and progress tracking
- `src/metadata_writer.py`: metadata writing for regular images and backup handling
- `src/xmp_writer.py`: XMP sidecar writing for RAW files
- `run.sh`: current local run example
