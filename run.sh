export PATH=/usr/local/bin:$PATH

# /volume1/homes/ben/scripts/synology-photo-ai-tags/.synology-photo-ai-tags/bin/python -m src \
#   --backend gemini \
#   --model gemini-2.5-flash \
#   --requests-per-minute 8 \
#   --batch-size 5 \
#   --max-files-per-run 220 \
#   --progress /var/services/homes/ben/Photos/.ai-tags-progress.json \
#   --root ~/Photos
#   # --root /Volumes/homes/ben/Photos/MobileBackup/iPhone/2026/test

# macOS + Ollama example:
python3 -m src \
  --backend ollama \
  --model qwen2.5vl:7b \
  --ollama-host http://localhost:11434 \
  --batch-size 1 \
  --requests-per-minute 30 \
  --request-timeout 300 \
  --progress /Volumes/homes/ben/Photos/.ai-tags-progress.json \
  --root /Volumes/homes/ben/Photos/
