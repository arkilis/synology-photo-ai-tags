export PATH=/usr/local/bin:$PATH

/volume1/homes/ben/scripts/synology-photo-ai-tags/.synology-photo-ai-tags/bin/python -m src \
  --model gemini-2.5-flash \
  --requests-per-minute 8 \
  --batch-size 5 \
  --max-files-per-run 220 \
  --progress /var/services/homes/ben/Photos/.ai-tags-progress.json \
  --root ~/Photos
  # --root /Volumes/homes/ben/Photos/MobileBackup/iPhone/2026/test
