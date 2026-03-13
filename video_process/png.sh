mkdir -p all_png

find . -type f -name "*.png" -exec sh -c '
for f do
  dir=$(basename "$(dirname "$f")")
  cp "$f" "all_png/${dir}.png"
done
' sh {} +