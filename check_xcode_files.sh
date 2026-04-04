#!/bin/bash
# check_xcode_files.sh
# Checks that all .swift files under apple/Application/ are registered in the Xcode project.

PBXPROJ="apple/etLLM.xcodeproj/project.pbxproj"
SOURCE_DIR="apple/Application"
MISSING=0

if [ ! -f "$PBXPROJ" ]; then
  echo "❌ project.pbxproj not found at: $PBXPROJ"
  exit 1
fi

while IFS= read -r -d '' file; do
  name=$(basename "$file")
  if ! grep -q "$name" "$PBXPROJ"; then
    echo "❌ Not in Xcode project: $name  ($file)"
    MISSING=1
  fi
done < <(find "$SOURCE_DIR" -name "*.swift" -print0)

if [ $MISSING -eq 0 ]; then
  echo "✅ All Swift files in $SOURCE_DIR are registered in Xcode project"
fi

exit $MISSING
