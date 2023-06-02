DATA_DIR="$(dirname $0)"
STEMS="bass drums guitar piano"
OUT_DIR="$DATA_DIR/slakh2100"

for STEM in $STEMS
do
	STEM_DIR="$DATA_DIR/${STEM}_22050"
	
	for SPLIT_DIR in $STEM_DIR/*
	do
		SPLIT=$(basename $SPLIT_DIR)
		
		for TRACK_FILE in "$SPLIT_DIR/"*.wav
		do 
			TRACK=$(basename "$TRACK_FILE" .wav)

			SOURCE_FILE="$TRACK_FILE"
			TARGET_FILE="$OUT_DIR/$SPLIT/$TRACK/$STEM.wav"

			echo "hard-linking file \"${SOURCE_FILE}\" to \"${TARGET_FILE}\""
			
			mkdir -p "$OUT_DIR/$SPLIT/$TRACK"
			ln "${SOURCE_FILE}" "${TARGET_FILE}" 
		done
	done
done
