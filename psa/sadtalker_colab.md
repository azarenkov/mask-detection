# Generating the PSA video with SadTalker (Google Colab, GPU)

1. Runtime → Change runtime type → **T4 GPU**.
2. Run:

```bash
!git clone https://github.com/OpenTalker/SadTalker.git
%cd SadTalker
!pip install -r requirements.txt
!bash scripts/download_models.sh
```

3. Upload `portrait.jpg` (participant's photo, with consent) and `speech.wav` (recorded PSA script).
4. Generate:

```bash
!python inference.py --driven_audio /content/speech.wav --source_image /content/portrait.jpg \
    --result_dir /content/psa_out --still --preprocess full --enhancer gfpgan
```

5. Download the `.mp4` from `/content/psa_out`, rename it to `psa_video.mp4`,
   put it into `MaskVision/assets/` and restart `app.py`.

## Example script (~30 s)

> Hi! Masks help stop the spread of respiratory infections. Wear your mask so it covers both
> your nose and your chin, keep it snug at the sides, and change it when it gets damp.
> Our AI system, MaskVision, checks whether a mask is worn — but only you can wear it right. Stay safe!
