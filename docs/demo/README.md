# Product demo video

`image-captioning-demo.mp4` is the 23-second technical demo embedded at the top of the repository README:
1920×1080, 30 fps, H.264 + AAC, about 3.2 MB. `image-captioning-demo.jpg` is its poster, and is also the video's
first frame, so players and link previews show the product rather than a blank intro.

## What it shows

1. **Hook.** The SPA's own headline, "Describe any image in natural language", and the project name.
2. **One live request.** A photo is dropped into Caption Studio, *Generate caption* is clicked, and the generated
   caption card lands.
3. **Behind the caption.** `POST /v1/captions` → FastAPI → InceptionV3 encoder → Transformer decoder → caption,
   served by model `v2.0.0` with weights pinned on Hugging Face Hub and loaded once at startup.
4. **Research to production.** The parity audit against the IEEE notebook, the pre-registered comparison with
   BLIP, ViT-GPT2 and GIT, and the post-deploy caption smoke test.

## What is real

- **The interface** is recreated from the SPA source (`frontend/src/App.jsx` and `frontend/src/components/`), with
  its copy, layout and Tailwind colours. Secondary text is brighter than in the app so it stays legible at video
  size.
- **The caption and its metadata** are one unedited response from the deployed Space, captured on
  2026-10-10 at 13:59:34 UTC:

  ```json
  {"caption": "a man riding a wave on top of a surfboard", "model_version": "v2.0.0",
   "decode_strategy": "greedy", "latency_ms": 910.63, "request_id": "fdiZ1i"}
  ```

  The first request after the Space had idled returned the same caption with 6856.28 ms latency; the warm repeat
  is the one shown.
- **The evidence lines** are real outputs: `[OK] parity audit: 4/4 checks passed` from
  `scripts/notebook_module_audit.py`, the comparison protocol in `docs/EVAL_METHODOLOGY.md` § 8, and the smoke
  step of deploy run `37830915847` (HTTP 200 from model `v2.0.0`).
- **No comparison scores are shown.** The Phase 3 comparison isn't a held-out, like-for-like comparison
  (`docs/EVAL_METHODOLOGY.md` § 8.5), so the video states only that it exists and how it is controlled.

## Credits

- Photo: "Surfer surfing the wave (Unsplash).jpg" by Austin Neill, CC0 1.0, via
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Surfer_surfing_the_wave_(Unsplash).jpg).
  Uploaded as `surfer.jpg` (1280×853, 226,530 bytes).
- Music: "Happy Beats / Business Moves" Vol. 12. Music by Sascha Ende at [ende.app](https://ende.app/en),
  [CC BY 4.0](https://ende.app/en/standard-license).
- Sound effects: [Kenney](https://kenney.nl/), CC0.
- Typefaces: Geist and Geist Mono, SIL Open Font License.

## How it was made

An HTML composition of the scenes above, rendered to MP4 with [HyperFrames](https://github.com/heygen-com/hyperframes)
(headless Chrome and FFmpeg). The composition workspace is regenerable and gitignored. This copy is re-encoded
at CRF 24 to stay under the repository's 5 MB file limit.

## Inline playback on GitHub

GitHub plays a video inline only when it was uploaded through its web editor (a
`github.com/user-attachments/assets/...` URL). It does not play a file stored in the repository, so the README
shows the poster linked to this file. To switch the README to an inline player, edit `README.md` on github.com,
drag `image-captioning-demo.mp4` into the editor (the upload limit is 10 MB), and replace the poster link with the
URL GitHub inserts.
