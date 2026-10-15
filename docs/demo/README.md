# Product demo video

`image-captioning-demo.mp4` is the 27-second technical demo embedded at the top of the repository README:
1920×1080, 30 fps, H.264 + AAC, about 4.3 MB. `image-captioning-demo.jpg` is its poster, and is also the video's
first frame, so players and link previews show the product rather than a blank intro.

## What it shows

1. **Hook.** The SPA's own headline, "Describe any image in natural language", and the project name.
2. **One live comparison, recorded in the deployed app.** The "Kitten on a computer keyboard" sample is picked
   from the *Try an example* gallery, *Compare both* is clicked, and the greedy and beam-search captions land
   one after the other with their measured latencies.
3. **Behind the caption.** `POST /v1/captions` → FastAPI → InceptionV3 encoder → Transformer decoder (greedy or
   beam search) → caption, served by model `v2.0.0` with weights pinned on Hugging Face Hub.
4. **Research to production.** The parity audit against the IEEE notebook, the pre-registered comparison with
   BLIP, ViT-GPT2 and GIT, and the post-deploy caption smoke test.

## What is real

- **The app scene is a screen recording** of <https://image-captioning-system.vercel.app/> in headless Chromium
  (Playwright, 1920×1080 viewport, no request mocking), made on 2026-10-15 at 20:21 UTC. The only additions at
  recording time were a cursor overlay that follows the real pointer events and a hidden scrollbar. The pre-paint
  frames are trimmed and the composition zooms in on the photo and the two captions once the second result lands.
- **The captions and latencies** are the two unedited responses from the deployed Space during that recording:

  ```json
  {"caption": "a cat is sitting on a keyboard of a computer", "model_version": "v2.0.0",
   "decode_strategy": "greedy", "latency_ms": 863.29, "request_id": "_B80Vu"}
  {"caption": "a close up of a cat laying on a keyboard", "model_version": "v2.0.0",
   "decode_strategy": "beam", "latency_ms": 1496.47, "request_id": "X9Y8n7"}
  ```

  A warm-up *Compare both* on the same image a minute earlier returned the same captions at 850.5 ms and
  1474.8 ms. Latencies are from one run on the Space's CPU hardware, not a benchmark.
- **The evidence lines** are real outputs: `[OK] parity audit: 4/4 checks passed` from
  `scripts/notebook_module_audit.py`, the comparison protocol in `docs/EVAL_METHODOLOGY.md` § 8, and the smoke
  step of deploy run `37830915847` (HTTP 200 from model `v2.0.0`).
- **No comparison scores are shown** outside the app's own footnote. The Phase 3 comparison isn't a held-out,
  like-for-like comparison (`docs/EVAL_METHODOLOGY.md` § 8.5), so the video states only that it exists and how it
  is controlled.

## Credits

- Photo: "Mango HaX0ring" by Slava, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), via
  [Flickr](https://www.flickr.com/photos/slava/496607907/); COCO 2017 image 418799, shipped with the SPA as
  `frontend/public/samples/000000418799.jpg` (see `frontend/src/samples.js`). The other four gallery thumbnails
  visible in the recording are credited in the app and in `frontend/src/samples.js`.
- Music: "Happy Beats / Business Moves" Vol. 12. Music by Sascha Ende at [ende.app](https://ende.app/en),
  [CC BY 4.0](https://ende.app/en/standard-license).
- Sound effects: [Kenney](https://kenney.nl/), CC0.
- Typefaces: Geist and Geist Mono, SIL Open Font License.

## How it was made

The app scene was recorded with Playwright (the Chromium already installed for the frontend E2E suite) and
transcoded to H.264 with FFmpeg. It sits inside an HTML composition with the other scenes, rendered to MP4 with
[HyperFrames](https://github.com/heygen-com/hyperframes) (headless Chrome and FFmpeg). The composition workspace
is regenerable and gitignored. This copy is re-encoded at CRF 24 to stay under the repository's 5 MB file limit.

The previous version (2026-10-10) recreated the interface from the SPA source and showed a single greedy
request on an uploaded surfer photo ("Surfer surfing the wave (Unsplash).jpg" by Austin Neill, CC0 1.0).

## Inline playback on GitHub

GitHub plays a video inline only when it was uploaded through its web editor (a
`github.com/user-attachments/assets/...` URL). It does not play a file stored in the repository, so the player in
`README.md` points at such an upload, not at this file. After replacing `image-captioning-demo.mp4`, edit
`README.md` on github.com, drag the new file into the editor (the upload limit is 10 MB), and replace the old
attachment URL with the one GitHub inserts.
