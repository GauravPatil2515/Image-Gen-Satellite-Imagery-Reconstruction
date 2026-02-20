# Satellite Imagery Reconstruction

GAN-based enhancement of low-resolution satellite imagery. Generates unique satellite terrain images from text descriptions using procedural terrain generation with optional AI-powered image synthesis.

## GitHub

[https://github.com/GauravPatil2515/Image-Gen-Satellite-Imagery-Reconstruction](https://github.com/GauravPatil2515/Image-Gen-Satellite-Imagery-Reconstruction)

## Demo

[https://your-app.onrender.com](https://your-app.onrender.com)

## Deploy on Render

1. Go to [render.com](https://render.com) → **New Web Service**
2. Connect repo: `https://github.com/GauravPatil2515/Image-Gen-Satellite-Imagery-Reconstruction.git`
4. Configure:
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
5. Add environment variables (optional, see below)
6. Click **Deploy**

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `PORT` | Auto-set | Server port (set by Render automatically) |
| `HUGGINGFACE_API_KEY` | Optional | HuggingFace Inference API for AI satellite image generation |
| `GROQ_API_KEY` | Optional | Groq API for AI-powered terrain analysis |
| `OPENROUTER_API_KEY` | Optional | OpenRouter API for AI-powered terrain analysis |

All AI keys are optional. The app works fully without them using local procedural GAN-style generation.

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8000

## How It Works

1. Enter a terrain description (e.g. "coastal city", "mountain forest", "desert oasis")
2. Click **Reconstruct**
3. The system shows a 3-stage pipeline: Low-Res → GAN Enhanced → Super-Resolution
4. Metrics (Water %, Land %, PSNR, SSIM) and terrain analysis are displayed
5. Each generation produces a unique image (time-seeded randomness)

## API

```
POST /solve
Content-Type: application/json

{"data": "coastal city with harbor"}

Response: {"output": "data:image/png;base64,...", "lowres": "...", "midres": "...", "analysis": "...", "method": "gan", "prompt": "coastal city with harbor"}
```
