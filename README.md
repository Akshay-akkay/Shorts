[![Shorts Thumbnail](https://i.ytimg.com/vi/HBbK-F-c8N8/oar2.jpg?sqp=-oaymwEoCJUDENAFSFqQAgHyq4qpAxcIARUAAIhC2AEB4gEKCBgQAhgGOAFAAQ==&rs=AOn4CLA3l0gi7G1975gZnhpB7M2_QMRfiA)](https://www.youtube.com/shorts/HBbK-F-c8N8)

# Shorts – Automated YouTube Clip Generator

---

📺 Demo

Shorts Demo ↗

Shorts is a **Python-powered toolkit** for rapidly creating short, shareable video clips from YouTube videos. It automates video download, caption parsing, and even suggests the best moments to clip using **Google Gemini’s LLM**. Perfect for content creators, social media managers, and anyone who wants to repurpose long-form video into high-impact shorts.

---

🚀 Key Features
* **YouTube to MP4:** Downloads videos (with captions) in 480p for fast processing.
* **Smart Clipping:** Specify start/end times or let the Gemini LLM suggest the most engaging segments (ideal for Shorts, Reels, TikTok).
* **Auto-Captions:** Parses VTT captions and overlays them as styled subtitles on your clips.
* **LLM Integration:** Uses **Gemini 1.5 Pro** to analyze video and recommend 1–3 highlight moments (API key required).
* **Batch & Custom Modes:** Clip multiple highlights in one go, or extract a specific segment with a single command.

---

🛠️ How It Works
* **Download:** Fetches the YouTube video and English captions using ‎`yt-dlp`‎.
* **Parse Captions:** Converts VTT subtitle files into timed caption segments.
* **Clip & Overlay:** Extracts your chosen segment(s), overlays captions, and exports ready-to-share MP4s.
* **AI Suggestions (Optional):** If enabled, uploads your video to Gemini, which analyzes and returns the best clip timings.

---

⚡ Example Usage

```bash
# Basic: Manual start/end
python clipper.py "[https://youtube.com/](https://youtube.com/)..." myclip.mp4 --start_time 30 --end_time 60

# AI-powered: LLM picks the best moments
export GEMINI_API_KEY=your-key
python clipper.py "[https://youtube.com/](https://youtube.com/)..." highlights.mp4 --use_llm

# Use browser cookies for private videos
python clipper.py "[https://youtube.com/](https://youtube.com/)..." out.mp4 --cookies-from-browser "chrome:Default"
