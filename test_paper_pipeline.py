from src.paper_pipeline import analyze_video

video_path = "uploads\Screen Recording 2025-12-03 172326.mp4"

result, confidence = analyze_video(video_path)

print("Result:", result)
print("Confidence:", confidence)