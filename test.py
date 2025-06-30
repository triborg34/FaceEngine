from ultralytics import solutions

app = solutions.SearchApp(
    
    # data = "path/to/img/directory" # Optional, build search engine with your own images
    device="cpu"  # configure the device for processing i.e "cpu" or "cuda"
)

app.run(debug=False)  # You can also use `debug=True` argument for testing