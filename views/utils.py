

 # farame count, fps, duration
def format_video_info(farame_count=1, duration=1):
    return """#### Video Information
    - **Frame Count**: {}
    - **Duration**: {} seconds
    - **FPS**: {}""".format(farame_count, duration, round(farame_count/max(duration, 1), 2))


def error_text(text):
    return f"<div style='color: red;'>{text}</div>"

def success_text(text):
    return f"<div style='color: green;'>{text}</div>"