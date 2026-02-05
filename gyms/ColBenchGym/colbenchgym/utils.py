"""
Utility functions for HTML webpage rendering and processing.
"""

import os
import re
import time
import traceback
from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.firefox.service import Service


def replace_urls(text):
    """Replace image URLs with standardized format."""
    # Regular expression to find the URLs
    pattern = r"https://source\.unsplash\.com/random/(\d+)x(\d+)/\?[\w=]+"

    # Function to replace each match with the new URL format
    def replace_match(match):
        width, height = match.groups()
        return f"https://picsum.photos/id/48/{width}/{height}"

    # Use re.sub to replace all occurrences in the text
    new_text = re.sub(pattern, replace_match, text)

    # Make sure that the new text has id 48 for all images
    # Define the regex pattern to match the URLs
    pattern = r"https://picsum\.photos/(\d+)/(\d+)"

    # Define the replacement pattern
    replacement = r"https://picsum.photos/id/48/\1/\2"

    # Use re.sub to replace all matches in the paragraph
    new_text = re.sub(pattern, replacement, new_text)

    return new_text


def get_driver():
    """Get Firefox webdriver for rendering HTML."""
    options = FirefoxOptions()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)
    return driver


def render_full_html(driver, html_snippet, temp_path, env_id=0):
    """Render HTML snippet to PNG image."""
    current_time = time.time()

    html_file_path = os.path.join(temp_path, f"{env_id}_{current_time}.html")
    image_path = os.path.join(temp_path, f"{env_id}_{current_time}.png")

    # Ensure temp directory exists
    os.makedirs(temp_path, exist_ok=True)

    # Save the HTML snippet to a temporary file
    with open(html_file_path, "w") as file:
        file.write(html_snippet)

    try:
        # Open the local HTML file
        driver.get(f"file://{html_file_path}")
        driver.get_full_page_screenshot_as_file(image_path)

        os.remove(html_file_path)
        return image_path
    except Exception as e:
        print(e)
        traceback.print_exc()
        if os.path.exists(html_file_path):
            os.remove(html_file_path)
        return None


def extract_html_snippet(paragraph):
    """Extract HTML snippet from text."""
    # Regular expression pattern to match the entire HTML content
    paragraph = replace_urls(paragraph)
    html_pattern = r"<html.*?>.*?</html>"

    # Search for the HTML snippet in the paragraph
    match = re.search(html_pattern, paragraph, re.DOTALL)

    if match:
        return paragraph.replace(match.group(0), "[SEE RENDERED HTML]"), match.group(0)
    else:
        html_pattern = r"<body.*?>.*?</body>"
        match = re.search(html_pattern, paragraph, re.DOTALL)
        if match:
            return paragraph.replace(match.group(0), "[SEE RENDERED HTML]"), match.group(0)
        else:
            return paragraph, None


def encode_image(image_path, gpt_client=False):
    """Encode image to base64."""
    import base64
    with open(image_path, "rb") as image_file:
        if gpt_client:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_image}"
        else:
            return f"data:image;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
