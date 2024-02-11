import cv2
import pytesseract

# Path to the Tesseract executable (For me it's linux)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def extract_text_from_image(image_path):

    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using histogram equalization
    enhanced_image = cv2.equalizeHist(gray_image)

    # Apply adaptive thresholding to handle variations in lighting and enhance text visibility
    _, thresholded_image = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Gaussian blur to reduce noise and enhance text visibility
    blurred_image = cv2.GaussianBlur(thresholded_image, (5, 5), 0)

    # Apply morphological operations to further enhance text visibility
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_image = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel)

    # Use adaptive thresholding again on the morphological result
    _, final_thresholded_image = cv2.threshold(morph_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the final thresholded image
    contours, _ = cv2.findContours(final_thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through each contour and extract text using Tesseract
    extracted_text = ""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = final_thresholded_image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config='--psm 6')
        extracted_text += text + "\n"

    return extracted_text

if __name__ == "__main__":
    # Path to image file
    image_path = './sample-image-2.png'

    extracted_text = extract_text_from_image(image_path)

    print("Extracted Text:")
    print(extracted_text)
