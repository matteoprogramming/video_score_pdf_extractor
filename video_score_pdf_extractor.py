import cv2
import numpy as np


def extract_adjacent_different_images(videopath, min_mean, step_frame):
    print("--------------")
    print("ANLYZING VIDEO")
    print("--------------")
    video = cv2.VideoCapture(videopath)
    if not video.isOpened():
        print("Impossibile to open the video")
        return []
    slides = list()
    ret, prev_frame = video.read()
    slides.append(prev_frame)
    frame_counter = 1
    while ret:
        print(f"Frames processed: {frame_counter} - Slides found: {len(slides)}", end='\r')
        frame_counter += 1
        ret, next_frame = video.read()
        for _ in range(step_frame):
                video.grab()        
        try:
            if ret and next_frame is not None and prev_frame.shape == next_frame.shape:
                diff = cv2.absdiff(prev_frame, next_frame)
                if diff.mean() > min_mean:
                    slides.append(next_frame)
        except cv2.error as e:
            print("Error while calculating the absolute difference:", e)
            continue
        prev_frame = next_frame
    print()
    print("Closing video...")
    print("--------------", end="\n\n")
    video.release()
    return slides


def remove_similar_images(slides, min_mean):
    print("--------------")
    print("Removing garbage")
    print("--------------")
    if len(slides) <= 0:
        print("No images given")
        return []
    unique_slides = list()
    for slide in slides:
        isgood = 1
        for u_slid in unique_slides:
            if u_slid.shape == slide.shape:
                try:
                    diff = cv2.absdiff(slide, u_slid)
                    mean = diff.mean()
                except cv2.error as e:
                    print("Error while calculating the absolute difference:", e)
                    continue
                if mean == 0 or mean < min_mean:
                    isgood *= 0
                    break
            else:
                continue
        if isgood:
            unique_slides.append(slide)
            print(f"Good slides found: {len(unique_slides)}", end="\r")
    print()
    print("--------------", end="\n\n")
    return unique_slides





def auto_crop_image(image, threshold=100):
    _, img_gray = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)
    row_means = np.mean(img_gray, axis=1)
    col_means = np.mean(img_gray, axis=0)
    left = np.argmax(col_means > 0)
    right = np.argmax(col_means[::-1] > 0)
    top = np.argmax(row_means > 0)
    bottom = np.argmax(row_means[::-1] > 0)
    cropped_image = img_gray[top:image.shape[0]-bottom, left:image.shape[1]-right]
    return cropped_image


def crop_staff(image):
    row_means = np.mean(image, axis=1)
    col_means = np.mean(image, axis=0)
    left = np.argmax(col_means < 254)
    right = np.argmax(col_means[::-1] <254)
    top = np.argmax(row_means < 254)
    bottom = np.argmax(row_means[::-1] < 254)
    cropped_image = image[top:image.shape[0]-bottom, left:image.shape[1]-right]
    return cropped_image


def save_images(folder_path, images):
    import os
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    for i, image in enumerate(images):
        path = os.path.join(folder_path, f"slide_{i}.jpg")
        print("[SAVED]",path)
        cv2.imwrite(path, image)


def resize_image(image, new_width):
    height, width = image.shape[:2]
    ratio = new_width / width
    new_height = int(height * ratio)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


##################
#   top border   #
##################
#   #        #   #
#   #        #   #
#   #        #   #
#   #        #   #
#   #        #   #
#   #        #   #
##################
#  down border   #
##################


def create_image(height, width, color=255):
    img = np.full((height, width), color, dtype=np.uint8)
    return img

def normalize_page(page, page_border, page_height, current_height):
    page_image = cv2.vconcat(page)
    lateral_border = create_image(page_image.shape[0], page_border)
    page_image = cv2.hconcat([lateral_border, page_image, lateral_border])
    up_border = create_image(page_border, page_image.shape[1]) 
    page_image = cv2.vconcat([up_border, page_image])
    down_border = create_image(page_height- (current_height+page_border), page_image.shape[1])
    page_image = cv2.vconcat([page_image, down_border])
    return page_image



def create_images_pages(images, page_height=3508, page_width=2480, page_border=120):
    print("--------------")
    print("PAGE CREATION ")
    print("--------------")
    real_width = page_width-(2*page_border)
    real_height = page_height-(2*page_border)
    current_height = 0
    pages = list()
    page = list()
    i = 0
    while i < len(images):
        image = images[i]
        resized_image = resize_image(image, real_width)
        resized_image_height = resized_image.shape[0]
        if current_height+resized_image_height<=real_height:
            page.append(resized_image)
            current_height+=resized_image_height
            i += 1
        else:
            page_image = normalize_page(page, page_border, page_height, current_height)
            pages.append(page_image)
            print(f"Pages created: {len(pages)}", end="\r")
            page = list()
            current_height = 0
    if len(page):
        page_image = normalize_page(page, page_border, page_height, current_height)
        pages.append(page_image)
        print(f"Pages created: {len(pages)}", end="\r")
    print()
    print("--------------", end="\n\n")
    return pages

def save_images(folder_path, images):
    import os
    print("--------------")
    print("SAVING IMAGES")
    print("--------------")
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    for i, image in enumerate(images):
        path = os.path.join(folder_path, f"slide_{i}.jpg")
        print("[SAVED]",path)
        cv2.imwrite(path, image)
    print("--------------", end="\n\n")


def export_pdf(images, filepath):
    from PIL import Image 
    images = [Image.fromarray(image) for image in images]
    images[0].save(filepath, "PDF" ,resolution=100.0, save_all=True, append_images=images[1:])


def main():
    print("--------------------------")
    print("--------------------------")
    print("|| PDF SCORES EXTRACTOR ||")
    print("--------------------------")
    print("--------------------------")
    print()
    video_path = input("Enter the path of the video>")
    filename = input("Enter the output PDF filename>")
    print("Pay attention! If the result is not good, modify the indicators in the script")
    print()
    step_frame = 50
    min_mean = 5
    slides = extract_adjacent_different_images(video_path, min_mean, step_frame)
    unique_images = remove_similar_images(slides, min_mean)
    staffs = [crop_staff(auto_crop_image(slide)) for slide in unique_images][1:] # put 1 instead of 0 if there is a title
    pages = create_images_pages(staffs)
    export_pdf(pages, filename)
    

if __name__ == "__main__":

    main()