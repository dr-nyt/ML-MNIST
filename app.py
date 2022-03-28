from pprint import pprint
import cv2
from tensorflow.keras.models import load_model
from pygame import image
from pygame.locals import *
import pygame
import sys
import numpy as np

WINDOW_SIZE_X = 640
WINDOW_SIZE_Y = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGE_SAVE = True
PREDICT = True
MODEL = load_model("bestmodel.h5")

LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

BOUNDARY_PADDING = 5

# Initialize pygame
pygame.init()
DISPLAY_SURFACE = pygame.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))
pygame.display.set_caption("MNIST Board")
FONT = pygame.font.Font("freesansbold.ttf", 18)

is_writing = False
x_coords = []
y_coords = []
image_count = 0

while True:
    # Handle events
    for event in pygame.event.get():
        # Close window when cross is clicked
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        # Generate shape as mouse moves
        if event.type == MOUSEMOTION and is_writing:
            x_coord, y_coord = event.pos
            pygame.draw.circle(
                DISPLAY_SURFACE, WHITE,
                (x_coord, y_coord), 4, 0
            )

            x_coords.append(x_coord)
            y_coords.append(y_coord)

        # Start drawing on mouse release
        if event.type == MOUSEBUTTONDOWN:
            is_writing = True

        # Stop drawing on mouse release
        if event.type == MOUSEBUTTONUP:
            is_writing = False
            # Sort x & y coordinates of drawn image
            x_coords = sorted(x_coords)
            y_coords = sorted(y_coords)

            # Generate bounding box around drawn image
            rect_min_x = max(x_coords[0] - BOUNDARY_PADDING, 0)
            rect_max_x = min(x_coords[-1] + BOUNDARY_PADDING, WINDOW_SIZE_X)
            rect_min_y = max(y_coords[0] - BOUNDARY_PADDING, 0)
            rect_max_y = min(y_coords[-1] + BOUNDARY_PADDING, WINDOW_SIZE_Y)
            print("x coord:", rect_min_x, rect_max_x)
            print("y coord:", rect_min_y, rect_max_y)

            # Clear coordinates of drawn image
            x_coords = []
            y_coords = []

            # Get pixel values of the whole window
            image_array = np.array(pygame.PixelArray(DISPLAY_SURFACE))
            pprint(image_array.shape)
            # Isolate the values inside the bounding box
            image_array = image_array[
                rect_min_x:rect_max_x,
                rect_min_y:rect_max_y
            ]
            pprint(image_array.shape)
            # Transpose and convert the values to float
            image_array = image_array.T.astype(np.float32)
            # Add padding around the image
            image = np.pad(
                image_array, (50, 50),
                "constant", constant_values=0
            )
            # Resize image to be 28x28 and normalized
            image = cv2.resize(image, (28, 28)) / 255

            if IMAGE_SAVE:
                cv2.imwrite(f"img/image-{image_count}.png", image_array)
                image_count += 1

            if PREDICT:
                # image = cv2.resize(image_array, (28, 28))
                label = LABELS[
                    np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))
                ]

                text_surface = FONT.render(label, True, RED, WHITE)
                text_rectangle = pygame.Rect(
                    (rect_min_x, rect_min_y), (rect_max_x - rect_min_x, rect_max_y - rect_min_y))
                # text_rectangle.left, text_rectangle.right = rect_min_x, rect_max_y

                DISPLAY_SURFACE.blit(
                    text_surface, (rect_min_x, max(rect_min_y - 18, 0)))

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAY_SURFACE.fill(BLACK)

    pygame.display.update()
