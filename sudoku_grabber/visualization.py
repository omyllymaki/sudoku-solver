import cv2


def add_points(input_image, points, radius=5, colour=(0, 0, 255)):
    """Draws circular points on an image."""
    image = input_image.copy()

    # Dynamically change to a colour image if necessary
    if len(colour) == 3:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for point in points:
        image = cv2.circle(image, tuple(int(x) for x in point), radius, colour, -1)
    return image


def show_images():
    cv2.waitKey(0)
    cv2.destroyAllWindows()