import cv2
import numpy as np

def find_pixel_center_and_yaw(png_name):
    #load the image...
    image = cv2.imread(png_name)
    # Define the lower and upper BGR range for red color
    lower_red = np.array([0, 0, 200])
    upper_red = np.array([50, 50, 255])

    #create a mask to extract the red regions
    red_mask = cv2.inRange(image, lower_red, upper_red)

    #find contours in the red regions
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #calculate the areas of all detected ovals
    areas = [cv2.contourArea(contour) for contour in contours]

    #get indices of the two largest areas
    indices = np.argsort(areas)[-2:]

    #draw the two largest ovals in green and calculate their centers
    green_centers = []
    for i in indices:
        ellipse = cv2.fitEllipse(contours[i])
        cv2.ellipse(image, ellipse, (0, 255, 0), 2)  #draw green oval
        green_centers.append(tuple(map(int, ellipse[0])))

    #calculate the midpoint between the centers of the green circles
    midpoint_x = (green_centers[0][0] + green_centers[1][0]) // 2
    midpoint_y = (green_centers[0][1] + green_centers[1][1]) // 2

    mid_pt = np.array([midpoint_x, midpoint_y])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 20, 0.01, 10)
    corners = np.intp(corners)

    coordinates_2d = corners.reshape(-1, 2)

    four_corner_dists = []
    for i in coordinates_2d:
        four_corner_dists.append(np.linalg.norm(mid_pt - i))

    #sorts list in descending order
    four_corner_dists.sort(reverse=True)

    #check if the image was successfully loaded
    if image is not None:
        #gets height and width of the image
        image_height, image_width, channels = image.shape
    else:
        print("Image not found or could not be loaded.")

    #the coordinates of the four corners of the PNG
    png_corners = np.array([
        [0, 0],                      # Top-left corner
        [0, image_height],           # Bottom-left corner
        [image_width, 0],            # Top-right corner
        [image_width, image_height]  # Bottom-right corner
    ])

    #create dictionary to store closest points
    closest_points = {}

    # Calculate distances for each PNG corner
    for i, png_corner in enumerate(png_corners):
        #calculat distances between the PNG corner and all points in the 'corners' array
        distances = np.sqrt((coordinates_2d[:, 0] - png_corner[0]) ** 2 + (coordinates_2d[:, 1] - png_corner[1]) ** 2)

        #finds index of the closest and furthest points
        closest_index = np.argmin(distances)

        #stores closest and furthest points in the dictionaries
        closest_points[f'Corner {i + 1}'] = corners[closest_index]

    #print("Closest Points to respective corner:")
    four_points_list = []
    for corner, point in closest_points.items():
        four_points_list.append((point[0][0], point[0][1]))

    x2, y2 = (four_points_list[0])
    x4, y4 = (four_points_list[3])
    x3, y3 = (four_points_list[1])
    x1, y1 = (four_points_list[2])

    #algebra for the intersection point of the two lines
    A1 = y3 - y1
    B1 = x1 - x3
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y2
    B2 = x2 - x4
    C2 = A2 * x2 + B2 * y2
    det = A1 * B2 - A2 * B1
    if det != 0:
        intersection_x = (B2 * C1 - B1 * C2) / det
        intersection_y = (A1 * C2 - A2 * C1) / det

        #intersecton coord to integer
        intersection_x = int(intersection_x)
        intersection_y = int(intersection_y)


    #calculate the slope of one line
    m1 = (y4 - y3) / (x4 - x3)

    #calculates the slope of other line
    m2 = (y2- y1) / (x2 - x1)

    #calculates relative orientation angle in radians
    delta_theta = np.arctan2(m2 - m1, 1 + m1 * m2)
    #converts result from radians to degrees
    delta_theta_degrees = -np.degrees(delta_theta)
    #relative YawOrientationAngle: {delta_theta_degrees} degrees
    print(intersection_x, intersection_y, int(delta_theta_degrees//1))


find_pixel_center_and_yaw('torpBoard1.png')