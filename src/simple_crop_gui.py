"""
This is a GUI implementation of the simple cropping algorithm. Upon clicking 
and dragging, a rectangle will show. This rectangle can be dragged and resized. 
To erase the current rectangle, press 'ESC'. When you are satisfied with the 
image, click 'RETURN'. This will close pygame and return the cropped image as 
opened by PIL.
"""

import os, sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional
from simple_crop import simple_crop
import click


class CropApp:
    """
    Class for a GUI-based image cropper
    """
    filename : str
    img_arr : np.ndarray
    cropper : Optional[List[Tuple[int, int]]]
    drag : Optional[Tuple[int, int]]
    drag_corner1 : bool
    drag_corner2 : bool

    def __init__(self, filename: str) -> None:
        """
        Constructor

        Args:
            filename: image file name
        """
        self.filename = filename
        with Image.open(filename) as img:
            self.img_arr = np.array(img)
        self.cropper = None
        self.drag = None
        self.drag_corner1 = False
        self.drag_corner2 = False

        # Initialize Pygame
        pygame.init()

        # Set window title
        pygame.display.set_caption("Simple Image Cropper")

        # Set window size
        self.surface = pygame.display.set_mode(img.size)
        self.clock = pygame.time.Clock()

        # event loop
        self.event_loop()

    @property
    def mouse_pos(self) -> Tuple[int, int]:
        """
        position of mouse
        """
        return pygame.mouse.get_pos()

    @property
    def cropper_rect(self) -> pygame.Rect:
        """
        pygame rectangle object representation of cropper
        """
        if self.cropper is None:
            raise ValueError("Cropper is None")

        top_left, bottom_right = self.cropper
        x1, y1 = top_left
        x2, y2 = bottom_right
        return pygame.Rect(x1, y1, x2-x1, y2-y1)
    
    @property
    def radius(self) -> int:
        """
        radius of cropper corner circles
        """
        return 6 + min(self.img_arr.shape[:2]) // 200

    def draw_window(self) -> None:
        """
        Draws contents of the window
        """
        displayImage = pygame.image.load(self.filename)
        self.surface.blit(displayImage, (0, 0))

        if self.cropper:
            top_left, bottom_right = self.cropper
            x1, y1 = top_left
            x2, y2 = bottom_right

            # drawing translucent crop rectangle
            crop_rect = pygame.Surface((x2-x1, y2-y1))
            crop_rect.fill((200, 200, 200))
            crop_rect.set_alpha(180)
            self.surface.blit(crop_rect, (x1, y1))

            # drawing top-left and bottom-right circles
            for coord in self.cropper:
                pygame.draw.circle(self.surface, color=(255, 0, 0),
                                   center=coord, radius=self.radius)
                pygame.draw.circle(self.surface, color=(0, 0, 0),
                                   center=coord, radius=self.radius, width=3)

            # displaying cropped image dimensions
            font = pygame.font.Font(None, size=40)
            text = font.render(f"{x2-x1} x {y2-y1}", True, 
                               (255, 255, 255), (0, 0, 0))
            text.set_alpha(100)
            text_rect = text.get_rect()
            text_rect.bottomleft = (0, self.img_arr.shape[0])
            self.surface.blit(text, dest=text_rect)
        
    def create_cropper(self) -> None:
        """
        Creates 1 x 1 cropper rectangle with its bottom-corner at the point of 
            the click to allow for dragging.

        Args:
            click_pos (Tuple[int, int]): Coordinate position of the initial
                mouse-down action.
        """
        x, y = self.mouse_pos
        h, w, _ = self.img_arr.shape

        if 2 <= x <= w - 2 and 2 <= y <= h - 2:
            self.cropper = [(x-1, y-1), self.mouse_pos]
            self.drag_corner2 = True

    def update_cropper(self) -> None:
        """
        Updates dimesions and position of cropper to change its size while
            making sure that it stays within the window bounds and that
            the top-left and bottom-right corners don't overlap.
        """
        if self.cropper is None:
            raise ValueError("Cropper is None")

        top_left, bottom_right = self.cropper
        x, y = self.mouse_pos

        # dragging cropper rectangle
        if self.drag:

            dx = x - self.drag[0]
            dy = y - self.drag[1]
            x1 = top_left[0] + dx
            y1 = top_left[1] + dy
            x2 = bottom_right[0] + dx
            y2 = bottom_right[1] + dy

            if x1 < 0:
                x1, x2 = 0, bottom_right[0] - top_left[0]
            if y1 < 0:
                y1, y2 = 0, bottom_right[1] - top_left[1]
            if x2 > self.img_arr.shape[1] - 1:
                x1 = self.img_arr.shape[1] - 1 - (bottom_right[0] - top_left[0])
                x2 = self.img_arr.shape[1] - 1
            if y2 > self.img_arr.shape[0] - 1:
                y1 = self.img_arr.shape[0] - 1 - (bottom_right[1] - top_left[1])
                y2 = self.img_arr.shape[0] - 1

            self.cropper = [(x1, y1), (x2, y2)]
            self.drag = self.mouse_pos

        # dragging top-left corner
        if self.drag_corner1:

            if x < 0:
                corner_x = 0
            elif x > bottom_right[0] - 1:
                corner_x = bottom_right[0] - 1
            else:
                corner_x = x

            if y < 0:
                corner_y = 0
            elif y > bottom_right[1] - 1:
                corner_y = bottom_right[1] - 1
            else:
                corner_y = y

            self.cropper[0] = (corner_x, corner_y)

        # dragging bottom-right corner
        if self.drag_corner2:

            if x < top_left[0] + 1:
                corner_x = top_left[0] + 1
            elif x > self.img_arr.shape[1] - 1:
                corner_x = self.img_arr.shape[1] - 1
            else:
                corner_x = x

            if y < top_left[1] + 1:
                corner_y = top_left[1] + 1
            elif y > self.img_arr.shape[0] - 1:
                corner_y = self.img_arr.shape[0] - 1
            else:
                corner_y = y

            self.cropper[1] = (corner_x, corner_y)

    def mouse_distance(self, p: Tuple[int, int]) -> float:
        """
        Calculates the distance between two points

        Inputs:
            p1, p2 (Tuple(int)): pixel coordinates of the two points

        Returns (float): pixelwise distance between the two points
        """
        x, y = self.mouse_pos
        px, py = p
        return ((x - px) ** 2 + (y - py) ** 2) ** 0.5

    def event_loop(self) -> None:
        """
        Handles user interactions

        Parameters: none beyond self

        Returns: nothing
        """
        while True:
            
            # process pygame events
            events = pygame.event.get()
            for event in events:

                # quitting program
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYUP:

                    # return cropped image if cropper is activated
                    if event.key == pygame.K_RETURN:
                        if self.cropper:
                            top_left, bottom_right = self.cropper
                            new_img_arr = simple_crop(self.img_arr, top_left,
                                                bottom_right)
                            new_img = Image.fromarray(new_img_arr)
                            new_img.show()
                            pygame.quit()
                            sys.exit()
                    
                    # delete current cropper
                    elif event.key == pygame.K_ESCAPE:
                        self.cropper = None

                elif event.type == pygame.MOUSEBUTTONDOWN:

                    # create cropper if it doesn't exist yet
                    if not self.cropper and event.button == 1:
                        self.create_cropper()

                    # activate corner dragging
                    elif self.mouse_distance(self.cropper[0]) <= self.radius:
                        self.drag_corner1 = True
                    elif self.mouse_distance(self.cropper[1]) <= self.radius:
                        self.drag_corner2 = True

                    # drag rectangle
                    elif self.cropper_rect.collidepoint(self.mouse_pos):
                        self.drag = self.mouse_pos

                    # recreate new cropper if click is outside original cropper
                    else:
                        self.create_cropper()

                elif event.type == pygame.MOUSEBUTTONUP:

                    # deactivate dragging
                    if self.drag:
                        self.drag = None

                    # deactivate corner dragging
                    if self.drag_corner1:
                        self.drag_corner1 = False
                    elif self.drag_corner2:
                        self.drag_corner2 = False

                elif event.type == pygame.MOUSEMOTION:

                    # update cropper
                    if self.cropper:
                        self.update_cropper()

            # updating display
            self.draw_window()
            pygame.display.update()
            self.clock.tick(24)


# click commands
@click.command(name="simple_crop_gui")
@click.option('-f', '--filename', type=click.Path(exists=True))

def cmd(filename: str) -> None:
    """
    command line operation
    """
    CropApp(filename)

if __name__ == "__main__":
    cmd()
