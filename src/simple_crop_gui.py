import os, sys
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
    drag : bool
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
        self.cropper = [(50, 50), (100, 100)]
        self.drag = False
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
        top_left, bottom_right = self.cropper
        x1, y1 = top_left
        x2, y2 = bottom_right
        return pygame.Rect(x1, y1, x2-x1, y2-y1)
    
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
            crop_rect.set_alpha(150)
            self.surface.blit(crop_rect, (x1, y1))
            
            # drawing top-left and bottom-right circles
            for coord in self.cropper:
                pygame.draw.circle(self.surface, color=(255, 0, 0), 
                                   center=coord, radius=8)
                pygame.draw.circle(self.surface, color=(0, 0, 0), 
                                   center=coord, radius=8, width=3)
    
    def create_cropper(self, click_pos: Tuple[int, int]) -> None:
        """
        Creates 0 x 0 cropper rectangle at the point of click

        Args:
            click_pos (Tuple[int, int]): Coordinate position of the initial 
                mouse-down action.
        """
        self.cropper = [click_pos] * 2
    
    def move_cropper():
        
        raise NotImplementedError
     
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
                    
                    if event.key == pygame.K_RETURN:
                        top_left, bottom_right = self.cropper
                        new_img_arr = simple_crop(self.img_arr, top_left, 
                                              bottom_right)
                        new_img = Image.fromarray(new_img_arr)
                        new_img.show()
                        pygame.quit()
                        sys.exit()
                    
                    elif event.key == pygame.K_ESCAPE:
                        self.cropper = None
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    
                    # create cropper if it doesn't exist yet
                    if not self.cropper and event.button == 1:
                        self.create_cropper(self.mouse_pos)
                    
                    # activate corner dragging
                    elif self.mouse_distance(self.cropper[0]) <= 8:
                        self.drag_corner1 = True
                    elif self.mouse_distance(self.cropper[1]) <= 8:
                        self.drag_corner2 = True
                    
                    # drag rectangle
                    elif self.cropper_rect.collidepoint(self.mouse_pos):
                        self.drag = True
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    
                    # activate corner dragging
                    if self.drag_corner1:
                        self.drag_corner1 = False
                    elif self.drag_corner2:
                        self.drag_corner2 = False

            # drag corners
            if self.drag_corner1:
                self.cropper[0] = self.mouse_pos
            if self.drag_corner2:
                self.cropper[1] = self.mouse_pos
            
            
            # updating display
            self.draw_window()
            pygame.display.update()
            self.clock.tick(24)
            print(self.drag_corner1, self.drag_corner2)
            print(self.cropper)


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