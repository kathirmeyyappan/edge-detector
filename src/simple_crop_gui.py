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
        
        # Initialize Pygame
        pygame.init()

        # Set window title
        pygame.display.set_caption("Simple Image Cropper")
        
        # Set window size
        self.surface = pygame.display.set_mode(img.size)
        self.clock = pygame.time.Clock()
        
        # event loop
        self.event_loop()
        
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
            
            
            crop_rect = pygame.Surface((x2-x1, y2-y1))
            crop_rect.fill((200, 200, 200))
            crop_rect.set_alpha(150)
            self.surface.blit(crop_rect, (x1, y1))
        
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
        
            # updating display
            self.draw_window()
            pygame.display.update()
            self.clock.tick(24)


# click commands
@click.command(name="simlpe_crop_gui")
@click.option('-f', '--filename', type=click.Path(exists=True))

def cmd(filename: str) -> None:
    """
    command line operation
    """
    CropApp(filename)

if __name__ == "__main__":
    cmd()