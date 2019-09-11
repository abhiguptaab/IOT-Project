import pygame
pygame.init()
pygame.mixer.init()
i=0
while(1):
    i=i+1
    sounda= pygame.mixer.Sound("beep.wav")
    sounda.play()