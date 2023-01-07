import pygame
from pygame import gfxdraw
import numpy as np
import matplotlib.pyplot as plt

def visualise(sizes, weights, runThread = True):
    # Initialize the game engine
    pygame.init()
 
    # Define the colors we will use in RGB format
    BLACK = (  0,   0,   0)
    WHITE = (255, 255, 255)
 
    # Set the height and width of the screen
    size = [1920, 1080]
    screen = pygame.display.set_mode(size)
 
    pygame.display.set_caption("Neural Network Weights")

    multiplier = 1.5

    #OFFSETS AND SIZES
    offset_x = int(50 * multiplier)
    offset_y = int(40 * multiplier)

    neuron_radius = int(20 * multiplier)
    neuron_border = int(2 * multiplier)
    neuron_offset_x = int(380 * multiplier)
    neuron_offset_y = int(70 * multiplier)

    l_offset = 0

    layerCounter = 0
    neuronCounter = 0

    #Y offset of each layer of neurons.
    y_layer_offsets = []
    for l in range(0, len(sizes)):
        max_neurons = np.max(sizes)
        current_layer = sizes[l]

        max_layer_height = max_neurons * neuron_offset_y
        current_layer_height = current_layer * neuron_offset_y

        offset = (max_layer_height * 0.5) - (current_layer_height * 0.5)

        y_layer_offsets.append(offset)

    clock = pygame.time.Clock()
    counter = 0
    #DRAW LOOP
    while runThread[0]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                runThread = False

        #Width array of each neural synapse.
        line_width_array = []
        max = 0
        min = 0
        for layer in weights:
            if np.max(layer) > max:
                max = np.max(layer)
            if np.min(layer) < min:
                min = np.min(layer)
        for layer in weights:
            layer_weights = ((layer + abs(min)) / (max + abs(min))) * 7
            line_width_array.append(layer_weights)

        #CLEAR SCREEN BEFORE EACH RENDER
        screen.fill((0, 40, 69))

        #WEIGHT RENDER
        for l in range(0, len(sizes)):
            #Iterate over layers
            for n in range(0, sizes[l]):
                if l > 0:
                    #Iterate over weights and render them.
                    for w in range(0, weights[l - 1][n].shape[0]):
                        start_pos = ((l - 1) * neuron_offset_x + offset_x + l_offset, w * neuron_offset_y + offset_y + l_offset + y_layer_offsets[l - 1])
                        end_pos = (l * neuron_offset_x + offset_x + l_offset, n * neuron_offset_y + offset_y + l_offset + y_layer_offsets[l])

                        #pygame.draw.line(screen, (255 - line_width_array[l - 1][n][w], 0, line_width_array[l - 1][n][w]), start_pos, end_pos, 2) #COLOR MODE
                        pygame.draw.line(screen, (210, 210, 210), start_pos, end_pos, int(line_width_array[l - 1][n][w]) + 1) #LINE WIDTH MODE

        #NEURON RENDER
        for l in range(0, len(sizes)):
            #Iterate over layers
            for n in range(0, sizes[l]):
                #Iterate over neurons and draw them
                neuron_x = l * neuron_offset_x + offset_x
                neuron_y = n * neuron_offset_y + offset_y + y_layer_offsets[l]
                pygame.draw.circle(screen, (0, 40, 69), (neuron_x, neuron_y), neuron_radius)
                pygame.draw.circle(screen, WHITE, (neuron_x, neuron_y), neuron_radius, neuron_border)
                #gfxdraw.aacircle(screen, neuron_x, neuron_y, neuron_radius, BLACK)
        
        clock.tick(4)
        pygame.display.flip()

        #SAVE AS IMAGE
        pygame.image.save(screen, 'img/neural_net_{}.jpeg'.format(counter))
        counter += 1
    pygame.quit()